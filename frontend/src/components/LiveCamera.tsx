"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Camera, CameraOff, Maximize2, Minimize2, Download } from "lucide-react";
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";
import TripwireLayer from "./TripwireLayer";

interface Track {
  id: number;
  bbox: [number, number, number, number];
  label: string;
  conf: number;
  class_id?: number;
  trail?: Array<{ x: number; y: number }>;
}

interface LiveCameraProps {
  tracks: Track[];
  showTrackHistory: boolean;
  enabledClasses?: Set<string>;
  onTrackUpdate?: (tracks: Track[]) => void;
  onMetricsUpdate?: (metrics: { fps: number; processingMs: number }) => void;
  onWsStatusChange?: (status: "offline" | "connecting" | "online") => void;
  onCrossing?: (trackId: number, label: string) => void;
  apiUrl: string;
}

const TRACK_COLORS = [
  "hsl(187, 100%, 42%)",
  "hsl(292, 84%, 61%)",
  "hsl(45, 100%, 50%)",
  "hsl(120, 70%, 50%)",
  "hsl(200, 100%, 60%)",
  "hsl(350, 80%, 60%)",
  "hsl(270, 80%, 60%)",
  "hsl(30, 100%, 55%)",
];

const LiveCamera = ({
  tracks,
  showTrackHistory,
  enabledClasses,
  onTrackUpdate,
  onMetricsUpdate,
  onWsStatusChange,
  onCrossing,
  apiUrl,
}: LiveCameraProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const animationRef = useRef<number | null>(null);
  const overlayAnimRef = useRef<number | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const lastProcessingTimeRef = useRef<number>(Date.now());
  const sendTimeRef = useRef<number>(0);
  const firstMessageRef = useRef<boolean>(false);

  const getTrackColor = (id: number) => TRACK_COLORS[id % TRACK_COLORS.length];

  const startCamera = async () => {
    setHasPermission(null);

    // Close existing WS to prevent orphaned connections
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "environment" },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try {
          await videoRef.current.play();
        } catch {
          // Video play may fail if element is detached
        }

        setIsStreaming(true);
        setHasPermission(true);
        firstMessageRef.current = false;

        const urlToUse = apiUrl || window.location.origin;
        const wsUrl = urlToUse.startsWith("https")
          ? urlToUse.replace("https://", "wss://")
          : urlToUse.replace("http://", "ws://");

        onWsStatusChange?.("connecting");
        const ws = new WebSocket(`${wsUrl}/ws/track`);

        ws.onopen = () => {
          // Connection established
        };

        ws.onmessage = (event) => {
          const roundTrip = performance.now() - sendTimeRef.current;
          const fps = sendTimeRef.current > 0 ? 1000 / roundTrip : 0;

          if (!firstMessageRef.current) {
            firstMessageRef.current = true;
            onWsStatusChange?.("online");
          }

          onMetricsUpdate?.({ fps, processingMs: roundTrip });

          try {
            const data = JSON.parse(event.data);
            if (data.tracks && onTrackUpdate) {
              onTrackUpdate(data.tracks);
            }
          } catch {
            // Ignore malformed JSON from server
          }
        };

        ws.onerror = () => {
          onWsStatusChange?.("offline");
        };

        ws.onclose = () => {
          onWsStatusChange?.("offline");
        };

        wsRef.current = ws;
      }
    } catch {
      setHasPermission(false);
      onWsStatusChange?.("offline");
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);

      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      onWsStatusChange?.("offline");
    }
  };

  const takeSnapshot = useCallback(() => {
    const video = videoRef.current;
    const overlayCanvas = canvasRef.current;
    if (!video || !overlayCanvas) return;

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth || 640;
    tempCanvas.height = video.videoHeight || 480;
    const ctx = tempCanvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    ctx.drawImage(overlayCanvas, 0, 0, tempCanvas.width, tempCanvas.height);

    tempCanvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `snapshot_${Date.now()}.png`;
      a.click();
      URL.revokeObjectURL(url);
    }, "image/png");
  }, []);

  const processFrame = useCallback(() => {
    const video = videoRef.current;
    if (!video || !isStreaming || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    const now = Date.now();
    if (now - lastProcessingTimeRef.current < 100) {
      animationRef.current = requestAnimationFrame(processFrame);
      return;
    }
    lastProcessingTimeRef.current = now;

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth || 640;
    tempCanvas.height = video.videoHeight || 480;
    const ctx = tempCanvas.getContext("2d");
    if (ctx) {
      ctx.drawImage(video, 0, 0);
      sendTimeRef.current = performance.now();
      const dataUrl = tempCanvas.toDataURL("image/jpeg", 0.6);
      try {
        wsRef.current.send(dataUrl);
      } catch {
        // WS closed between readyState check and send
      }
    }

    animationRef.current = requestAnimationFrame(processFrame);
  }, [isStreaming]);

  const tracksRef = useRef<Track[]>(tracks);
  const showTrackHistoryRef = useRef(showTrackHistory);
  const enabledClassesRef = useRef(enabledClasses);
  tracksRef.current = tracks;
  showTrackHistoryRef.current = showTrackHistory;
  enabledClassesRef.current = enabledClasses;

  const drawOverlays = useCallback(function draw() {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / (video.videoWidth || 640);
    const scaleY = canvas.height / (video.videoHeight || 480);

    const currentTracks = tracksRef.current;
    const currentEnabled = enabledClassesRef.current;

    // Filter by enabled classes
    const visibleTracks = currentEnabled && currentEnabled.size > 0
      ? currentTracks.filter((t) => currentEnabled.has(t.label))
      : currentTracks;

    visibleTracks.forEach((track) => {
      const [x1, y1, x2, y2] = track.bbox.map((val, i) =>
        i % 2 === 0 ? val * scaleX : val * scaleY
      );
      const color = getTrackColor(track.id);
      const width = x2 - x1;
      const height = y2 - y1;

      // Gradient trail
      if (showTrackHistoryRef.current && track.trail && track.trail.length > 1) {
        for (let i = 1; i < track.trail.length; i++) {
          const alpha = (i / track.trail.length) * 0.8;
          ctx.beginPath();
          ctx.strokeStyle = color;
          ctx.globalAlpha = alpha;
          ctx.lineWidth = 2;
          ctx.moveTo(track.trail[i - 1].x * scaleX, track.trail[i - 1].y * scaleY);
          ctx.lineTo(track.trail[i].x * scaleX, track.trail[i].y * scaleY);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }

      // Bounding box with glow
      ctx.shadowColor = color;
      ctx.shadowBlur = 15;
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, width, height);

      // Corner accents
      const cornerSize = 15;
      ctx.lineWidth = 4;

      ctx.beginPath();
      ctx.moveTo(x1, y1 + cornerSize);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x1 + cornerSize, y1);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(x2 - cornerSize, y1);
      ctx.lineTo(x2, y1);
      ctx.lineTo(x2, y1 + cornerSize);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(x1, y2 - cornerSize);
      ctx.lineTo(x1, y2);
      ctx.lineTo(x1 + cornerSize, y2);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(x2 - cornerSize, y2);
      ctx.lineTo(x2, y2);
      ctx.lineTo(x2, y2 - cornerSize);
      ctx.stroke();

      ctx.shadowBlur = 0;

      // Label
      const labelValue = `${track.label.toUpperCase()} [ID: ${track.id}] - ${Math.round((track.conf || 0) * 100)}%`;
      ctx.font = "bold 14px Rajdhani";
      const labelWidth = ctx.measureText(labelValue).width + 16;
      const labelHeight = 24;

      ctx.fillStyle = color;
      ctx.globalAlpha = 0.9;
      ctx.fillRect(x1, y1 - labelHeight - 4, labelWidth, labelHeight);
      ctx.globalAlpha = 1;

      ctx.fillStyle = "#0a0a0f";
      ctx.fillText(labelValue, x1 + 8, y1 - 10);
    });

    overlayAnimRef.current = requestAnimationFrame(draw);
  }, []);

  // processFrame loop
  useEffect(() => {
    if (isStreaming) {
      animationRef.current = requestAnimationFrame(processFrame);
    }
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    };
  }, [isStreaming, processFrame]);

  // drawOverlays loop (separate to avoid spawning multiple chains)
  useEffect(() => {
    if (isStreaming) {
      overlayAnimRef.current = requestAnimationFrame(drawOverlays);
    }
    return () => {
      if (overlayAnimRef.current) {
        cancelAnimationFrame(overlayAnimRef.current);
        overlayAnimRef.current = null;
      }
    };
  }, [isStreaming, drawOverlays]);

  // Build tripwire track points from visible tracks
  const tripwireTracks = tracks
    .filter((t) => !enabledClasses || enabledClasses.size === 0 || enabledClasses.has(t.label))
    .map((t) => ({
      id: t.id,
      label: t.label,
      cx: (t.bbox[0] + t.bbox[2]) / 2 / (videoRef.current?.videoWidth || 640),
      cy: (t.bbox[1] + t.bbox[3]) / 2 / (videoRef.current?.videoHeight || 480),
    }));

  return (
    <div
      className={cn(
        "glass-panel overflow-hidden transition-all duration-300",
        isFullscreen && "fixed inset-4 z-50"
      )}
    >
      <div className="relative aspect-video bg-background/80">
        {!isStreaming ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div
              className={cn(
                "w-24 h-24 rounded-full flex items-center justify-center mb-6 transition-all duration-300",
                hasPermission === false
                  ? "bg-destructive/20"
                  : "bg-neon-cyan/20 animate-pulse-glow"
              )}
            >
              {hasPermission === false ? (
                <CameraOff className="w-12 h-12 text-destructive" />
              ) : (
                <Camera className="w-12 h-12 text-neon-cyan" />
              )}
            </div>
            <h3 className="font-display text-xl font-semibold text-foreground mb-2">
              {hasPermission === false ? "CAMERA ACCESS DENIED" : "INITIALIZE CAMERA"}
            </h3>
            <p className="text-muted-foreground font-body mb-6 text-center max-w-md">
              {hasPermission === false
                ? "Please enable camera permissions in your browser settings"
                : "Click below to activate real-time object detection"}
            </p>
            <Button
              variant="neon"
              size="lg"
              onClick={startCamera}
            >
              <Camera className="w-5 h-5 mr-2" />
              START LIVE FEED
            </Button>
          </div>
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
              style={{ zIndex: 5 }}
            />

            {/* Tripwire layer — positioned on top, receives pointer events */}
            <TripwireLayer
              tracks={tripwireTracks}
              onCrossing={onCrossing ?? (() => {})}
            />

            {/* Scanline effect */}
            <div className="absolute inset-0 pointer-events-none opacity-20" style={{ zIndex: 1 }}>
              <div className="absolute inset-0 bg-gradient-to-b from-transparent via-neon-cyan/5 to-transparent h-[200%] animate-scan" />
            </div>

            {/* Controls overlay */}
            <div className="absolute top-4 right-4 flex gap-2" style={{ zIndex: 20 }}>
              <Button
                variant="glass"
                size="icon"
                onClick={takeSnapshot}
                title="Save snapshot"
              >
                <Download className="w-4 h-4" />
              </Button>
              <Button
                variant="glass"
                size="icon"
                onClick={() => setIsFullscreen(!isFullscreen)}
              >
                {isFullscreen ? (
                  <Minimize2 className="w-4 h-4" />
                ) : (
                  <Maximize2 className="w-4 h-4" />
                )}
              </Button>
              <Button variant="glass" size="icon" onClick={stopCamera}>
                <CameraOff className="w-4 h-4" />
              </Button>
            </div>

            {/* Live indicator */}
            <div
              className="absolute top-4 left-4 flex items-center gap-2 px-3 py-1.5 rounded-full bg-destructive/90 backdrop-blur-sm"
              style={{ zIndex: 20 }}
            >
              <div className="w-2 h-2 rounded-full bg-foreground animate-pulse" />
              <span className="text-xs font-body font-bold text-foreground">LIVE</span>
            </div>

            {/* Tripwire hint */}
            <div
              className="absolute bottom-4 left-4 text-[10px] text-muted-foreground/70 font-body"
              style={{ zIndex: 20 }}
            >
              DRAG to draw tripwire • DBLCLICK to clear
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default LiveCamera;
