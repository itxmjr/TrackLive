"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import { toast } from "sonner";

interface TrackPoint {
  id: number;
  label: string;
  cx: number;
  cy: number;
}

interface Tripwire {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface TripwireLayerProps {
  tracks: TrackPoint[];
  onCrossing: (trackId: number, label: string) => void;
}

function segmentsIntersect(
  ax: number, ay: number, bx: number, by: number,
  cx: number, cy: number, dx: number, dy: number
): boolean {
  const d1x = bx - ax, d1y = by - ay;
  const d2x = dx - cx, d2y = dy - cy;
  const cross = d1x * d2y - d1y * d2x;
  if (Math.abs(cross) < 1e-10) return false;
  const t = ((cx - ax) * d2y - (cy - ay) * d2x) / cross;
  const u = ((cx - ax) * d1y - (cy - ay) * d1x) / cross;
  return t >= 0 && t <= 1 && u >= 0 && u <= 1;
}

const TripwireLayer = ({ tracks, onCrossing }: TripwireLayerProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tripwire, setTripwire] = useState<Tripwire | null>(null);
  const [isFlashing, setIsFlashing] = useState(false);
  const drawStart = useRef<{ x: number; y: number } | null>(null);
  const prevCenters = useRef<Map<number, { x: number; y: number }>>(new Map());
  const flashTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync canvas dimensions with parent via ResizeObserver
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (canvas.width !== width || canvas.height !== height) {
          canvas.width = width;
          canvas.height = height;
        }
      }
    });
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  // Detect crossings on each tracks update
  useEffect(() => {
    if (!tripwire) return;

    tracks.forEach((track) => {
      const prev = prevCenters.current.get(track.id);
      if (prev) {
        if (
          segmentsIntersect(
            prev.x, prev.y, track.cx, track.cy,
            tripwire.x1, tripwire.y1, tripwire.x2, tripwire.y2
          )
        ) {
          onCrossing(track.id, track.label);
          toast(`${track.label.toUpperCase()} [ID:${track.id}] crossed tripwire`, {
            duration: 2500,
            style: {
              background: "hsl(220 25% 8%)",
              border: "1px solid hsl(45 100% 50% / 0.6)",
              color: "hsl(200 100% 95%)",
            },
          });
          // Flash tripwire amber
          if (flashTimeout.current) clearTimeout(flashTimeout.current);
          setIsFlashing(true);
          flashTimeout.current = setTimeout(() => setIsFlashing(false), 500);
        }
      }
    });

    // Update prev centers
    const newMap = new Map<number, { x: number; y: number }>();
    tracks.forEach((t) => newMap.set(t.id, { x: t.cx, y: t.cy }));
    prevCenters.current = newMap;
  }, [tracks, tripwire, onCrossing]);

  // Draw tripwire
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!tripwire) return;

    const { x1, y1, x2, y2 } = tripwire;
    const w = canvas.width;
    const h = canvas.height;

    const px1 = x1 * w, py1 = y1 * h;
    const px2 = x2 * w, py2 = y2 * h;

    const color = isFlashing ? "hsl(45, 100%, 50%)" : "hsl(187, 100%, 42%)";

    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 4]);
    ctx.beginPath();
    ctx.moveTo(px1, py1);
    ctx.lineTo(px2, py2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Direction tick marks (perpendicular lines)
    const len = Math.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2);
    if (len > 0) {
      const nx = -(py2 - py1) / len;
      const ny = (px2 - px1) / len;
      const numTicks = Math.floor(len / 30);
      for (let i = 1; i <= numTicks; i++) {
        const t = i / (numTicks + 1);
        const mx = px1 + t * (px2 - px1);
        const my = py1 + t * (py2 - py1);
        ctx.beginPath();
        ctx.moveTo(mx, my);
        ctx.lineTo(mx + nx * 8, my + ny * 8);
        ctx.stroke();
      }
    }

    ctx.restore();
  }, [tripwire, isFlashing]);

  const getNormalizedCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) / rect.width,
      y: (e.clientY - rect.top) / rect.height,
    };
  };

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    drawStart.current = getNormalizedCoords(e);
  }, []);

  const handleMouseUp = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawStart.current) return;
    const end = getNormalizedCoords(e);
    const start = drawStart.current;
    drawStart.current = null;
    const dx = end.x - start.x, dy = end.y - start.y;
    if (Math.sqrt(dx * dx + dy * dy) < 0.02) return; // Too short
    setTripwire({ x1: start.x, y1: start.y, x2: end.x, y2: end.y });
    prevCenters.current.clear();
  }, []);

  const handleDoubleClick = useCallback(() => {
    setTripwire(null);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full"
      style={{ cursor: tripwire ? "default" : "crosshair", zIndex: 10 }}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onDoubleClick={handleDoubleClick}
    />
  );
};

export default TripwireLayer;
