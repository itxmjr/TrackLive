"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Header from "@/components/Header";
import ModeToggle from "@/components/ModeToggle";
import VideoUpload from "@/components/VideoUpload";
import LiveCamera from "@/components/LiveCamera";
import ControlPanel from "@/components/ControlPanel";
import StatsSidebar from "@/components/StatsSidebar";
import FpsChart, { FpsDataPoint } from "@/components/FpsChart";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";

const queryClient = new QueryClient();
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Track {
  id: number;
  bbox: [number, number, number, number];
  label: string;
  conf: number;
  class_id?: number;
  trail?: Array<{ x: number; y: number }>;
}

const MAX_FPS_HISTORY = 60;

export default function Home() {
  const [mode, setMode] = useState<"upload" | "live">("upload");
  const [showTrackHistory, setShowTrackHistory] = useState(true);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [iouThreshold, setIouThreshold] = useState(0.45);
  const [modelSize, setModelSize] = useState("small");

  // WS status
  const [wsStatus, setWsStatus] = useState<"offline" | "connecting" | "online">("offline");

  // Tracking data
  const [tracks, setTracks] = useState<Track[]>([]);
  const [stats, setStats] = useState({
    totalTracked: 0,
    currentFps: 0,
    activeIds: [] as number[],
    processingTime: 0,
  });

  // Analytics
  const [fpsHistory, setFpsHistory] = useState<FpsDataPoint[]>([]);

  // Class filter
  const [classCounts, setClassCounts] = useState<Record<string, number>>({});
  const [classIdMap, setClassIdMap] = useState<Record<string, number>>({});
  const [enabledClasses, setEnabledClasses] = useState<Set<string>>(new Set());

  // Dwell time
  const firstSeenRef = useRef<Map<number, number>>(new Map());
  const [maxDwellMs, setMaxDwellMs] = useState(0);

  // Track which classes we've auto-enabled already
  const seenClassesRef = useRef<Set<string>>(new Set());
  // Stable ref to current tracks count for metrics callback
  const tracksCountRef = useRef(0);

  // Tripwire crossings
  const [crossingsCount, setCrossingsCount] = useState(0);

  // Debounce ref for class sync
  const classSyncTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleTrackUpdate = useCallback((newTracks: Track[]) => {
    setTracks(newTracks);

    const now = Date.now();

    // Update classCounts + classIdMap
    const newCounts: Record<string, number> = {};
    const newClassIds: Record<string, number> = {};
    newTracks.forEach((t) => {
      newCounts[t.label] = (newCounts[t.label] ?? 0) + 1;
      if (t.class_id !== undefined) newClassIds[t.label] = t.class_id;
    });
    setClassIdMap((prev) => {
      let changed = false;
      const next = { ...prev };
      Object.entries(newClassIds).forEach(([label, id]) => {
        if (next[label] === undefined) { next[label] = id; changed = true; }
      });
      return changed ? next : prev;
    });
    setClassCounts((prev) => {
      const merged = { ...prev };
      Object.entries(newCounts).forEach(([k, v]) => {
        merged[k] = Math.max(merged[k] ?? 0, v);
      });
      return merged;
    });

    // Auto-enable only first-time classes (respect user toggles for known classes)
    setEnabledClasses((prev) => {
      const next = new Set(prev);
      let changed = false;
      newTracks.forEach((t) => {
        if (!seenClassesRef.current.has(t.label)) {
          seenClassesRef.current.add(t.label);
          next.add(t.label);
          changed = true;
        }
      });
      return changed ? next : prev;
    });

    // Dwell time
    const activeIds = new Set(newTracks.map((t) => t.id));
    newTracks.forEach((t) => {
      if (!firstSeenRef.current.has(t.id)) {
        firstSeenRef.current.set(t.id, now);
      }
    });
    // Prune dead tracks to prevent unbounded growth
    for (const id of firstSeenRef.current.keys()) {
      if (!activeIds.has(id)) {
        firstSeenRef.current.delete(id);
      }
    }
    const dwellTimes = newTracks.map((t) => now - (firstSeenRef.current.get(t.id) ?? now));
    setMaxDwellMs(dwellTimes.length > 0 ? Math.max(...dwellTimes) : 0);

    // Update stable ref for metrics callback
    tracksCountRef.current = newTracks.length;

    // Stats
    setStats((prev) => {
      const maxId = newTracks.length > 0 ? Math.max(...newTracks.map((t) => t.id)) : 0;
      return {
        ...prev,
        totalTracked: Math.max(prev.totalTracked, maxId + 1),
        activeIds: newTracks.map((t) => t.id),
      };
    });
  }, []);

  const handleMetricsUpdate = useCallback(
    ({ fps, processingMs }: { fps: number; processingMs: number }) => {
      setStats((prev) => ({ ...prev, currentFps: fps, processingTime: processingMs }));
      setFpsHistory((prev) => {
        const next = [
          ...prev,
          { t: Date.now(), fps: Math.round(fps * 10) / 10, tracks: tracksCountRef.current },
        ];
        return next.length > MAX_FPS_HISTORY ? next.slice(-MAX_FPS_HISTORY) : next;
      });
    },
    [] // stable — reads tracksCountRef, not tracks state
  );

  const handleCrossing = useCallback((_trackId: number, _label: string) => {
    setCrossingsCount((n) => n + 1);
  }, []);

  const handleClassToggle = useCallback(
    (className: string) => {
      setEnabledClasses((prev) => {
        const next = new Set(prev);
        if (next.has(className)) {
          next.delete(className);
        } else {
          next.add(className);
        }
        return next;
      });
    },
    []
  );

  const handleEnableAllClasses = useCallback(() => {
    setEnabledClasses(new Set(Object.keys(classCounts)));
  }, [classCounts]);

  const handleDisableAllClasses = useCallback(() => {
    setEnabledClasses(new Set());
  }, []);

  // Sync enabled classes to backend (debounced)
  useEffect(() => {
    if (Object.keys(classCounts).length === 0) return;
    if (classSyncTimeout.current) clearTimeout(classSyncTimeout.current);
    classSyncTimeout.current = setTimeout(async () => {
      try {
        const allClasses = Object.keys(classCounts);
        const disabledClasses = allClasses.filter((c) => !enabledClasses.has(c));
        const enabledIds =
          enabledClasses.size === 0
            ? null
            : allClasses
                .filter((c) => enabledClasses.has(c))
                .map((c) => classIdMap[c])
                .filter((id) => id !== undefined);
        const disabledIsAll = disabledClasses.length === allClasses.length;
        await fetch(`${API_URL}/update-settings`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            classes: disabledIsAll ? [] : enabledIds?.length ? enabledIds : null,
          }),
        });
      } catch {
        // ignore
      }
    }, 500);
  }, [enabledClasses, classCounts, classIdMap]);

  // Sync other settings with backend
  useEffect(() => {
    const updateBackendSettings = async () => {
      try {
        await fetch(`${API_URL}/update-settings`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_name: `yolov8${
              modelSize === "nano"
                ? "n"
                : modelSize === "small"
                ? "s"
                : modelSize === "medium"
                ? "m"
                : "l"
            }.pt`,
            confidence_threshold: confidenceThreshold,
            iou_threshold: iouThreshold,
          }),
        });
      } catch {
        // Ignore settings sync failures
      }
    };
    updateBackendSettings();
  }, [modelSize, confidenceThreshold, iouThreshold]);

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-background cyber-grid">
        <Header wsStatus={wsStatus} />

        <main className="container mx-auto px-4 py-6">
          {/* Mode Toggle */}
          <div className="flex justify-center mb-6">
            <ModeToggle mode={mode} onModeChange={setMode} />
          </div>

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
            {/* Main View */}
            <div className="md:col-span-7 xl:col-span-8 space-y-6">
              {mode === "upload" ? (
                <VideoUpload apiUrl={API_URL} />
              ) : (
                <LiveCamera
                  tracks={tracks}
                  showTrackHistory={showTrackHistory}
                  enabledClasses={enabledClasses}
                  onTrackUpdate={handleTrackUpdate}
                  onMetricsUpdate={handleMetricsUpdate}
                  onWsStatusChange={setWsStatus}
                  onCrossing={handleCrossing}
                  apiUrl={API_URL}
                />
              )}
            </div>

            {/* Sidebar with Tabs */}
            <div className="md:col-span-5 xl:col-span-4 content-start">
              <Tabs defaultValue="controls" className="w-full">
                <TabsList className="w-full mb-4 bg-card/60 border border-border/50">
                  <TabsTrigger
                    value="controls"
                    className="flex-1 font-display text-xs tracking-wider data-[state=active]:text-neon-cyan"
                  >
                    CONTROLS
                  </TabsTrigger>
                  <TabsTrigger
                    value="analytics"
                    className="flex-1 font-display text-xs tracking-wider data-[state=active]:text-neon-cyan"
                  >
                    ANALYTICS
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="controls" className="space-y-4 mt-0">
                  <ControlPanel
                    showTrackHistory={showTrackHistory}
                    onShowTrackHistoryChange={setShowTrackHistory}
                    confidenceThreshold={confidenceThreshold}
                    onConfidenceThresholdChange={setConfidenceThreshold}
                    iouThreshold={iouThreshold}
                    onIouThresholdChange={setIouThreshold}
                    modelSize={modelSize}
                    onModelSizeChange={setModelSize}
                    classCounts={classCounts}
                    classIdMap={classIdMap}
                    enabledClasses={enabledClasses}
                    onClassToggle={handleClassToggle}
                    onEnableAllClasses={handleEnableAllClasses}
                    onDisableAllClasses={handleDisableAllClasses}
                  />
                </TabsContent>

                <TabsContent value="analytics" className="space-y-4 mt-0">
                  <StatsSidebar
                    totalTracked={stats.totalTracked}
                    currentFps={stats.currentFps}
                    activeIds={stats.activeIds}
                    processingTime={stats.processingTime}
                    crossingsCount={crossingsCount}
                    maxDwellMs={maxDwellMs}
                  />
                  <FpsChart data={fpsHistory} />
                </TabsContent>
              </Tabs>
            </div>
          </div>

          {/* Footer */}
          <footer className="mt-12 text-center text-sm text-muted-foreground font-body">
            <p>TRACKLIVE • Real-Time Object Detection & Tracking</p>
            <p className="mt-1 text-xs">
              Powered by YOLOv8 + SORT • WebSocket Pipeline
            </p>
          </footer>
        </main>
      </div>
    </QueryClientProvider>
  );
}
