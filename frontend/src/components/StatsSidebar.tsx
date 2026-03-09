import { Activity, Gauge, Users, TrendingUp, Clock, AlertTriangle, Timer } from "lucide-react";
import { cn } from "@/lib/utils";

interface StatsSidebarProps {
  totalTracked: number;
  currentFps: number;
  activeIds: number[];
  processingTime: number;
  crossingsCount?: number;
  maxDwellMs?: number;
}

const StatsSidebar = ({
  totalTracked,
  currentFps,
  activeIds,
  processingTime,
  crossingsCount = 0,
  maxDwellMs = 0,
}: StatsSidebarProps) => {
  const fpsStatus = currentFps >= 25 ? "optimal" : currentFps >= 15 ? "good" : "low";

  const formatDwell = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  return (
    <div className="glass-panel p-4 lg:p-5 space-y-4 lg:space-y-5 animate-fade-in">
      <div className="flex items-center gap-3 pb-4 border-b border-border/50">
        <div className="w-8 h-8 rounded-lg bg-neon-magenta/20 flex items-center justify-center">
          <Activity className="w-4 h-4 text-neon-magenta" />
        </div>
        <h3 className="font-display text-sm font-bold tracking-wider text-foreground">
          LIVE STATISTICS
        </h3>
      </div>

      {/* Total Objects Tracked */}
      <div className="glass-panel p-3 lg:p-4 neon-border-cyan">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-neon-cyan" />
            <span className="text-xs font-body text-muted-foreground uppercase tracking-wider">
              Total Tracked
            </span>
          </div>
        </div>
        <p className="text-2xl lg:text-3xl font-display font-bold neon-text-cyan">
          {totalTracked.toLocaleString()}
        </p>
      </div>

      {/* Current FPS */}
      <div
        className={cn(
          "glass-panel p-3 lg:p-4 border transition-all duration-300",
          fpsStatus === "optimal" && "neon-border-cyan",
          fpsStatus === "good" && "border-yellow-500/50 shadow-[0_0_15px_hsl(45,100%,50%,0.3)]",
          fpsStatus === "low" && "border-destructive/50 shadow-[0_0_15px_hsl(0,84%,60%,0.3)]"
        )}
      >
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <Gauge className="w-4 h-4 text-neon-magenta" />
            <span className="text-xs font-body text-muted-foreground uppercase tracking-wider">
              Current FPS
            </span>
          </div>
          <span
            className={cn(
              "text-xs px-2 py-0.5 rounded-full font-semibold",
              fpsStatus === "optimal" && "bg-neon-cyan/20 text-neon-cyan",
              fpsStatus === "good" && "bg-yellow-500/20 text-yellow-400",
              fpsStatus === "low" && "bg-destructive/20 text-destructive"
            )}
          >
            {fpsStatus.toUpperCase()}
          </span>
        </div>
        <div className="flex items-baseline gap-2">
          <p
            className={cn(
              "text-2xl lg:text-3xl font-display font-bold",
              fpsStatus === "optimal" && "neon-text-cyan",
              fpsStatus === "good" && "text-yellow-400",
              fpsStatus === "low" && "text-destructive"
            )}
          >
            {currentFps.toFixed(1)}
          </p>
          <span className="text-sm text-muted-foreground">fps</span>
        </div>

        {/* FPS Bar */}
        <div className="mt-2 h-1.5 lg:h-2 bg-muted rounded-full overflow-hidden">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-300",
              fpsStatus === "optimal" && "bg-gradient-to-r from-neon-cyan to-neon-cyan/50",
              fpsStatus === "good" && "bg-gradient-to-r from-yellow-500 to-yellow-500/50",
              fpsStatus === "low" && "bg-gradient-to-r from-destructive to-destructive/50"
            )}
            style={{ width: `${Math.min((currentFps / 60) * 100, 100)}%` }}
          />
        </div>
      </div>

      {/* Active IDs & Processing Time Row */}
      <div className="grid grid-cols-2 gap-3">
        {/* Active IDs */}
        <div className="glass-panel p-3 neon-border-magenta">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5">
              <Users className="w-3.5 h-3.5 text-neon-magenta" />
              <span className="text-[10px] font-body text-muted-foreground uppercase tracking-wider">
                Active IDs
              </span>
            </div>
          </div>
          <span className="text-xl lg:text-2xl font-display font-bold neon-text-magenta">
            {activeIds.length}
          </span>
        </div>

        {/* Processing Time */}
        <div className="glass-panel p-3">
          <div className="flex items-center gap-1.5 mb-2">
            <Clock className="w-3.5 h-3.5 text-muted-foreground" />
            <span className="text-[10px] font-body text-muted-foreground uppercase tracking-wider">
              Inference
            </span>
          </div>
          <div className="flex items-baseline gap-1">
            <p className="text-xl lg:text-2xl font-display font-bold text-foreground">
              {processingTime.toFixed(1)}
            </p>
            <span className="text-xs text-muted-foreground">ms</span>
          </div>
        </div>
      </div>

      {/* Tripwire Crossings + Dwell Time Row */}
      <div className="grid grid-cols-2 gap-3">
        {/* Crossings */}
        <div className="glass-panel p-3 neon-border-amber">
          <div className="flex items-center gap-1.5 mb-2">
            <AlertTriangle className="w-3.5 h-3.5 text-neon-amber" />
            <span className="text-[10px] font-body text-muted-foreground uppercase tracking-wider">
              Crossings
            </span>
          </div>
          <span className="text-xl lg:text-2xl font-display font-bold neon-text-amber">
            {crossingsCount}
          </span>
        </div>

        {/* Max Dwell Time */}
        <div className="glass-panel p-3 neon-border-amber">
          <div className="flex items-center gap-1.5 mb-2">
            <Timer className="w-3.5 h-3.5 text-neon-amber" />
            <span className="text-[10px] font-body text-muted-foreground uppercase tracking-wider">
              Max Dwell
            </span>
          </div>
          <span className="text-xl lg:text-2xl font-display font-bold neon-text-amber">
            {formatDwell(maxDwellMs)}
          </span>
        </div>
      </div>
    </div>
  );
};

export default StatsSidebar;
