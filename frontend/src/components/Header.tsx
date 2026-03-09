import { Activity } from "lucide-react";
import { cn } from "@/lib/utils";

interface HeaderProps {
  wsStatus?: "offline" | "connecting" | "online";
}

const Header = ({ wsStatus = "offline" }: HeaderProps) => {
  const dotColor =
    wsStatus === "online"
      ? "bg-neon-cyan"
      : wsStatus === "connecting"
      ? "bg-yellow-400"
      : "bg-destructive";

  const label =
    wsStatus === "online"
      ? "WS LIVE"
      : wsStatus === "connecting"
      ? "CONNECTING"
      : "WS OFFLINE";

  return (
    <header className="glass-panel border-b border-border/30 px-4 sm:px-6 py-3 sm:py-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg bg-gradient-to-br from-neon-cyan to-neon-magenta flex items-center justify-center animate-glow-pulse">
              <Activity className="w-5 h-5 sm:w-6 sm:h-6 text-background" />
            </div>
          </div>
          <div>
            <h1 className="font-display text-xl font-bold tracking-wider neon-text-cyan">
              TRACKLIVE
            </h1>
            <p className="hidden sm:block text-xs text-muted-foreground font-body tracking-wide">
              REAL-TIME OBJECT DETECTION SYSTEM
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-muted/50 border border-border/50">
            <div className={cn("w-2 h-2 rounded-full", dotColor, wsStatus !== "offline" && "animate-pulse")} />
            <span className="text-xs font-body text-muted-foreground">{label}</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
