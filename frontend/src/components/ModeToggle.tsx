import { Upload, Video } from "lucide-react";
import { cn } from "@/lib/utils";

interface ModeToggleProps {
  mode: "upload" | "live";
  onModeChange: (mode: "upload" | "live") => void;
}

const ModeToggle = ({ mode, onModeChange }: ModeToggleProps) => {
  return (
    <div className="glass-panel p-1 inline-flex rounded-lg">
      <button
        onClick={() => onModeChange("upload")}
        className={cn(
          "flex items-center gap-2 px-4 py-2.5 rounded-md font-body font-semibold text-sm transition-all duration-300",
          mode === "upload"
            ? "bg-neon-cyan text-background shadow-[0_0_15px_hsl(var(--neon-cyan)/0.5)]"
            : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
        )}
      >
        <Upload className="w-4 h-4" />
        <span className="hidden sm:inline">UPLOAD VIDEO</span>
      </button>
      <button
        onClick={() => onModeChange("live")}
        className={cn(
          "flex items-center gap-2 px-4 py-2.5 rounded-md font-body font-semibold text-sm transition-all duration-300",
          mode === "live"
            ? "bg-neon-magenta text-background shadow-[0_0_15px_hsl(var(--neon-magenta)/0.5)]"
            : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
        )}
      >
        <Video className="w-4 h-4" />
        <span className="hidden sm:inline">LIVE CAMERA</span>
      </button>
    </div>
  );
};

export default ModeToggle;
