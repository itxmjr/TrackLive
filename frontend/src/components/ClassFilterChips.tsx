"use client";

import { cn } from "@/lib/utils";
import { Button } from "./ui/button";

interface ClassFilterChipsProps {
  classCounts: Record<string, number>;
  classIdMap: Record<string, number>;
  enabledClasses: Set<string>;
  onToggle: (className: string) => void;
  onEnableAll: () => void;
  onDisableAll: () => void;
}

const CLASS_COLORS = [
  "hsl(187, 100%, 42%)",
  "hsl(292, 84%, 61%)",
  "hsl(45, 100%, 50%)",
  "hsl(120, 70%, 50%)",
  "hsl(200, 100%, 60%)",
  "hsl(350, 80%, 60%)",
  "hsl(270, 80%, 60%)",
  "hsl(30, 100%, 55%)",
];

const ClassFilterChips = ({
  classCounts,
  enabledClasses,
  onToggle,
  onEnableAll,
  onDisableAll,
}: ClassFilterChipsProps) => {
  const classes = Object.entries(classCounts);

  if (classes.length === 0) {
    return (
      <div className="text-xs text-muted-foreground italic py-2">
        No classes detected yet
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-[10px] text-neon-cyan hover:bg-neon-cyan/10"
          onClick={onEnableAll}
        >
          ALL
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-[10px] text-muted-foreground hover:bg-muted/50"
          onClick={onDisableAll}
        >
          NONE
        </Button>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {classes.map(([name, count], idx) => {
          const isEnabled = enabledClasses.has(name);
          const color = CLASS_COLORS[idx % CLASS_COLORS.length];

          return (
            <button
              key={name}
              onClick={() => onToggle(name)}
              className={cn(
                "flex items-center gap-1.5 px-3 sm:px-2 py-2 sm:py-1 min-h-[36px] sm:min-h-0 rounded-full border text-xs sm:text-[11px] font-medium transition-all duration-150",
                isEnabled
                  ? "chip-active-cyan text-foreground"
                  : "border-border/40 bg-muted/20 text-muted-foreground opacity-60"
              )}
            >
              <span
                className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                style={{ background: isEnabled ? color : "hsl(220 20% 40%)" }}
              />
              <span className="capitalize">{name}</span>
              <span
                className={cn(
                  "px-1 rounded text-[9px] font-bold",
                  isEnabled ? "bg-neon-cyan/20 text-neon-cyan" : "bg-muted/50 text-muted-foreground"
                )}
              >
                {count}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default ClassFilterChips;
