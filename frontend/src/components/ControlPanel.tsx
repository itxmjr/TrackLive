import { Settings, Eye, Sliders, Box, Filter } from "lucide-react";
import { Switch } from "./ui/switch";
import { Slider } from "./ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Label } from "./ui/label";
import ClassFilterChips from "./ClassFilterChips";

interface ControlPanelProps {
  showTrackHistory: boolean;
  onShowTrackHistoryChange: (value: boolean) => void;
  confidenceThreshold: number;
  onConfidenceThresholdChange: (value: number) => void;
  iouThreshold: number;
  onIouThresholdChange: (value: number) => void;
  modelSize: string;
  onModelSizeChange: (value: string) => void;
  classCounts?: Record<string, number>;
  classIdMap?: Record<string, number>;
  enabledClasses?: Set<string>;
  onClassToggle?: (className: string) => void;
  onEnableAllClasses?: () => void;
  onDisableAllClasses?: () => void;
}

const ControlPanel = ({
  showTrackHistory,
  onShowTrackHistoryChange,
  confidenceThreshold,
  onConfidenceThresholdChange,
  iouThreshold,
  onIouThresholdChange,
  modelSize,
  onModelSizeChange,
  classCounts = {},
  classIdMap = {},
  enabledClasses = new Set(),
  onClassToggle,
  onEnableAllClasses,
  onDisableAllClasses,
}: ControlPanelProps) => {
  return (
    <div className="glass-panel p-4 lg:p-5 space-y-4 lg:space-y-5 animate-slide-in-right">
      <div className="flex items-center gap-3 pb-4 border-b border-border/50">
        <div className="w-8 h-8 rounded-lg bg-neon-cyan/20 flex items-center justify-center">
          <Settings className="w-4 h-4 text-neon-cyan" />
        </div>
        <h3 className="font-display text-sm font-bold tracking-wider text-foreground">
          CONTROL PANEL
        </h3>
      </div>

      {/* Track History Toggle */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Eye className="w-4 h-4 text-neon-magenta" />
            <Label className="font-body font-semibold text-sm">Show Track History</Label>
          </div>
          <Switch
            checked={showTrackHistory}
            onCheckedChange={onShowTrackHistoryChange}
          />
        </div>
        <p className="text-xs text-muted-foreground pl-6">
          Display movement trails for tracked objects
        </p>
      </div>

      {/* Confidence Threshold */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-neon-cyan" />
            <Label className="font-body font-semibold text-sm">Confidence Threshold</Label>
          </div>
          <span className="text-sm font-mono text-neon-cyan">
            {(confidenceThreshold * 100).toFixed(0)}%
          </span>
        </div>
        <Slider
          value={[confidenceThreshold]}
          onValueChange={([value]) => onConfidenceThresholdChange(value)}
          min={0}
          max={1}
          step={0.05}
          className="py-2"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>Low</span>
          <span>High</span>
        </div>
      </div>

      {/* IOU Threshold */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-neon-magenta" />
            <Label className="font-body font-semibold text-sm">IOU Threshold</Label>
          </div>
          <span className="text-sm font-mono text-neon-magenta">
            {(iouThreshold * 100).toFixed(0)}%
          </span>
        </div>
        <Slider
          value={[iouThreshold]}
          onValueChange={([value]) => onIouThresholdChange(value)}
          min={0}
          max={1}
          step={0.05}
          className="py-2"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>Loose</span>
          <span>Strict</span>
        </div>
      </div>

      {/* Model Selection */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <Box className="w-4 h-4 text-neon-cyan" />
          <Label className="font-body font-semibold text-sm">YOLO Model</Label>
        </div>
        <Select value={modelSize} onValueChange={onModelSizeChange}>
          <SelectTrigger className="bg-muted/50 border-border/50 focus:ring-neon-cyan">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="nano">
              <div className="flex items-center justify-between w-full">
                <span>YOLOv8n (Nano)</span>
                <span className="text-xs text-muted-foreground group-focus:text-accent-foreground/70 ml-4">~3.2M params</span>
              </div>
            </SelectItem>
            <SelectItem value="small">
              <div className="flex items-center justify-between w-full">
                <span>YOLOv8s (Small)</span>
                <span className="text-xs text-muted-foreground group-focus:text-accent-foreground/70 ml-4">~11.2M params</span>
              </div>
            </SelectItem>
            <SelectItem value="medium">
              <div className="flex items-center justify-between w-full">
                <span>YOLOv8m (Medium)</span>
                <span className="text-xs text-muted-foreground group-focus:text-accent-foreground/70 ml-4">~25.9M params</span>
              </div>
            </SelectItem>
            <SelectItem value="large">
              <div className="flex items-center justify-between w-full">
                <span>YOLOv8l (Large)</span>
                <span className="text-xs text-muted-foreground group-focus:text-accent-foreground/70 ml-4">~43.7M params</span>
              </div>
            </SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Larger models are more accurate but slower
        </p>
      </div>

      {/* Class Filter */}
      <div className="space-y-3 pt-2 border-t border-border/30">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-neon-cyan" />
          <Label className="font-body font-semibold text-sm">Class Filter</Label>
        </div>
        <ClassFilterChips
          classCounts={classCounts}
          classIdMap={classIdMap}
          enabledClasses={enabledClasses}
          onToggle={onClassToggle ?? (() => {})}
          onEnableAll={onEnableAllClasses ?? (() => {})}
          onDisableAll={onDisableAllClasses ?? (() => {})}
        />
      </div>
    </div>
  );
};

export default ControlPanel;
