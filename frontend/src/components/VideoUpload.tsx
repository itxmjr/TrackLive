"use client";
import { useState, useCallback } from "react";
import { Upload, FileVideo, Play, RotateCcw } from "lucide-react";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { cn } from "@/lib/utils";

interface VideoStats {
  unique_track_ids: number;
  avg_fps: number;
  total_frames: number;
  class_counts: Record<string, number>;
}

interface VideoUploadProps {
  apiUrl: string;
}

const VideoUpload = ({ apiUrl }: VideoUploadProps) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessed, setIsProcessed] = useState(false);
  const [processedUrl, setProcessedUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [stats, setStats] = useState<VideoStats | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleFileSelect = useCallback(async (selectedFile: File) => {
    setFile(selectedFile);
    setIsUploading(true);
    setUploadProgress(0);
    setStats(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${apiUrl}/process-video`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        const taskId = data.task_id;
        setTaskId(taskId);
        const outputUrl = data.output_url;

        // Poll for completion
        const pollInterval = setInterval(async () => {
          try {
            const statusResponse = await fetch(`${apiUrl}/task-status/${taskId}`);
            if (statusResponse.ok) {
              const statusData = await statusResponse.json();
              const taskInfo = statusData.status; // Now it's a dict

              if (taskInfo.status === "completed") {
                clearInterval(pollInterval);
                setUploadProgress(100);
                setIsUploading(false);
                setIsProcessed(true);
                setStats(taskInfo.stats);
                setProcessedUrl(`${apiUrl}${outputUrl}?t=${Date.now()}`);
              } else if (taskInfo.status.startsWith("failed")) {
                clearInterval(pollInterval);
                setError(`Processing failed: ${taskInfo.status}`);
                setIsUploading(false);
              } else if (taskInfo.status === "processing") {
                setUploadProgress(taskInfo.progress || 0);
                if (taskInfo.stats) setStats(taskInfo.stats);
              }
            }
          } catch {
            // Polling may fail transiently
          }
        }, 1000);

      } else {
        setError("Upload failed");
        setIsUploading(false);
      }
    } catch {
      setError("Network error");
      setIsUploading(false);
    }
  }, [apiUrl]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith("video/")) {
      handleFileSelect(droppedFile);
    }
  }, [handleFileSelect]);

  const handleReset = () => {
    setFile(null);
    setUploadProgress(0);
    setIsUploading(false);
    setIsProcessed(false);
    setProcessedUrl(null);
    setError(null);
    setStats(null);
    setTaskId(null);
  };

  if (isProcessed && file) {
    return (
      <div className="glass-panel p-6 animate-fade-in space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-neon-cyan/20 flex items-center justify-center">
              <FileVideo className="w-5 h-5 text-neon-cyan" />
            </div>
            <div>
              <p className="font-body font-semibold text-foreground">{file.name}</p>
              <p className="text-xs text-muted-foreground">
                {(file.size / (1024 * 1024)).toFixed(2)} MB • Processed
              </p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={handleReset} className="hover:bg-neon-cyan/10">
            <RotateCcw className="w-4 h-4" />
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2 space-y-4">
            <div className="relative aspect-video rounded-lg overflow-hidden bg-background/50 border border-border/50 group">
              <video
                src={processedUrl || URL.createObjectURL(file)}
                className="w-full h-full object-contain"
                controls
                autoPlay
                key={processedUrl}
              />
              <div className="absolute top-4 right-4 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <Button variant="glass" size="sm" className="text-xs h-7">ORIGINAL</Button>
                <Button variant="neon" size="sm" className="text-xs h-7">TRACKED</Button>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="glass-panel p-4 bg-background/40 border-border/30 h-full flex flex-col">
              <h3 className="font-display text-sm font-bold tracking-wider text-neon-cyan mb-4 flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-neon-cyan animate-pulse" />
                DETECTION ANALYSIS
              </h3>

              {stats ? (
                <div className="space-y-4 flex-1">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 rounded bg-muted/30 border border-border/20">
                      <p className="text-[10px] text-muted-foreground uppercase">Unique Objects</p>
                      <p className="text-xl font-display font-bold text-foreground">{stats.unique_track_ids}</p>
                    </div>
                    <div className="p-3 rounded bg-muted/30 border border-border/20">
                      <p className="text-[10px] text-muted-foreground uppercase">Avg Perf</p>
                      <p className="text-xl font-display font-bold text-foreground">{Math.round(stats.avg_fps)} <span className="text-xs text-muted-foreground">FPS</span></p>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-widest">Class Breakdown</p>
                    <div className="max-h-[200px] overflow-y-auto pr-2 custom-scrollbar space-y-2">
                      {(() => {
                        const entries = Object.entries(stats.class_counts || {});
                        const total = entries.reduce((s, [, c]) => s + c, 0);
                        return entries.map(([name, count]) => (
                          <div key={name} className="space-y-1">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-medium text-foreground capitalize">{name}</span>
                              <span className="text-[10px] text-neon-cyan font-bold">{count}</span>
                            </div>
                            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                              <div
                                className="h-full bg-gradient-to-r from-neon-cyan to-neon-magenta rounded-full transition-all"
                                style={{ width: `${total > 0 ? (count / total) * 100 : 0}%` }}
                              />
                            </div>
                          </div>
                        ));
                      })()}
                    </div>
                  </div>

                  <div className="mt-auto pt-4 border-t border-border/20">
                    <div className="flex justify-between text-[10px] text-muted-foreground mb-1">
                      <span>TOTAL FRAMES</span>
                      <span>{stats.total_frames}</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center text-center opacity-50">
                  <Play className="w-8 h-8 mb-2 text-muted-foreground animate-pulse" />
                  <p className="text-xs">Gathering analysis data...</p>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="flex gap-3">
          <Button
            variant="cyber"
            className="flex-1 shadow-[0_0_15px_rgba(5,213,250,0.2)]"
            disabled={!taskId}
            onClick={() => {
              if (taskId) {
                const a = document.createElement("a");
                a.href = `${apiUrl}/export/tracks/${encodeURIComponent(taskId)}?format=csv`;
                a.download = `tracks_${taskId}.csv`;
                a.click();
              }
            }}
          >
            <Play className="w-4 h-4 mr-2" />
            DOWNLOAD REPORT
          </Button>
          <Button variant="outline" onClick={handleReset} className="px-8">
            UPLOAD NEW
          </Button>
        </div>
      </div>
    );
  }


  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        "glass-panel p-8 border-2 border-dashed transition-all duration-300 cursor-pointer group",
        isDragOver
          ? "border-neon-cyan bg-neon-cyan/5 shadow-[0_0_30px_hsl(var(--neon-cyan)/0.3)]"
          : "border-border/50 hover:border-neon-cyan/50 hover:bg-muted/20"
      )}
    >
      {isUploading ? (
        <div className="text-center animate-fade-in">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-neon-cyan/20 flex items-center justify-center">
            <FileVideo className="w-8 h-8 text-neon-cyan animate-pulse" />
          </div>
          <p className="font-body font-semibold text-foreground mb-2">
            Processing: {file?.name}
          </p>
          <Progress value={uploadProgress} className="h-2 mb-2" />
          <p className="text-sm text-muted-foreground">
            {Math.round(uploadProgress)}% Complete
          </p>
        </div>
      ) : error ? (
        <div className="text-center animate-fade-in">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-destructive/20 flex items-center justify-center">
            <RotateCcw className="w-8 h-8 text-destructive" />
          </div>
          <p className="font-body font-semibold text-destructive mb-2">
            Error: {error}
          </p>
          <Button variant="outline" size="sm" onClick={handleReset}>
            TRY AGAIN
          </Button>
        </div>
      ) : (
        <div className="text-center">
          <div className={cn(
            "w-20 h-20 mx-auto mb-4 rounded-full flex items-center justify-center transition-all duration-300",
            isDragOver
              ? "bg-neon-cyan/30 shadow-[0_0_20px_hsl(var(--neon-cyan)/0.5)]"
              : "bg-muted/50 group-hover:bg-neon-cyan/20"
          )}>
            <Upload className={cn(
              "w-10 h-10 transition-all duration-300",
              isDragOver ? "text-neon-cyan scale-110" : "text-muted-foreground group-hover:text-neon-cyan"
            )} />
          </div>
          <h3 className="font-display text-lg font-semibold text-foreground mb-2">
            DROP VIDEO FILE HERE
          </h3>
          <p className="text-muted-foreground font-body mb-4">
            or click to browse your files
          </p>
          <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
            <span className="px-2 py-1 rounded bg-muted/50">MP4</span>
            <span className="px-2 py-1 rounded bg-muted/50">AVI</span>
            <span className="px-2 py-1 rounded bg-muted/50">MOV</span>
            <span className="px-2 py-1 rounded bg-muted/50">WEBM</span>
          </div>
          <input
            type="file"
            accept="video/*"
            className="absolute inset-0 opacity-0 cursor-pointer"
            onChange={(e) => {
              const selectedFile = e.target.files?.[0];
              if (selectedFile) handleFileSelect(selectedFile);
            }}
          />
        </div>
      )}
    </div>
  );
};

export default VideoUpload;
