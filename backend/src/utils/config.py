"""
Configuration settings for object detection and tracking.

Centralizes all parameters to ensure a single source of truth,
simplify hyperparameter tuning, and improve reproducibility.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DetectorConfig:
    """YOLO detector configuration."""
    model_name: str = "yolov8n.pt"  # n=nano, s=small, m=medium, l=large, x=xlarge
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45  # For NMS (Non-Max Suppression)
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    classes: tuple[int, ...] | None = None  # None = all classes, (0,) = persons only


@dataclass(frozen=True)
class TrackerConfig:
    """SORT tracker configuration."""
    max_age: int = 30  # Frames to keep track alive without detection
    min_hits: int = 3  # Minimum detections before track is confirmed
    iou_threshold: float = 0.3  # Minimum IoU for matching


@dataclass(frozen=True)
class VideoConfig:
    """Video processing configuration."""
    output_fps: int = 30
    display_scale: float = 1.0  # Resize display window
    save_output: bool = False
    output_dir: Path = Path("outputs")


@dataclass
class Config:
    """Main configuration container."""
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    video: VideoConfig = VideoConfig()


# Default configuration instance
default_config = Config()