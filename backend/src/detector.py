"""
YOLO-based object detection module.

Uses ultralytics YOLOv8 for state-of-the-art detection.
Outputs standardized detection format for tracker consumption.
"""

import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO

from .utils.config import DetectorConfig


@dataclass
class Detection:
    """
    Single object detection result.
    
    Attributes:
        bbox: Bounding box as [x1, y1, x2, y2] (top-left, bottom-right)
        confidence: Detection confidence score (0-1)
        class_id: COCO class ID
        class_name: Human-readable class name
    """
    bbox: np.ndarray  # Shape: (4,)
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """Return bbox as (x1, y1, x2, y2) integers."""
        return tuple(self.bbox.astype(int))
    
    @property
    def width(self) -> float:
        """Bounding box width."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Bounding box height."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        """Bounding box area in pixels."""
        return self.width * self.height
    
    @property
    def center(self) -> tuple[float, float]:
        """Center point (cx, cy) of bounding box."""
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        return (cx, cy)
    
    def to_tracker_format(self) -> np.ndarray:
        """
        Convert to format expected by SORT tracker.
        
        Returns:
            Array of [x1, y1, x2, y2, confidence]
        """
        return np.array([*self.bbox, self.confidence])


class Detector:
    """
    YOLO-based object detector.
    
    Wraps ultralytics YOLO model with clean interface
    for our tracking pipeline.
    """
    
    # COCO dataset class names (80 classes)
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]
    
    def __init__(self, config: DetectorConfig | None = None):
        """
        Initialize detector with configuration.
        
        Args:
            config: Detector configuration. Uses defaults if None.
        """
        self.config = config or DetectorConfig()
        self._model: YOLO | None = None
    
    def load_model(self) -> None:
        """
        Load YOLO model weights.
        
        First call downloads weights automatically if not present.
        """
        print(f"Loading YOLO model: {self.config.model_name}")
        
        # Ultralytics handles download automatically
        self._model = YOLO(self.config.model_name)
        
        # Set device
        if self.config.device != "auto":
            self._model.to(self.config.device)
        
        print(f"Model loaded successfully on {self._get_device()}")
    
    def _get_device(self) -> str:
        """Get the device model is running on."""
        if self._model is None:
            return "not loaded"
        return str(next(self._model.model.parameters()).device)
    
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run detection on a single frame.
        
        Args:
            frame: BGR image as numpy array (OpenCV format)
            
        Returns:
            List of Detection objects
        """
        if self._model is None:
            self.load_model()
        
        results = self._model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=list(self.config.classes) if self.config.classes else None,
            verbose=False
        )
        
        detections = []
        
        result = results[0]
        
        # Extract boxes (if any detections)
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                detection = Detection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=self.COCO_CLASSES[class_id]
                )
                detections.append(detection)
        
        return detections
    
    def warmup(self, frame: np.ndarray) -> None:
        """
        Warmup the model with a dummy inference.
        
        First inference is slower due to memory allocation.
        Call this before timing-critical code.
        """
        if self._model is None:
            self.load_model()
        
        # Run one inference to warmup
        _ = self.detect(frame)
        print("Model warmup complete")
    
    def __repr__(self) -> str:
        return f"Detector(model={self.config.model_name}, device={self._get_device()})"