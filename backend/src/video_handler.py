"""
Video input/output handling using OpenCV.

Responsibilities:
- Read frames from video files
- Display processed frames
- Optionally save output video
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator
from dataclasses import dataclass

from .utils.config import VideoConfig


@dataclass
class FrameData:
    """Container for frame information."""
    frame: np.ndarray
    frame_number: int
    timestamp_ms: float


class VideoHandler:
    """Handles video input and output operations."""
    
    def __init__(self, source: str | Path, config: VideoConfig | None = None):
        """
        Initialize video handler.
        
        Args:
            source: Path to video file
            config: Video configuration settings
        """
        self.source = Path(source)
        self.config = config or VideoConfig()
        self._cap: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None
        
        # Video properties (populated on open)
        self.width: int = 0
        self.height: int = 0
        self.fps: float = 0
        self.total_frames: int = 0
    
    def open(self) -> None:
        """Open video source for reading."""
        if not self.source.exists():
            raise FileNotFoundError(f"Video not found: {self.source}")
        
        self._cap = cv2.VideoCapture(str(self.source))
        
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.source}")
        
        # Extract video properties
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def frames(self) -> Generator[FrameData, None, None]:
        """
        Generate frames from video.
        
        Yields:
            FrameData objects containing frame and metadata
        """
        if self._cap is None:
            self.open()
        
        frame_number = 0
        
        while True:
            ret, frame = self._cap.read()
            
            if not ret:
                break
            
            timestamp_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            
            yield FrameData(
                frame=frame,
                frame_number=frame_number,
                timestamp_ms=timestamp_ms
            )
            
            frame_number += 1
    
    def display(self, frame: np.ndarray, window_name: str = "Detection & Tracking") -> bool:
        """
        Display frame in window.
        
        Args:
            frame: Frame to display
            window_name: Window title
            
        Returns:
            False if user pressed 'q' to quit, True otherwise
        """
        if self.config.display_scale != 1.0:
            frame = cv2.resize(frame, None, 
                             fx=self.config.display_scale,
                             fy=self.config.display_scale)
        
        cv2.imshow(window_name, frame)
        
        # Wait for 1ms and check for 'q' key
        key = cv2.waitKey(1) & 0xFF
        return key != ord('q')
    
    def close(self) -> None:
        """Release all resources."""
        if self._cap is not None:
            self._cap.release()
        if self._writer is not None:
            self._writer.release()
        cv2.destroyAllWindows()
    
    def __enter__(self) -> "VideoHandler":
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def __repr__(self) -> str:
        return f"VideoHandler(source={self.source}, {self.width}x{self.height}@{self.fps}fps)"