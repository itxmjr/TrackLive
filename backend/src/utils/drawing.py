"""
Visualization utilities for drawing detections and tracks.

Uses OpenCV for rendering bounding boxes, labels, and tracking IDs.
"""

import cv2
import numpy as np
from typing import Sequence

# Type alias for color (BGR format)
Color = tuple[int, int, int]


class Colors:
    """Color palette for visualization."""
    
    # Primary colors (BGR format - OpenCV uses BGR, not RGB!)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    
    # Extended palette for multiple tracks
    PALETTE = [
        (255, 128, 0),    # Orange
        (255, 153, 255),  # Pink
        (255, 178, 102),  # Light orange
        (230, 230, 0),    # Yellow
        (255, 153, 153),  # Light red
        (153, 255, 153),  # Light green
        (153, 153, 255),  # Light blue
        (0, 255, 127),    # Spring green
        (255, 0, 127),    # Rose
        (127, 0, 255),    # Violet
    ]
    
    @classmethod
    def get_color(cls, idx: int) -> Color:
        """Get color for given index (cycles through palette)."""
        return cls.PALETTE[idx % len(cls.PALETTE)]


def draw_detection(
    frame: np.ndarray,
    bbox: Sequence[int],
    label: str,
    confidence: float,
    color: Color = Colors.GREEN,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a single detection on frame.
    
    Args:
        frame: Image to draw on (modified in place)
        bbox: Bounding box as (x1, y1, x2, y2)
        label: Class label text
        confidence: Confidence score (0-1)
        color: Box and text color (BGR)
        thickness: Line thickness
        
    Returns:
        Frame with detection drawn
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Prepare label text
    text = f"{label}: {confidence:.2f}"
    
    # Get text size for background rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, font_thickness
    )
    
    # Draw filled rectangle behind text
    cv2.rectangle(
        frame,
        (x1, y1 - text_height - 10),
        (x1 + text_width + 5, y1),
        color,
        -1  # Filled
    )
    
    # Draw text
    cv2.putText(
        frame,
        text,
        (x1 + 2, y1 - 5),
        font,
        font_scale,
        Colors.WHITE,
        font_thickness,
        cv2.LINE_AA
    )
    
    return frame


def draw_track(
    frame: np.ndarray,
    bbox: Sequence[int],
    track_id: int,
    label: str | None = None,
    color: Color | None = None,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a tracked object with ID.
    
    Args:
        frame: Image to draw on (modified in place)
        bbox: Bounding box as (x1, y1, x2, y2)
        track_id: Unique tracking ID
        label: Optional class label
        color: Box color (auto-assigned from palette if None)
        thickness: Line thickness
        
    Returns:
        Frame with track drawn
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Auto-assign color based on track ID
    if color is None:
        color = Colors.get_color(track_id)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Prepare ID text
    if label:
        text = f"ID:{track_id} {label}"
    else:
        text = f"ID:{track_id}"
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(
        text, font, font_scale, font_thickness
    )
    
    # Draw filled background
    cv2.rectangle(
        frame,
        (x1, y1 - text_height - 10),
        (x1 + text_width + 5, y1),
        color,
        -1
    )
    
    # Draw ID text
    cv2.putText(
        frame,
        text,
        (x1 + 2, y1 - 5),
        font,
        font_scale,
        Colors.WHITE,
        font_thickness,
        cv2.LINE_AA
    )
    
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter on frame."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        Colors.GREEN,
        2,
        cv2.LINE_AA
    )
    return frame


def draw_frame_info(
    frame: np.ndarray,
    frame_number: int,
    total_frames: int,
    num_detections: int
) -> np.ndarray:
    """Draw frame information overlay."""
    h, w = frame.shape[:2]
    
    # Frame counter
    text = f"Frame: {frame_number}/{total_frames}"
    cv2.putText(frame, text, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, Colors.WHITE, 2, cv2.LINE_AA)
    
    # Detection count
    text = f"Detections: {num_detections}"
    cv2.putText(frame, text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, Colors.WHITE, 2, cv2.LINE_AA)
    
    return frame