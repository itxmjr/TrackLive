"""
SORT: Simple Online and Realtime Tracking.

Implementation based on the paper:
"Simple Online and Realtime Tracking" by Bewley et al.
https://arxiv.org/abs/1602.00763

Key components:
1. Kalman Filter: Predicts object motion
2. Hungarian Algorithm: Matches detections to tracks
3. IoU: Measures overlap between boxes
"""

import numpy as np
from dataclasses import dataclass, field
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Optional

from .utils.config import TrackerConfig


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        bbox1: First box as [x1, y1, x2, y2]
        bbox2: Second box as [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    if union <= 0:
        return 0.0
    
    return intersection / union


def iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate IoU matrix between two sets of boxes.
    
    Args:
        boxes1: Array of shape (N, 4)
        boxes2: Array of shape (M, 4)
        
    Returns:
        IoU matrix of shape (N, M)
    """
    n = len(boxes1)
    m = len(boxes2)
    
    if n == 0 or m == 0:
        return np.zeros((n, m))
    
    iou_matrix = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = iou(boxes1[i], boxes2[j])
    
    return iou_matrix


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """
    Convert [x1, y1, x2, y2] to [cx, cy, area, aspect_ratio].
    
    Kalman filter tracks center position, area, and aspect ratio
    because these change smoothly during motion.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2
    cy = bbox[1] + h / 2
    area = w * h
    aspect_ratio = w / (h + 1e-6)
    
    return np.array([cx, cy, area, aspect_ratio]).reshape((4, 1))


def convert_z_to_bbox(z: np.ndarray) -> np.ndarray:
    """
    Convert [cx, cy, area, aspect_ratio] back to [x1, y1, x2, y2].
    """
    cx, cy, area, aspect_ratio = z.flatten()[:4]
    
    # Derive width and height from area and aspect ratio
    # area = w * h, aspect_ratio = w / h
    # Therefore: w = sqrt(area * aspect_ratio), h = sqrt(area / aspect_ratio)
    w = np.sqrt(area * aspect_ratio)
    h = area / (w + 1e-6)
    
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    
    return np.array([x1, y1, x2, y2])


class KalmanBoxTracker:
    """
    Kalman Filter for tracking a single bounding box.
    
    State vector: [cx, cy, area, aspect_ratio, vx, vy, v_area]
    - (cx, cy): center position
    - area: bounding box area
    - aspect_ratio: width / height (assumed constant)
    - (vx, vy): velocity
    - v_area: rate of area change
    """
    
    _count = 0  # Global track ID counter
    
    def __init__(self, bbox: np.ndarray, class_id: int = 0, class_name: str = "object"):
        """
        Initialize tracker with initial bounding box.
        
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
            class_id: Detected class ID
            class_name: Detected class name
        """
        # Initialize Kalman Filter
        # State: [cx, cy, area, ratio, vx, vy, v_area] - 7 dimensions
        # Measurement: [cx, cy, area, ratio] - 4 dimensions
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (constant velocity model)
        # Next state = current state + velocity * dt
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # cx' = cx + vx
            [0, 1, 0, 0, 0, 1, 0],  # cy' = cy + vy
            [0, 0, 1, 0, 0, 0, 1],  # area' = area + v_area
            [0, 0, 0, 1, 0, 0, 0],  # ratio' = ratio (constant)
            [0, 0, 0, 0, 1, 0, 0],  # vx' = vx
            [0, 0, 0, 0, 0, 1, 0],  # vy' = vy
            [0, 0, 0, 0, 0, 0, 1],  # v_area' = v_area
        ])
        
        # Measurement matrix (we observe cx, cy, area, ratio)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        
        # Measurement noise covariance
        self.kf.R[2:, 2:] *= 10.0  # Area and ratio are less reliable
        
        # Initial covariance - high uncertainty for velocities
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        
        # Process noise covariance
        self.kf.Q[-1, -1] *= 0.01  # Area velocity is very stable
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state with first detection
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        
        # Track metadata
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1
        
        self.class_id = class_id
        self.class_name = class_name
        
        # Track lifecycle
        self.hits = 1           # Total successful matches
        self.hit_streak = 1     # Consecutive matches
        self.age = 0            # Frames since creation
        self.time_since_update = 0  # Frames since last match
        
        # History for trajectory visualization
        self.history: List[np.ndarray] = []
    
    def predict(self) -> np.ndarray:
        """
        Advance state and return predicted bounding box.
        
        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        # Handle negative area prediction
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        # Reset hit streak if no recent update
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        
        # Store prediction in history
        predicted_bbox = self.get_state()
        self.history.append(predicted_bbox)
        
        return predicted_bbox
    
    def update(self, bbox: np.ndarray) -> None:
        """
        Update state with matched detection.
        
        Args:
            bbox: Matched detection [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        # Update Kalman filter with measurement
        self.kf.update(convert_bbox_to_z(bbox))
    
    def get_state(self) -> np.ndarray:
        """
        Get current bounding box estimate.
        
        Returns:
            Current state as [x1, y1, x2, y2]
        """
        return convert_z_to_bbox(self.kf.x)


@dataclass
class Track:
    """
    Output representation of a tracked object.
    
    Clean interface for downstream consumers.
    """
    track_id: int
    bbox: np.ndarray
    class_id: int
    class_name: str
    hits: int
    age: int
    
    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """Return bbox as integer tuple."""
        return tuple(self.bbox.astype(int))


class SORTTracker:
    """
    SORT: Simple Online and Realtime Tracking.
    
    Manages multiple object tracks using Kalman filtering
    and Hungarian algorithm for data association.
    """
    
    def __init__(self, config: TrackerConfig | None = None):
        """
        Initialize SORT tracker.
        
        Args:
            config: Tracker configuration
        """
        self.config = config or TrackerConfig()
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
    
    def update(
        self,
        detections: np.ndarray,
        class_ids: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: Array of shape (N, 5) with [x1, y1, x2, y2, confidence]
            class_ids: Optional array of class IDs for each detection
            class_names: Optional list of class names for each detection
            
        Returns:
            List of active Track objects
        """
        self.frame_count += 1
        
        # Handle empty detections
        if len(detections) == 0:
            detections = np.empty((0, 5))
        
        # Default class info
        if class_ids is None:
            class_ids = np.zeros(len(detections), dtype=int)
        if class_names is None:
            class_names = ["object"] * len(detections)
        
        # Get predicted locations from existing trackers
        predicted_boxes = []
        trackers_to_remove = []
        
        for i, tracker in enumerate(self.trackers):
            predicted_bbox = tracker.predict()
            
            # Check for invalid predictions (NaN or negative size)
            if np.any(np.isnan(predicted_bbox)):
                trackers_to_remove.append(i)
            else:
                predicted_boxes.append(predicted_bbox)
        
        # Remove invalid trackers
        for i in reversed(trackers_to_remove):
            self.trackers.pop(i)
        
        predicted_boxes = np.array(predicted_boxes) if predicted_boxes else np.empty((0, 4))
        
        # Match detections to trackers using IoU
        matched, unmatched_dets, unmatched_trks = self._associate_detections(
            detections[:, :4], predicted_boxes
        )
        
        # Update matched trackers with assigned detections
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(detections[det_idx, :4])
            # Update class info (in case of class change or refinement)
            self.trackers[trk_idx].class_id = class_ids[det_idx]
            self.trackers[trk_idx].class_name = class_names[det_idx]
        
        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            new_tracker = KalmanBoxTracker(
                detections[det_idx, :4],
                class_id=class_ids[det_idx],
                class_name=class_names[det_idx]
            )
            self.trackers.append(new_tracker)
        
        # Build output: confirmed tracks only
        active_tracks = []
        trackers_to_remove = []
        
        for i, tracker in enumerate(self.trackers):
            # Only return tracks that have been confirmed
            if tracker.hit_streak >= self.config.min_hits or self.frame_count <= self.config.min_hits:
                if tracker.time_since_update < 1:  # Was just updated
                    active_tracks.append(Track(
                        track_id=tracker.id,
                        bbox=tracker.get_state(),
                        class_id=tracker.class_id,
                        class_name=tracker.class_name,
                        hits=tracker.hits,
                        age=tracker.age
                    ))
            
            # Remove dead tracks
            if tracker.time_since_update > self.config.max_age:
                trackers_to_remove.append(i)
        
        # Clean up dead trackers
        for i in reversed(trackers_to_remove):
            self.trackers.pop(i)
        
        return active_tracks
    
    def _associate_detections(
        self,
        detections: np.ndarray,
        trackers: np.ndarray
    ) -> tuple[list, list, list]:
        """
        Match detections to tracked objects using Hungarian algorithm.
        
        Args:
            detections: Array of detection boxes (N, 4)
            trackers: Array of predicted tracker boxes (M, 4)
            
        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_trackers)
        """
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(trackers)))
        
        # Compute IoU matrix
        iou_matrix = iou_batch(detections, trackers)
        
        # Hungarian algorithm (scipy uses cost, so we use 1 - IoU)
        # We want to maximize IoU, which equals minimizing (1 - IoU)
        cost_matrix = 1 - iou_matrix
        
        det_indices, trk_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches below threshold
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(trackers)))
        
        for det_idx, trk_idx in zip(det_indices, trk_indices):
            if iou_matrix[det_idx, trk_idx] >= self.config.iou_threshold:
                matched.append((det_idx, trk_idx))
                unmatched_dets.remove(det_idx)
                unmatched_trks.remove(trk_idx)
        
        return matched, unmatched_dets, unmatched_trks
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker._count = 0
    
    def __repr__(self) -> str:
        return f"SORTTracker(active_tracks={len(self.trackers)}, frame={self.frame_count})"