"""
Object Detection and Tracking - Main Entry Point.

This module provides a complete pipeline for:
1. Reading video frames
2. Detecting objects using YOLO
3. Tracking objects using SORT
4. Visualizing results in real-time
5. Optionally saving output video

Usage:
    python main.py --video path/to/video.mp4
    python main.py --video path/to/video.mp4 --save --output output.mp4
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import cv2

from .video_handler import VideoHandler
from .detector import Detector, Detection
from .tracker import SORTTracker, Track
from .utils.config import (
    Config,
    DetectorConfig,
    TrackerConfig,
    VideoConfig,
)
from .utils.drawing import (
    draw_track,
    draw_fps,
    draw_frame_info,
    Colors,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time Object Detection and Tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input/Output
    parser.add_argument(
        "--video", "-v",
        type=Path,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path to output video file (if --save is used)"
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save output video"
    )
    
    # Detector settings
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Filter specific class IDs (e.g., 0 for person, 2 for car)"
    )
    
    # Tracker settings
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Maximum frames to keep track alive without detection"
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Minimum detections before track is confirmed"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Minimum IoU for matching detections to tracks"
    )
    
    # Display settings
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Display window scale factor"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display window (useful for headless processing)"
    )
    
    return parser.parse_args()


def create_video_writer(
    output_path: Path,
    width: int,
    height: int,
    fps: float
) -> cv2.VideoWriter:
    """Create video writer for saving output."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )
    
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")
    
    return writer


def run_pipeline(
    video_path: Path,
    config: Config,
    output_path: Path | None = None,
    display: bool = True
) -> dict:
    """
    Run the detection and tracking pipeline.
    
    Args:
        video_path: Input video path
        config: Pipeline configuration
        output_path: Optional output video path
        display: Whether to display results
        
    Returns:
        Dictionary with processing statistics
    """
    # Initialize components
    detector = Detector(config=config.detector)
    tracker = SORTTracker(config=config.tracker)
    video_writer = None
    
    # Statistics
    stats = {
        "total_frames": 0,
        "total_detections": 0,
        "total_tracks": 0,
        "avg_fps": 0.0,
        "unique_track_ids": set(),
    }
    
    fps_list = []
    frame_time = time.time()
    
    try:
        with VideoHandler(video_path, config=config.video) as video:
            print(f"\n{'='*60}")
            print(f"  Object Detection & Tracking Pipeline")
            print(f"{'='*60}")
            print(f"  Input:      {video_path.name}")
            print(f"  Resolution: {video.width}x{video.height}")
            print(f"  FPS:        {video.fps:.2f}")
            print(f"  Frames:     {video.total_frames}")
            print(f"  Model:      {config.detector.model_name}")
            print(f"{'='*60}\n")
            
            # Setup output writer
            if output_path:
                video_writer = create_video_writer(
                    output_path,
                    video.width,
                    video.height,
                    video.fps
                )
                print(f"Saving output to: {output_path}")
            
            # Warmup
            first_frame_data = next(video.frames())
            detector.warmup(first_frame_data.frame)
            
            # Reset video
            video.close()
            video.open()
            
            # Main processing loop
            for frame_data in video.frames():
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - frame_time + 1e-6)
                frame_time = current_time
                fps_list.append(fps)
                
                # 1. DETECT
                detections = detector.detect(frame_data.frame)
                
                # 2. PREPARE detections for tracker
                if detections:
                    det_array = np.array([d.to_tracker_format() for d in detections])
                    class_ids = np.array([d.class_id for d in detections])
                    class_names = [d.class_name for d in detections]
                else:
                    det_array = np.empty((0, 5))
                    class_ids = np.array([])
                    class_names = []
                
                # 3. TRACK
                tracks = tracker.update(det_array, class_ids, class_names)
                
                # 4. VISUALIZE
                output_frame = frame_data.frame.copy()
                
                for track in tracks:
                    draw_track(
                        output_frame,
                        track.xyxy,
                        track.track_id,
                        label=track.class_name
                    )
                    stats["unique_track_ids"].add(track.track_id)
                
                # Draw overlays
                draw_fps(output_frame, fps)
                draw_frame_info(
                    output_frame,
                    frame_data.frame_number,
                    video.total_frames,
                    len(tracks)
                )
                
                # Draw track count
                cv2.putText(
                    output_frame,
                    f"Tracks: {len(tracks)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    Colors.CYAN,
                    2,
                    cv2.LINE_AA
                )
                
                # 5. OUTPUT
                if video_writer:
                    video_writer.write(output_frame)
                
                if display:
                    if not video.display(output_frame):
                        print("\nUser interrupted")
                        break
                
                # Update stats
                stats["total_frames"] += 1
                stats["total_detections"] += len(detections)
                stats["total_tracks"] += len(tracks)
                
                # Progress update every 100 frames
                if frame_data.frame_number % 100 == 0:
                    progress = (frame_data.frame_number / video.total_frames) * 100
                    print(f"Progress: {progress:.1f}% | Frame: {frame_data.frame_number} | "
                          f"FPS: {fps:.1f} | Tracks: {len(tracks)}")
    
    finally:
        if video_writer:
            video_writer.release()
            print(f"\nOutput saved to: {output_path}")
    
    # Finalize stats
    stats["avg_fps"] = np.mean(fps_list) if fps_list else 0
    stats["unique_track_ids"] = len(stats["unique_track_ids"])
    
    return stats


def print_stats(stats: dict) -> None:
    """Print processing statistics."""
    print(f"\n{'='*60}")
    print(f"  Processing Complete!")
    print(f"{'='*60}")
    print(f"  Total Frames:      {stats['total_frames']}")
    print(f"  Average FPS:       {stats['avg_fps']:.2f}")
    print(f"  Total Detections:  {stats['total_detections']}")
    print(f"  Unique Tracks:     {stats['unique_track_ids']}")
    print(f"{'='*60}\n")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Validate input
    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)
    
    # Build configuration from arguments
    config = Config(
        detector=DetectorConfig(
            model_name=args.model,
            confidence_threshold=args.confidence,
            classes=tuple(args.classes) if args.classes else None,
        ),
        tracker=TrackerConfig(
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_threshold,
        ),
        video=VideoConfig(
            display_scale=args.scale,
        ),
    )
    
    # Determine output path
    output_path = None
    if args.save:
        output_path = args.output or args.video.with_stem(f"{args.video.stem}_tracked")
    
    # Run pipeline
    try:
        stats = run_pipeline(
            video_path=args.video,
            config=config,
            output_path=output_path,
            display=not args.no_display,
        )
        print_stats(stats)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()