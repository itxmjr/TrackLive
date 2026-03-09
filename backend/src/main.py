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
import logging
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

logger = logging.getLogger(__name__)


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
    display: bool = True,
    progress_callback: callable = None
) -> dict:
    """
    Run the detection and tracking pipeline.

    Args:
        video_path: Input video path
        config: Pipeline configuration
        output_path: Optional output video path
        display: Whether to display results
        progress_callback: Optional function called with (current_frame, total_frames, stats)

    Returns:
        Dictionary with processing statistics
    """
    detector = Detector(config=config.detector)
    tracker = SORTTracker(config=config.tracker)
    video_writer = None

    stats = {
        "total_frames": 0,
        "total_detections": 0,
        "total_tracks": 0,
        "avg_fps": 0.0,
        "unique_track_ids": 0,
        "class_counts": {},
        "processed_at": time.time(),
        "frame_tracks": [],
    }

    fps_list = []
    frame_time = time.time()
    unique_ids = set()

    try:
        with VideoHandler(video_path, config=config.video) as video:
            logger.info(
                "Pipeline started — input=%s resolution=%dx%d fps=%.2f frames=%d model=%s",
                video_path.name, video.width, video.height, video.fps,
                video.total_frames, config.detector.model_name,
            )

            if output_path:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*"avc1")
                    video_writer = cv2.VideoWriter(
                        str(output_path),
                        fourcc,
                        video.fps,
                        (video.width, video.height)
                    )
                    if not video_writer.isOpened():
                        raise Exception("avc1 failed")
                except:
                    logger.warning("avc1 codec failed, falling back to mp4v")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(output_path),
                        fourcc,
                        video.fps,
                        (video.width, video.height)
                    )

                if not video_writer.isOpened():
                    raise RuntimeError(f"Failed to create video writer: {output_path}")
                logger.info("Saving output to: %s", output_path)

            try:
                first_frame_data = next(video.frames())
                detector.warmup(first_frame_data.frame)
                video.close()
                video.open()
            except StopIteration:
                logger.error("Video has no frames")
                return stats

            for frame_data in video.frames():
                current_time = time.time()
                fps = 1.0 / (current_time - frame_time + 1e-6)
                frame_time = current_time
                fps_list.append(fps)

                detections = detector.detect(frame_data.frame)

                if detections:
                    det_array = np.array([d.to_tracker_format() for d in detections])
                    class_ids = np.array([d.class_id for d in detections])
                    class_names = [d.class_name for d in detections]
                    for d in detections:
                        stats["class_counts"][d.class_name] = stats["class_counts"].get(d.class_name, 0) + 1
                else:
                    det_array = np.empty((0, 5))
                    class_ids = np.array([])
                    class_names = []

                tracks = tracker.update(det_array, class_ids, class_names)

                output_frame = frame_data.frame.copy()
                for track in tracks:
                    draw_track(output_frame, track.xyxy, track.track_id, label=track.class_name)
                    unique_ids.add(track.track_id)
                    stats["frame_tracks"].append({
                        "frame": frame_data.frame_number,
                        "track_id": track.track_id,
                        "label": track.class_name,
                        "x1": float(track.bbox[0]),
                        "y1": float(track.bbox[1]),
                        "x2": float(track.bbox[2]),
                        "y2": float(track.bbox[3]),
                    })

                draw_fps(output_frame, fps)
                draw_frame_info(output_frame, frame_data.frame_number, video.total_frames, len(tracks))

                if video_writer:
                    video_writer.write(output_frame)

                if display:
                    if not video.display(output_frame):
                        break

                stats["total_frames"] += 1
                stats["total_detections"] += len(detections)
                stats["total_tracks"] += len(tracks)
                stats["unique_track_ids"] = len(unique_ids)
                stats["avg_fps"] = np.mean(fps_list)

                if progress_callback:
                    progress_callback(frame_data.frame_number, video.total_frames, stats)

                if frame_data.frame_number % 100 == 0:
                    progress = (frame_data.frame_number / video.total_frames) * 100
                    logger.info("Progress: %.1f%% | Frame: %d | FPS: %.1f", progress, frame_data.frame_number, fps)

    finally:
        if video_writer:
            video_writer.release()

            if output_path and output_path.exists():
                temp_output = output_path.with_name(f"temp_{output_path.name}")
                output_path.rename(temp_output)

                try:
                    import subprocess
                    cmd = [
                        "ffmpeg", "-i", str(temp_output),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "veryfast", "-crf", "23",
                        "-y", str(output_path)
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    logger.info("FFmpeg remuxing complete: %s", output_path)
                except Exception as e:
                    logger.error("FFmpeg remuxing failed: %s", e)
                    if temp_output.exists() and not output_path.exists():
                        temp_output.rename(output_path)
                finally:
                    if temp_output.exists():
                        temp_output.unlink()

    return stats


def print_stats(stats: dict) -> None:
    """Print processing statistics."""
    logger.info(
        "Processing complete — frames=%d avg_fps=%.2f detections=%d unique_tracks=%d",
        stats["total_frames"], stats["avg_fps"],
        stats["total_detections"], stats["unique_track_ids"],
    )


def main() -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if not args.video.exists():
        logger.error("Video not found: %s", args.video)
        sys.exit(1)

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

    output_path = None
    if args.save:
        output_path = args.output or args.video.with_stem(f"{args.video.stem}_tracked")

    try:
        stats = run_pipeline(
            video_path=args.video,
            config=config,
            output_path=output_path,
            display=not args.no_display,
        )
        print_stats(stats)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error("Error: %s", e)
        raise


if __name__ == "__main__":
    main()
