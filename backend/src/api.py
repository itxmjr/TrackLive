"""
FastAPI Backend for Object Detection and Tracking.

Provides REST endpoints for video processing and WebSockets for real-time camera tracking.
"""

import asyncio
import base64
import binascii
import csv
import io
import json
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import mimetypes
import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

from .detector import Detector, DetectorConfig
from .tracker import SORTTracker, TrackerConfig
from .utils.config import Config, VideoConfig
from .main import run_pipeline

logger = logging.getLogger(__name__)

detector = Detector(config=DetectorConfig(model_name="yolov8n.pt"))
tracker = SORTTracker(config=TrackerConfig())

# Storage for processing tasks
processing_tasks: dict[str, dict] = {}

mimetypes.init()
mimetypes.add_type('video/mp4', '.mp4')

# Create directories
UPLOAD_DIR = Path("data/uploads")
RESULT_DIR = Path("data/results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Task cleanup interval (1 hour)
TASK_TTL_SECONDS = 3600


class UpdateSettingsRequest(BaseModel):
    model_name: Optional[str] = None
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    iou_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_age: Optional[int] = Field(default=None, ge=1)
    min_hits: Optional[int] = Field(default=None, ge=0)
    classes: Optional[list[int]] = None


async def cleanup_old_tasks():
    """Periodically remove completed/failed tasks older than TTL."""
    while True:
        await asyncio.sleep(TASK_TTL_SECONDS)
        now = time.time()
        expired = [
            tid for tid, task in processing_tasks.items()
            if task.get("status") in ("completed", "failed")
            and now - task.get("completed_at", now) > TASK_TTL_SECONDS
        ]
        for tid in expired:
            del processing_tasks[tid]
        if expired:
            logger.info("Cleaned up %d expired tasks", len(expired))


@asynccontextmanager
async def lifespan(app: FastAPI):
    detector.load_model()
    cleanup_task = asyncio.create_task(cleanup_old_tasks())
    yield
    cleanup_task.cancel()


app = FastAPI(title="TrackLive API", lifespan=lifespan)

cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for results
app.mount("/results", StaticFiles(directory="data/results", html=True), name="results")


@app.get("/health")
async def health_check():
    return {"status": "ok", "model": detector.config.model_name}


@app.post("/update-settings")
async def update_settings(settings: UpdateSettingsRequest):
    """Update detector and tracker settings."""
    global detector, tracker

    model_changed = settings.model_name is not None and settings.model_name != detector.config.model_name

    # Apply detector config whenever any detector setting changes
    if model_changed or settings.confidence_threshold is not None or settings.iou_threshold is not None or settings.classes is not None:
        classes_tuple = detector.config.classes
        if settings.classes is not None:
            classes_tuple = tuple(settings.classes) if settings.classes else None

        detector.config = DetectorConfig(
            model_name=settings.model_name if settings.model_name is not None else detector.config.model_name,
            confidence_threshold=settings.confidence_threshold if settings.confidence_threshold is not None else detector.config.confidence_threshold,
            iou_threshold=settings.iou_threshold if settings.iou_threshold is not None else detector.config.iou_threshold,
            classes=classes_tuple,
        )

        if model_changed:
            detector.load_model()

    if settings.max_age is not None or settings.min_hits is not None or settings.iou_threshold is not None:
        tracker.config = TrackerConfig(
            max_age=settings.max_age if settings.max_age is not None else tracker.config.max_age,
            min_hits=settings.min_hits if settings.min_hits is not None else tracker.config.min_hits,
            iou_threshold=settings.iou_threshold if settings.iou_threshold is not None else tracker.config.iou_threshold,
        )

    return {"status": "success"}


def process_video_task(file_path: Path, output_path: Path, task_id: str):
    """Background task to process video."""
    try:
        processing_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "stats": None,
        }

        def progress_update(current, total, stats):
            progress = int((current / total) * 100)
            processing_tasks[task_id]["progress"] = progress
            if current % 20 == 0:
                processing_tasks[task_id]["stats"] = stats.copy()

        config = Config(
            detector=detector.config,
            tracker=tracker.config,
            video=VideoConfig(display_scale=1.0),
        )

        results = run_pipeline(
            video_path=file_path,
            config=config,
            output_path=output_path,
            display=False,
            progress_callback=progress_update,
        )

        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["progress"] = 100
        processing_tasks[task_id]["stats"] = results
        processing_tasks[task_id]["completed_at"] = time.time()
    except Exception as e:
        logger.error("Error processing video %s: %s", task_id, e)
        processing_tasks[task_id] = {
            "status": f"failed: {str(e)}",
            "progress": 0,
            "stats": None,
            "completed_at": time.time(),
        }


@app.post("/process-video")
async def process_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Handle video upload and process it in the background."""
    task_id = f"{file.filename}_{int(asyncio.get_event_loop().time())}"
    file_path = UPLOAD_DIR / file.filename
    output_filename = f"tracked_{file.filename}"
    output_path = RESULT_DIR / output_filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(process_video_task, file_path, output_path, task_id)

    return {
        "task_id": task_id,
        "filename": file.filename,
        "output_url": f"/results/{output_filename}",
        "status": "queued",
    }


@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Check the status of a processing task."""
    status = processing_tasks.get(task_id, "not_found")
    return {"task_id": task_id, "status": status}


@app.get("/export/tracks/{task_id}")
async def export_tracks(task_id: str, format: str = Query(default="csv")):
    """Export track data for a completed task as CSV or JSON."""
    task = processing_tasks.get(task_id)
    if not task:
        return JSONResponse({"error": "task not found"}, status_code=404)
    if task.get("status") != "completed":
        return JSONResponse({"error": "task not completed"}, status_code=400)

    frame_tracks = task.get("stats", {}).get("frame_tracks", [])

    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["frame", "track_id", "label", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(frame_tracks)
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=tracks_{task_id}.csv"},
        )
    else:
        return Response(
            content=json.dumps(frame_tracks),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=tracks_{task_id}.json"},
        )


@app.websocket("/ws/track")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time tracking via WebSocket.
    Expects base64 encoded JPEG frames.
    """
    await websocket.accept()
    tracker.reset()

    try:
        while True:
            data = await websocket.receive_text()

            try:
                if "," in data:
                    header, encoded = data.split(",", 1)
                else:
                    encoded = data

                try:
                    data_bytes = base64.b64decode(encoded)
                except binascii.Error:
                    await websocket.send_json({"error": "invalid base64"})
                    continue

                nparr = np.frombuffer(data_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                detections = await asyncio.to_thread(detector.detect, frame)

                if detections:
                    det_array = np.array([d.to_tracker_format() for d in detections])
                    class_ids = np.array([d.class_id for d in detections])
                    class_names = [d.class_name for d in detections]
                else:
                    det_array = np.empty((0, 5))
                    class_ids = np.array([])
                    class_names = []

                tracks = await asyncio.to_thread(tracker.update, det_array, class_ids, class_names)

                response = {
                    "tracks": [
                        {
                            "id": t.track_id,
                            "bbox": t.bbox.tolist(),
                            "label": t.class_name,
                            "class_id": t.class_id,
                            "conf": float(np.max(confs)) if (confs := [d.confidence for d in detections if d.class_id == t.class_id]) else 0.0,
                            "trail": t.trail,
                        } for t in tracks
                    ]
                }

                await websocket.send_json(response)

            except Exception as e:
                logger.exception("Error processing frame")
                try:
                    await websocket.send_json({"error": str(e)})
                except Exception:
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
