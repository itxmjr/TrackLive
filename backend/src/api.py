"""
FastAPI Backend for Object Detection and Tracking.

Provides REST endpoints for video processing and WebSockets for real-time camera tracking.
"""

import cv2
import numpy as np
import base64
import json
import asyncio
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import shutil

from .detector import Detector, DetectorConfig
from .tracker import SORTTracker, TrackerConfig
from .utils.config import Config

app = FastAPI(title="Real Time Object Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = Detector(config=DetectorConfig(model_name="yolov8n.pt"))
tracker = SORTTracker(config=TrackerConfig())

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    detector.load_model()

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": detector.config.model_name}

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    """
    Handle video upload and process it.
    In a real app, this would return a job ID and process in background.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    # Note: In a production scenario, we'd use a BackgroundTask or Celery
    # For now, we return a simple confirmation.
    return {
        "filename": file.filename,
        "message": "Video received. Sequential processing would happen here.",
        "path": str(tmp_path)
    }

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
            # Receive frame
            data = await websocket.receive_text()
            
            try:
                header, encoded = data.split(",", 1)
                data_bytes = base64.b64decode(encoded)
                nparr = np.frombuffer(data_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                    
                detections = detector.detect(frame)
                
                if detections:
                    det_array = np.array([d.to_tracker_format() for d in detections])
                    class_ids = np.array([d.class_id for d in detections])
                    class_names = [d.class_name for d in detections]
                else:
                    det_array = np.empty((0, 5))
                    class_ids = np.array([])
                    class_names = []
                    
                tracks = tracker.update(det_array, class_ids, class_names)
                
                response = {
                    "tracks": [
                        {
                            "id": t.track_id,
                            "bbox": t.bbox.tolist(),
                            "label": t.class_name,
                            "hits": t.hits
                        } for t in tracks
                    ]
                }
                
                await websocket.send_json(response)
                
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        print("Websocket disconnected")
