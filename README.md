# Real Time Object Tracker

Real-time object detection and tracking using YOLOv8 and SORT.

## Overview

This project provides a complete pipeline for real-time object detection and tracking in videos. It uses the YOLOv8 model for object detection and the SORT (Simple Online and Realtime Tracking) algorithm for tracking objects across frames.

## Key Features

- **High-Performance Detection**: Powered by YOLOv8.
- **Robust Tracking**: Uses SORT with Kalman Filtering for accurate object association.
- **Efficient Backend**: Built with FastAPI for high-performance and asynchronous support.
- **WebSocket Integration**: Supports real-time tracking for camera streams.

## Deployment on Hugging Face Spaces

This project is configured to run on Hugging Face Spaces using Docker.

### Local Setup (Backend)

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -e .
   ```
3. Run the API:
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 7860
   ```

### WebSocket Endpoint

The real-time tracking WebSocket endpoint is available at `/ws/track`. It expects base64 encoded JPEG frames and returns object tracks in JSON format.

## Technology Stack

- **Python**
- **FastAPI**
- **YOLOv8** (Ultralytics)
- **OpenCV**
- **Numpy**
- **Kalman Filter** (FilterPy)
