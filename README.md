# TrackLive

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)
[![Deploy](https://img.shields.io/badge/Live-tracklive.vercel.app-000?logo=vercel&logoColor=white)](https://tracklive.vercel.app)

Real-time object detection and tracking in the browser. Stream your webcam through a WebSocket pipeline powered by YOLOv8 and SORT, or upload a video for offline processing with full tracking overlays and CSV export.

<div align="center">

<!-- Replace with your demo GIF: ![TrackLive Demo](assets/demo.gif) -->
<img src="assets/demo.gif" alt="TrackLive Demo" width="720" />

<br />

[**Live Demo**](https://tracklive.vercel.app) &nbsp;&middot;&nbsp; [Report Bug](https://github.com/itxmjr/TrackLive/issues) &nbsp;&middot;&nbsp; [Request Feature](https://github.com/itxmjr/TrackLive/issues)

</div>

## Features

- **Live WebSocket streaming** -- sub-200ms round-trip camera-to-overlay pipeline
- **SORT multi-object tracking** -- Kalman-filtered bounding boxes with persistent IDs
- **Trajectory trails** -- per-track history rendered on a canvas overlay
- **Virtual tripwire** -- draw a line on the feed; get toast alerts + crossing count when objects cross
- **Class filtering** -- toggle COCO classes on/off with instant backend sync
- **Video upload mode** -- drag-and-drop a video, get a tracked output with detection analytics
- **CSV / JSON export** -- download per-frame track data for any processed video
- **Cyberpunk UI** -- glassmorphism panels, neon accents, responsive down to 320px

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | FastAPI, YOLOv8 (Ultralytics), SORT (FilterPy Kalman filter), OpenCV |
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS v4, Radix UI + shadcn/ui, Recharts |
| **Infra** | Docker Compose, Vercel (frontend), Uvicorn |

## Architecture

```
Browser (camera)
  |  base64 JPEG
  v
WebSocket (/ws/track)
  |
  |-> YOLOv8  detect()
  |
  |-> SORT    update()
  |
  '-> JSON response  { tracks: [{ id, bbox, label, trail }] }
          |
          v
     Canvas overlay (bounding boxes, trails, tripwire)
```

## Getting Started

### Prerequisites

- Python 3.13+
- Node.js 20+
- ffmpeg (optional, for video remuxing)

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn src.api:app --reload
```

The API starts at `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

### Environment Variables

Copy `.env.example` to `.env` at the project root and adjust as needed:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
CORS_ORIGINS=http://localhost:3000
```

## Docker

Spin up both services with Docker Compose:

```bash
docker compose up --build
```

| Service | Port |
|---------|------|
| Backend | `http://localhost:7860` |
| Frontend | `http://localhost:3000` |

The frontend reads `NEXT_PUBLIC_API_URL` from the compose environment, which defaults to `http://localhost:7860`.

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + current model name |
| `POST` | `/update-settings` | Update detector/tracker config |
| `POST` | `/process-video` | Upload video for background processing |
| `GET` | `/task-status/{task_id}` | Poll processing progress |
| `GET` | `/export/tracks/{task_id}?format=csv\|json` | Download track data |
| `WS` | `/ws/track` | Real-time camera tracking stream |

## Project Structure

```
TrackLive/
├── backend/
│   ├── src/
│   │   ├── api.py            # FastAPI app, WebSocket + REST endpoints
│   │   ├── detector.py        # YOLOv8 wrapper
│   │   ├── tracker.py         # KalmanBoxTracker + SORTTracker
│   │   ├── main.py            # Offline video pipeline
│   │   ├── video_handler.py   # Video I/O utilities
│   │   └── utils/
│   │       └── config.py      # DetectorConfig, TrackerConfig, VideoConfig
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx       # Main page, all state + wiring
│   │   │   ├── layout.tsx     # Root layout
│   │   │   └── globals.css    # Cyberpunk theme tokens
│   │   └── components/
│   │       ├── LiveCamera.tsx         # Camera + WS + overlay canvas
│   │       ├── TripwireLayer.tsx      # Virtual tripwire canvas
│   │       ├── ControlPanel.tsx       # Settings panel
│   │       ├── ClassFilterChips.tsx   # Per-class toggle chips
│   │       ├── FpsChart.tsx           # Rolling FPS + track count chart
│   │       ├── StatsSidebar.tsx       # Stats display
│   │       ├── VideoUpload.tsx        # Drag-and-drop video upload
│   │       └── ui/                    # shadcn/ui primitives
│   └── Dockerfile
├── docker-compose.yml
├── LICENSE
└── README.md
```

## Configuration

Detector and tracker settings can be updated at runtime via `POST /update-settings`:

```json
{
  "detector": {
    "model_name": "yolov8n.pt",
    "confidence": 0.25,
    "iou_threshold": 0.45,
    "classes": [0, 2, 5]
  },
  "tracker": {
    "max_age": 30,
    "min_hits": 3,
    "iou_threshold": 0.3
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `yolov8n.pt` | YOLO model variant (`n`, `s`, `m`, `l`, `x`) |
| `confidence` | `0.25` | Minimum detection confidence |
| `iou_threshold` | `0.45` | NMS IoU threshold |
| `classes` | all | COCO class IDs to detect |
| `max_age` | `30` | Frames before a lost track is removed |
| `min_hits` | `3` | Detections before a track is confirmed |

## License

[MIT](LICENSE) -- M Jawad ur Rehman
