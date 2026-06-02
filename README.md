# FaceGuard AI

A real-time AI-powered face recognition and security alert system that monitors CCTV/IP camera feeds, identifies known individuals, and dispatches instant Telegram notifications with face-crop photos when a detection triggers an alert.

Built as a production-grade system demonstrating the integration of computer vision, deep learning, IoU-based object tracking, AI-driven decision making, and event-driven alerting — all running on a single Python process with Docker support.

---

## Demo

```
[Camera Feed]
      │
      ▼
┌─────────────┐     no person     ┌─────────────┐
│  YOLOv8n    │──────────────────▶│  Skip Frame │
│ Person Gate │                   └─────────────┘
└──────┬──────┘
       │ person detected
       ▼
┌─────────────────┐
│  InsightFace    │  RetinaFace detection + ArcFace embedding
│  Face Pipeline  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  IoU Tracker    │  Assigns persistent IDs across frames
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  FAISS Search   │  Cosine similarity → majority-vote identity
└──────┬──────────┘
       │
       ▼
┌─────────────────┐     ignore     ┌────────────────┐
│  Decision Engine│───────────────▶│  Log to Postgres│
│  (rule-based /  │                └────────────────┘
│   LLM optional) │
└──────┬──────────┘
       │ notify
       ▼
┌─────────────────┐
│  Telegram Bot   │  Sends photo + structured alert message
└─────────────────┘
```

---

## Key Features

- **YOLOv8n person pre-filter** — skips the expensive face pipeline entirely when no person is in frame; drastically reduces CPU load on idle cameras
- **InsightFace (buffalo_l)** — production-grade RetinaFace detection + ArcFace recognition at `det_size=(640,640)` for long-range detection
- **FAISS flat index** — sub-millisecond nearest-neighbour search across hundreds of registered identities
- **Adaptive similarity threshold** — threshold relaxes for small/far/occluded faces (scaled by InsightFace `det_score`)
- **IoU tracking** — face identity persists across frames without running recognition every tick (re-identifies every ~3 s)
- **Dual decision engines** — rule-based (zero latency) or optional LLM-powered via LangChain + GPT-4o-mini
- **Non-blocking Telegram alerts** — photo + caption sent in a daemon thread; never stalls the camera loop
- **PostgreSQL event log** — every detection and alert stored with UUID, timestamp, confidence, and reason
- **Docker Compose** — one-command deployment with Postgres, health checks, and persistent volumes
- **Auto-reconnect** — recovers from RTSP stream drops without restarting the process

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Face Detection | [InsightFace](https://github.com/deepinsight/insightface) — RetinaFace |
| Face Recognition | InsightFace — ArcFace (buffalo_l) |
| Person Pre-filter | [YOLOv8n](https://github.com/ultralytics/ultralytics) |
| Vector Search | [FAISS](https://github.com/facebookresearch/faiss) (IndexFlatIP) |
| AI Decision Layer | [LangChain](https://github.com/langchain-ai/langchain) + OpenAI GPT-4o-mini *(optional)* |
| Video I/O | OpenCV |
| Alerting | Telegram Bot API |
| Database | PostgreSQL + SQLAlchemy |
| Containerisation | Docker + Docker Compose |
| Runtime | Python 3.10 |

---

## Project Structure

```
ai-face-alert-system/
│
├── vision/
│   ├── camera.py          # Main loop: YOLO → InsightFace → tracker → alerts
│   ├── recognizer.py      # FAISS index build + face identify (majority vote)
│   └── tracker.py         # IoU-based multi-face tracker
│
├── brain/
│   ├── decision.py        # Rule-based alert engine (time, confidence, area)
│   └── agent.py           # Optional: LLM decision engine (GPT-4o-mini)
│
├── notifications/
│   └── telegram.py        # Telegram Bot: send text + photo alerts
│
├── api/
│   └── system.py          # Orchestrates: detect → decide → log → notify
│
├── db/
│   └── models.py          # SQLAlchemy models: Person, Detection, Alert
│
├── known_faces/           # Reference photos — see known_faces/README.md
│
├── .env.example           # Environment variable template
├── docker-compose.yml     # Full stack: app + postgres
├── Dockerfile
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL (or use Docker Compose which includes it)
- A Telegram bot token ([create one via @BotFather](https://t.me/BotFather))
- An IP camera with RTSP stream, or a webcam

### 1. Clone & configure

```bash
git clone https://github.com/AREEBMEHDI/faceguard-ai.git
cd faceguard-ai

cp .env.example .env
# Edit .env and fill in your RTSP_URL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
```

### 2. Add known faces

```bash
mkdir -p known_faces/your_name
# Copy 3–10 photos of the person into that folder
```

See [known_faces/README.md](known_faces/README.md) for photo guidelines.

### 3a. Run with Docker (recommended)

```bash
docker compose up --build
```

This starts PostgreSQL and the app container. The InsightFace model (~300 MB) downloads automatically on first run.

### 3b. Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start Postgres separately, then:
python -m vision.camera
```

---

## Configuration

All configuration is via environment variables (`.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `RTSP_URL` | `0` | RTSP stream URL or `0` for webcam |
| `HEADLESS` | `false` | `true` disables the OpenCV display window (required for servers) |
| `TELEGRAM_BOT_TOKEN` | — | Telegram bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | — | Target chat/group ID for alerts |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost/ai_face_alert` | PostgreSQL connection string |
| `OPENAI_API_KEY` | — | Required only if using `brain/agent.py` (LLM mode) |

---

## Decision Engine

The system ships with two interchangeable decision engines in `brain/`:

### Rule-based (default — `decision.py`)

Zero-latency, no external API calls. Triggers alerts based on:
- Detection confidence threshold (`< 0.27` → ignore)
- Time of day (after 22:00 or before 06:00 → high alert)
- Sensitive area flag → high alert

### LLM-powered (optional — `agent.py`)

Uses LangChain + GPT-4o-mini to contextually analyse each event and return a structured JSON decision. Swap it in by changing the import in `api/system.py`:

```python
# from brain.decision import decide      # rule-based
from brain.agent import decide            # LLM-powered
```

---

## Alert Format

When an alert fires, the Telegram bot sends a photo of the detected face alongside a structured message:

```
🚨 FACE ALERT — HIGH
👤 Person: `John Doe`
📷 Camera: `rtsp://...`
🕐 Time: `02:30`
📊 Confidence: `0.91`
📍 Sensitive area: `Yes`
⚡ Action: `notify`
📝 Reason: `After-hours detection`
```

---

## Architecture Notes

**Why YOLO as a pre-filter?**  
InsightFace runs RetinaFace at `det_size=(640,640)` which is computationally expensive. On a security camera watching an empty room 95% of the time, running it every frame wastes CPU. YOLOv8n (5 MB, CPU) acts as a cheap gate — if no person is detected, the frame is skipped entirely.

**Why FAISS instead of simple cosine loops?**  
FAISS `IndexFlatIP` with `normalize_L2` is equivalent to exact cosine search but scales to thousands of embeddings without any speed penalty. It also supports easy batch queries for future multi-camera expansion.

**Why IoU tracking instead of re-running recognition every frame?**  
Face recognition (embedding lookup) is the most expensive step per face. The IoU tracker assigns a stable ID to each face box across frames, so recognition only runs on new tracks or every ~90 frames (~3 s at 30 fps) rather than 30 times per second per face.

**macOS `spawn` multiprocessing**  
PyTorch and AVFoundation both use background threads on macOS. Setting `torch.multiprocessing.set_start_method("spawn")` before any import, and warming up YOLO before opening the camera, prevents the crash that occurs when AVFoundation's capture thread races PyTorch's lazy initialisation.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Areeb Mehdi** — AI Engineer  
[GitHub](https://github.com/AREEBMEHDI) · [Email](mailto:mehdibusinesslogicpk@gmail.com)
