# Must be set before any torch import to prevent AVFoundation thread conflict.
# PyTorch's lazy initialisation forks internally; on macOS that crashes if
# AVFoundation background threads are already running.
import torch
torch.multiprocessing.set_start_method("spawn", force=True)

from vision.recognizer import load_known_faces, detect_faces, identify
from vision.tracker import FaceTracker
from api.system import process_event
from dotenv import load_dotenv
import cv2
import datetime
import numpy as np
import os
import time

load_dotenv()

_HEADLESS = os.getenv("HEADLESS", "false").lower() == "true"

# ---------------------------------------------------------------------------
# YOLOv8n person pre-filter
# Runs on every frame. If no person is detected, the expensive InsightFace
# pipeline is skipped entirely — ideal for security cameras watching empty
# rooms for most of the day.
#
# Device is CPU so it doesn't share Metal state with anything else.
# YOLO is warmed up on a dummy frame here (module load time) so all of
# PyTorch's lazy initialisation completes BEFORE the camera opens its
# AVFoundation background capture thread.
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO

    _device = "cpu"
    _yolo = YOLO("yolov8n.pt")

    # Warm up: forces every lazy torch/ort call to happen now, before camera
    _yolo.predict(np.zeros((480, 640, 3), dtype=np.uint8),
                  classes=[0], conf=0.40, device=_device, verbose=False)

    USE_YOLO = True
    print(f"✅ YOLOv8n ready  (device: {_device})")
except Exception as e:
    USE_YOLO = False
    print(f"⚠️  YOLOv8n unavailable ({e}), skipping person pre-filter")


def _person_present(frame):
    """True if YOLO sees ≥1 person (class 0) at ≥40 % confidence."""
    results = _yolo.predict(frame, classes=[0], conf=0.40,
                            device=_device, verbose=False)
    return len(results[0].boxes) > 0


# ---------------------------------------------------------------------------
# Box colours
# ---------------------------------------------------------------------------
_GREEN = (0, 220, 0)
_RED   = (0, 0, 220)
_GRAY  = (160, 160, 160)


def _open_stream(url):
    """Open an RTSP (or local) stream and return the VideoCapture object."""
    if url.startswith("rtsp://"):
        # Use FFMPEG backend for RTSP; AVFoundation does not support network streams
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(int(url) if url.isdigit() else url, cv2.CAP_AVFOUNDATION)
    return cap


def start_camera():
    rtsp_url = os.getenv("RTSP_URL", "0")
    print(f"🎥 Connecting to: {rtsp_url}")

    cap = _open_stream(rtsp_url)
    if not cap.isOpened():
        print("❌ Could not open stream")
        return

    print("✅ Stream opened")

    load_known_faces()

    tracker = FaceTracker(
        iou_threshold=0.35,
        max_lost=20,
        rerecognize_every=90,   # re-identify every ~3 s at 30 fps
    )

    last_alert_time = 0
    fps_count = 0
    fps_t0 = time.time()
    fps = 0.0
    consecutive_failures = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= 30:
                print("⚠️  Stream lost — reconnecting…")
                cap.release()
                time.sleep(3)
                cap = _open_stream(rtsp_url)
                consecutive_failures = 0
            time.sleep(0.05)
            continue
        consecutive_failures = 0

        # ── YOLO pre-filter ──────────────────────────────────────────────
        if USE_YOLO and not _person_present(frame):
            if not _HEADLESS:
                _draw_fps(frame, fps)
                cv2.imshow("AI Face Alert", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            fps_count += 1
            continue

        # ── Face detection (InsightFace CPU) ─────────────────────────────
        detections = detect_faces(frame)

        # ── IoU tracking ─────────────────────────────────────────────────
        active_tracks = tracker.update(detections)

        # ── Recognition (new / stale tracks only) ────────────────────────
        for track in active_tracks:
            if tracker.needs_recognition(track):
                name, conf = identify(track.embedding, track.det_score)
                tracker.confirm(track, name, conf)

        # ── Draw ──────────────────────────────────────────────────────────
        for track in active_tracks:
            x1, y1, x2, y2 = track.box
            known = track.name not in ("...", "Unknown")
            color = _GREEN if known else (_GRAY if track.name == "..." else _RED)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if track.name not in ("...", "Unknown"):
                label = f"{track.name}  {track.confidence:.2f}  q:{track.det_score:.2f}"
            else:
                label = f"{track.name}  q:{track.det_score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)

            # ── Alert trigger ─────────────────────────────────────────────
            if known and time.time() - last_alert_time > 5:
                event = {
                    "person":         track.name,
                    "camera":         os.getenv("RTSP_URL", "Camera-1"),
                    "time":           datetime.datetime.now().strftime("%H:%M"),
                    "confidence":     track.confidence,
                    "sensitive_area": True,
                }
                # Crop the face region and send it with the Telegram alert
                fx1, fy1, fx2, fy2 = (
                    max(0, x1), max(0, y1),
                    min(frame.shape[1], x2), min(frame.shape[0], y2)
                )
                face_crop = frame[fy1:fy2, fx1:fx2].copy()
                decision = process_event(event, face_crop=face_crop)
                print(f"[ALERT] {track.name} ({track.confidence:.2f}) → {decision}")
                last_alert_time = time.time()

        # ── FPS overlay ───────────────────────────────────────────────────
        fps_count += 1
        elapsed = time.time() - fps_t0
        if elapsed >= 1.0:
            fps = fps_count / elapsed
            fps_count = 0
            fps_t0 = time.time()

        if not _HEADLESS:
            _draw_fps(frame, fps)
            cv2.imshow("AI Face Alert", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not _HEADLESS:
        cv2.destroyAllWindows()


def _draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


if __name__ == "__main__":
    start_camera()
