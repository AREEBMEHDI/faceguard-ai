import os
import io
import requests
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

_BASE = f"https://api.telegram.org/bot{_TOKEN}"

_LEVEL_EMOJI = {
    "high":   "🚨",
    "medium": "⚠️",
    "low":    "ℹ️",
}


def _configured():
    return bool(_TOKEN and _CHAT_ID
                and _TOKEN != "your_bot_token_here"
                and _CHAT_ID != "your_chat_id_here")


def send_alert(event: dict, decision: dict, face_crop=None):
    """
    Send a Telegram alert.

    event    – dict with person, camera, time, confidence, sensitive_area
    decision – dict with alert_level, action, reason
    face_crop – optional BGR numpy array of the detected face (will be sent as photo)
    """
    if not _configured():
        print("⚠️  Telegram not configured — add BOT_TOKEN and CHAT_ID to .env")
        return

    level   = decision.get("alert_level", "low")
    emoji   = _LEVEL_EMOJI.get(level, "🔔")
    action  = decision.get("action", "")
    reason  = decision.get("reason", "")

    text = (
        f"{emoji} *FACE ALERT — {level.upper()}*\n"
        f"👤 Person: `{event['person']}`\n"
        f"📷 Camera: `{event['camera']}`\n"
        f"🕐 Time: `{event['time']}`\n"
        f"📊 Confidence: `{event['confidence']:.2f}`\n"
        f"📍 Sensitive area: `{'Yes' if event.get('sensitive_area') else 'No'}`\n"
        f"⚡ Action: `{action}`\n"
        f"📝 Reason: `{reason}`"
    )

    if face_crop is not None and face_crop.size > 0:
        _send_photo(text, face_crop)
    else:
        _send_message(text)


def _send_message(text: str):
    try:
        resp = requests.post(
            f"{_BASE}/sendMessage",
            json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=5,
        )
        if not resp.ok:
            print(f"Telegram error: {resp.text}")
    except Exception as e:
        print(f"Telegram send failed: {e}")


def _send_photo(caption: str, bgr_image: np.ndarray):
    try:
        # Encode frame as JPEG in memory — no temp file needed
        _, buf = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        photo_bytes = io.BytesIO(buf.tobytes())
        photo_bytes.name = "alert.jpg"

        resp = requests.post(
            f"{_BASE}/sendPhoto",
            data={"chat_id": _CHAT_ID, "caption": caption, "parse_mode": "Markdown"},
            files={"photo": photo_bytes},
            timeout=10,
        )
        if not resp.ok:
            print(f"Telegram photo error: {resp.text}")
    except Exception as e:
        print(f"Telegram photo send failed: {e}")
