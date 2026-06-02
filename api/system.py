from brain.decision import decide
from db.models import Detection, Alert, engine
from notifications.telegram import send_alert
from sqlalchemy.orm import sessionmaker
import threading

Session = sessionmaker(bind=engine)


def process_event(event, face_crop=None):
    session = Session()

    detection = Detection(
        person_name=event["person"],
        camera=event["camera"],
        confidence=event["confidence"]
    )
    session.add(detection)
    session.commit()

    decision = decide(event)

    alert = Alert(
        alert_level=decision["alert_level"],
        action=decision["action"],
        reason=decision["reason"]
    )
    session.add(alert)
    session.commit()
    session.close()

    # Send Telegram notification in a background thread so it never
    # blocks the camera loop — network calls can take 1-3 seconds.
    if decision["action"] == "notify":
        threading.Thread(
            target=send_alert,
            args=(event, decision, face_crop),
            daemon=True,
        ).start()

    return decision
