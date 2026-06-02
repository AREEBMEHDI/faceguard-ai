from datetime import datetime

def decide(event):
    confidence = event.get("confidence", 0)
    hour = int(event.get("time", "00:00").split(":")[0])
    sensitive = event.get("sensitive_area", False)

    alert_level = "low"
    action = "ignore"
    reason = "Normal detection"

    if confidence < 0.27:
        return {
            "alert_level": "low",
            "action": "ignore",
            "reason": "Low confidence"
        }

    if hour >= 22 or hour <= 6:
        alert_level = "high"
        action = "notify"
        reason = "After-hours detection"

    if sensitive:
        alert_level = "high"
        action = "notify"
        reason = "Sensitive area detection"

    return {
        "alert_level": alert_level,
        "action": action,
        "reason": reason
    }
