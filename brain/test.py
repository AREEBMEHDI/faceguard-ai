from decision import decide

event = {
    "person": "Areeb",
    "camera": "Gate-3",
    "time": "02:30",
    "confidence": 0.91,
    "sensitive_area": True
}

print(decide(event))
