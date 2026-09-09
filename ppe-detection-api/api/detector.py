"""
detector.py -- wraps the fine-tuned RT-DETR model in a plain function
that both the /detect endpoint AND the Part B reasoning layer call.
Keeping this separate from main.py means Part B never talks to FastAPI
directly -- it calls this module's function -- which is a cleaner
architecture point you can make in the verbal defense.
"""
from pathlib import Path
from typing import List, Dict

from ultralytics import RTDETR

_MODEL = None
WEIGHTS_PATH = Path("models/best.pt")  # update after training


def get_model() -> RTDETR:
    global _MODEL
    if _MODEL is None:
        _MODEL = RTDETR(str(WEIGHTS_PATH))
    return _MODEL


def detect(image_path: str, conf_threshold: float = 0.25) -> List[Dict]:
    """
    Runs inference and returns a list of structured detections:
    [{"class": "Hardhat", "confidence": 0.91, "box": [x1, y1, x2, y2]}, ...]

    conf_threshold default of 0.25 matches Ultralytics' own default and is
    a reasonable starting point for review; you should sweep this against
    your val set precision/recall curve and justify whatever you land on
    in the memo rather than leaving it as an unexamined default.
    """
    model = get_model()
    results = model.predict(image_path, conf=conf_threshold, verbose=False)
    detections = []
    for result in results:
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls.item())
            detections.append({
                "class": names[cls_id],
                "confidence": round(float(box.conf.item()), 4),
                "box": [round(v, 1) for v in box.xyxy[0].tolist()],
            })
    return detections
