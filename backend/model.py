
from ultralytics import YOLO
import numpy as np

MODEL_PATH = "best.pt"

class CircuitGuardModel:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)

    def predict(self, image: np.ndarray, conf: float = 0.25):
        """
        Run YOLO inference on an image
        """
        results = self.model(image, conf=conf)
        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                detections.append({
                    "class_id": int(box.cls[0]),
                    "class_name": self.model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": [float(x) for x in box.xyxy[0]]
                })

        return detections
