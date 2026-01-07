from ultralytics import YOLO
from PIL import Image
import io

# Load model ONCE
model = YOLO("models/yolov8m.pt")

def run_inference(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(image)

    detections = []

    for r in results:
        for box in r.boxes:
            detections.append({
                "class_id": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })

    return detections
