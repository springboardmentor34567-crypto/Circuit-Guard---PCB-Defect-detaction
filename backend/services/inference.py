import os
import time
import json
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

MODEL_PATH = "/home/omen/Downloads/ML/pcb_detector/ML/runs/detect/yolov11m/weights/last.pt"
CONF_THRESHOLD = 0.25

model = YOLO(MODEL_PATH)

def run_inference(image_path, output_dir):
    base = os.path.basename(image_path)
    name = os.path.splitext(base)[0]

    annotated_path = os.path.join(output_dir, f"{name}_annotated.jpg")
    json_path = os.path.join(output_dir, f"{name}.json")

    start = time.time()
    results = model.predict(image_path, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
    res = results[0]
    inference_ms = round((time.time() - start) * 1000, 2)

    defects = []
    for box in res.boxes:
        cls_id = int(box.cls[0])
        defects.append({
            "label": model.names[cls_id],
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist()
        })

    img = res.plot()
    Image.fromarray(img).save(annotated_path)

    metadata = {
        "file_name": base,
        "timestamp": datetime.now().isoformat(),
        "inference_ms": inference_ms,
        "defect_count": len(defects),
        "defects": defects
    }

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata, annotated_path, json_path
