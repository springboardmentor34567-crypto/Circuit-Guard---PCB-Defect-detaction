from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import base64

app = FastAPI(title="PCB Defect Detection API")

MODEL_PATH = "C:/Users/chand/Desktop/PCB Project/models/best.pt"

CLASSES = [
    "spurious_copper",
    "missing_hole",
    "short",
    "spur",
    "open_circuit",
    "mouse_bite"
]

model = YOLO(MODEL_PATH)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model(img)[0]
    annotated_img = results.plot()

    detections = []
    class_list = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])

        detections.append({
            "Class": CLASSES[cls],
            "Confidence (%)": round(conf * 100, 2),
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2)
        })
        class_list.append(CLASSES[cls])

    _, buffer = cv2.imencode(".jpg", annotated_img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({
        "detections": detections,
        "image": img_base64,
        "total": len(detections),
        "unique_classes": list(set(class_list))
    })
