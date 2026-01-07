from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
import os
import base64

app = FastAPI()

# ---------- CONFIG ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# ---------- LOAD MODEL ----------
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ---------- IMAGE PREPROCESS ----------
def read_image(file: UploadFile):
    image_bytes = file.file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    return img

# ---------- INFERENCE ----------
def run_inference(img):
    results = model(img)[0]
    return results

# ---------- FORMAT RESPONSE ----------
def format_boxes(results):
    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": float(box.conf[0]),
            "type": model.names[int(box.cls[0])]
        })
    return boxes

# ---------- ROUTES ----------
@app.get("/")
def root():
    return {"status": "PCB Defect API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = read_image(file)

    results = run_inference(img)
    boxes = format_boxes(results)

    # 🔹 YOLO annotated image (WITH boxes & labels)
    annotated_img = results.plot()

    # 🔹 Encode image to base64
    _, buffer = cv2.imencode(".png", annotated_img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "boxes": boxes,
        "image": img_base64
    }
