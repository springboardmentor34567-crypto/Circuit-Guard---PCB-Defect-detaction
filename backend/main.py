# backend/main.py

import os
import time
import base64
import cv2
import torch
from typing import List
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, UploadFile, File
from model import CircuitGuardModel
from utils import read_image

# CPU OPTIMIZATION 

torch.set_num_threads(os.cpu_count())
torch.backends.mkldnn.enabled = True


# FASTAPI APP

app = FastAPI(
    title="CircuitGuard API",
    description="PCB Defect Detection using YOLO",
    version="1.4"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# LOAD MODEL 

model = CircuitGuardModel()

# SINGLE IMAGE ENDPOINT

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image = await read_image(file)

    t0 = time.time()

    with torch.no_grad():
        result = model.model.predict(
            image,
            batch=1,
            imgsz=960,          
            conf=0.25,
            device="cpu",
            verbose=False
        )[0]

    infer_time = time.time() - t0
    print(f"[YOLO] Single inference: {infer_time:.2f} sec")

    detections = []
    if result.boxes:
        for box in result.boxes:
            detections.append({
                "class_id": int(box.cls[0]),
                "class_name": model.model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": [float(x) for x in box.xyxy[0]]
            })

    annotated = result.plot()
    _, buffer = cv2.imencode(".png", annotated)
    annotated_b64 = base64.b64encode(buffer).decode()

    return {
        "total_defects": len(detections),
        "detections": detections,
        "annotated_image": annotated_b64
    }



# BATCH IMAGE ENDPOINT 

@app.post("/detect-batch")
async def detect_batch(files: List[UploadFile] = File(...)):
    images = []
    filenames = []

   
    for file in files:
        img = await read_image(file)
        images.append(img)
        filenames.append(file.filename)

    t0 = time.time()

  
    with torch.no_grad():
        results = model.model.predict(
            images,
            batch=len(images),  
            imgsz=960,          
            conf=0.25,
            device="cpu",
            verbose=False
        )

    infer_time = time.time() - t0
    print(
        f"[YOLO] Batch inference: {infer_time:.2f} sec "
        f"({infer_time / len(images):.2f} sec/image)"
    )

    responses = []

    for idx, result in enumerate(results):
        detections = []

        if result.boxes:
            for box in result.boxes:
                detections.append({
                    "class_id": int(box.cls[0]),
                    "class_name": model.model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": [float(x) for x in box.xyxy[0]]
                })

        annotated = result.plot()
        _, buffer = cv2.imencode(".png", annotated)
        annotated_b64 = base64.b64encode(buffer).decode()

        responses.append({
            "filename": filenames[idx],
            "total_defects": len(detections),
            "detections": detections,
            "annotated_image": annotated_b64
        })

    return responses
