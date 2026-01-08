import os
import time
import io
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
from typing import List

app = FastAPI(title="🚀 PCB Defect Detection API")

# --- 1. HARDWARE & MODEL SETUP ---
# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Initializing model on: {device.upper()}")

# Load model once at startup
model = YOLO("models/best.pt")
model.to(device)

# Warm-up run: Prevents the first real request from being slow
# We run a tiny blank image through the model to wake up the GPU
model.predict(source=torch.zeros(1, 3, 640, 640).to(device), verbose=False)

# --- 2. FILE SYSTEM SETUP ---
RESULT_DIR = "static/results"
os.makedirs(RESULT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    start_time = time.perf_counter()
    
    images = []
    filenames = []
    
    # Pre-processing loop
    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        images.append(image)
        filenames.append(file.filename)

    # 3. BATCH INFERENCE (The fast part)
    # verbose=False removes terminal clutter, making it slightly faster
    results = model.predict(source=images, conf=0.25, device=device, verbose=False)
    
    all_detections = []

    # 4. POST-PROCESSING (Saving results)
    for i, res in enumerate(results):
        output_filename = f"detected_{filenames[i]}"
        output_path = os.path.join(RESULT_DIR, output_filename)
        
        # res.save() is a blocking I/O operation. 
        # If this is still slow, the bottleneck is your Disk (Hard Drive)
        res.save(filename=output_path)

        image_detections = []
        for box in res.boxes:
            image_detections.append({
                "defect": model.names[int(box.cls[0])],
                "confidence": f"{float(box.conf[0]):.2%}",
                "location": [round(x, 1) for x in box.xyxy[0].tolist()]
            })
        
        all_detections.append({
            "filename": filenames[i],
            "defect_count": len(image_detections),
            "detections": image_detections,
            "result_url": f"http://127.0.0.1:8000/static/results/{output_filename}"
        })

    total_time_ms = (time.perf_counter() - start_time) * 1000 

    return {
        "status": "success",
        "device_used": device,
        "total_images": len(files),
        "total_speed_ms": round(total_time_ms, 2),
        "avg_speed_per_image_ms": round(total_time_ms / len(files), 2),
        "results": all_detections
    }