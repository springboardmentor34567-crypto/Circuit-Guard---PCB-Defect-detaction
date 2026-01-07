from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
import os
import json
import csv
import zipfile
import base64
from datetime import datetime
from ultralytics import YOLO

app = FastAPI(title="PCB Defect Detection System")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load YOLO model
MODEL_PATH = "models/best.pt"
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Class names mapping
CLASS_NAMES = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper"
}


class DownloadImageRequest(BaseModel):
    image_data: str
    filename: str


class DownloadCSVRequest(BaseModel):
    detections: List[dict]
    filename: str


class DownloadZipRequest(BaseModel):
    results: List[dict]


@app.get("/")
async def home(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/detect-progressive")
async def detect_progressive(file: UploadFile = File(...)):
    """
    Detect defects in a single image - Progressive processing
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Run detection
        results = model(img)[0]
        
        # Process detections
        detections = []
        annotated_img = img.copy()
        
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = detection
            class_id = int(cls)
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            
            detections.append({
                "defect_type": class_name,
                "confidence": float(conf),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            })
            
            # Draw bounding box with thicker lines
            color = (0, 255, 0)
            thickness = 4  # Increased from 2 to 4
            cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Draw label with background for better visibility
            label = f"{class_name}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.5  # Increased to 1.5 for much bigger text
            font_thickness = 4  # Increased to 4 for bolder text
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(annotated_img, 
                         (int(x1), int(y1) - text_height - baseline - 10),
                         (int(x1) + text_width + 10, int(y1)),
                         color, -1)  # Filled rectangle
            
            # Draw text on background
            cv2.putText(annotated_img, label, (int(x1) + 5, int(y1) - 5),
                       font, font_scale, (0, 0, 0), font_thickness)  # Black text on green background
        
        # Convert images to base64
        _, original_buffer = cv2.imencode('.jpg', img)
        original_base64 = base64.b64encode(original_buffer).decode('utf-8')
        
        _, annotated_buffer = cv2.imencode('.jpg', annotated_img)
        annotated_base64 = base64.b64encode(annotated_buffer).decode('utf-8')
        
        return {
            "filename": file.filename,
            "detections": detections,
            "defect_count": len(detections),
            "original_image": f"data:image/jpeg;base64,{original_base64}",
            "annotated_image": f"data:image/jpeg;base64,{annotated_base64}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/download/image/{image_type}")
async def download_image(image_type: str, request: DownloadImageRequest):
    """Download image (original or annotated)"""
    try:
        # Extract base64 data
        image_data = request.image_data
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"attachment; filename={request.filename}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/download/csv")
async def download_csv(request: DownloadCSVRequest):
    """Download CSV file"""
    try:
        # Create CSV in memory
        output = io.StringIO()
        if request.detections:
            fieldnames = list(request.detections[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(request.detections)
        
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={request.filename}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/download/zip")
async def download_zip(request: DownloadZipRequest):
    """Download all annotated images as ZIP"""
    try:
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for idx, result in enumerate(request.results):
                if 'error' in result:
                    continue
                
                # Get annotated image
                image_data = result.get('annotated_image', '')
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                # Decode and add to ZIP
                image_bytes = base64.b64decode(image_data)
                filename = result.get('filename', f'image_{idx}.jpg')
                zip_file.writestr(f"annotated_{filename}", image_bytes)
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=all_annotated_images.zip"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("🚀 Starting FastAPI PCB Defect Detection System...")
    print("📍 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)