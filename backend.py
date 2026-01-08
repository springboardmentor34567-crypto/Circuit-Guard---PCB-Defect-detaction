from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import glob
from PIL import Image
import io

# --- CONFIGURATION ---
app = FastAPI(title="CircuitGuard API (YOLO Only)")

TEMPLATE_DIR = "database/templates"
MODEL_PATH = "models/pcb_defect_classifier/weights/best.pt"

# Global Model Variable
model = None

# Colors for Bounding Boxes (BGR)
COLORS = {
    "Missing_hole": (0, 0, 0), "Mouse_bite": (128, 0, 128),
    "Open_circuit": (0, 0, 255), "Short": (0, 255, 255),
    "Spur": (0, 200, 0), "Spurious_copper": (0, 165, 255)
}

# --- LIFECYCLE: LOAD MODEL ON STARTUP ---
@app.on_event("startup")
def startup_event():
    global model
    print("⏳ Loading YOLOv8 Model...")
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"✅ Model Loaded Successfully: {MODEL_PATH}")
    else:
        print(f"❌ CRITICAL ERROR: Model not found at {MODEL_PATH}")

# --- HELPER FUNCTIONS ---
def find_template(pcb_id):
    # Try exact match
    path = os.path.join(TEMPLATE_DIR, f"{pcb_id}.jpg")
    if os.path.exists(path): return path
    # Try fuzzy match
    candidates = glob.glob(os.path.join(TEMPLATE_DIR, f"{pcb_id}*.jpg"))
    if candidates: return candidates[0]
    return None

def image_to_base64(img_bgr):
    """Converts OpenCV image to Base64 string to send over API"""
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

# --- API ENDPOINT ---
@app.post("/inspect")
async def inspect_endpoint(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded on server."}

    # 1. Read Uploaded Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_test = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Find Template based on filename
    filename = file.filename
    pcb_id = filename.split('_')[0]
    template_path = find_template(pcb_id)

    if not template_path:
        return {"error": f"Template ID '{pcb_id}' not found in database."}

    # 3. Load Template & Align
    img_temp = cv2.imread(template_path)
    # Resize test to match template dimensions exactly
    img_test = cv2.resize(img_test, (img_temp.shape[1], img_temp.shape[0]))

    # 4. Subtraction Pipeline (High Sensitivity)
    gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    gray_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_temp, gray_test)
    
    # Threshold = 30 (Sensitive to Spurs)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Cleaning (Dilation only, to keep thin spurs)
    mask = cv2.dilate(thresh, np.ones((3,3), np.uint8), iterations=2)
    
    # 5. Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated = img_test.copy()
    results_data = []

    for cnt in contours:
        if cv2.contourArea(cnt) > 100: # Filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Padding: 45px (Big box as requested)
            pad = 45
            vx1, vy1 = max(0, x-pad), max(0, y-pad)
            vx2, vy2 = min(img_test.shape[1], x+w+pad), min(img_test.shape[0], y+h+pad)
            
            # Crop ROI for YOLO
            roi = img_test[vy1:vy2, vx1:vx2]
            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            
            # Predict
            pred = model.predict(roi_pil, verbose=False)[0]
            cls = pred.names[pred.probs.top1]
            conf = pred.probs.top1conf.item()
            
            # Draw Visuals
            color = COLORS.get(cls, (0,0,255))
            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), color, 3)
            
            # Label
            label_text = f"{cls} {conf:.0%}"
            cv2.putText(annotated, label_text, (vx1, vy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            results_data.append({
                "Filename": filename,
                "Defect Type": cls,
                "Confidence": f"{conf:.1%}",
                "Location": f"({x}, {y})",
                "Dimensions": f"{w}x{h}"
            })

    # 6. Return Result
    return {
        "status": "success",
        "annotated_image": image_to_base64(annotated),
        "defects": results_data
    }