from PIL import Image
import base64
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from collections import Counter
import shutil
import os

# ================== APP ==================
app = FastAPI(title="PCB Defect Detection API")

# ================== CORS ==================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== MODEL ==================
model = YOLO("model/best.pt")

# ================== DIRECTORIES ==================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================== ROOT ==================
@app.get("/")
def root():
    return {"status": "Backend running"}

# ================== PREDICT API ==================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Save uploaded image
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO
    results = model(file_path)
    r = results[0]

    # Collect defects
    defects = []
    if r.boxes is not None:
        for c in r.boxes.cls:
            defects.append(model.names[int(c)])

    defect_counts = Counter(defects)

    # Create annotated image
    annotated_img = r.plot()[:, :, ::-1]  # BGR → RGB
    pil_img = Image.fromarray(annotated_img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "status": "success",
        "defects_detected": dict(defect_counts),
        "total_defects": sum(defect_counts.values()),
        "annotated_image": img_base64
    }
