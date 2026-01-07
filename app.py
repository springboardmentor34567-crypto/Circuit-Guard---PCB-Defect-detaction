# backend/app.py
import io
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from utils import draw_boxes_on_image, ensure_dir

app = FastAPI()
# set default model path here (change if needed)
DEFAULT_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH") or r"C:\Users\M. Sucharitha\Desktop\dataset\runs\detect\train2\weights\best.pt"
MODEL = YOLO(DEFAULT_MODEL_PATH)  # loads model on import (CPU or GPU)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
ensure_dir(OUTPUT_DIR)

@app.get("/")
def health():
    return {"status": "ok", "model": str(DEFAULT_MODEL_PATH)}

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25):
    """
    Accepts an image file upload. Returns JSON with boxes and saves
    an annotated output image in backend/outputs/.
    """
    contents = await file.read()
    img_array = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Unable to decode image"}, status_code=400)

    # Convert to BGR->RGB for ultralytics if needed (Ultralytics accepts numpy BGR/RGB automatically)
    # Run prediction
    results = MODEL.predict(source=img, conf=conf, save=False, stream=False, imgsz=640)

    # results is a list (one element)
    res = results[0]

    # extract boxes, scores, class ids, class names
    boxes = res.boxes
    items = []
    for b in boxes:
        xyxy = b.xyxy.tolist()[0]  # [xmin, ymin, xmax, ymax]
        score = float(b.conf.tolist()[0])
        cls_id = int(b.cls.tolist()[0])
        cls_name = MODEL.names[cls_id] if hasattr(MODEL, "names") else str(cls_id)
        xmin, ymin, xmax, ymax = map(float, xyxy)
        items.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "score": score,
            "bbox": [xmin, ymin, xmax, ymax]
        })

    # Draw boxes on image and save
    out_fname = os.path.join(OUTPUT_DIR, f"pred_{file.filename}")
    img_with_boxes = draw_boxes_on_image(img.copy(), items)
    cv2.imwrite(out_fname, img_with_boxes)

    return {
        "predictions": items,
        "output_image": out_fname
    }
