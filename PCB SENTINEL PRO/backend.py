from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io
import base64
import json

app = FastAPI()

# 1. Load the Model (Happens only once when server starts)
try:
    model = YOLO("./MODEL/best.pt")
except:
    print("Model not found. Make sure best.pt is in the folder.")   

@app.get("/")
def home():
    return {"message": "PCB Defect Detection API is Running"}

# 2. The Detection Endpoint
@app.post("/detect/")
async def detect_defects(file: UploadFile = File(...), conf: float = 0.25, iou: float = 0.45):
    # Read the image file uploaded by the user
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Run Inference
    results = model.predict(image, conf=conf, iou=iou)
    result = results[0]

    # --- Process Results ---
    
    # A. Create the Annotated Image
    annotated_frame = result.plot()
    annotated_pil = Image.fromarray(annotated_frame[..., ::-1])
    
    # Convert image to Base64 string (to send over API)
    img_buffer = io.BytesIO()
    annotated_pil.save(img_buffer, format="JPEG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

    # B. Extract Defect Data (for the table)
    defects = []
    for box in result.boxes:
        c_id = int(box.cls[0].item())
        defects.append({
            "Type": model.names[c_id],
            "Confidence": round(box.conf[0].item(), 4),
            "Box": [round(x, 1) for x in box.xyxy[0].tolist()]
        })

    # Return JSON response
    return {
        "filename": file.filename,
        "defect_count": len(defects),
        "defects": defects,
        "annotated_image_base64": img_str
    }