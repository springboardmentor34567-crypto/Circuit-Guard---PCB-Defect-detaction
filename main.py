from fastapi import FastAPI, File, UploadFile
from app.inference import run_inference

app = FastAPI()

@app.get("/")
def home():
    return {"message": "YOLOv8 Backend is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    results = run_inference(image_bytes)
    return {
        "detections": results
    }
