import os
import shutil
import zipfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from backend.services.inference import run_inference
from backend.services.reports import create_pdf
from backend.services.pcb_gate import is_pcb
from backend.database import SessionLocal
from backend.models import PCBResult

app = FastAPI(title="PCB Defect Detection API")

app.mount("/static", StaticFiles(directory="backend/static"), name="static")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.get("/")
def serve_ui():
    return FileResponse("frontend/index.html")


OUTPUT_DIR = "backend/static/processed_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def to_static_url(path: str):
    return "/" + path.replace("backend/", "").replace("\\", "/")


def create_zip(output_dir: str, zip_path: str):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in os.listdir(output_dir):
            if f.endswith((".pdf", ".json", "_annotated.jpg")):
                zipf.write(
                    os.path.join(output_dir, f),
                    arcname=f
                )


@app.post("/process")
async def process_images(files: list[UploadFile] = File(...)):
    db: Session = SessionLocal()
    results = []
    pcb_files_exist = False

    try:
        for file in files:
            image_path = os.path.join(OUTPUT_DIR, file.filename)

            with open(image_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # =====================
            # PCB GATE (MODEL 1)
            # =====================
            pcb_result = is_pcb(image_path)

            if not pcb_result["is_pcb"]:
                results.append({
                    "file": file.filename,
                    "status": "NOT_PCB",
                    "pcb_confidence": pcb_result["pcb_conf"],
                    "defect_count": 0,
                    "defects": [],
                    "image_path": to_static_url(image_path),
                    "annotated_path": None,
                    "pdf_path": None,
                    "json_path": None,
                    "inference_ms": 0
                })
                continue

            # =====================
            # YOLO DEFECT MODEL (MODEL 2)
            # =====================
            metadata, annotated_path, json_path = run_inference(
                image_path, OUTPUT_DIR
            )

            pdf_path = image_path.replace(".jpg", ".pdf")
            create_pdf(pdf_path, image_path, annotated_path, metadata)

            pcb_files_exist = True

            record = PCBResult(
                file_name=metadata["file_name"],
                image_path=image_path,
                annotated_path=annotated_path,
                pdf_path=pdf_path,
                defect_count=metadata["defect_count"],
                inference_ms=metadata["inference_ms"],
            )
            db.add(record)
            db.commit()

            results.append({
                "file": metadata["file_name"],
                "status": "DEFECT" if metadata["defect_count"] > 0 else "OK",
                "pcb_confidence": pcb_result["pcb_conf"],
                "defect_count": metadata["defect_count"],
                "defects": metadata["defects"],
                "image_path": to_static_url(image_path),
                "annotated_path": to_static_url(annotated_path),
                "pdf_path": to_static_url(pdf_path),
                "json_path": to_static_url(json_path),
                "inference_ms": metadata["inference_ms"]
            })

        zip_url = None
        if pcb_files_exist:
            zip_path = os.path.join(OUTPUT_DIR, "report_package.zip")
            create_zip(OUTPUT_DIR, zip_path)
            zip_url = "/static/processed_results/report_package.zip"

    finally:
        db.close()

    return JSONResponse({
        "total_files": len(results),
        "zip_url": zip_url,
        "results": results
    })
