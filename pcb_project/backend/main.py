from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import os

from backend.storage import create_job_dirs
from backend.inference import run_inference

# Combined generators
from backend.formatter import generate_csv, generate_txt

# Separate generators
from backend.formatter import generate_separate_csv, generate_separate_txt

# PDF & ZIP
from backend.generators import (
    generate_detailed_pdf,
    generate_separate_pdfs,
    generate_zip
)

app = FastAPI()


@app.post("/analyze")
async def analyze(files: list[UploadFile] = File(...)):
    # ---------- Safety check ----------
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # ---------- Create job folders ----------
    job_id, dirs = create_job_dirs()

    # ---------- Save uploaded images ----------
    image_paths = []
    for f in files:
        save_path = os.path.join(dirs["originals"], f.filename)

        with open(save_path, "wb") as buf:
            shutil.copyfileobj(f.file, buf)

        f.file.seek(0)  # reset file pointer
        image_paths.append(save_path)

    # ---------- Run inference ----------
    data, originals, annotated = run_inference(image_paths, dirs)

    # ---------- Combined outputs ----------
    csv_combined = generate_csv(data, dirs["base"])
    txt_combined = generate_txt(data, dirs["base"])
    pdf_combined = generate_detailed_pdf(
        data, originals, annotated, dirs["base"]
    )

    # ---------- Separate outputs ----------
    csv_separate = generate_separate_csv(data, dirs["base"])
    txt_separate = generate_separate_txt(data, dirs["base"])
    pdf_separate = generate_separate_pdfs(
        data, originals, annotated, dirs["base"]
    )

    # ---------- ZIP (only for multiple images) ----------
    zip_path = None
    if len(annotated) > 1:
        zip_path = generate_zip(dirs["base"])

    # ---------- Response ----------
    return {
        "job_id": job_id,

        # Detection results
        "data": data,

        # Image paths
        "original_images": originals,
        "annotated_images": annotated,

        # Combined downloads
        "csv": csv_combined,
        "txt": txt_combined,
        "pdf": pdf_combined,

        # Separate downloads (ALWAYS dicts)
        "csv_separate": csv_separate or {},
        "txt_separate": txt_separate or {},
        "pdf_separate": pdf_separate or {},

        # ZIP
        "zip": zip_path
    }


@app.get("/download")
def download(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path,
        filename=os.path.basename(path),
        media_type="application/octet-stream"
    )
