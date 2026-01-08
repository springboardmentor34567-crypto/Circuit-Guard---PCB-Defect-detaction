import os
import uuid

BASE_DIR = "backend/temp_results"

def create_job_dirs():
    # Ensure base directory exists
    os.makedirs(BASE_DIR, exist_ok=True)

    job_id = str(uuid.uuid4())
    base = os.path.join(BASE_DIR, job_id)

    dirs = {
        "base": base,
        "originals": os.path.join(base, "originals"),
        "annotated": os.path.join(base, "annotated")
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    return job_id, dirs
