# backend/utils.py

import cv2
import numpy as np
from fastapi import UploadFile

async def read_image(file: UploadFile) -> np.ndarray:
    """
    Convert uploaded image to OpenCV format
    """
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    return image

def resize_for_inference(img, max_size=640):
    """
    Resize image while preserving aspect ratio.
    Ensures max(width, height) == max_size.
    """
    h, w = img.shape[:2]

    if max(h, w) <= max_size:
        return img  # no resize needed

    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
