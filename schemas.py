# backend/schemas.py

from pydantic import BaseModel
from typing import List

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]

class DetectionResponse(BaseModel):
    total_defects: int
    detections: List[Detection]
