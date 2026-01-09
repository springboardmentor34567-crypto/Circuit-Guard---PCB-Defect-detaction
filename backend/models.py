from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from backend.database import Base

class PCBResult(Base):
    __tablename__ = "pcb_results"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String)
    image_path = Column(String)
    annotated_path = Column(String)
    pdf_path = Column(String)
    defect_count = Column(Integer)
    inference_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
