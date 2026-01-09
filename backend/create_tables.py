from backend.database import engine, Base
from backend.models import PCBResult

Base.metadata.create_all(bind=engine)
