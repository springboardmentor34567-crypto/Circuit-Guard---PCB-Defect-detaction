PROJECT_NAME: CircuitGuard – Intelligent PCB Defect Detection System

OVERVIEW:
CircuitGuard is a production-oriented PCB defect detection system designed
to demonstrate real-world deployment of deep learning models.
The system integrates a YOLO-based computer vision model with a FastAPI backend
and a Streamlit frontend to deliver an end-to-end defect inspection pipeline.
It enables automated quality inspection, visual analytics, and report generation
for printed circuit boards (PCBs).

KEY_HIGHLIGHTS:
- End-to-end ML system (Frontend + Backend + Model)
- Clear client–server separation using REST APIs
- Real-time defect detection with visual feedback
- Industry-style deployment workflow
- Scalable and modular architecture

TECHNOLOGY_STACK:
- Programming Language: Python 3.11
- Deep Learning Model: YOLO (Ultralytics)
- Backend Framework: FastAPI
- Frontend Framework: Streamlit
- Image Processing: OpenCV, PIL
- Server: Uvicorn
- Version Control: Git, GitHub (LFS enabled for model files)

SUPPORTED_DEFECT_TYPES:
- Missing Hole
- Mouse Bite
- Open Circuit
- Short Circuit
- Spur
- Spurious Copper

SYSTEM_ARCHITECTURE:
User
 → Streamlit Frontend (UI & Visualization)
 → REST API (HTTP POST /predict)
 → FastAPI Backend (Inference Engine)
 → YOLO Model (best.pt)
 → JSON Response + Annotated Outputs
 → Frontend Dashboard & Downloads

PROJECT_STRUCTURE:
CircuitGuard/
├── app.py                     # Streamlit frontend application
├── pcb-defect-backend/
│   ├── main.py                # FastAPI backend service
│   ├── model/
│   │   └── best.pt            # Trained YOLO model (LFS tracked)
│   └── uploads/               # Images stored by backend for traceability
├── screenshots/               # Application UI screenshots
├── requirements.txt
├── packages.txt
├── runtime.txt
└── README.md

FRONTEND_CAPABILITIES:
- Upload single or multiple PCB images
- Sends images to backend via REST API
- Displays original and annotated images
- Visualizes defect statistics (bar & donut charts)
- Enables export of annotated images and reports

BACKEND_CAPABILITIES:
- Accepts images via POST /predict endpoint
- Saves uploaded images for verification and audit
- Executes YOLO inference on the backend
- Returns structured JSON responses to the frontend

API_SPECIFICATION:
Endpoint: POST /predict
Input:
- Multipart form-data
- Image file (PNG, JPG, JPEG)

Sample_Response:
{
  "status": "success",
  "defects_detected": {
    "spur": 1
  },
  "total_defects": 1
}

MODEL_INFORMATION:
Model_Name: YOLO (Ultralytics)
Input_Type: PCB top-view images
Performance_Metrics:
- mAP@50: 0.98
- Precision: 0.97
- Recall: 0.97

INTEGRATION_PROOF:
- Images uploaded from the frontend are saved in:
  pcb-defect-backend/uploads/
- Inference is executed exclusively in the backend
- Results are returned via REST API and rendered in frontend
- Confirms true frontend–backend communication (not local-only execution)

KNOWN_LIMITATIONS:
- Some visualization logic remains frontend-driven
- Backend response metadata can be extended further
- Current setup is optimized for single-node inference

FUTURE_ENHANCEMENTS:
- Fully backend-driven annotation rendering
- Database integration for inspection history
- User authentication and access control
- Containerization using Docker
- Cloud deployment (AWS / Azure / GCP)
- Asynchronous batch processing for large-scale inspection

AUTHOR:
Name: Prashant Yadav
Degree: B.Tech – Computer Science & Engineering (Artificial Intelligence)
Project_Type: Internship / Applied Machine Learning Project

PROJECT_GOAL:
To demonstrate practical deployment of a computer vision model
in a real-world, production-style architecture with clean
engineering practices and scalable design.

