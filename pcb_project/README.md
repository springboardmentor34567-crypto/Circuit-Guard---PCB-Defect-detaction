Circuit Guard – PCB Defect Detection System

This project focuses on detecting defects in Printed Circuit Boards (PCBs) using a deep learning–based approach.
The system automates PCB inspection and presents defect detection results through a simple frontend interface.

Project Overview

Printed Circuit Boards are critical components in electronic systems, and defects in PCBs can lead to failures.
This project aims to detect common PCB defects using computer vision and deep learning techniques, improving accuracy and inspection efficiency.

The application is divided into:

Backend: Handles model inference and API services

Frontend: Provides a user interface for uploading images and viewing results

Project Structure
Circuit-Guard---PCB-Defect-detaction/
│
├── backend/                # Backend logic and inference code
├── frontend/               # Frontend user interface
├── final_report/           # Project report documentation
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Ignored files and folders

Technologies Used

Python

Deep Learning (YOLO-based model)

OpenCV

FastAPI (Backend)

Streamlit (Frontend)

Prerequisites

Python 3.9 or above

pip (Python package manager)

Git

Setup Instructions
Step 1: Clone the Repository
git clone https://github.com/springboardmentor34567-crypto/Circuit-Guard---PCB-Defect-detaction.git

cd Circuit-Guard---PCB-Defect-detaction

Step 2: Create a Virtual Environment (Optional but Recommended)
python -m venv venv

venv\Scripts\activate


Note: The virtual environment is not included in the repository as per submission guidelines.

Step 3: Install Dependencies
pip install -r requirements.txt

Running the Backend

The backend is responsible for handling inference and API requests.

Step 4: Start the Backend Server
cd backend

uvicorn main:app --reload


Backend will start at: http://127.0.0.1:8000

API documentation (if enabled): http://127.0.0.1:8000/docs

Running the Frontend

The frontend allows users to upload PCB images and view defect detection results.

Step 5: Start the Frontend Application

Open a new terminal and navigate to the project root:

cd Circuit-Guard---PCB-Defect-detaction

streamlit run frontend/app.py


Frontend will open in the browser automatically

Default URL: http://localhost:8501

Usage Instructions

Start the backend server

Start the frontend application

Upload PCB images through the frontend

View detected defects and results on the interface

Notes

Datasets are not included in this repository as per submission guidelines.

Trained model files are not included due to size and reproducibility constraints.

The project report is available in the final_report/ directory.

Author

Name: Sudarshan

Branch: sudarshan_s

License

This project is submitted for academic evaluation purposes only.