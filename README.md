# 🚀 Circuit Guard – PCB Defect Detection System

This project focuses on detecting defects in **Printed Circuit Boards (PCBs)** using a **deep learning–based approach**.  
The system automates PCB inspection and presents defect detection results through a **simple and interactive frontend interface**.

---

## 📌 Project Overview

Printed Circuit Boards are critical components in electronic systems, and defects in PCBs can lead to system failures.  
This project aims to **detect common PCB defects** using **computer vision and deep learning techniques**, improving inspection accuracy and efficiency.

### 🔹 Application Components

- **Backend**
  - Handles model inference
  - Provides API services

- **Frontend**
  - Allows users to upload PCB images
  - Displays defect detection results

---

## 📂 Project Structure

Circuit-Guard---PCB-Defect-detaction/
├── backend/ # Backend logic and inference code
├── frontend/ # Frontend user interface
├── final_report/ # Project report documentation
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Ignored files and folders

---

## 🛠️ Technologies Used

- **Python**
- **Deep Learning (YOLO-based model)**
- **OpenCV**
- **FastAPI** (Backend)
- **Streamlit** (Frontend)

---

## ✅ Prerequisites

- Python 3.9 or above  
- pip (Python package manager)  
- Git  

---

## ⚙️ Setup Instructions

### 🔹 Step 1: Clone the Repository

git clone https://github.com/springboardmentor34567-crypto/Circuit-Guard---PCB-Defect-detaction.git

cd Circuit-Guard---PCB-Defect-detaction

### 🔹 Step 2: Create a Virtual Environment (Optional)

python -m venv venv

venv\Scripts\activate

Note: Virtual environments are not included in the repository.

### 🔹 Step 3: Install Dependencies

pip install -r requirements.txt

### ▶️ Running the Backend

cd backend
uvicorn main:app --reload

Backend URL: http://127.0.0.1:8000

API Docs: http://127.0.0.1:8000/docs

### 🖥️ Running the Frontend
Open a new terminal, then run:

cd Circuit-Guard---PCB-Defect-detaction

streamlit run frontend/app.py

Frontend URL: http://localhost:8501

### 🧪 Usage Instructions

Start the backend server

Start the frontend application

Upload PCB images

View detected defects and results

### 📝 Notes
Datasets are not included as per submission guidelines

Trained model files are not included due to size constraints

Project report is available in the final_report/ directory

### 👤 Author
Name: Sudarshan

Branch: sudarshan_s

### 📜 License
This project is submitted for academic evaluation purposes only.
