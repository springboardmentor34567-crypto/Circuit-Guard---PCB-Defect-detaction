# 🛡️ CircuitGuard Pro: Automated PCB Defect Detection System

**An Industrial-Grade AI solution for detecting and classifying PCB defects with 98.1% accuracy.**

---

## 📖 Overview
**CircuitGuard Pro** is an automated optical inspection (AOI) tool designed to replace manual PCB inspection. It combines **Computer Vision (OpenCV)** for precise defect localization and **Deep Learning (YOLOv8)** for classification.

The system is built using a **Microservices Architecture**, separating the heavy inference logic (FastAPI) from the user interface (Streamlit), allowing for scalable, batch-processing capabilities optimized for **Apple Silicon (M4)** hardware.

---

## 🚀 Key Features
- **Hybrid Detection Pipeline:** Uses Reference-Based Image Subtraction for localization + YOLOv8 for classification.
- **Microservices Architecture:** Decoupled Backend (FastAPI) and Frontend (Streamlit) for high performance.
- **Batch Processing:** Inspect multiple PCBs simultaneously with auto-template matching.
- **Hardware Accelerated:** Optimized for Apple M4 Neural Engine using PyTorch MPS backend (~10ms inference).
- **Automated Reporting:** Generates downloadable ZIP packages containing annotated images and CSV logs.
- **6 Defect Classes:** Accurately identifies Open Circuit, Short, Mouse Bite, Spur, Pin Hole, and Spurious Copper.

---

## 🛠️ Tech Stack
- **Language:** Python 3.9+
- **Frontend:** Streamlit
- **Backend API:** FastAPI (Uvicorn Server)
- **Computer Vision:** OpenCV (cv2)
- **AI Model:** Ultralytics YOLOv8 (Transfer Learning)
- **Data Handling:** Pandas, NumPy

---

## 📂 Project Structure
```text
CircuitGuard_Pro/
├── backend.py              # FastAPI Inference Server (The Brain)
├── frontend.py             # Streamlit User Interface (The Face)
├── requirements.txt        # Project Dependencies
├── README.md               # Documentation
│
├── models/
│   └── pcb_defect_classifier/
│       └── weights/
│           └── best.pt     # Trained YOLOv8 Model (98.1% Acc)
│
├── database/
│   └── templates/          # Reference "Golden" PCB Images (e.g., 00041.jpg)
│
└── 01_smart_process.py     # (Utility) Script used for data preprocessing