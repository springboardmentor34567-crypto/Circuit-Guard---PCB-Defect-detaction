# 🛡️ CircuitGuard: PCB Defect Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blueviolet?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Status](https://img.shields.io/badge/Status-COMPLETED-success?style=for-the-badge)

---

## 📖 Overview

**CircuitGuard** is an AI-powered defect detection system designed to identify faults in **Printed Circuit Boards (PCBs)** with high accuracy.  
It leverages **YOLOv8 deep learning models** to automate inspection, reduce manual effort, and ensure quality assurance in electronics manufacturing.

The project evolved through two major phases:
- **🚀 Streamlit Prototype** – A rapid proof-of-concept interface for quick experimentation and validation.
- **🌐 Flask Web Application** – A production-ready, full-stack solution with database integration, analytics dashboards, and report generation.

---

### 🚀 Streamlit Version (Rapid Prototype)
<img src="https://github.com/user-attachments/assets/099f69d7-b44b-43f2-92eb-e8b6c4562076" width="80%" alt="Streamlit Demo">

---

### 🌐 Flask Web App (Final Production Build)
<img src="https://github.com/user-attachments/assets/6ec57688-71a0-4146-80c4-ff2b4981d0e2" width="80%" alt="Flask Demo">

---

## 🎓 Academic Context

This project was developed as a **capstone deliverable** during the **Infosys Springboard Internship (8 weeks)**.  
It represents the practical application of:
- **Deep Learning** (YOLOv8, Transfer Learning)
- **Computer Vision** (Bounding box detection, confidence scoring)
- **Full-Stack Development** (Flask, SQLAlchemy, Jinja2)
- **Data Engineering** (Dataset preparation, XML-to-YOLO conversion)

---

## 📅 Project Milestones & Status

| Milestone | Module | Task Description | Status |
| :--- | :--- | :--- | :--- |
| **Milestone 1** | Dataset Preparation | XML annotations converted to YOLO format, folder restructuring, and class mapping. | ![Done](https://img.shields.io/badge/DONE-success?style=flat-square) |
| **Milestone 2** | Model Training (YOLOv8) | Transfer learning applied with high-resolution (1024px) training for precise detection. | ![Done](https://img.shields.io/badge/DONE-success?style=flat-square) |
| **Milestone 3** | Inference & Validation | Bounding box generation, confidence scoring, and model persistence. | ![Done](https://img.shields.io/badge/DONE-success?style=flat-square) |
| **Milestone 4** | Backend API Development | Flask-based REST API enabling real-time inference. | ![Done](https://img.shields.io/badge/DONE-success?style=flat-square) |
| **Milestone 5** | Frontend Web UI & Deployment | Streamlit prototype + Flask/Jinja2 dashboard with SQL history tracking. | ![Done](https://img.shields.io/badge/DONE-success?style=flat-square) |

---

## 🛠️ Tech Stack

<div align="center">

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-1E90FF?style=for-the-badge&logo=yolo&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/index.html)
[![Jinja2](https://img.shields.io/badge/Jinja2-B41717?style=for-the-badge&logo=jinja&logoColor=white)](https://jinja.palletsprojects.com/)

</div>

---

## 📂 Project Structure & File Architecture

CircuitGuard follows a **Model-View-Controller (MVC)** architecture with **Flask Blueprints** for modularity.  
Each component is designed for scalability, maintainability, and clarity.

```text
PCB_DEFECT_DETECTOR/
├── models/
│   └── model_loader.py       # Loads YOLOv8 model once at startup for efficient inference.
├── routes/
│   ├── dashboard.py          # Handles homepage logic and navigation.
│   ├── inspection.py         # Manages image uploads, runs inference, and displays results.
│   ├── metrices.py           # Fetches SQL data to visualize defect statistics.
│   └── reports.py            # Generates downloadable PDF/CSV inspection reports.
├── services/
│   ├── analytics.py          # SQLAlchemy integration for scan history and defect analytics.
│   ├── inference.py          # Core AI worker: preprocessing, YOLOv8 prediction, bounding box rendering.
│   └── reports.py            # Formats inspection data into user-friendly reports.
├── static/
│   ├── css/style.css         # Custom styling for UI.
│   ├── js/main.js            # Frontend logic for previews and async API calls.
│   ├── images/background.jpg # UI assets.
│   ├── reports_storage/      # Temporary storage for generated reports.
│   ├── results/              # Processed images with bounding boxes.
│   └── uploads/              # Raw user-uploaded images.
├── templates/
│   ├── base.html             # Master layout (navbar, footer).
│   ├── dashboard.html        # Landing page.
│   ├── inspection.html       # Upload interface for defect detection.
│   ├── metrices.html         # Analytics dashboard with charts.
│   └── reports.html          # Report listing and download page.
├── pcb_yolo_model_v1.pt      # Trained YOLOv8 weights.
├── app.py                    # Entry point: initializes Flask app, registers Blueprints, configures DB.
├── config.py                 # Centralized settings (paths, DB URLs, secrets).
├── requirements.txt          # Dependencies list.
└── streamlit.py              # Legacy prototype for rapid testing.
