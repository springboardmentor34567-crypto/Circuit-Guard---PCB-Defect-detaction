# 🔍 Real-Time PCB Defect Detection System

An industrial-grade automated quality control solution utilizing **YOLOv8** for high-precision defect detection, served via a **FastAPI** backend and an interactive **Streamlit** frontend.

## 📌 Overview
This project addresses the critical challenge of manual inspection in PCB manufacturing. By leveraging the **DeepPCB dataset**, the system identifies six common defect types with high accuracy and real-time inference speeds. The architecture is decoupled: a high-performance **FastAPI** server handles the deep learning computations, while a sleek **Streamlit** web interface provides a user-friendly experience for operators.

### **Defect Classes Detected:**
* Missing Hole
* Mouse Bite
* Open Circuit
* Short Circuit
* Spur
* Spurious Copper

---

## 🚀 Key Features
* **Decoupled Architecture:** Scalable **FastAPI** backend and a responsive **Streamlit** frontend.
* **Batch Processing:** Upload and analyze multiple PCB images simultaneously.
* **Real-Time Visualization:** Annotated results with colored bounding boxes and confidence labels.
* **Detailed Analytics:** Automatic generation of a **Confidence Table** (defect counts, class names, and raw coordinates).
* **Data Portability:** Export all processed images and data reports in a single **ZIP file** for audit trails.

---

## 🏗️ System Architecture
The system follows a modern **Client-Server model**:
1.  **Frontend (Streamlit):** Handles user interaction, image uploads, and rendering of detection results.
2.  **Backend (FastAPI):** Receives images via API, processes them through the YOLOv8 engine, and returns structured JSON data.
3.  **Model (YOLOv8):** Optimized with **C2f modules** and **Anchor-Free detection** for superior performance on small-scale industrial defects.



---

## 📊 Performance Metrics
Validated on the DeepPCB dataset:
* **Precision:** 95.47%
* **Recall:** 94.02%
* **mAP50:** 95.03%
* **mAP50-95:** 90.62%

---

## 📂 Project Structure
```text
pcb-defect-detection/
├── models/             # Production weights (e.g., best.pt)
├── training/           # Model development phase (Scripts/Notebooks)
│   └── train_yolo.py   # Script used to train the YOLO model
├── src/
│   ├── main.py        # FastAPI backend logic (Inference Engine)
│   └── app.py         # Streamlit frontend UI (User Interface)
├── data/              # Sample images and pcb_data.yaml config
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
🛠️ Installation & Setup
1. Clone the Repository
Bash

git clone [https://github.com/your-username/pcb-defect-detection.git](https://github.com/your-username/pcb-defect-detection.git)
cd pcb-defect-detection
2. Install Dependencies
Bash

pip install -r requirements.txt
3. Run the Backend (Terminal 1)
Bash

uvicorn src.main:app --reload --port 8000
4. Run the Frontend (Terminal 2)
Bash

streamlit run src.app.py
🛡️ Challenges & Solutions
Small Target Detection: Utilized the YOLOv8 C2f backbone to preserve spatial features for minute defects like "spurs."

Process Efficiency: Implemented an asynchronous FastAPI service to ensure the UI remains responsive during heavy batch inference.

Data Management: Developed a ZIP-export feature to provide manufacturers with an immediate, organized audit trail of all inspections.

🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated. 