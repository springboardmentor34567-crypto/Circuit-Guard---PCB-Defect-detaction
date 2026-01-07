# PCB Defect Detection System 🔍

A neat, professional, and high-performance system for detecting defects on Printed Circuit Boards (PCBs) using the **YOLOv8** object detection model. This project provides two user authentication interfaces: a modern **Streamlit** dashboard and a robust **Flask** web application.

## 📌 Project Overview
Printed Circuit Boards (PCBs) are critical components in modern electronics. Visual inspection for defects is often manual and prone to errors. This project automates the detection of common PCB defects using computer vision and deep learning.

**Key Features:**
- **High Accuracy:** Powered by a fine-tuned YOLOv8s model.
- **Dual Interface:** Choose between Streamlit (for quick, interactive use) or Flask (for a traditional web app structure).
- **Batch Processing:** Upload multiple images and process them simultaneously.
- **Detailed Results:** View original vs. annotated images side-by-side with bounding boxes and confidence scores.
- **Export Options:** Download results as individual images, YOLO-formatted `.txt` annotations, or a complete ZIP archive.

## 📊 Dataset
The model was trained on the **PCB Defects Dataset**, available on Kaggle.
- **Source:** [PCB Defects Dataset (Kaggle)](https://www.kaggle.com/datasets/akhatova/pcb-defects)
- **Defect Classes (6 Types):**
  1. `Missing_hole` 🔴
  2. `Mouse_bite` 🟠
  3. `Open_circuit` 🟡
  4. `Short` 🟢
  5. `Spur` 🔵
  6. `Spurious_copper` 🟣

## 🧠 Model Pipeline Details
The core detection engine is built using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) on PyTorch.

- **Base Model:** `yolov8s.pt` (Small version for balance of speed and accuracy).
- **Fine-tuning Strategy:**
  - **Epochs:** 50 (Fine-tuning phase)
  - **Batch Size:** 16
  - **Image Size:** 640x640
  - **Backbone Freezing:** First 10 layers frozen to preserve low-level feature extraction capabilities.
  - **Optimizer:** AdamW

### 📈 Performance Metrics
The model achieves exceptional performance on the validation set:
- **mAP@50:** ~95.1% 🚀
- **mAP@50-95:** ~52.2%
- **Precision:** ~94.0%
- **Recall:** ~92.3%
- **Classification Accuracy:** ~99.0%

*(Detailed training logs and evaluation reports are available in the `notebooks/` directory).*

## 🛠️ Tech Stack
- **Deep Learning:** PyTorch, Ultralytics YOLOv8
- **Web Frameworks:** Streamlit, Flask, Jinja2
- **Data Processing:** OpenCV, PIL, NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn

## 📂 Project Structure
```
.
├── app.py                      # Streamlit Application (Main Interface)
├── flask_app.py                # Flask Application (Alternative Interface)
├── requirements_streamlit.txt  # Dependencies for Streamlit
├── requirements_flask.txt      # Dependencies for Flask
├── notebooks/                  # Training & Experimentation
│   ├── 01_prepare_yolo_dataset.ipynb
│   ├── 04_train_yolov8_ultralytics.ipynb  # Main training notebook
│   └── ...
├── artifacts/                  # Trained models & weights
│   └── ultralytics/pcb_yolov8_s/weights/best.pt
├── static/                     # Flask static assets (CSS, JS)
├── templates/                  # Flask HTML templates
└── README.md                   # Project Documentation
```

## 🚀 Installation & Usage

### 1. Prerequisites
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone <repository_url>
cd <repository_name>

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Running the Streamlit App (Recommended)
The Streamlit app offers a highly interactive and polished UI with a carousel view for results.

1.  **Run the App:**
    ```bash
    streamlit run app.py
    ```
2.  **Access:** Open your browser at `http://localhost:8501`.

### 3. Running the Flask App
The Flask app provides a standard web application server structure.

1.  **Run the App:**
    ```bash
    python flask_app.py
    ```
2.  **Access:** Open your browser at `http://localhost:5000`.

## 📝 How to Use
1.  **Load Model:** Upon launching, ensure the model path is correct (default is `artifacts/.../best.pt`) and click "Load Model".
2.  **Upload:** Drag and drop one or multiple PCB images (`.jpg`, `.png`).
3.  **Process:** Click the "Process" button. The system will detect defects in all images.
4.  **Analyze & Download:**
    - Navigate to the **Results** tab.
    - View the summary statistics (Total detections, defects per image).
    - Downloads individual annotated images or a bulk `.zip` file containing all images and YOLO labels.

---
*Created by Indrasai* ✨
