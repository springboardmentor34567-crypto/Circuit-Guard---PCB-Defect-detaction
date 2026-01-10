🛡️ Circuit Guard – PCB Defect Detection System
📌 Project Overview

Circuit Guard is an intelligent PCB (Printed Circuit Board) defect detection system that uses deep learning (YOLOv8) to automatically identify and localize manufacturing defects in PCB images.
The system helps improve quality control by reducing manual inspection effort, increasing accuracy, and enabling faster defect detection.

This project provides an end-to-end pipeline including dataset preparation, model training, evaluation, and a web-based interface for real-time defect detection.

🎯 Objectives

Detect PCB defects accurately using computer vision

Reduce manual inspection time and human errors

Classify multiple PCB defect types

Provide a user-friendly interface for defect visualization

Generate reliable performance metrics for evaluation

🧠 Defects Detected

The model detects the following PCB defects:

Missing Hole

Mouse Bite

Open Circuit

Short

Spur

Spurious Copper

🏗️ Project Architecture
Circuit Guard
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│
├── model/
│   ├── yolov8n.pt
│   └── best.pt
│
├── training/
│   ├── train.py
│   ├── data.yaml
│
├── evaluation/
│   ├── metrics.csv
│   ├── confusion_matrix.png
│
├── frontend/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── results/
│   ├── detected_images/
│
├── requirements.txt
└── README.md

🛠️ Technologies Used

Programming Language: Python

Deep Learning Framework: PyTorch

Object Detection Model: YOLOv8 (Ultralytics)

Image Processing: OpenCV

Web Framework: Flask / Gradio

Visualization: Matplotlib

Dataset Annotation: YOLO format

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/circuit-guard.git
cd circuit-guard

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Verify Installation
python -c "from ultralytics import YOLO; print('YOLOv8 Installed Successfully')"

🚀 Model Training

To train the YOLOv8 model:

python train.py


Training includes:

Data loading

Image augmentation

Loss optimization

Validation after each epoch

📊 Model Evaluation

Performance is evaluated using:

mAP@50

Precision

Recall

Confusion Matrix

Sample Results
Defect Class	mAP@50
Missing Hole	0.9886
Mouse Bite	0.9793
Open Circuit	0.9711
Short	0.9646
Spur	0.9570
Spurious Copper	0.9511
🌐 Web Application

The web interface allows users to:

Upload PCB images

Run defect detection

View bounding boxes and labels

Download results

To launch the web app:

python app.py

📸 Sample Output

Original PCB image

Bounding box annotated image

Defect class labels with confidence scores

🔐 Advantages

High accuracy and reliability

Fast real-time inference

Scalable for industrial applications

Easy to use and deploy

🚧 Limitations

Performance depends on dataset quality

Requires GPU for faster training

Limited to predefined defect classes

🔮 Future Enhancements

Integrate SAM for precise defect segmentation

Add real-time camera inspection

Cloud deployment

Defect severity analysis

Automated report generation

👨‍💻 Author

Prasanna Kumar
Department of Computer Science / AI
Project: Final Year / Academic Project

📜 License

This project is for academic and research purposes only.
