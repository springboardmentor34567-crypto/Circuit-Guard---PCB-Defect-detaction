# 🚀 CircuitGuard: PCB Defect Detection using Deep Learning

## 📌 Overview

Printed Circuit Boards (PCBs) form the backbone of modern electronic systems. Even minor manufacturing defects can lead to device malfunction, short circuits, or long-term reliability failures. Ensuring PCB quality is therefore a critical requirement in electronics manufacturing.

**CircuitGuard** is an **AI-powered PCB defect detection system** that automates PCB inspection using **deep learning and computer vision**, providing fast, accurate, and scalable defect detection.


## ❗ Problem Statement

Traditional PCB inspection methods rely heavily on **manual visual inspection**, which presents several challenges:

* Time-consuming and labor-intensive
* Inconsistent inspection quality
* Prone to human fatigue and error
* Not scalable for high-volume production

As manufacturing throughput increases, manual inspection becomes impractical and unreliable.

## 💡 Proposed Solution

CircuitGuard eliminates these limitations by introducing an **end-to-end automated inspection pipeline**.

The system:

* Uses a **YOLO-based deep learning model** for real-time defect detection
* Applies **image processing techniques** to analyze PCB images
* Automatically **detects and classifies PCB defects with high precision**
* Exposes model inference through a **FastAPI backend**
* Provides an intuitive **Streamlit-based frontend** for easy interaction and visualization


## ⚙️ Key Features

* 🔍 Automated PCB defect detection and classification
* 🧠 YOLO-based deep learning model for real-time inference
* ⚡ High-speed and accurate defect localization
* 🌐 **FastAPI backend** for scalable and efficient model serving
* 🖥️ **Streamlit frontend** for user-friendly inspection and result visualization
* 📊 Consistent performance compared to manual inspection


## 🏗️ System Architecture (High Level)

1. **PCB Image Upload (Streamlit UI)**
2. **Request Handling (FastAPI Backend)**
3. **Image Preprocessing**
4. **YOLO-Based Defect Detection Model**
5. **Defect Classification & Bounding Box Generation**
6. **Results Visualization on Frontend**


## 🧰 Tech Stack

### 🔹 Backend

* FastAPI
* Deep Learning Inference Engine
* YOLO Model

### 🔹 Frontend

* Streamlit

### 🔹 Core Technologies

* Python
* Computer Vision
* Image Processing
* Deep Learning


## 🎯 Use Cases

* Electronics manufacturing quality control
* Automated PCB inspection pipelines
* AI-driven industrial automation
* Academic and research projects in computer vision


## 📈 Impact

By automating PCB inspection, CircuitGuard:

* Reduces inspection time and operational cost
* Minimizes human error
* Improves long-term product reliability
* Scales efficiently with increasing production demands


## 🤝 Contribution

Contributions, enhancements, and suggestions are welcome.
Feel free to fork the repository and submit a pull request.

