# 🔍 PCB Defect Detection System

A powerful web-based PCB (Printed Circuit Board) defect detection system built with **FastAPI** and **YOLOv8 deep learning model**.  
This system automatically detects and classifies 6 types of defects in PCB images with high accuracy in real-time.

---

## 🌟 Features

### Core Functionality
- **Real-time Defect Detection**: Upload PCB images and get instant defect analysis
- **Batch Processing**: Process multiple images simultaneously with progressive results
- **High Accuracy**: Powered by YOLOv8 trained on 1000+ PCB images
- **Visual Analytics**: Interactive charts showing defect distribution
- **Grid & List View**: Two viewing modes for better user experience

### Defect Types Detected
- ⚠️ **Missing Hole** - Drilling defects
- 🐭 **Mouse Bite** - Edge irregularities  
- ⚡ **Open Circuit** - Broken connections
- 🔌 **Short Circuit** - Unwanted connections
- 📍 **Spur** - Extra copper protrusions
- 💎 **Spurious Copper** - Unwanted copper residue

### Download Options
- **Individual Downloads**
  - Annotated images with bounding boxes
  - Original images
  - CSV reports per image
- **Batch Downloads**
  - Combined CSV report for all images
  - ZIP file with all annotated images

### User Experience
- **Modern UI**: Beautiful gradient design with smooth animations
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Drag & Drop**: Easy file upload with drag-and-drop support
- **Real-time Feedback**: Live progress indicators and status updates
- **Modal View**: Click thumbnails to view detailed results
- **API Documentation**: Interactive Swagger UI at `/docs`

---

## 🏗️ Architecture

```
┌─────────────┐
│   Browser   │ ← User Interface (HTML/CSS/JS)
│  Port 8000  │
└──────┬──────┘
       │ HTTP Request
       ▼
┌─────────────────────────────────┐
│      FastAPI Server             │
│  ┌──────────┐  ┌─────────────┐ │
│  │ Frontend │  │   Backend   │ │
│  │Templates │  │   API       │ │
│  └──────────┘  └──────┬──────┘ │
│                       │         │
│                ┌──────▼──────┐  │
│                │  YOLOv8 AI  │  │
│                │   Model     │  │
│                └─────────────┘  │
└─────────────────────────────────┘
       │
       ▼ Results (JSON + Base64)
┌─────────────┐
│   Browser   │ ← Displays Results
└─────────────┘
```

 

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- 4GB+ RAM recommended for ML model inference
- Modern web browser (Chrome, Firefox, Edge)
- (Optional) NVIDIA GPU with CUDA for faster processing

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/deepakpatidar1210/PCB-Defect-Detect.git
cd PCB-Defect-Detect
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```


#### 4. Run the Application
```bash
python main.py
```

#### 5. Access the Application
Open your browser and navigate to:
- **Main App**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📖 Usage Guide

### Step 1: Upload Images
- Click the upload area or drag & drop PCB images
- Supported formats: PNG, JPG, JPEG
- Multiple images can be uploaded at once

### Step 2: Detect Defects
- Click the **🚀 Start Detection** button
- Watch images being processed progressively
- Real-time progress bar shows processing status

### Step 3: View Results
**List View (Default):**
- Side-by-side comparison (Original vs Annotated)
- Detection table with defect details
- Individual download buttons

**Grid View:**
- Thumbnail gallery of all results
- Click any image to open detailed modal
- Perfect for quick overview

### Step 4: Download Reports
**Individual Downloads:**
- 💾 Download Annotated Image
- 📥 Download Original Image  
- 📄 Download CSV Report

**Batch Downloads:**
- 📊 Download Combined CSV (all detections)
- 📦 Download All Images (ZIP)

### Step 5: View Analytics
- Overall summary statistics
- Interactive bar chart (Chart.js)
- Defect distribution breakdown
- Images processed count

---

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **YOLOv8** - Object detection model (Ultralytics)
- **OpenCV** - Image processing
- **NumPy** - Numerical operations
- **Pillow** - Image handling

### Frontend
- **Jinja2** - Template engine
- **HTML5/CSS3** - Structure and styling
- **JavaScript** - Interactive functionality
- **Chart.js** - Data visualization
- **Fetch API** - AJAX requests

### AI/ML
- **YOLOv8** - Pre-trained and fine-tuned
- **PyTorch** - Deep learning framework
- **CUDA** - GPU acceleration (optional)

### Training Platform
- **Google Colab** - Cloud GPU training
- **Data Augmentation** - Mosaic, Mixup, Rotation
- **Iterative Training** - 10 → 20 → 50 epochs

---

## 📁 Project Structure

```
PCB-DEFECT-DETECTION/
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
├── README.md                 # Documentation
├── templates/
│   └── index.html            # Frontend HTML
├── static/
│   ├── style.css             # Styling
│   └── script.js             # Frontend logic
├── models/
│   └── best.pt               # YOLOv8 trained model
├── uploads/                  # Temporary storage (auto-created)
└── results/                  # Detection results (auto-created)
```

---

## 📊 Model Training Details

### Dataset
- **Source**: PCB defect dataset with XML annotations
- **Classes**: 6 defect types
- **Split**: 80% Training, 20% Validation
- **Format**: Converted from VOC XML to YOLO format

### Training Process
1. **Initial Training**: 10 epochs with YOLOv8s base model
2. **Continued Training**: +10 epochs for refinement
3. **Fine-tuning**: +20 epochs with adjusted hyperparameters
4. **Final Training**: +50 epochs with data augmentation

### Hyperparameters
- **Batch Size**: 4-8
- **Image Size**: 640x640
- **Optimizer**: SGD with cosine learning rate
- **Learning Rate**: 0.001 (initial)
- **Data Augmentation**: Mosaic, Mixup, HSV, Rotation, Scale

### Results
- **Accuracy**: 95%+ on validation set
- **Inference Speed**: <2 seconds per image (CPU)
- **Model Size**: ~25MB (best.pt)

---

 
 

## 📄 License

This project is open source and available under the MIT License.

---


## 👨‍💻 Author

**Deepak Patidar**  
GitHub: [@deepakpatidar1210](https://github.com/deepakpatidar1210)

---

 

## 🌟 Show Your Support

If you find this project helpful, please:
- ⭐ Star this repository
- 🔗 Share with others
- 🐛 Report bugs
- 💡 Suggest new features

---

<p align="center">
  <strong>Made with ❤️ by Deepak Patidar</strong><br>
  🚀 PCB Defect Detection System - Powered by AI
</p>
