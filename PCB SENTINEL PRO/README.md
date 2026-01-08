# 🔍 PCB Sentinel Pro
**A Powerful Dual-Architecture PCB Defect Detection System**

An enterprise-grade PCB (Printed Circuit Board) defect detection system built with FastAPI backend and Streamlit frontend, powered by YOLOv8 deep learning model. This system automatically detects and classifies defects in PCB images with high accuracy in real-time, featuring both REST API access and an interactive web dashboard.

---

## 🌟 Features

### Core Functionality
- ⚡ **Real-time Defect Detection**: Upload PCB images and get instant defect analysis
- 📦 **Batch Processing**: Process multiple images simultaneously with progressive results
- 🎯 **High Accuracy**: Powered by fine-tuned YOLOv8 model
- 📊 **Visual Analytics**: Interactive dashboard with defect distribution and severity badges
- 🔄 **Dual Interface**: REST API for integration + Streamlit UI for visualization
- 🔍 **Search & Filter**: Quickly find specific images in batch results

### Defect Types Detected
- ⚠️ **Missing Hole** - Drilling defects
- 🐭 **Mouse Bite** - Edge irregularities  
- ⚡ **Open Circuit** - Broken connections
- 🔌 **Short Circuit** - Unwanted connections
- 📍 **Spur** - Extra copper protrusions
- 💎 **Spurious Copper** - Unwanted copper residue

### Defect Severity Classification
- 🔴 **HIGH**: `open_circuit`, `short`, `missing_hole` (Critical failures)
- 🟡 **MEDIUM**: `mouse_bite`, `spur` (Quality issues)
- 🟢 **LOW**: `spurious_copper` (Minor defects)

### Download Options

#### Individual Downloads (Per Image)
- 🖼️ Annotated images with bounding boxes
- 📄 Text report with defect details
- 📦 ZIP bundle (image + report)

#### Batch Downloads
- 📊 Combined CSV report for all images
- 📦 Master ZIP with all images, reports, and summary
- 📋 Batch summary with PASS/FAIL status

### User Experience
- 🎨 **Modern UI**: Beautiful dark theme with orange accent colors
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile
- 🖱️ **Drag & Drop**: Easy file upload with multi-select support
- 📈 **Real-time Feedback**: Live progress bars and status indicators
- 🔎 **Modal View**: Click thumbnails for high-resolution inspection
- 📖 **API Documentation**: Interactive Swagger UI at `/docs`
- ⚙️ **Adjustable Parameters**: Tune confidence and IoU thresholds

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│          Streamlit Dashboard            │
│           (applifinal.py)               │
│      Port 8501 - User Interface         │
│  ┌─────────────────────────────────┐   │
│  │  • File Upload & Batch Queue    │   │
│  │  • Results Grid/List View       │   │
│  │  • Download Manager             │   │
│  │  • Search & Filter              │   │
│  └──────────┬──────────────────────┘   │
└─────────────┼──────────────────────────┘
              │ HTTP POST /detect/
              │ (Image + conf/iou params)
              ▼
┌─────────────────────────────────────────┐
│         FastAPI Backend                 │
│           (backend.py)                  │
│      Port 8000 - REST API               │
│  ┌─────────────────────────────────┐   │
│  │     YOLOv8 Inference Engine     │   │
│  │   ┌─────────────────────────┐   │   │
│  │   │   MODEL/best.pt         │   │   │
│  │   │   (Trained Weights)     │   │   │
│  │   └─────────────────────────┘   │   │
│  │  • Image preprocessing          │   │
│  │  • Defect detection             │   │
│  │  • Annotation rendering         │   │
│  │  • Base64 encoding              │   │
│  └──────────┬──────────────────────┘   │
└─────────────┼──────────────────────────┘
              │ JSON Response
              │ (Defects + Base64 image)
              ▼
┌─────────────────────────────────────────┐
│      Streamlit Dashboard                │
│  • Displays annotated results           │
│  • Severity badges & statistics         │
│  • Export options (ZIP/CSV/TXT)         │
└─────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- ✅ Python 3.10 or higher (3.11–3.13 recommended)
- ✅ pip (Python package manager)
- ✅ 4GB+ RAM recommended for ML model inference
- ✅ Modern web browser (Chrome, Firefox, Edge)
- ⚡ (Optional) NVIDIA GPU with CUDA for faster processing

### Installation

#### 1. Clone or Download the Repository
```bash
cd "d:\infosys spring board\PCB SENTINEL PRO"
```

#### 2. Create Virtual Environment
```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install fastapi uvicorn[standard] ultralytics pillow requests pandas streamlit python-multipart
```

**GPU Acceleration (Optional)**  
If you have an NVIDIA GPU, install CUDA-enabled PyTorch:
```bash
# Visit https://pytorch.org/get-started/locally/ for platform-specific instructions
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Verify Model Weights
Ensure `MODEL/best.pt` exists. If training a new model, see [Model Training](#-model-training-details) section.

---

## 🎮 Running the Application

You need **two terminals** running simultaneously:

### Terminal 1: Start FastAPI Backend
```bash
uvicorn backend:app --host 127.0.0.1 --port 8000 --reload
```

**Verify Backend is Running:**
- Health Check: http://127.0.0.1:8000/
- Interactive API Docs: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

### Terminal 2: Start Streamlit Frontend
```bash
streamlit run applifinal.py
```

**Access the Dashboard:**
- Main App: http://localhost:8501
- The app will auto-open in your default browser

---

## 📖 Usage Guide

### Step 1: Upload Images
1. Click the **"📂 Drop PCB Images Here"** area or drag files directly
2. Supported formats: **PNG, JPG, JPEG**
3. Multiple images can be uploaded at once (batch mode)

### Step 2: Configure Detection Parameters (Optional)
Open the **sidebar** (☰) to adjust:
- **Confidence Threshold** (0.0 - 1.0, default: 0.25)  
  Higher = fewer false positives, may miss defects
- **IoU Threshold** (0.0 - 1.0, default: 0.45)  
  Controls overlapping box suppression

### Step 3: Start Detection
1. Click **🚀 Analyze N Images via API** button
2. Watch the progress bar as images are processed
3. Backend performs YOLOv8 inference and returns results

### Step 4: View Results

#### Results View Options
- **Grid Layout**: Thumbnail gallery with quick overview
- **List Layout**: Side-by-side original vs annotated comparison

#### Interactive Elements
- **🔍 Zoom**: Click to view high-resolution annotated image in modal
- **🔢 Count Button**: Shows defect count with PASS/FAIL status
- **📄 Details**: Opens modal with:
  - Per-defect breakdown (Type, Confidence, Bounding Box)
  - Severity badges (High/Medium/Low)
  - Download ZIP (image + report)

#### Search & Filter
Use the **🔍 Search** box to filter images by filename

### Step 5: Download Reports

#### Individual Downloads (in Details modal)
- **⬇️ Download This Report**: ZIP with annotated image + text report

#### Batch Downloads (bottom of results)
- **📦 Download Batch Report**: Master ZIP containing:
  - `images/` folder with all annotated images
  - `reports/` folder with per-image text reports
  - `Master_Summary.txt` with batch statistics

---

## 🛠️ Technology Stack

### Backend
| Technology | Purpose |
|-----------|---------|
| FastAPI | Modern Python web framework for REST API |
| Uvicorn | ASGI server for production deployment |
| YOLOv8 (Ultralytics) | Object detection deep learning model |
| PyTorch | Deep learning framework |
| Pillow | Image processing and manipulation |
| python-multipart | File upload handling |

### Frontend
| Technology | Purpose |
|-----------|---------|
| Streamlit | Interactive web dashboard framework |
| Pandas | Data manipulation and CSV export |
| Requests | HTTP client for API communication |
| Base64 | Image encoding/decoding |

### AI/ML
| Component | Details |
|----------|---------|
| YOLOv8 | Pre-trained and fine-tuned for PCB defects |
| PyTorch | Deep learning backend |
| CUDA | GPU acceleration (optional) |
| Training Platform | Google Colab with cloud GPUs |

---

## 📁 Project Structure

```
PCB SENTINEL PRO/
├── applifinal.py                    # Streamlit frontend (UI)
├── backend.py                       # FastAPI backend (REST API)
├── README.md                        # Documentation (this file)
├── requirements.txt                 # Python dependencies (optional)
├── MODEL/
│   └── best.pt                      # YOLOv8 trained model weights
├── REPORT/                          # Output folder (auto-created)
├── YOLOMODEL FINETUNING/
│   └── YOLOv8 Colab Notebook.ipynb  # Training/fine-tuning notebook
└── __pycache__/                     # Python cache (auto-generated)
```

---

## 📊 Model Training Details

### Dataset
- **Source**: PCB defect dataset with annotated bounding boxes
- **Classes**: 6 defect types (missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper)
- **Split**: 80% Training, 20% Validation
- **Format**: YOLO format (converted from VOC XML if needed)

### Training Process
The model was trained iteratively on Google Colab using the provided notebook:

1. **Initial Training**: 10 epochs with YOLOv8s base model
2. **Continued Training**: +10 epochs for refinement
3. **Fine-tuning**: +20 epochs with adjusted hyperparameters
4. **Final Training**: +50 epochs with data augmentation

### Hyperparameters
```yaml
Model: YOLOv8s (small)
Batch Size: 4-8 (depending on GPU memory)
Image Size: 640x640
Optimizer: SGD with cosine learning rate
Learning Rate: 0.001 (initial)
Data Augmentation:
  - Mosaic
  - Mixup
  - HSV color jitter
  - Random rotation
  - Random scale
```

### Results
- **Accuracy**: 90%+ on validation set
- **Inference Speed**: <2 seconds per image (CPU), <0.5s (GPU)
- **Model Size**: ~25MB (`best.pt`)

### Retraining Your Model
1. Open [YOLOMODEL FINETUNING/YOLOv8 Colab Notebook.ipynb](YOLOMODEL%20FINETUNING/YOLOv8%20Colab%20Notebook.ipynb)
2. Upload your annotated dataset (YOLO format)
3. Adjust hyperparameters as needed
4. Run training cells
5. Download `best.pt` and replace `MODEL/best.pt`
6. Restart the backend server

---

## 🌐 REST API Reference

### Base URL
```
http://127.0.0.1:8000
```

### Endpoints

#### GET `/`
**Health Check**
```bash
curl http://127.0.0.1:8000/
```
**Response:**
```json
{
  "message": "PCB Defect Detection API is Running"
}
```

#### POST `/detect/`
**Detect Defects in Uploaded Image**

**Request (multipart/form-data):**
- `file` (required): Image file (PNG, JPG, JPEG)
- `conf` (optional): Confidence threshold (float, default: 0.25)
- `iou` (optional): IoU threshold (float, default: 0.45)

**Example (curl):**
```bash
curl -X POST "http://127.0.0.1:8000/detect/" \
  -F "file=@sample.jpg" \
  -F "conf=0.25" \
  -F "iou=0.45"
```

**Example (PowerShell):**
```powershell
$form = @{
    file = Get-Item -Path ".\sample.jpg"
    conf = "0.25"
    iou = "0.45"
}
Invoke-RestMethod -Uri "http://127.0.0.1:8000/detect/" -Method Post -Form $form
```

**Response (JSON):**
```json
{
  "filename": "board.jpg",
  "defect_count": 2,
  "defects": [
    {
      "Type": "short",
      "Confidence": 0.9123,
      "Box": [120.5, 85.3, 245.8, 198.6]
    },
    {
      "Type": "spur",
      "Confidence": 0.7734,
      "Box": [340.2, 150.7, 425.1, 210.3]
    }
  ],
  "annotated_image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

---

## 🐛 Troubleshooting

### Backend Issues

**❌ "Model not found" error**
```bash
# Solution: Verify model file exists
ls MODEL/best.pt
# If missing, download or train the model
```

**❌ Backend won't start / Port already in use**
```bash
# Solution: Kill process on port 8000
# Windows PowerShell
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process

# Linux/Mac
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn backend:app --port 8001
```

**❌ Connection refused from Streamlit**
- Ensure backend is running (check Terminal 1)
- Verify `API_URL` in `applifinal.py` matches backend address
- Check firewall settings blocking localhost

### Frontend Issues

**❌ Streamlit won't start**
```bash
# Solution: Reinstall Streamlit
pip uninstall streamlit
pip install streamlit

# Or specify port manually
streamlit run applifinal.py --server.port 8502
```

**❌ "Cannot connect to Backend" error**
- Start backend first (Terminal 1), then frontend (Terminal 2)
- Check backend health at http://127.0.0.1:8000/

### Installation Issues

**❌ PyTorch / Ultralytics installation fails**
```bash
# Solution: Try Python 3.11 or 3.12
python --version

# Or install specific torch version
pip install torch==2.1.0 torchvision==0.16.0
```

**❌ CUDA not detected (GPU not used)**
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA toolkit from NVIDIA
# Then reinstall PyTorch with CUDA support
```

### Performance Issues

**❌ Slow inference (>5 seconds per image)**
- Use a smaller image size (resize before upload)
- Increase confidence threshold (fewer detections = faster)
- Use GPU acceleration (see GPU setup above)
- Close other resource-intensive applications

**❌ Out of memory errors**
- Reduce batch size in training notebook
- Lower image resolution
- Use CPU instead of GPU (slower but more stable)

---

## 📄 License
This project is open source and available under the **MIT License**.

---

## 👨‍💻 Author
**Infosys Springboard Team**  
*PCB Sentinel Pro - Enterprise Defect Detection*

---

## 🌟 Show Your Support
If you find this project helpful:
- ⭐ Star this repository
- 🔗 Share with others
- 🐛 Report bugs via Issues
- 💡 Suggest new features
- 🤝 Contribute improvements

---

## 🙏 Credits
- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **FastAPI**: [Sebastián Ramírez](https://fastapi.tiangolo.com/)
- **Streamlit**: [Streamlit Inc.](https://streamlit.io/)
- **PyTorch**: [Meta AI Research](https://pytorch.org/)
