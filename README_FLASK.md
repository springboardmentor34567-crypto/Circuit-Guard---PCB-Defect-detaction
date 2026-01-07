# PCB Defect Detection - Flask Web Application

A modern Flask web application for detecting defects in PCB images using YOLOv8.

## Features

- 🎨 **Clean & Modern UI** - Professional color palette and responsive design
- 📤 **Batch Image Upload** - Upload multiple PCB images at once
- 🔍 **Real-time Processing** - Process images with progress tracking
- 🖼️ **Side-by-side Comparison** - View original vs annotated images
- 📊 **Results Carousel** - Navigate through results with ease
- 💾 **Individual Downloads** - Download images and annotations separately
- 📦 **ZIP Export** - Download all results as a ZIP file

## Installation

1. Install dependencies:
```bash
pip install -r requirements_flask.txt
```

2. Ensure the YOLOv8 model is trained and available at:
   `artifacts/ultralytics/pcb_yolov8_s/weights/best.pt`

## Running the Application

```bash
python flask_app.py
```

The application will start on `http://localhost:5000`

## Usage

1. **Load Model**: Click "🚀 Load Model" button in the sidebar
2. **Upload Images**: Drag and drop or click to select PCB images
3. **Configure Settings**: Adjust confidence threshold and image size
4. **Process**: Click "🔎 Process All Images & Generate Annotations"
5. **View Results**: Navigate to the Results tab to see processed images
6. **Download**: Download individual files or the complete ZIP

## Project Structure

```
.
├── flask_app.py              # Main Flask application
├── templates/
│   ├── base.html            # Base template
│   ├── index.html           # Upload & processing page
│   └── results.html         # Results display page
├── static/
│   ├── css/
│   │   └── style.css        # Stylesheet with color palette
│   └── js/
│       ├── main.js          # Common utilities
│       ├── upload.js         # Upload & processing logic
│       └── results.js       # Results carousel logic
├── uploads/                 # Temporary upload storage
└── results/                 # Processed results storage
```

## Color Palette

The application uses a clean, professional color scheme:
- **Primary**: Blue (#2563eb) - Trust & Technology
- **Success**: Green (#10b981) - Success states
- **Error**: Red (#ef4444) - Error states
- **Neutral**: Slate grays for backgrounds and text

## API Endpoints

- `GET /` - Main page
- `POST /api/load_model` - Load YOLOv8 model
- `POST /api/upload` - Upload image files
- `POST /api/process` - Process uploaded images
- `GET /api/download_image/<filename>` - Download annotated image
- `GET /api/download_annotation/<filename>` - Download annotation file
- `GET /api/download_zip` - Download all results as ZIP
- `GET /results` - Results page
- `GET /results/<filename>` - Serve result images

## Notes

- Model is loaded on application startup
- Uploaded files are stored in session-based directories
- Results are cleaned up automatically (you may want to add cleanup logic)
- Maximum file size: 100MB per upload

