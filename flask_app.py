"""
PCB Defect Detection - Flask Web Application

To run this app:
    python flask_app.py

Make sure you have:
    1. Trained the YOLOv8 model (see notebooks/04_train_yolov8_ultralytics.ipynb)
    2. Installed all dependencies: pip install -r requirements_flask.txt
"""

import os
import json
import hashlib
import tempfile
import zipfile
from pathlib import Path
from io import BytesIO
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'pcb-defect-detector-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
# Create temporary directories for storage to avoid triggering Flask reloader
TEMP_DIR = Path(tempfile.gettempdir()) / "pcb_defect_app"
app.config['UPLOAD_FOLDER'] = TEMP_DIR / 'uploads'
app.config['RESULTS_FOLDER'] = TEMP_DIR / 'results'

# Create necessary directories
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['RESULTS_FOLDER'].mkdir(parents=True, exist_ok=True)

print(f"Storage directories initialized at: {TEMP_DIR}")

# Class ID to name mapping
CLASS_ID_TO_NAME = {
    0: "Missing_hole",
    1: "Mouse_bite",
    2: "Open_circuit",
    3: "Short",
    4: "Spur",
    5: "Spurious_copper",
}

# Global model storage
model = None
model_path = "artifacts/ultralytics/pcb_yolov8_s/weights/best.pt"
device = 0 if torch.cuda.is_available() else "cpu"


def load_model(path=None):
    """Load the YOLOv8 model"""
    global model, model_path
    
    if path:
        model_path = path

    try:
        if not Path(model_path).exists():
            return False, f"Model file not found at: {model_path}"
        model = YOLO(model_path)
        return True, None
    except Exception as e:
        return False, f"Error loading model: {str(e)}"


def convert_to_yolo_format(box, img_width, img_height):
    """Convert YOLOv8 box (xyxy format) to YOLO annotation format (normalized cx, cy, w, h)"""
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    cx = (x1 + x2) / 2.0 / img_width
    cy = (y1 + y2) / 2.0 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    class_id = int(box.cls[0])
    return class_id, cx, cy, w, h


def generate_yolo_annotation(result, img_width, img_height):
    """Generate YOLO format annotation string from YOLOv8 result"""
    annotation_lines = []
    for box in result.boxes:
        class_id, cx, cy, w, h = convert_to_yolo_format(box, img_width, img_height)
        annotation_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(annotation_lines)


def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG', 'PNG', 'JPEG'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         model_loaded=model is not None,
                         device="GPU (CUDA)" if torch.cuda.is_available() else "CPU",
                         gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


@app.route('/api/load_model', methods=['POST'])
def api_load_model():
    """API endpoint to load the model"""
    global model
    
    data = request.get_json() or {}
    path = data.get('model_path')
    
    success, error = load_model(path)
    if success:
        return jsonify({'success': True, 'message': 'Model loaded successfully!'})
    else:
        return jsonify({'success': False, 'message': error}), 400


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Handle file uploads"""
    if 'files[]' not in request.files:
        return jsonify({'success': False, 'message': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'success': False, 'message': 'No files selected'}), 400
    
    # Create session ID for this upload
    session_id = hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    session['upload_session_id'] = session_id
    
    # Save uploaded files
    uploaded_files = []
    upload_dir = Path(app.config['UPLOAD_FOLDER']) / session_id
    upload_dir.mkdir(exist_ok=True)
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = upload_dir / filename
            file.save(filepath)
            uploaded_files.append({
                'name': filename,
                'path': str(filepath),
                'size': filepath.stat().st_size
            })
        
    session['uploaded_files'] = uploaded_files
    return jsonify({
        'success': True,
        'session_id': session_id,
        'files': [f['name'] for f in uploaded_files],
        'count': len(uploaded_files)
    })


@app.route('/api/process', methods=['POST'])
def api_process():
    """Process uploaded images"""
    if model is None:
        return jsonify({'success': False, 'message': 'Model not loaded. Please load the model first.'}), 400
    
    data = request.get_json()
    conf_threshold = float(data.get('conf_threshold', 0.25))
    img_size = int(data.get('img_size', 640))
    
    if 'uploaded_files' not in session:
        return jsonify({'success': False, 'message': 'No files uploaded'}), 400
    
    uploaded_files = session['uploaded_files']
    session_id = session.get('upload_session_id')
    
    results_dir = Path(app.config['RESULTS_FOLDER']) / session_id
    results_dir.mkdir(exist_ok=True)
    images_dir = results_dir / 'images'
    labels_dir = results_dir / 'labels'
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    results = []
    processed_count = 0
    total_detections = 0
    
    for file_info in uploaded_files:
        try:
            # Load image
            image = Image.open(file_info['path'])
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_width, img_height = image.size
            
            # Save original image
            original_filename = file_info['name']
            original_path = images_dir / f"original_{original_filename}"
            image.save(original_path)
            
            # Run inference
            inference_results = model.predict(
                source=str(file_info['path']),
                imgsz=img_size,
                conf=conf_threshold,
                device=device,
                save=False,
                verbose=False
            )
            
            if inference_results and len(inference_results) > 0:
                result = inference_results[0]
                
                # Generate annotated image
                plotted_img = result.plot()
                if plotted_img is not None and plotted_img.size > 0:
                    plotted_pil = Image.fromarray(plotted_img)
                    annotated_filename = f"annotated_{Path(original_filename).stem}.png"
                    annotated_path = images_dir / annotated_filename
                    plotted_pil.save(annotated_path)
                    
                    # Generate annotation file
                    annotation_content = generate_yolo_annotation(result, img_width, img_height)
                    annotation_filename = f"{Path(original_filename).stem}.txt"
                    annotation_path = labels_dir / annotation_filename
                    annotation_path.write_text(annotation_content)
                    
                    num_detections = len(result.boxes)
                    total_detections += num_detections
                    
                    results.append({
                        'original_name': original_filename,
                        'original_filename': f"original_{original_filename}",
                        'annotated_name': annotated_filename,
                        'annotated_filename': annotated_filename,
                        'annotation_name': annotation_filename,
                        'annotation_filename': annotation_filename,
                        'detections': num_detections,
                        'annotation_content': annotation_content,
                        'status': 'success'
                    })
                    processed_count += 1
            else:
                # No detections
                annotated_filename = f"annotated_{Path(original_filename).stem}.png"
                annotated_path = images_dir / annotated_filename
                image.save(annotated_path)
                
                annotation_filename = f"{Path(original_filename).stem}.txt"
                annotation_path = labels_dir / annotation_filename
                annotation_path.write_text("")
                
                results.append({
                    'original_name': original_filename,
                    'original_filename': f"original_{original_filename}",
                    'annotated_name': annotated_filename,
                    'annotated_filename': annotated_filename,
                    'annotation_name': annotation_filename,
                    'annotation_filename': annotation_filename,
                    'detections': 0,
                    'annotation_content': '',
                    'status': 'success'
                })
                processed_count += 1
                
        except Exception as e:
            results.append({
                'original_name': file_info['name'],
                'status': 'error',
                'error': str(e)
            })
    
    # Create ZIP file
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for img_file in images_dir.glob("annotated_*"):
            zip_file.write(img_file, f"images/{img_file.name}")
        for txt_file in labels_dir.glob("*.txt"):
            zip_file.write(txt_file, f"labels/{txt_file.name}")
    
    zip_buffer.seek(0)
    zip_path = results_dir / 'results.zip'
    with open(zip_path, 'wb') as f:
        f.write(zip_buffer.getvalue())
    
    summary = {
        'processed_count': processed_count,
        'total_detections': total_detections,
        'avg_detections': total_detections / processed_count if processed_count > 0 else 0
    }
    
    session['results'] = results
    session['summary'] = summary
    session['results_dir'] = str(results_dir)
    session['zip_path'] = f"results/{session_id}/results.zip"
    
    return jsonify({
        'success': True,
        'results': results,
        'summary': summary,
        'zip_path': session['zip_path']
    })


@app.route('/api/download_image/<path:filename>')
def api_download_image(filename):
    """Download individual annotated image"""
    session_id = session.get('upload_session_id')
    if not session_id:
        return jsonify({'error': 'No session found'}), 404
    
    file_path = Path(app.config['RESULTS_FOLDER']) / session_id / 'images' / filename
    if file_path.exists():
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


@app.route('/api/download_annotation/<path:filename>')
def api_download_annotation(filename):
    """Download individual annotation file"""
    session_id = session.get('upload_session_id')
    if not session_id:
        return jsonify({'error': 'No session found'}), 404
    
    file_path = Path(app.config['RESULTS_FOLDER']) / session_id / 'labels' / filename
    if file_path.exists():
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


@app.route('/api/download_zip')
def api_download_zip():
    """Download ZIP file with all results"""
    if 'zip_path' not in session:
        return jsonify({'error': 'No ZIP file available'}), 404
    
    zip_path = Path(app.config['RESULTS_FOLDER']) / session['upload_session_id'] / 'results.zip'
    if zip_path.exists():
        return send_file(zip_path, as_attachment=True, download_name='pcb_detection_results.zip')
    return jsonify({'error': 'ZIP file not found'}), 404


@app.route('/results')
def results():
    """Results page"""
    if 'results' not in session:
        return redirect(url_for('index'))
    
    results_data = session.get('results', [])
    summary = session.get('summary', {})
    
    # Calculate summary if not present
    if not summary:
        processed_count = len([r for r in results_data if r.get('status') == 'success'])
        total_detections = sum(r.get('detections', 0) for r in results_data)
        avg_detections = total_detections / processed_count if processed_count > 0 else 0
        summary = {
            'processed_count': processed_count,
            'total_detections': total_detections,
            'avg_detections': avg_detections
        }
    
    return render_template('results.html', 
                         results=results_data,
                         summary=summary)


@app.route('/results/<path:filename>')
def serve_result_image(filename):
    """Serve result images"""
    session_id = session.get('upload_session_id')
    if not session_id:
        print("Error serving image: No session found")
        return jsonify({'error': 'No session found'}), 404
    
    file_path = Path(app.config['RESULTS_FOLDER']) / session_id / 'images' / filename
    print(f"Serving image: {file_path}")
    
    if file_path.exists():
        return send_file(file_path)
    print(f"Error: File not found at {file_path}")
    return jsonify({'error': 'File not found'}), 404



@app.route('/api/clear', methods=['POST'])
def api_clear():
    """Clear session data and files"""
    session_id = session.get('upload_session_id')
    if session_id:
        # Try to remove directories
        try:
            upload_dir = Path(app.config['UPLOAD_FOLDER']) / session_id
            results_dir = Path(app.config['RESULTS_FOLDER']) / session_id
            
            if upload_dir.exists():
                import shutil
                shutil.rmtree(upload_dir)
            if results_dir.exists():
                import shutil
                shutil.rmtree(results_dir)
        except Exception as e:
            print(f"Error clearing files: {e}")
            
    # Clear session keys
    session.pop('uploaded_files', None)
    session.pop('results', None)
    session.pop('summary', None)
    session.pop('upload_session_id', None)
    session.pop('zip_path', None)
    
    return jsonify({'success': True, 'message': 'Session cleared'})


if __name__ == '__main__':
    # Load model on startup
    print("Loading model...")
    success, error = load_model()
    if success:
        print("Model loaded successfully!")
    else:
        print(f"Warning: {error}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

