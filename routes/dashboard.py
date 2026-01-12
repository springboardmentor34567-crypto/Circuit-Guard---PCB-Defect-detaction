from flask import Blueprint, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
from services.inference import run_batch_inference, get_all_results
from services.analytics import get_dashboard_stats

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
def index():
    return render_template('dashboard.html')

@dashboard_bp.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('files[]')
    saved_count = 0
    
    # Clear old uploads for demo
    for f in os.listdir(current_app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(current_app.config['UPLOAD_FOLDER'], f))

    for file in files:
        if file and file.filename.split('.')[-1].lower() in current_app.config['ALLOWED_EXTENSIONS']:
            filename = secure_filename(file.filename)
            file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
            saved_count += 1
            
    return jsonify({'message': f'Uploaded {saved_count} files successfully', 'count': saved_count})

@dashboard_bp.route('/predict', methods=['POST'])
def predict():
    data = request.json
    conf_low = data.get('conf_low', 0.25)
    conf_high = data.get('conf_high', 1.0)
    
    summary = run_batch_inference(current_app.config['UPLOAD_FOLDER'], conf_low, conf_high)
    stats = get_dashboard_stats()
    
    return jsonify({'summary': summary, 'stats': stats})