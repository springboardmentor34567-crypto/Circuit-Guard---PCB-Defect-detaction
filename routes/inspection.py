import os
import glob
from flask import Blueprint, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename
from services.inference import get_all_results, get_cached_result, run_batch_inference

# Define Blueprint
inspection_bp = Blueprint('inspection', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==========================================
#  1. VIEW ROUTES (UI Pages)
# ==========================================

@inspection_bp.route('/')
def index():
    """Renders the main inspection page with search."""
    query = request.args.get('search', '').lower()
    results = get_all_results()
    
    filtered = results
    if query:
        filtered = []
        for r in results:
            # Search in filename
            if query in r['filename'].lower():
                filtered.append(r)
                continue
            # Search in defect labels
            found_defect = False
            for d in r.get('defects', []):
                if query in d['label'].lower():
                    found_defect = True
                    break
            if found_defect:
                filtered.append(r)
    
    return render_template('inspection.html', results=filtered, search_query=query)

@inspection_bp.route('/item/<file_id>')
def detail(file_id):
    """Returns the modal content for a specific image."""
    item = get_cached_result(file_id)
    return render_template('partials/inspection_modal.html', item=item)

# ==========================================
#  2. ACTION ROUTE
# ==========================================

@inspection_bp.route('/predict', methods=['POST'])
def predict():
    """Handles file upload and runs inference."""
    # 1. Get the correct upload path from app config
    upload_path = current_app.config['UPLOAD_FOLDER']

    # 2. Clear old uploads to keep folder clean
    existing_files = glob.glob(os.path.join(upload_path, '*'))
    for f in existing_files:
        try: os.remove(f)
        except: pass

    # 3. Save Uploaded Files
    # CRITICAL: Your JS must use formData.append('files[]', file)
    uploaded_files = request.files.getlist('files[]')
    
    if not uploaded_files:
        print("Error: No files received. Check JS FormData key is 'files[]'")
        return jsonify({'error': 'No files uploaded'}), 400

    saved_count = 0
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(upload_path, filename))
            saved_count += 1
            
    if saved_count == 0:
        return jsonify({'error': 'No valid images found (only png, jpg, jpeg)'}), 400

    # 4. Get Settings from Frontend (Convert to float!)
    try:
        conf = float(request.form.get('conf_threshold', 0.25))
        iou = float(request.form.get('iou_threshold', 0.45))
    except ValueError:
        conf = 0.25
        iou = 0.45
    
    # 5. Run the Service
    # Now passing the correct config path
    results = run_batch_inference(upload_path, conf, iou)
    
    return jsonify(results)