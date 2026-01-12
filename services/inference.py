import os
import glob
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from models.model_loader import ModelLoader
from database import save_scan

# In-memory cache
RESULTS_CACHE = {}

# --- CONFIGURATION: YOLO COLORS (BGR Format) ---
COLORS = {
    'Missing_hole':    (255, 56, 56),    # Blue-ish
    'Mouse_bite':      (56, 255, 56),    # Green
    'Open_circuit':    (56, 56, 255),    # Red
    'Short':           (255, 150, 56),   # Cyan/Orange mix
    'Spur':            (200, 56, 255),   # Purple
    'Spurious_copper': (56, 255, 255),   # Yellow
    'default':         (0, 0, 255)       # Red fallback
}

def get_color(label):
    return COLORS.get(label, COLORS['default'])

def run_batch_inference(upload_folder, conf_low, conf_high):
    model = ModelLoader.get_instance()
    image_paths = glob.glob(os.path.join(upload_folder, '*'))
    
    batch_results = []
    
    for img_path in image_paths:
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = os.path.basename(img_path)
            
            try:
                # 1. Get raw predictions
                # *** CRITICAL: Run at 1024px for tight boxes ***
                # (Ensure models/model_loader.py supports this!)
                detections = model.predict(
                    img_path, 
                    float(conf_low), 
                    float(conf_high),
                    imgsz=1024  # <--- THIS MAKES THE BOXES TIGHT
                )
                
                # 2. Draw boxes (Using Dynamic Scaling)
                processed_data = process_image_with_opencv(img_path, detections)
                
                # 3. Save & Cache Results
                status = 'Fail' if detections else 'Pass'
                max_conf = max([d.get('confidence', 0) for d in detections]) if detections else 0.0
                
                try:
                    save_scan(filename, status, detections, max_conf)
                except Exception:
                    pass 

                result_entry = {
                    'id': filename,
                    'filename': filename,
                    'status': status,
                    'defects': detections,
                    'thumbnail': processed_data['thumbnail'], 
                    'cutouts': processed_data['cutouts'],
                    'confidence': max_conf
                }
                
                batch_results.append(result_entry)
                RESULTS_CACHE[filename] = result_entry

            except Exception as e:
                print(f"❌ Error processing file {filename}: {e}")

    return {
        'total': len(batch_results),
        'defected': sum(1 for r in batch_results if r['status'] == 'Fail'),
        'results': batch_results
    }

def process_image_with_opencv(img_path, detections):
    """
    Reads image, draws COLOR-CODED boxes with DYNAMIC SCALING.
    """
    try:
        # --- A. Create Cutouts ---
        img_pil = Image.open(img_path).convert('RGB')
        cutouts = []
        for d in detections:
            try:
                box = list(map(int, d['bbox'])) 
                crop = img_pil.crop(tuple(box))
                cbuff = BytesIO()
                crop.save(cbuff, format="JPEG")
                cutouts.append({
                    'label': d['label'],
                    'image': base64.b64encode(cbuff.getvalue()).decode('utf-8')
                })
            except Exception:
                continue

        # --- B. Draw Smart Bounding Boxes ---
        img_cv = cv2.imread(img_path)
        if img_cv is None: return {'thumbnail': '', 'cutouts': []}

        h, w = img_cv.shape[:2]
        
        # *** DYNAMIC SCALING MATH ***
        # 1. Thickness: Scales with image size (Min 2px)
        line_thickness = max(2, int(min(h, w) * 0.005))
        
        # 2. Font Scale: Scales with image size (Min 0.5)
        font_scale = max(0.5, min(h, w) * 0.001)

        for d in detections:
            box = list(map(int, d['bbox'])) 
            label_text = f"{d['label']} {d['confidence']:.2f}"
            color = get_color(d['label'])

            # Draw Box
            cv2.rectangle(img_cv, (box[0], box[1]), (box[2], box[3]), color, line_thickness)
            
            # Draw Label Background
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, int(line_thickness/2)))
            
            y1 = box[1]
            if y1 - text_h - 10 < 0:
                y1 = box[1] + text_h + 10 # Draw inside if at top edge
                
            cv2.rectangle(img_cv, 
                          (box[0], y1 - text_h - int(10 * font_scale)), 
                          (box[0] + text_w, y1), 
                          color, -1)

            # Draw Text
            cv2.putText(img_cv, label_text, 
                        (box[0], y1 - int(5 * font_scale)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (255, 255, 255), max(1, int(line_thickness/2)))

        # --- C. Create Thumbnail ---
        if w > 800:
            scale = 800 / w
            img_cv = cv2.resize(img_cv, (800, int(h * scale)))

        _, buffer = cv2.imencode('.jpg', img_cv)
        thumb_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {'thumbnail': thumb_b64, 'cutouts': cutouts}

    except Exception as e:
        print(f"Drawing Error: {e}")
        return {'thumbnail': '', 'cutouts': []}

def get_cached_result(file_id):
    return RESULTS_CACHE.get(file_id)

def get_all_results():
    return list(RESULTS_CACHE.values())