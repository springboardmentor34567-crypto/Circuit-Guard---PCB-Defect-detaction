import cv2
import numpy as np
import os
import glob
import shutil

# --- CONFIGURATION ---
RAW_DATASET = "PCB_DATASET" 
TEMPLATE_SOURCE = os.path.join(RAW_DATASET, "pcb_used")
DEFECT_SOURCE = os.path.join(RAW_DATASET, "images")
OUTPUT_DIR = "dataset_for_training"

# --- 1. SETUP OUTPUT FOLDERS ---
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 Starting Smart Processing (Crop & Sort)...")

# --- 2. GET CLASS LIST ---
classes = [d for d in os.listdir(DEFECT_SOURCE) if os.path.isdir(os.path.join(DEFECT_SOURCE, d))]
print(f"   Found classes: {classes}")

total_crops = 0

# --- 3. PROCESSING LOOP ---
for class_name in classes:
    # Create class folder
    class_save_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_save_dir, exist_ok=True)
    
    # Get images in this class
    image_path = os.path.join(DEFECT_SOURCE, class_name)
    images = glob.glob(os.path.join(image_path, "*.jpg"))
    
    print(f"   Processing '{class_name}': {len(images)} images...")
    
    for test_img_path in images:
        filename = os.path.basename(test_img_path)
        pcb_id = filename.split('_')[0]
        
        # FIND MATCHING TEMPLATE
        # Try finding 00041.jpg or 00041_temp.jpg
        temp_candidates = [
            os.path.join(TEMPLATE_SOURCE, f"{pcb_id}.jpg"),
            os.path.join(TEMPLATE_SOURCE, f"{pcb_id}_temp.jpg")
        ]
        
        temp_path = None
        for t in temp_candidates:
            if os.path.exists(t):
                temp_path = t
                break
        
        if not temp_path:
            continue # Skip if no template found
            
        # LOAD & PROCESS
        img_test = cv2.imread(test_img_path)
        img_temp = cv2.imread(temp_path)
        
        if img_test is None or img_temp is None: continue
        
        # Resize to match
        if img_test.shape != img_temp.shape:
            img_test = cv2.resize(img_test, (img_temp.shape[1], img_temp.shape[0]))
            
        # Subtraction
        gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
        gray_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_temp, gray_test)
        _, thresh = cv2.threshold(diff, 70, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)
        
        # Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        crop_count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 450: # Filter Noise
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Add Padding (20px)
                pad = 20
                x1, y1 = max(0, x-pad), max(0, y-pad)
                x2, y2 = min(img_test.shape[1], x+w+pad), min(img_test.shape[0], y+h+pad)
                
                roi = img_test[y1:y2, x1:x2]
                
                # Save Crop
                save_name = f"{filename}_{crop_count}.jpg"
                cv2.imwrite(os.path.join(class_save_dir, save_name), roi)
                crop_count += 1
                total_crops += 1

print(f"\n✅ Processing Complete! Created {total_crops} cropped images.")