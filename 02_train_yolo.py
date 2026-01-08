from ultralytics import YOLO
import os
import shutil
import random

# --- CONFIG ---
SOURCE_DIR = "dataset_for_training" # The folder we just created
SPLIT_DIR = "yolo_ready"

def prepare_data():
    """Splits data into train/val for YOLO"""
    if os.path.exists(SPLIT_DIR): shutil.rmtree(SPLIT_DIR)
    
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    for cls in classes:
        os.makedirs(f"{SPLIT_DIR}/train/{cls}", exist_ok=True)
        os.makedirs(f"{SPLIT_DIR}/val/{cls}", exist_ok=True)
        
        images = [f for f in os.listdir(os.path.join(SOURCE_DIR, cls)) if f.endswith(('.jpg', '.png'))]
        random.shuffle(images)
        
        # 80/20 Split
        idx = int(len(images) * 0.8)
        
        for img in images[:idx]:
            shutil.copy(f"{SOURCE_DIR}/{cls}/{img}", f"{SPLIT_DIR}/train/{cls}/{img}")
        for img in images[idx:]:
            shutil.copy(f"{SOURCE_DIR}/{cls}/{img}", f"{SPLIT_DIR}/val/{cls}/{img}")
            
    print("✅ Data Split Complete.")

def train():
    prepare_data()
    
    # Load Pre-trained model
    model = YOLO('yolov8s-cls.pt')
    
    # Train
    model.train(
        data=SPLIT_DIR,
        epochs=10,
        imgsz=128,
        device='mps',  # Use Mac M4 GPU
        project='models',
        name='pcb_defect_classifier'
    )

if __name__ == "__main__":
    train()