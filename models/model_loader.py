import os
import threading
from ultralytics import YOLO

class PCBModel:
    """
    Real Wrapper for Ultralytics YOLO to load pcb_yolo_model_v1.pt
    """
    def __init__(self):
        # Path to your model in the main directory
        model_path = "pcb_yolo_model_v1.pt"
        
        print(f"Loading Real Model from: {model_path}...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found at {model_path}. Please check the file path.")
            
        self.model = YOLO(model_path)
        
        # Load class names directly from the model
        self.classes = self.model.names 
        print(f"✅ Model Loaded Successfully! Classes: {self.classes}")
        self.loaded = True

    # --- UPDATED PREDICT METHOD TO ACCEPT imgsz ---
    def predict(self, image_path, conf_low, conf_high, imgsz=640):
        """
        Runs actual inference on the image.
        Accepts 'imgsz' to support high-res inference (e.g., 1024).
        """
        try:
            # Run YOLO prediction
            # We now pass 'imgsz' down to the ultralytics model
            results = self.model.predict(
                source=image_path, 
                conf=conf_low, 
                imgsz=imgsz,   # <--- THIS PASSES THE SIZE TO YOLO
                verbose=False
            )
            
            result = results[0] # We are processing one image at a time
            formatted_detections = []
            
            if result.boxes:
                for box in result.boxes:
                    # Extract values
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                    
                    # Filter by Upper Confidence Threshold as well (from UI slider)
                    if conf <= conf_high:
                        formatted_detections.append({
                            'class_id': cls_id,
                            'label': self.classes.get(cls_id, f"Unknown_{cls_id}"),
                            'confidence': round(conf, 4),
                            'bbox': [int(x) for x in xyxy] # Ensure integers for PIL cropping
                        })
            
            return formatted_detections

        except Exception as e:
            print(f"Error during inference on {image_path}: {e}")
            return []

class ModelLoader:
    _instance = None
    _lock = threading.Lock()

    @staticmethod
    def get_instance():
        if ModelLoader._instance is None:
            with ModelLoader._lock:
                if ModelLoader._instance is None:
                    ModelLoader._instance = PCBModel()
        return ModelLoader._instance