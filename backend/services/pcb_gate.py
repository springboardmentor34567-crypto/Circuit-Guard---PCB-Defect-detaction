from ultralytics import YOLO

# ===== CONFIG =====
PCB_MODEL_PATH = "/home/omen/Downloads/ML/pcb_detector/ML/pcb_classifier/runs/classify/train2/weights/best.pt"
PCB_THRESHOLD = 0.95
DEVICE = 0
# ==================

pcb_model = YOLO(PCB_MODEL_PATH)

def is_pcb(image_path: str):
    results = pcb_model.predict(
        source=image_path,
        device=DEVICE,
        verbose=False
    )

    r = results[0]
    probs = r.probs

    pcb_conf = probs.data[1].item()      # index 1 = pcb
    nonpcb_conf = probs.data[0].item()

    return {
        "is_pcb": pcb_conf >= PCB_THRESHOLD,
        "pcb_conf": pcb_conf,
        "nonpcb_conf": nonpcb_conf
    }
