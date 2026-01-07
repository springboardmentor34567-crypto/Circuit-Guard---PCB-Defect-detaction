# backend/utils.py
import os
import cv2

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def draw_boxes_on_image(img_bgr, items, color=(0,255,0), thickness=2):
    """
    img_bgr: OpenCV BGR image
    items: list of dicts with keys 'bbox' [xmin,ymin,xmax,ymax], 'class_name', 'score'
    Returns annotated BGR image
    """
    for it in items:
        xmin, ymin, xmax, ymax = map(int, it['bbox'])
        label = f"{it['class_name']}:{it['score']:.2f}"
        cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), color, thickness)
        # draw filled rectangle for label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (xmin, ymin - h - 6), (xmin + w, ymin), color, -1)
        cv2.putText(img_bgr, label, (xmin, ymin - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return img_bgr
