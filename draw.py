# utils/draw.py
import cv2
import numpy as np

def draw_boxes(pil_image, detections):
    """
    Draws bounding boxes on the ORIGINAL pixel space.
    
    detections format:
    {
        "class_id": int,
        "confidence": float,
        "bbox": [x_min, y_min, x_max, y_max]
    }
    """
    img = np.array(pil_image).copy()

    h, w = img.shape[:2]

    # Auto thickness based on image size
    thickness = max(2, int((w + h) / 900))

    for det in detections:
        bbox = det["bbox"]

        # YOLO native format is already in pixel coords (your backend returns this)
        x1, y1, x2, y2 = map(int, bbox)

        # Clamp bounds to avoid errors
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # COLOR (randomized but stable)
        color = (0, 255, 0)  

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Label text
        label = f"ID {det['class_id']} ({det['confidence']:.2f})"

        cv2.putText(
            img,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            thickness,
            cv2.LINE_AA
        )

    return img



