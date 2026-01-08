from ultralytics import YOLO
import cv2
import os

# Load model once
model = YOLO("models/best.pt")


def run_inference(image_paths, dirs):
    detections = []
    annotated_paths = []
    original_paths = []

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        original_paths.append(img_path)

        # Run YOLO inference
        result = model(img_path)[0]

        # Save annotated image
        annotated_img = result.plot()
        ann_path = os.path.join(dirs["annotated"], fname)
        cv2.imwrite(ann_path, annotated_img)
        annotated_paths.append(ann_path)

        # If no detections, continue safely
        if result.boxes is None:
            continue

        for box in result.boxes:
            # Extract bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Extract class and confidence safely
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            detections.append({
                "image": fname,
                "class": model.names[cls_id],
                "confidence": round(conf, 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

    return detections, original_paths, annotated_paths
