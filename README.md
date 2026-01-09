PCB Defect Detection and Classification Application

This project presents an end-to-end **PCB defect detection system** built using modern deep learning techniques. The system detects and localizes manufacturing defects in Printed Circuit Boards (PCBs) using YOLO-based object detection models and provides real-time inference through a web-based application.

The solution integrates **data preprocessing, model training, backend inference, and frontend visualization** into a single pipeline suitable for industrial inspection workflows.

---

## Defect Types Covered

The system is trained to detect the following six PCB defect classes:

- Missing hole  
- Mouse bite  
- Open circuit  
- Short  
- Spur  
- Spurious copper  

---

## Project Objectives

- Develop a complete PCB defect detection pipeline
- Prepare a YOLO-compatible dataset from the DeepPCB source
- Train and evaluate multiple YOLO models
- Integrate a PCB vs Non-PCB classifier to reduce false detections
- Build a frontend and backend for real-time inference
- Generate structured outputs (images, metadata, reports)

---

## Dataset Preparation

- Converted Pascal VOC XML annotations to YOLO format
- Ensured correct image–annotation alignment
- Normalized bounding boxes and class IDs
- Created an 80/20 train–validation split
- Verified consistency across all six defect classes

The final dataset is clean, reproducible, and compatible with **YOLOv8 and YOLOv11** training pipelines.

---

## Model Training and Evaluation

Multiple YOLO models were trained and compared to identify the best-performing architecture.

### Models Trained
- YOLOv8s (baseline)
- YOLOv8m
- YOLOv8 (freeze + finetune strategy)
- YOLOv11s
- YOLOv11m

### Final Evaluation Metrics

| Model | mAP@50 | mAP@50–95 |
|------|--------|-----------|
| YOLOv8s | 0.92399 | 0.48410 |
| YOLOv8m | 0.92979 | 0.49594 |
| YOLOv8 (Finetuned) | 0.93454 | 0.50423 |
| YOLOv11s | 0.94464 | 0.50462 |
| **YOLOv11m** | **0.96510** | **0.51379** |

**YOLOv11m** achieved the best overall performance and is used as the final detection model.

---

## PCB vs Non-PCB Classification

To improve robustness in real-world usage, a **binary image classifier** was introduced before defect detection.

### Purpose
- Reject non-PCB images
- Reduce false positives
- Avoid unnecessary computation

### Workflow
1. Input image is first classified as PCB / Non-PCB
2. Only valid PCB images proceed to defect detection
3. Improves system accuracy and reliability

---

## System Architecture

### Backend
- Built using **FastAPI**
- Handles:
  - Image preprocessing
  - PCB classification
  - Defect detection
  - Metadata generation
  - PDF / JSON report creation
- Supports ZIP download of inference results

### Frontend
- Initial testing using Gradio
- Final implementation using **HTML, CSS, JavaScript**
- Features:
  - Multi-image upload
  - Real-time inference
  - Interactive results table
  - Per-image defect analysis
  - Charts and visual summaries
  - Downloadable reports (PDF, JSON, ZIP)
  - State persistence across navigation

---

## Inference Workflow

1. User uploads images through the frontend  
2. PCB vs Non-PCB classifier validates inputs  
3. YOLOv11m performs defect detection  
4. Annotated images and metadata are generated  
5. Reports are created and packaged  
6. Results are displayed interactively in the UI  

---

## Challenges and Solutions

| Challenge | Solution |
|---------|----------|
| Dataset misalignment | Feature-based alignment and resizing |
| Noise in subtraction images | Gaussian blur and morphological filters |
| Overfitting in early training | Cosine LR, AdamW, early stopping |
| Limited GPU memory | Reduced batch size, AMP enabled |
| Confusion between small defects | Improved class balancing and augmentations |

---

## Applications

- Automated PCB inspection
- Assembly-line quality control
- Robotics-based inspection systems
- Industrial defect filtering
- Predictive maintenance workflows

---

## Industry Impact

AI-based PCB inspection reduces manual inspection effort, improves detection accuracy, increases consistency, and lowers production costs, making it suitable for real manufacturing environments.

---

## Final Summary

- YOLOv11m delivered the highest detection accuracy
- Fine-tuned YOLOv8 improved over baseline models
- YOLOv11 architecture significantly outperformed YOLOv8
- The complete pipeline from data to deployment is functional and production-ready

---

## Conclusion

This project demonstrates a complete and practical PCB defect detection system using state-of-the-art YOLO architectures. With further optimization such as pruning or quantization, the system can be deployed in real industrial settings.

---

## Author

**Babulal Dudi**  
Computer Science & Engineering  
Interests: Machine Learning, Computer Vision, Backend Systems

