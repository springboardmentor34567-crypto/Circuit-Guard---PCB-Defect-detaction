import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SAM model
sam_checkpoint = "./checkpoints/sam_vit_b_01ec64.pth"  # path inside your project
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam.to(device=device)

# Initialize mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# Paths (use your YOLO dataset path)
dataset_dir = "./dataset/images"      # your YOLO images
output_dir = "./dataset/sam_masks"    # masks will be saved here
os.makedirs(output_dir, exist_ok=True)

# Process all images
for img_name in os.listdir(dataset_dir):
    if img_name.endswith((".jpg", ".png")):
        img_path = os.path.join(dataset_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate SAM masks
        masks = mask_generator.generate(image)

        # Save masks
        for i, mask in enumerate(masks):
            mask_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_mask_{i}.png")
            cv2.imwrite(mask_path, mask['segmentation'].astype(np.uint8) * 255)

        print(f"[SAM] Processed {img_name}, {len(masks)} masks generated.")
