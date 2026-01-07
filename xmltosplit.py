import os
import xml.etree.ElementTree as ET
from PIL import Image
import shutil
import random

# ---------------- SETTINGS ----------------
ANNOTATIONS_DIR = "annotations"   # Folder with XML files
IMAGES_DIR = "images"            # Folder with image files
OUTPUT_DIR = "dataset"           # Root folder for YOLO training
VAL_SPLIT = 0.2                  # 20% of data for validation

os.makedirs(OUTPUT_DIR, exist_ok=True)

# YOLO expects this structure
IMAGE_TRAIN_DIR = os.path.join(OUTPUT_DIR, "images", "train")
IMAGE_VAL_DIR   = os.path.join(OUTPUT_DIR, "images", "val")
LABEL_TRAIN_DIR = os.path.join(OUTPUT_DIR, "labels", "train")
LABEL_VAL_DIR   = os.path.join(OUTPUT_DIR, "labels", "val")

for d in [IMAGE_TRAIN_DIR, IMAGE_VAL_DIR, LABEL_TRAIN_DIR, LABEL_VAL_DIR]:
    os.makedirs(d, exist_ok=True)

# ------------------------------------------

classes = []

def xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return x_center / img_w, y_center / img_h, w / img_w, h / img_h

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image size
    size = root.find('size')
    if size is not None:
        width = int(size.findtext('width'))
        height = int(size.findtext('height'))
    else:
        img_name = root.findtext('filename')
        img_path = os.path.join(IMAGES_DIR, img_name)
        with Image.open(img_path) as im:
            width, height = im.size

    objects = []
    for obj in root.findall('object'):
        cls = obj.findtext('name')
        if cls not in classes:
            classes.append(cls)
        cls_id = classes.index(cls)
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.findtext('xmin'))
        ymin = float(bndbox.findtext('ymin'))
        xmax = float(bndbox.findtext('xmax'))
        ymax = float(bndbox.findtext('ymax'))
        x, y, w, h = xyxy_to_yolo(xmin, ymin, xmax, ymax, width, height)
        objects.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return objects

# ----------------- MAIN -------------------
xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".xml")]
random.shuffle(xml_files)

val_count = int(len(xml_files) * VAL_SPLIT)

for i, xml_file in enumerate(xml_files):
    label_lines = parse_xml(os.path.join(ANNOTATIONS_DIR, xml_file))
    base_name = os.path.splitext(xml_file)[0]

    # Determine split
    if i < val_count:
        img_dest = IMAGE_VAL_DIR
        label_dest = LABEL_VAL_DIR
    else:
        img_dest = IMAGE_TRAIN_DIR
        label_dest = LABEL_TRAIN_DIR

    # Copy image
    # Try common extensions if not specified
    img_name = base_name
    img_path = None
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        temp_path = os.path.join(IMAGES_DIR, base_name + ext)
        if os.path.exists(temp_path):
            img_path = temp_path
            img_name = base_name + ext
            break
    if img_path is None:
        print(f"[WARN] Image for {base_name} not found, skipping.")
        continue
    shutil.copy(img_path, os.path.join(img_dest, img_name))

    # Write label file
    txt_path = os.path.join(label_dest, base_name + ".txt")
    with open(txt_path, 'w') as f:
        f.write("\n".join(label_lines))

# Save classes.txt
with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w") as f:
    for cls in classes:
        f.write(cls + "\n")

print("Done!")
print(f"YOLO dataset ready in '{OUTPUT_DIR}'")
print("Folder structure:")
print("dataset/images/train")
print("dataset/images/val")
print("dataset/labels/train")
print("dataset/labels/val")
print("classes.txt contains class names.")
