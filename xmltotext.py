import os
import xml.etree.ElementTree as ET
from PIL import Image

# --- Folders ---
ANNOTATIONS_DIR = "annotations"  # XML files
IMAGES_DIR = "images"            # Image files
LABELS_DIR = "labels"            # Output YOLO txt files

os.makedirs(LABELS_DIR, exist_ok=True)

# Collect all classes dynamically
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
        # Try reading from image
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

# Process all XML files
for xml_file in os.listdir(ANNOTATIONS_DIR):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
        label_lines = parse_xml(xml_path)
        # Save to labels folder with same basename
        base_name = os.path.splitext(xml_file)[0]
        txt_path = os.path.join(LABELS_DIR, base_name + ".txt")
        with open(txt_path, 'w') as f:
            f.write("\n".join(label_lines))

# Save classes.txt
with open("classes.txt", "w") as f:
    for cls in classes:
        f.write(cls + "\n")

print("Conversion completed!")
print(f"YOLO labels saved in '{LABELS_DIR}'")
print("Classes saved in 'classes.txt'")
