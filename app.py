"""
PCB Defect Detection - Streamlit Web Interface

To run this app:
    streamlit run app.py

Make sure you have:
    1. Trained the YOLOv8 model (see notebooks/04_train_yolov8_ultralytics.ipynb)
    2. Installed all dependencies: pip install -r requirements_streamlit.txt
"""

import sys
import os
import hashlib

import streamlit as st
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tempfile
import zipfile
from io import BytesIO


CLASS_ID_TO_NAME = {
    0: "Missing_hole",
    1: "Mouse_bite",
    2: "Open_circuit",
    3: "Short",
    4: "Spur",
    5: "Spurious_copper",
}

# Page configuration
st.set_page_config(
    page_title="🔍 PCB Defect Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1565a0;
    }
    .info-box {
        background-color: #1e3a5f;
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box h4 {
        color: #ffffff;
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    .info-box ol {
        color: #ffffff;
        margin: 0;
        padding-left: 1.5rem;
    }
    .info-box li {
        color: #ffffff;
        margin-bottom: 0.5rem;
        line-height: 1.6;
    }
    .info-box strong {
        color: #64b5f6;
        font-weight: bold;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔍 PCB Defect Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload PCB images to detect defects using YOLOv8 AI Model (Batch Processing Supported)</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    # Model path selection
    model_path = st.text_input(
        "📁 Model Path",
        value="artifacts/ultralytics/pcb_yolov8_s/weights/best.pt",
        help="Path to the trained YOLOv8 model weights"
    )
    
    # Confidence threshold
    conf_threshold = st.slider(
        "🎯 Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # Image size
    img_size = st.selectbox(
        "📐 Image Size",
        options=[640],  # can be adjusted to requirements
        index=0,
        help="Input image size for the model"
    )
    
    st.divider()
    
    # Device selection
    device = 0 if torch.cuda.is_available() else "cpu"
    device_display = "🖥️ GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
    st.info(f"{device_display} will be used for inference")
    
    if torch.cuda.is_available():
        st.success(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    st.divider()
    
    # Class information
    st.subheader("📋 Defect Classes")
    classes = [
        "🔴 Missing_hole",
        "🟠 Mouse_bite",
        "🟡 Open_circuit",
        "🟢 Short",
        "🔵 Spur",
        "🟣 Spurious_copper"
    ]
    for cls in classes:
        st.text(cls)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = None
if 'uploaded_img_index' not in st.session_state:
    st.session_state.uploaded_img_index = 0
if 'annotated_img_index' not in st.session_state:
    st.session_state.annotated_img_index = 0
if 'previous_file_names' not in st.session_state:
    st.session_state.previous_file_names = []
if 'current_upload_session' not in st.session_state:
    st.session_state.current_upload_session = None
if 'upload_key' not in st.session_state:
    st.session_state.upload_key = 0  # used to reset file_uploader


# Load model function
@st.cache_resource
def load_model(model_path):
    """Load the YOLOv8 model"""
    try:
        if not Path(model_path).exists():
            return None, f"❌ Model file not found at: {model_path}"
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, f"❌ Error loading model: {str(e)}"


# Helper function to convert YOLOv8 boxes to YOLO annotation format
def convert_to_yolo_format(box, img_width, img_height):
    """
    Convert YOLOv8 box (xyxy format) to YOLO annotation format (normalized cx, cy, w, h)
    """
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    
    cx = (x1 + x2) / 2.0 / img_width
    cy = (y1 + y2) / 2.0 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    
    class_id = int(box.cls[0])
    
    return class_id, cx, cy, w, h


# Function to generate YOLO annotation file content
def generate_yolo_annotation(result, img_width, img_height):
    """
    Generate YOLO format annotation string from YOLOv8 result
    """
    annotation_lines = []
    for box in result.boxes:
        class_id, cx, cy, w, h = convert_to_yolo_format(box, img_width, img_height)
        annotation_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    
    return "\n".join(annotation_lines)


# Helper function to convert PIL Image to bytes for download
def image_to_bytes(image, format='PNG'):
    """Convert PIL Image to bytes"""
    img_buffer = BytesIO()
    image.save(img_buffer, format=format)
    img_buffer.seek(0)
    return img_buffer.getvalue()


# New helper: display results carousel: Original | Annotated | Annotations
def display_results_carousel(
    original_images,
    annotated_images,
    annotated_image_names,
    annotation_contents,
    original_image_names,
    image_extensions=None,
    key_prefix="results"
):
    """
    Carousel-style viewer:
      [Original image] | [Annotated image]
      ------------------------------------
      [Annotations text + download buttons]
    """

    if not annotated_images or len(annotated_images) == 0:
        st.info("No annotated images to display")
        return

    n = len(annotated_images)

    if len(original_image_names) != n:
        original_image_names = [f"Image {i+1}" for i in range(n)]

    index_key = f"{key_prefix}_carousel_index"
    if index_key not in st.session_state:
        st.session_state[index_key] = 0

    current_index = st.session_state[index_key] % n

    # CSS for annotation box
    st.markdown("""
    <style>
    .annotation-container {
        max-height: 260px;
        overflow-y: auto;
        background-color: #000;
        padding: 0.75rem;
        border-radius: 4px;
        border: 1px solid #ddd;
        font-family: monospace;
        font-size: 0.85em;
        color: #0f0;
        white-space: pre-wrap;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with filename + counter
    original_name = original_image_names[current_index]
    st.markdown(
        f"<div style='text-align:center; font-weight:600; padding:0.3rem;'>"
        f"📷 {original_name} &nbsp;&nbsp;|&nbsp;&nbsp; Result {current_index+1} of {n}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Navigation + images
    col_prev, col_imgs, col_next = st.columns([1, 10, 1])

    with col_prev:
        if st.button("◀️", key=f"{key_prefix}_prev", width='stretch'):
            st.session_state[index_key] = (current_index - 1) % n
            st.rerun()

    with col_imgs:
        c1, c2 = st.columns(2)

        def _resize(im, max_width=360):
            im = im.copy()
            if im.width > max_width:
                ratio = max_width / im.width
                new_h = int(im.height * ratio)
                im = im.resize((max_width, new_h), Image.Resampling.LANCZOS)
            return im

        # Original image
        with c1:
            st.markdown("**Original**")
            if original_images and original_images[current_index] is not None:
                st.image(_resize(original_images[current_index]), width='stretch')
            else:
                st.info("Original image not available")

        # Annotated image
        with c2:
            st.markdown("**Annotated**")
            try:
                display_img = _resize(annotated_images[current_index])
                st.image(display_img, width='stretch')

                # Download annotated image
                if image_extensions and current_index < len(image_extensions):
                    img_format = image_extensions[current_index].upper()
                    if img_format == 'JPG':
                        img_format = 'JPEG'
                        mime_type = 'image/jpeg'
                    elif img_format == 'PNG':
                        mime_type = 'image/png'
                    else:
                        img_format = 'PNG'
                        mime_type = 'image/png'
                else:
                    img_format = 'PNG'
                    mime_type = 'image/png'

                img_bytes = image_to_bytes(annotated_images[current_index], format=img_format)
                annotated_filename = annotated_image_names[current_index]

                st.download_button(
                    label="📥 Download annotated image",
                    data=img_bytes,
                    file_name=annotated_filename,
                    mime=mime_type,
                    key=f"{key_prefix}_download_img_{current_index}",
                    width='stretch',
                )
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")

    with col_next:
        if st.button("▶️", key=f"{key_prefix}_next", width='stretch'):
            st.session_state[index_key] = (current_index + 1) % n
            st.rerun()

    st.markdown("---")

    # Annotations + txt download
    st.markdown("**📄 YOLO Annotations**")
    annotation_text = annotation_contents[current_index] if current_index < len(annotation_contents) else ""

    if annotation_text:
        lines = [ln.strip() for ln in annotation_text.strip().splitlines() if ln.strip()]
        
        pretty_lines = []
        for idx, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) == 5:
                cls_id_str, cx, cy, w, h = parts
                try:
                    cls_id = int(cls_id_str)
                except ValueError:
                    cls_id = None
                cls_name = CLASS_ID_TO_NAME.get(cls_id, "unknown")
                pretty_lines.append(
                    f"#{idx}  class={cls_id_str} ({cls_name})  "
                    f"cx={cx}, cy={cy}, w={w}, h={h}"
                )
            else:
                # fallback if the line is weird
                pretty_lines.append(line)
        
        html_content = "<br>".join(pretty_lines)
        st.markdown(
            f"<div class='annotation-container'><code>{html_content}</code></div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No detections (empty annotation file)", icon="ℹ️")


    annotation_filename = Path(original_image_names[current_index]).stem + ".txt"
    st.download_button(
        label="📥 Download .txt annotation",
        data=annotation_text.encode("utf-8"),
        file_name=annotation_filename,
        mime="text/plain",
        key=f"{key_prefix}_download_txt_{current_index}",
        width='stretch',
    )

def display_image_grid(images, image_names, max_per_row=3):
    """
    Display uploaded images as small thumbnails in a grid.
    """
    if not images:
        st.info("No images to display")
        return

    # Fallback names
    if len(image_names) != len(images):
        image_names = [f"Image {i+1}" for i in range(len(images))]

    def _resize(im, max_width=220):
        im = im.copy()
        if im.width > max_width:
            ratio = max_width / im.width
            new_h = int(im.height * ratio)
            im = im.resize((max_width, new_h), Image.Resampling.LANCZOS)
        return im

    for i in range(0, len(images), max_per_row):
        row_imgs = images[i:i + max_per_row]
        row_names = image_names[i:i + max_per_row]
        cols = st.columns(len(row_imgs))
        for col, img, name in zip(cols, row_imgs, row_names):
            with col:
                thumb = _resize(img)
                st.image(thumb, width='stretch')
                # Short caption with ellipsis if name is long
                short_name = name if len(name) <= 28 else name[:25] + "..."
                st.caption(f"🖼 {short_name}")


# Load model button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Load Model", width='stretch'):
        if not model_path or not model_path.strip():
            st.error("❌ Please enter a valid model path")
        else:
            with st.spinner("⏳ Loading model..."):
                model, error = load_model(model_path.strip())
                if error:
                    st.error(error)
                    st.session_state.model_loaded = False
                    st.session_state.model = None
                else:
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")

# Main content area
if st.session_state.model_loaded:
    st.markdown("---")
    tab_upload, tab_results = st.tabs(["📤 Upload & Preview", "📊 Results"])
    
    # ----------------- UPLOAD & PREVIEW TAB -----------------
    with tab_upload:
        st.subheader("📤 Upload PCB Images")
        
        # Optional: tip about test folder
        test_dir = Path("PCB_DATASET/test")
        if test_dir.exists():
            example_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.JPG"))
            if example_images:
                st.info("💡 Tip: You can also use images from the test folder for quick testing!")
        
        col_upload, col_clear = st.columns([4, 1])
        with col_upload:
            uploaded_files = st.file_uploader(
                "Choose image file(s) - You can select multiple images",
                type=['jpg', 'jpeg', 'png'],
                help="Upload one or more PCB images to detect defects",
                accept_multiple_files=True,
                key=f"uploader_{st.session_state.upload_key}",
            )
        
        with col_clear:
            if st.button("🧹 Clear images", width='stretch'):
                st.session_state.upload_key += 1
                st.session_state.processed_results = None
                st.session_state.current_upload_session = None
                st.session_state.previous_file_names = []
                if 'results_carousel_index' in st.session_state:
                    del st.session_state['results_carousel_index']
                if 'results_carousel_index_results' in st.session_state:
                    del st.session_state['results_carousel_index_results']
                st.rerun()
        
        if uploaded_files and len(uploaded_files) > 0:
            # Keep track of filenames to detect change
            current_names = sorted([f.name for f in uploaded_files])
            if st.session_state.previous_file_names != current_names:
                st.session_state.processed_results = None
                st.session_state.previous_file_names = current_names
                if 'results_carousel_index' in st.session_state:
                    del st.session_state['results_carousel_index']
            
            st.info(f"📁 {len(uploaded_files)} image(s) uploaded")
            
            # Process button for batch processing
            if st.button("🔎 Process All Images & Generate Annotations", type="primary", width='stretch'):
                if st.session_state.model is None:
                    st.error("❌ Model not loaded. Please load the model first.")
                    st.stop()
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    images_dir = temp_path / "images"
                    labels_dir = temp_path / "labels"
                    images_dir.mkdir(exist_ok=True)
                    labels_dir.mkdir(exist_ok=True)
                    
                    processed_count = 0
                    total_detections = 0
                    processing_progress = st.progress(0)
                    status_text = st.empty()
                    
                    results_summary = []
                    annotated_images = []
                    annotated_image_names = []
                    annotation_contents = []
                    original_image_names = []
                    image_extensions = []
                    original_images = []  # NEW: store original images for side-by-side display
                    
                    try:
                        for idx, uploaded_file in enumerate(uploaded_files):
                            progress = (idx + 1) / len(uploaded_files)
                            processing_progress.progress(progress)
                            status_text.text(f"Processing image {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                            
                            try:
                                uploaded_file.seek(0)
                                file_ext = Path(uploaded_file.name).suffix or '.jpg'
                                if file_ext.lower() not in ['.jpg', '.jpeg', '.png']:
                                    file_ext = '.jpg'
                                
                                image = Image.open(uploaded_file)
                                if image.mode != 'RGB':
                                    image = image.convert('RGB')
                                img_width, img_height = image.size
                                
                                # store original image
                                original_images.append(image.copy())
                                
                                image_filename = Path(uploaded_file.name).stem + file_ext
                                image_path = images_dir / image_filename
                                image.save(image_path)
                                
                                # Run inference
                                results = st.session_state.model.predict(
                                    source=str(image_path),
                                    imgsz=img_size,
                                    conf=conf_threshold,
                                    device=device,
                                    save=False,
                                    verbose=False
                                )
                                
                                if results and len(results) > 0:
                                    result = results[0]
                                    
                                    # Annotated image
                                    plotted_img = result.plot()
                                    if plotted_img is not None and plotted_img.size > 0:
                                        plotted_pil = Image.fromarray(plotted_img)
                                        annotated_filename = Path(uploaded_file.name).stem + "_annotated" + file_ext
                                        annotated_path = images_dir / annotated_filename
                                        plotted_pil.save(annotated_path)
                                        
                                        annotated_images.append(plotted_pil.copy())
                                        annotated_image_names.append(annotated_filename)
                                    
                                    # Annotation file
                                    annotation_content = generate_yolo_annotation(result, img_width, img_height)
                                    annotation_filename = Path(uploaded_file.name).stem + ".txt"
                                    annotation_path = labels_dir / annotation_filename
                                    annotation_path.write_text(annotation_content)
                                    
                                    annotation_contents.append(annotation_content)
                                    original_image_names.append(uploaded_file.name)
                                    image_extensions.append(file_ext.lower().replace('.', ''))
                                    
                                    num_detections = len(result.boxes)
                                    total_detections += num_detections
                                    
                                    results_summary.append({
                                        "Image": uploaded_file.name,
                                        "Detections": num_detections,
                                        "Status": "✅ Processed"
                                    })
                                    
                                    processed_count += 1
                                else:
                                    # No detections
                                    annotation_content = ""
                                    annotation_filename = Path(uploaded_file.name).stem + ".txt"
                                    annotation_path = labels_dir / annotation_filename
                                    annotation_path.write_text(annotation_content)
                                    
                                    annotated_filename = Path(uploaded_file.name).stem + "_annotated" + file_ext
                                    annotated_path = images_dir / annotated_filename
                                    image.save(annotated_path)
                                    
                                    annotated_images.append(image.copy())
                                    annotated_image_names.append(annotated_filename)
                                    
                                    annotation_contents.append(annotation_content)
                                    original_image_names.append(uploaded_file.name)
                                    image_extensions.append(file_ext.lower().replace('.', ''))
                                    
                                    results_summary.append({
                                        "Image": uploaded_file.name,
                                        "Detections": 0,
                                        "Status": "✅ Processed (No defects)"
                                    })
                                    
                                    processed_count += 1
                            
                            except Exception as e:
                                results_summary.append({
                                    "Image": uploaded_file.name,
                                    "Detections": 0,
                                    "Status": f"❌ Error: {str(e)}"
                                })
                                continue
                        
                        # Create zip in memory
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for img_file in images_dir.glob("*_annotated*"):
                                zip_file.write(img_file, f"images/{img_file.name}")
                            for txt_file in labels_dir.glob("*.txt"):
                                zip_file.write(txt_file, f"labels/{txt_file.name}")
                        
                        zip_buffer.seek(0)
                        zip_data = zip_buffer.getvalue()
                        zip_buffer.close()
                        
                        st.session_state.processed_results = {
                            'zip_data': zip_data,
                            'summary': results_summary,
                            'processed_count': processed_count,
                            'total_detections': total_detections,
                            'file_names': [f.name for f in uploaded_files],
                            'annotated_images': annotated_images,
                            'annotated_image_names': annotated_image_names,
                            'annotation_contents': annotation_contents,
                            'original_image_names': original_image_names,
                            'image_extensions': image_extensions,
                            'original_images': original_images,
                        }
                        
                        processing_progress.progress(1.0)
                        status_text.text("✅ Processing complete! Go to the '📊 Results' tab to view details.")
                    
                    except Exception as e:
                        st.error(f"❌ Error during batch processing: {str(e)}")
            
            # Preview uploaded images (only if not processed yet)
            if not st.session_state.processed_results:
                st.markdown("---")
                st.subheader("📷 Uploaded Images Preview")
                
                uploaded_images = []
                uploaded_image_names = []
                
                for uploaded_file in uploaded_files:
                    uploaded_file.seek(0)
                    try:
                        image = Image.open(uploaded_file)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        uploaded_images.append(image)
                        uploaded_image_names.append(uploaded_file.name)
                    except Exception as e:
                        st.error(f"Error loading {uploaded_file.name}: {str(e)}")
                
                if uploaded_images:
                    # display_image_carousel(uploaded_images, uploaded_image_names, "uploaded")
                    display_image_grid(uploaded_images, uploaded_image_names, max_per_row=3)
        else:
            st.info("📂 Upload images to begin.", icon="ℹ️")
    
    # ----------------- RESULTS TAB -----------------
    with tab_results:
        if st.session_state.processed_results:
            results_data = st.session_state.processed_results
            
            st.subheader("📊 Processing Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Images Processed", results_data['processed_count'])
            with col2:
                st.metric("Total Detections", results_data['total_detections'])
            with col3:
                avg_detections = results_data['total_detections'] / results_data['processed_count'] if results_data['processed_count'] > 0 else 0
                st.metric("Avg Detections/Image", f"{avg_detections:.1f}")
            
            st.markdown("### 📋 Detailed Results")
            st.dataframe(
                results_data['summary'],
                width='stretch',
                hide_index=True
            )
            
            st.markdown("---")
            st.subheader("💾 Download Results")
            
            zip_data = results_data.get('zip_data')
            if zip_data:
                zip_buffer = BytesIO(zip_data)
                st.download_button(
                    label="📦 Download ZIP (Annotated Images + Annotation Files)",
                    data=zip_buffer,
                    file_name="pcb_detection_results.zip",
                    mime="application/zip",
                    width='stretch',
                    type="primary"
                )
            else:
                st.error("❌ ZIP data not available. Please reprocess the images.")
            
            st.info("📁 The ZIP file contains:\n"
                    "- `images/` folder: All annotated images (with bounding boxes)\n"
                    "- `labels/` folder: YOLO format annotation files (.txt) for each image")
            
            st.markdown("---")
            st.subheader("🖼️ Results: Original vs Annotated + Annotations")
            st.caption("Use navigation buttons (◀️ ▶️) to browse each image's original, annotated view, and its YOLO annotations.")
            
            annotated_images = results_data.get('annotated_images', [])
            annotated_image_names = results_data.get('annotated_image_names', [])
            annotation_contents = results_data.get('annotation_contents', [])
            original_image_names = results_data.get('original_image_names', [])
            image_extensions = results_data.get('image_extensions', [])
            original_images = results_data.get('original_images', [])
            
            if annotated_images and len(annotated_images) > 0:
                display_results_carousel(
                    original_images=original_images,
                    annotated_images=annotated_images,
                    annotated_image_names=annotated_image_names,
                    annotation_contents=annotation_contents,
                    original_image_names=original_image_names,
                    image_extensions=image_extensions,
                    key_prefix="results"
                )
            else:
                st.info("No annotated images available. Please reprocess the images.")
        else:
            st.info("No results yet. Upload images and click **'🔎 Process All Images & Generate Annotations'** in the **Upload & Preview** tab.", icon="ℹ️")

else:
    st.markdown("""
    <div class="info-box">
        <h4>ℹ️ Instructions:</h4>
        <ol>
            <li>Click <strong>"🚀 Load Model"</strong> to load the trained YOLOv8 model</li>
            <li>Upload one or more PCB images using the file uploader (batch processing supported)</li>
            <li>Click <strong>"🔎 Process All Images & Generate Annotations"</strong> to analyze all images</li>
            <li>Go to the <strong>"📊 Results"</strong> tab to:
                <ul>
                    <li>See summary metrics</li>
                    <li>Download a ZIP with annotated images + YOLO .txt labels</li>
                    <li>View a carousel: Original | Annotated | Annotations</li>
                </ul>
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "🔍 PCB Defect Detection System | Powered by YOLOv8 & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
