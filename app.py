import base64
from io import BytesIO
import requests
import os
from collections import Counter
import io
import zipfile

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import altair as alt

# ------------------ CONFIG ------------------
API_URL = "http://127.0.0.1:8000/predict"
LOCAL_MODEL_PATH = r"C:\Users\asus\OneDrive\Desktop\yolo deploy\best.pt"
CLOUD_MODEL_PATH = "best.pt"

MODEL_PATH = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else CLOUD_MODEL_PATH

CONFIDENCE = 0.25
IOU = 0.45

st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🎄",
    layout="wide"
)



# ------------------ CUSTOM STYLING ------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Bitcount+Prop+Single:wght@400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: #f8fbff;
        font-family: 'Poppins', sans-serif;
        color: #102a43;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8f5ff 0%, #e7fff7 100%);
        border-right: 1px solid #d0e2ff;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #102a43 !important;
    }

    /* Code block for best.pt: light background, dark text */
    [data-testid="stSidebar"] pre, [data-testid="stSidebar"] code {
        background: #e5e7eb !important;
        color: #111827 !important;
    }

    /* Top toolbar (Share, etc.) */
    [data-testid="stToolbar"] * {
        color: #e5e7eb !important;
    }

    h2, h3 {
        font-weight: 600;
        color: #13406b;
        font-family: 'Space Grotesk', 'Poppins', system-ui, -apple-system, sans-serif;
    }

    /* CTA buttons with light animation */
    .stButton>button {
        border-radius: 999px;
        padding: 0.5rem 1.25rem;
        border: none;
        font-weight: 500;
        background: #85c5ff;
        color: #0f172a;
        box-shadow: 0 8px 14px rgba(148, 163, 184, 0.28);
        transition: transform 0.18s ease-out, box-shadow 0.18s ease-out, background 0.18s ease-out;
        animation: pulse-soft 2.4s ease-in-out infinite;
    }

    .stButton>button:hover {
        background: #63b1ff;
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 12px 22px rgba(148, 163, 184, 0.38);
    }

    @keyframes pulse-soft {
        0% {
            transform: translateY(0);
            box-shadow: 0 8px 14px rgba(148, 163, 184, 0.25);
        }
        50% {
            transform: translateY(-1px);
            box-shadow: 0 12px 22px rgba(148, 163, 184, 0.4);
        }
        100% {
            transform: translateY(0);
            box-shadow: 0 8px 14px rgba(148, 163, 184, 0.25);
        }
    }

    /* Download buttons – light so text is visible */
    [data-testid="stDownloadButton"] > button {
        background: #e5e7eb !important;
        color: #111827 !important;
        border-radius: 999px !important;
        border: 1px solid #cbd5f5 !important;
        font-weight: 500;
    }

    .upload-box {
        border-radius: 18px;
        border: 1px dashed #a3c9ff;
        padding: 1.5rem;
        background: #ffffff;
    }

    /* File uploader text & Browse button */
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] label {
        color: #f9fafb !important;
    }
    [data-testid="stFileUploader"] button {
        background: #111827 !important;
        color: #f9fafb !important;
        border-radius: 999px !important;
        border: none !important;
    }

    .metric-card {
        border-radius: 18px;
        padding: 0.75rem 1rem;
        background: #ffffff;
        border: 1px solid #dbeafe;
    }

    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
        margin-bottom: 0.1rem;
    }

    .metric-value {
        font-size: 1.15rem;
        font-weight: 600;
        color: #111827;
        font-family: 'Space Grotesk', 'Poppins', system-ui, sans-serif;
    }

    .logo-circle {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: #e0f2fe;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 32px;
        margin-bottom: 0.4rem;
    }

    .header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 0.2rem;
        margin-bottom: 0.6rem;
    }

    .main-title {
        font-family: 'Space Grotesk', system-ui, -apple-system,
                     BlinkMacSystemFont, 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 2.8rem;
        text-align: center;
        color: #13406b;
        letter-spacing: 0.03em;
    }

    .subtitle-text {
        font-size: 0.95rem;
        color: #334e68;
        text-align: center;
    }

    .instruction-card {
        border-radius: 18px;
        background: #ffffff;
        border: 1px solid #dbeafe;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .instruction-card ol {
        margin-left: 1.1rem;
        padding-left: 0.5rem;
    }
    .instruction-card li {
        margin-bottom: 0.25rem;
    }

    .defect-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.4rem;
    }
    .defect-badge {
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        background: #e0f2fe;
        font-size: 0.8rem;
        color: #13406b;
    }

    /* Robotic success message */
    .robot-success {
        margin: 1rem 0 0.4rem 0;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        background: linear-gradient(90deg, #0f172a 0%, #1f2937 55%, #16a34a 100%);
        color: #e5f9ff;
        font-family: 'JetBrains Mono', SFMono-Regular, Menlo, monospace;
        font-size: 0.9rem;
        letter-spacing: 0.09em;
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
    }
    .robot-success::after {
        content: "";
        position: absolute;
        inset: 0;
        background: repeating-linear-gradient(
            0deg,
            rgba(148, 163, 184, 0.0),
            rgba(148, 163, 184, 0.0) 2px,
            rgba(148, 163, 184, 0.25) 3px
        );
        mix-blend-mode: soft-light;
        opacity: 0.4;
        pointer-events: none;
        animation: scanlines 6s linear infinite;
    }
    @keyframes scanlines {
        0% { transform: translateY(-3px); }
        100% { transform: translateY(3px); }
    }
    .robot-label {
        color: #a7f3d0;
        margin-right: 0.75rem;
    }

    /* Clear info strip under robot banner */
    .status-strip {
        margin: 0.1rem 0 1.2rem 0;
        padding: 0.65rem 1.1rem;
        border-radius: 999px;
        background: #d1fae5;
        color: #064e3b;
        font-size: 0.9rem;
        font-weight: 500;
    }

    /* Keep Altair charts responsive and fully visible */
    .vega-embed, .vega-embed canvas {
        max-width: 100% !important;
    }
    /* Fix YOLO yellow warning box */
    div[role="alert"] {
        background-color: #ffffff !important;  /* white background */
        color: #111827 !important;             /* black text */
        border: 1px solid #d1d5db !important;
    }
    div[role="alert"] * {
        color: #111827 !important;
    }
    /* Hide ONLY Streamlit default remove (black ❌) button */
    button[title="Remove file"],
    button[aria-label="Remove file"] {
        display: none !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ SESSION STATE ------------------
if "full_results_df" not in st.session_state:
    st.session_state["full_results_df"] = None
if "annotated_images" not in st.session_state:
    st.session_state["annotated_images"] = []
if "show_download" not in st.session_state:
    st.session_state["show_download"] = False
if "image_results" not in st.session_state:
    st.session_state["image_results"] = []


# ------------------ MODEL LOADING & INFERENCE ------------------
@st.cache_resource
def load_model(path: str):
    """Load YOLO model once and cache it."""
    return YOLO(path)


def run_inference(model, image):
    """Run detection and return plotted image + raw result."""
    results = model.predict(image, conf=CONFIDENCE, iou=IOU)
    r = results[0]
    plotted = r.plot()  # BGR numpy array
    plotted = plotted[:, :, ::-1]  # BGR -> RGB
    pil_img = Image.fromarray(plotted)
    return pil_img, r


def get_class_counts(result, class_names):
    """Return a dict: {class_name: count} for one result."""
    if len(result.boxes) == 0:
        return {}
    cls_indices = result.boxes.cls.tolist()
    labels = [class_names[int(i)] for i in cls_indices]
    counts = Counter(labels)
    return dict(counts)


def get_defect_locations(result, class_names, image_name):
    """Return rows with defect type, confidence and bounding box coords + image name."""
    if len(result.boxes) == 0:
        return []

    boxes = result.boxes
    xyxy = boxes.xyxy.tolist()
    cls_indices = boxes.cls.tolist()
    confs = boxes.conf.tolist()

    rows = []
    for coords, c, cf in zip(xyxy, cls_indices, confs):
        x1, y1, x2, y2 = coords
        rows.append({
            "Image": image_name,
            "Defect type": class_names[int(c)],
            "Confidence": round(float(cf), 2),
            "x1": round(float(x1), 1),
            "y1": round(float(y1), 1),
            "x2": round(float(x2), 1),
            "y2": round(float(y2), 1),
        })

    return rows


# ------------------ SIDEBAR ------------------
with st.sidebar:
    # ---------- MODEL PERFORMANCE ----------
    st.subheader("📊 Model Performance")
    st.markdown("""
    **mAP@50:** 0.9823  
    **mAP@50–95:** 0.5598  
    **Precision:** 0.9714  
    **Recall:** 0.9765
    """)

    st.markdown("---")

    # ---------- HOW TO USE ----------
    st.subheader("🧭 How to Use")
    st.markdown("""
    1. Upload clear PCB images (top view).
    2. Wait for the AI to process defects.
    3. Review annotated outputs.
    4. Download images or ZIP results.
    """)

    st.markdown("---")

    # ---------- SETTINGS (BOTTOM) ----------
    st.subheader("⚙️ Settings")
    st.caption("Model loaded successfully")

# ------------------ MAIN LAYOUT ------------------
st.markdown(
    """
    <div class="header-container">
        <div class="logo-circle">🎄</div>
        <div class="main-title">CircuitGuard – PCB Defect Detection</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Subtitle ✅ (ONLY ONCE)
st.markdown(
    """
    <p class="subtitle-text">
    Detect and highlight <strong>PCB defects</strong> such as missing hole, mouse bite,
    open circuit, short, spur and spurious copper using a YOLO-based deep learning model.
    </p>
    """,
    unsafe_allow_html=True,
)

# Defect badges ✅ (ONLY ONCE)
st.markdown(
    """
    **Defect types detected by this model:**
    <div class="defect-badges">
      <span class="defect-badge">Missing hole</span>
      <span class="defect-badge">Mouse bite</span>
      <span class="defect-badge">Open circuit</span>
      <span class="defect-badge">Short</span>
      <span class="defect-badge">Spur</span>
      <span class="defect-badge">Spurious copper</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ================= UPLOAD HANDLING (SINGLE SOURCE OF TRUTH) =================

# 1️⃣ FILE UPLOADER
uploaded_files = st.file_uploader(
    "Upload PCB images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

# 2️⃣ SESSION STATE INIT (ONLY ONCE)
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

if "image_results" not in st.session_state:
    st.session_state["image_results"] = []

# 3️⃣ ADD NEW FILES (NO DUPLICATES)
if uploaded_files:
    for f in uploaded_files:
        if f.name not in [x.name for x in st.session_state["uploaded_files"]]:
            st.session_state["uploaded_files"].append(f)




# ------------------ DETECTION & DISPLAY ------------------
if "uploaded_files" in st.session_state and st.session_state["uploaded_files"]:
    try:
        model = load_model(MODEL_PATH)
        class_names = model.names  # dict: {id: name}
    except Exception as e:
        st.error(f"Error loading model from `{MODEL_PATH}`: {e}")
    else:
        global_counts = Counter()
        all_rows = []
        
        # 🔥 RESET RESULTS BEFORE FRESH DETECTION
        st.session_state["image_results"] = []
        global_counts = Counter()

        # Run detection for all remaining images
        for file in st.session_state["uploaded_files"]:
            img = Image.open(file).convert("RGB")

            with st.spinner(f"Running detection on {file.name}..."):

                files = {
                    "file": (file.name, file.getvalue(), file.type)
                }

                response = requests.post(API_URL, files=files)

                if response.status_code != 200:
                    st.error(f"Backend error for {file.name}")
                    continue

                api_result = response.json()
                defect_counts = api_result["defects_detected"]
                total_defects = api_result["total_defects"]

                global_counts.update(defect_counts)

                api_result = response.json()

                defect_counts = api_result["defects_detected"]
                total_defects = api_result["total_defects"]

                # -------- decode annotated image --------
                img_base64 = api_result["annotated_image"]
                img_bytes = base64.b64decode(img_base64)
                annotated_img = Image.open(BytesIO(img_bytes))

                existing_names = [r["name"] for r in st.session_state["image_results"]]

                if file.name not in existing_names:
                    st.session_state["image_results"].append(
                        {
                            "name": file.name,
                            "original": img,
                            "annotated": annotated_img,
                            "defect_counts": api_result["defects_detected"],
                            "total": api_result["total_defects"],
                        }
                    )


        # Build full results DF for export (all images)
        if all_rows:
            full_results_df = pd.DataFrame(all_rows)
            st.session_state["full_results_df"] = full_results_df
            st.session_state["annotated_images"] = [
                (res["name"], res["annotated"]) for res in st.session_state["image_results"]

            ]
        else:
            st.session_state["full_results_df"] = None
            st.session_state["annotated_images"] = []

        # Robotic animated success banner + clear info strip
        if st.session_state.get("image_results"):
            st.markdown(
                """
                <div class="robot-success">
                <span class="robot-label">[SYSTEM]</span>
                DEFECT SCAN COMPLETE — ANALYSIS DASHBOARD ONLINE.
                </div>
                <div class="status-strip">
                Detection complete. Scroll down to view detailed results and download options.
                </div>
                """,
                unsafe_allow_html=True,
            )
        # ---------------- DOWNLOAD ALL ANNOTATED IMAGES ----------------
        if st.session_state.get("image_results"):
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for res in st.session_state["image_results"]:
                    img_bytes = io.BytesIO()
                    res["annotated"].save(img_bytes, format="PNG")
                    img_bytes.seek(0)

                    base = os.path.splitext(res["name"])[0]
                    zf.writestr(f"annotated_{base}.png", img_bytes.getvalue())

            zip_buffer.seek(0)

            st.download_button(
                "⬇️ Download all annotated images (ZIP)",
                data=zip_buffer,
                file_name="annotated_images.zip",
                mime="application/zip",
            )


        # Overview grid
        if st.session_state["image_results"]:
            st.markdown("### Annotated results overview")
            grid_cols = st.columns(3)
            for idx, res in enumerate(st.session_state["image_results"]):

                col = grid_cols[idx % 3]
                with col:
                    st.image(
                        res["annotated"],
                        caption=res["name"],
                        use_column_width=True,
                    )

            # Detailed before/after for each image
            st.markdown("### Detailed view per image")
            for idx, res in enumerate(st.session_state["image_results"]):
                st.markdown(f"#### 🖼️ Image: {res['name']}")
                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        res["original"],
                        caption="Original image",
                        use_column_width=True,
                    )

                with col2:
                    st.image(
                        res["annotated"],
                        caption="Annotated detections",
                        use_column_width=True,
                    )

                    # single annotated image download
                    img_bytes = io.BytesIO()
                    res["annotated"].save(img_bytes, format="PNG")
                    img_bytes.seek(0)
                    base = os.path.splitext(res["name"])[0]
                    st.download_button(
                        "Download annotated image",
                        data=img_bytes,
                        file_name=f"annotated_{base}.png",
                        mime="image/png",
                        key=f"download_single_{idx}",
                    )

                
                if res["total"] == 0:
                    st.success("No defects detected in this image.")
                else:
                    st.info(f"Detected **{res['total']}** potential defect(s).")


                

                st.markdown("---")

                       # --------- Overall charts (bar then donut) ----------
        if sum(global_counts.values()) > 0:
            st.subheader("Overall defect distribution across all uploaded images")
            global_df = pd.DataFrame(
                {
                    "Defect Type": list(global_counts.keys()),
                    "Count": list(global_counts.values()),
                }
            )

            # 1️⃣ Bar chart
            bar_chart = (
                alt.Chart(global_df)
                .mark_bar(size=45)
                .encode(
                    x=alt.X(
                        "Defect Type:N",
                        sort="-y",
                        axis=alt.Axis(labelAngle=0),
                    ),
                    y=alt.Y("Count:Q"),
                    tooltip=["Defect Type", "Count"],
                )
                .properties(
                    height=260,
                    padding={"left": 5, "right": 5, "top": 10, "bottom": 10},
                )
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(bar_chart, use_container_width=True)

            # some spacing
            st.markdown("#### Defect type share")

            # 2️⃣ Donut chart directly under it
            donut_chart = (
                alt.Chart(global_df)
                .mark_arc(innerRadius=55, outerRadius=100)
                .encode(
                    theta=alt.Theta("Count:Q", stack=True),
                    color=alt.Color(
                        "Defect Type:N",
                        legend=alt.Legend(title="Defect type"),
                    ),
                    tooltip=["Defect Type", "Count"],
                )
                .properties(
                    height=260,
                    padding={"left": 0, "right": 0, "top": 10, "bottom": 10},
                )
            )
            st.altair_chart(donut_chart, use_container_width=True)

        else:
            st.info("No defects detected in any of the uploaded images.")



        # -------- Export flow: Finish + Download (CSV + annotated images) --------
        if st.session_state["full_results_df"] is not None:
            st.markdown("### Export results")
            if st.button("Finish defect detection"):
                st.session_state["show_download"] = True

            if st.session_state["show_download"]:
                full_results_df = st.session_state["full_results_df"]
                annotated_images = st.session_state["annotated_images"]

                csv_bytes = full_results_df.to_csv(index=False).encode("utf-8")

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    # CSV
                    zf.writestr("circuitguard_detection_results.csv", csv_bytes)
                    # Annotated images
                    for name, pil_img in annotated_images:
                        img_bytes_io = io.BytesIO()
                        pil_img.save(img_bytes_io, format="PNG")
                        img_bytes_io.seek(0)
                        base = os.path.splitext(name)[0]
                        zf.writestr(f"annotated_{base}.png", img_bytes_io.getvalue())

                zip_buffer.seek(0)

                st.download_button(
                    "Download results (CSV + annotated images, ZIP)",
                    data=zip_buffer,
                    file_name="circuitguard_results.zip",
                    mime="application/zip",
                )
if "uploaded_files" not in st.session_state or not st.session_state["uploaded_files"]:
    st.info("Upload one or more PCB images to start detection.")


