import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import requests
import base64
import io
import zipfile

FASTAPI_URL = "http://127.0.0.1:8000/predict"

CLASSES = [
    "spurious_copper",
    "missing_hole",
    "short",
    "spur",
    "open_circuit",
    "mouse_bite"
]

st.set_page_config(
    page_title="PCB Defect Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("⚙️ Model Info")
    st.write("Model: **YOLOv8**")
    st.write("mAP50: **96%**")
    st.write("Classes:")
    st.write(CLASSES)

st.title("🔍 PCB Defect Detection Dashboard")
st.write("Upload PCB images to detect defects and analyze predictions.")

uploaded_files = st.file_uploader(
    "Upload PCB Image(s)",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

image_results = []

# -----------------------------
# PROCESS IMAGES (API CALL)
# -----------------------------
if uploaded_files:

    for uploaded_file in uploaded_files:

        with st.spinner(f"Running model on {uploaded_file.name}"):
            response = requests.post(
                FASTAPI_URL,
                files={"file": uploaded_file.getvalue()}
            )

        data = response.json()

        img_bytes = base64.b64decode(data["image"])
        img_np = np.frombuffer(img_bytes, np.uint8)
        annotated_img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        original_img = cv2.imdecode(
            np.frombuffer(uploaded_file.getvalue(), np.uint8),
            cv2.IMREAD_COLOR
        )

        df_conf = pd.DataFrame(data["detections"])

        # Save everything in memory only
        image_results.append({
            "name": uploaded_file.name,
            "original": original_img,
            "detected": annotated_img,
            "df": df_conf,
            "class_list": data["unique_classes"],
            # store in-memory for ZIP
            "image_bytes": img_bytes,
            "log_string": df_conf.to_string(index=False)
        })

    # -----------------------------
    # IMAGE INDEX SELECTOR
    # -----------------------------
    st.markdown("---")
    st.markdown("## 🧭 Select Image for Detailed Analysis")

    labels = [f"{i+1}. {img['name']}" for i, img in enumerate(image_results)]

    idx = st.selectbox(
        "Select Image Index",
        range(len(labels)),
        format_func=lambda x: labels[x]
    )

    selected = image_results[idx]

    # -----------------------------
    # DETAILED VIEW
    # -----------------------------
    st.markdown(f"## 📌 Detailed View: {selected['name']}")

    col1, col2 = st.columns(2)
    with col1:
        st.image(selected["original"], channels="BGR", caption="Original Image")
    with col2:
        st.image(selected["detected"], channels="BGR", caption="Detected Image")

    st.markdown("### 📊 Prediction Summary")
    c1, c2 = st.columns(2)
    c1.metric("Total Detections", len(selected["df"]))
    c2.metric("Unique Defect Types", len(selected["class_list"]))

    st.markdown("### 📄 Confidence + Bounding Box Table")
    st.dataframe(selected["df"], use_container_width=True)

    st.markdown("### 📊 Confidence Histogram & Pie Chart")
    h1, h2 = st.columns(2)

    with h1:
        fig1, ax1 = plt.subplots(figsize=(3, 3))
        ax1.hist(selected["df"]["Confidence (%)"], bins=10)
        st.pyplot(fig1)

    with h2:
        counts = selected["df"]["Class"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(3, 3))
        ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
        st.pyplot(fig2)

    # -----------------------------
    # PER IMAGE ZIP (IN-MEMORY)
    # -----------------------------
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        # Add image
        z.writestr(f"{selected['name']}_detected.jpg", selected["image_bytes"])
        # Add log
        z.writestr(f"{selected['name']}_log.txt", selected["log_string"])
    buf.seek(0)

    st.download_button(
        "📦 Download ZIP (Image + Log)",
        data=buf,
        file_name=f"{selected['name']}_output.zip",
        mime="application/zip"
    )

    # -----------------------------
    # ALL FILES ZIP (IN-MEMORY)
    # -----------------------------
    all_buf = io.BytesIO()
    with zipfile.ZipFile(all_buf, "w") as z:
        for img in image_results:
            z.writestr(f"{img['name']}_detected.jpg", img["image_bytes"])
            z.writestr(f"{img['name']}_log.txt", img["log_string"])
    all_buf.seek(0)

    st.markdown("---")
    st.markdown("## 📦 Download All Outputs (ZIP)")
    st.download_button(
        "⬇ Download ZIP (All Images + Logs)",
        data=all_buf,
        file_name="pcb_results.zip",
        mime="application/zip"
    )
