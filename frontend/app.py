
import io
import time
import zipfile
import base64
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import requests
from PIL import Image
import pandas as pd
import altair as alt

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# CONFIG 
API_BATCH_URL = "http://127.0.0.1:8000/detect-batch"
MAX_IMAGE_WIDTH = 1024
MAX_WORKERS = min(4, os.cpu_count() or 2)

st.set_page_config(
    page_title="CircuitGuard – PCB Defect Detection",
    page_icon="🛡️",
    layout="wide"
)
st.markdown("", unsafe_allow_html=True)


st.session_state.setdefault("open_row_idx", None)
st.session_state.setdefault("batch_done", False)
st.session_state.setdefault("batch_results", None)
st.session_state.setdefault("originals", None)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap');

    body, html {
        font-family: 'Space Grotesk', sans-serif;
    }

    .header-container {
        display:flex;
        flex-direction:column;
        align-items:center;
        margin-top:0.5rem;
        margin-bottom:1rem;
    }

    .logo-circle {
        width:64px;
        height:64px;
        border-radius:50%;
        background:#e0f2fe;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:32px;
        margin-bottom:6px;
    }

    .metric-card {
        border-radius:16px;
        padding:0.75rem 1rem;
        background:#fff;
        border:1px solid #dbeafe;
    }

    .metric-label {
        font-size:0.75rem;
        text-transform:uppercase;
        color:#6b7280;
    }

    .metric-value {
        font-size:1.1rem;
        font-weight:600;
    }

    .result-row {
        background:#ffffff;
        border:1px solid #eef6ff;
        border-radius:12px;
        padding:12px 16px;
        margin-bottom:10px;
        box-shadow:0 4px 10px rgba(0,0,0,0.03);
    }

    .stButton>button {
        border-radius:999px;
        font-weight:600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data(show_spinner=False)
def load_raw_image(raw_bytes: bytes, size: tuple):
    return Image.frombytes("RGB", size, raw_bytes)

@st.cache_data(show_spinner=False)
def load_encoded_image(encoded_bytes: bytes):
    return Image.open(io.BytesIO(encoded_bytes))

@st.cache_data(show_spinner=False)
def cached_df(rows):
    return pd.DataFrame(rows).drop(columns=["Image"])


@st.cache_data(show_spinner=False)
def decode_annotated(encoded_bytes):
    return Image.open(io.BytesIO(encoded_bytes))


st.markdown(
    """
    <div class="header-container">
        <div class="logo-circle">🛡️</div>
        <div style="
            font-family:'Segoe UI Black', cursive !important;
            font-size:2.7rem;
            letter-spacing:2px;
            color:#13406b;
            text-align:center;
        ">
            CircuitGuard – PCB Defect Detection
        </div>
    </div>
    """,
    unsafe_allow_html=True
)




cols = st.columns(4)
metrics = [
    ("mAP@50", "0.9823"),
    ("mAP@50–95", "0.5598"),
    ("Precision", "0.9714"),
    ("Recall", "0.9765"),
]
for c, (k, v) in zip(cols, metrics):
    with c:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>{k}</div>"
            f"<div class='metric-value'>{v}</div></div>",
            unsafe_allow_html=True
        )



st.markdown(
    "<div style='background:white;border:1px solid #e5efff;border-radius:14px;"
    "padding:18px 22px;margin:20px 0;box-shadow:0 6px 18px rgba(0,0,0,0.04)'>"
    "<div style='font-size:1.1rem;font-weight:600;color:#13406b;margin-bottom:10px'>"
    "🛠️ How to use CircuitGuard</div>"
    "<ol style='margin:0;padding-left:18px;color:#334155;font-size:0.95rem;line-height:1.6'>"
    "<li>Upload one or more PCB images (PNG / JPG).</li>"
    "<li>Wait while defects are detected automatically.</li>"
    "<li>Review annotated images and defect details.</li>"
    "<li>Download a complete inspection report.</li>"
    "</ol>"
    "<div style='margin-top:10px;font-size:0.85rem;color:#64748b'>"
    "💡 Tip: Higher resolution images give better defect detection.</div>"
    "</div>",
    unsafe_allow_html=True
)


st.markdown("### Upload PCB Images")
uploaded_files = st.file_uploader(
    "Upload images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)


def generate_pdf(result):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"<b>Image:</b> {result['name']}", styles["Heading2"]))
    elements.append(Spacer(1, 12))

    for title, img in [("Original Image", result["original_img"]),
                       ("Annotated Image", result["annotated_img"])]:
        img_buf = io.BytesIO()
        img.resize((800, int(800 * img.height / img.width))) \
           .save(img_buf, format="JPEG", quality=65, optimize=True)
        img_buf.seek(0)

        elements.append(Paragraph(title, styles["Heading4"]))
        elements.append(RLImage(img_buf, width=250, height=150))
        elements.append(Spacer(1, 12))

    if result["loc_rows"]:
        df = pd.DataFrame(result["loc_rows"]).drop(columns=["Image"])
        table = Table([df.columns.tolist()] + df.values.tolist())
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONT", (0,0), (-1,0), "Helvetica-Bold")
        ]))
        elements.append(Paragraph("Defect Locations", styles["Heading4"]))
        elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer


if uploaded_files and not st.session_state.get("batch_done", False):

    with st.spinner("Running batch defect detection..."):

        originals = {}
        files_payload = []

        for f in uploaded_files:
            img = Image.open(f).convert("RGB")

           
            if img.width > MAX_IMAGE_WIDTH:
                new_h = int((MAX_IMAGE_WIDTH / img.width) * img.height)
                img = img.resize((MAX_IMAGE_WIDTH, new_h))

            originals[f.name] = img

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            buf.seek(0)

            files_payload.append(
                ("files", (f.name, buf.getvalue(), "image/jpeg"))
            )

       
        try:
            response = requests.post(
                API_BATCH_URL,
                files=files_payload,
                timeout=(10, 900)  
            )
            response.raise_for_status()

        except requests.exceptions.ReadTimeout:
            st.error(
                "⏳ Processing is taking longer than expected.\n\n"
                "Try fewer images or wait longer."
            )
            st.stop()

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Backend error: {e}")
            st.stop()

    
        st.session_state["batch_results"] = response.json()
        st.session_state["originals"] = originals
        st.session_state["batch_done"] = True



if st.session_state["batch_done"]:
    batch_results = st.session_state["batch_results"]
    originals = st.session_state["originals"]

    image_results = []
    all_rows = []
    global_counts = Counter()

    for idx, res in enumerate(batch_results):
        name = res["filename"]

        ann_bytes = base64.b64decode(res["annotated_image"])
        ann_img = decode_annotated(ann_bytes)

        rows = []
        for d in res["detections"]:
            rows.append({
                "Image": name,
                "Defect type": d["class_name"],
                "Confidence": round(d["confidence"], 2),
                "x1": round(d["bbox"][0], 1),
                "y1": round(d["bbox"][1], 1),
                "x2": round(d["bbox"][2], 1),
                "y2": round(d["bbox"][3], 1),
            })

        global_counts.update([r["Defect type"] for r in rows])
        all_rows.extend(rows)

        orig = originals[name]

        image_results.append({
            "idx": idx,
            "name": name,
            "original_bytes": orig.tobytes(),
            "original_size": orig.size,
            "annotated_bytes": ann_bytes,
            "loc_rows": rows,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })

    st.session_state.image_results = image_results
    st.session_state.all_rows = all_rows



    summary = [{
        "idx": r["idx"],
        "name": r["name"],
        "count": len(r["loc_rows"]),
        "max_conf": max([x["Confidence"] for x in r["loc_rows"]], default=0),
        "res": r
    } for r in image_results]

    st.markdown("### Search & Filter")
    c1, c2, c3 = st.columns(3)
    with c1:
        search_name = st.text_input("Search by image name")
    with c2:
        search_defect = st.selectbox(
            "Filter by defect type",
            ["All"] + sorted(global_counts.keys())
        )
    with c3:
        min_defects = st.number_input("Minimum defects", min_value=0, value=0)

    filtered = []
    for s in summary:
        if search_name and search_name.lower() not in s["name"].lower():
            continue
        if search_defect != "All":
            if search_defect not in [r["Defect type"] for r in s["res"]["loc_rows"]]:
                continue
        if s["count"] < min_defects:
            continue
        filtered.append(s)

    st.markdown(f"**Showing {len(filtered)} of {len(summary)} images**")

    header = st.columns([4, 1.5, 1.5, 2])
    header[0].markdown("**Image Name**")
    header[1].markdown("**No. of Defects**")
    header[2].markdown("**Max Confidence**")
    header[3].markdown("**Processed At**")

    results_container = st.container(height=360)

    with results_container:
        for s in filtered:
            r = s["res"]
            idx = s["idx"]

            st.markdown("<div class='result-row'>", unsafe_allow_html=True)
            cols = st.columns([4, 1.5, 1.5, 2])

            if cols[0].button(r["name"], key=f"btn_{idx}"):
                st.session_state["open_row_idx"] = (
                    None if st.session_state["open_row_idx"] == idx else idx
                )

            cols[1].markdown(f"<span style='font-size:16px'>{s['count']}</span>", unsafe_allow_html=True)
            cols[2].markdown(f"<span style='font-size:16px'>{round(s['max_conf'],2)}</span>", unsafe_allow_html=True)
            cols[3].markdown(f"<span style='font-size:16px'>{r['processed_at']}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.session_state["open_row_idx"] == idx:
                orig_img = load_raw_image(r["original_bytes"], r["original_size"])
                ann_img = load_encoded_image(r["annotated_bytes"])

                c1, c2 = st.columns(2)
                with c1:
                    st.image(orig_img, caption="Original", use_container_width=True)
                with c2:
                    st.image(ann_img, caption="Annotated", use_container_width=True)

                if r["loc_rows"]:
                    st.dataframe(cached_df(r["loc_rows"]), use_container_width=True)
                st.markdown("---")

    st.subheader("Overall defect distribution")

    df = pd.DataFrame({
        "Defect": list(global_counts.keys()),
        "Count": list(global_counts.values())
    })

    col1, col2 = st.columns(2)
    with col1:
        bar = (
            alt.Chart(df)
            .mark_bar(size=22, cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Defect:N", sort="-y"),
                y="Count:Q",
                tooltip=["Defect", "Count"]
            )
            .properties(height=300)
        )
        st.altair_chart(bar, use_container_width=True)

    with col2:
        pie = (
            alt.Chart(df)
            .mark_arc(innerRadius=60)
            .encode(
                theta="Count:Q",
                color="Defect:N",
                tooltip=["Defect", "Count"]
            )
            .properties(height=300)
        )
        st.altair_chart(pie, use_container_width=True)


has_results = (
    "image_results" in st.session_state
    and len(st.session_state.image_results) > 0
)

st.subheader("Download Results")

if not uploaded_files:
    st.info("Upload images to enable download.")

elif not has_results:
    st.info("Run defect detection to enable download.")

else:
    if st.button("Prepare Download"):
        st.info("Preparing download…")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            
            zf.writestr(
                "defect_locations.csv",
                pd.DataFrame(st.session_state.all_rows).to_csv(index=False)
            )

            
            for r in st.session_state.image_results:
                r["original_img"] = load_raw_image(
                    r["original_bytes"], r["original_size"]
                )
                r["annotated_img"] = load_encoded_image(
                    r["annotated_bytes"]
                )

                pdf = generate_pdf(r)
                zf.writestr(f"{r['name']}.pdf", pdf.getvalue())

        st.success("Download ready")

        st.download_button(
            label="⬇️ Download ZIP",
            data=zip_buffer.getvalue(),
            file_name="circuitguard_results.zip",
            mime="application/zip"
        )
