import streamlit as st
import requests  # NEW: Bridge to FastAPI
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from io import BytesIO
from fpdf import FPDF
import zipfile, os, datetime, base64, tempfile, math
import time 

# ---------------- USER CONFIG ----------------
# API URL replaces local MODEL_PATH
API_URL = "http://127.0.0.1:8000/predict"
CLASS_NAMES = ["mouse bite", "spur", "missing hole", "short", "open ckt", "spurous copper"]
CLASS_COLORS = {
    n: tuple(int(x) for x in np.random.RandomState(abs(hash(n)) % (2**32)).randint(50, 230, size=3))
    for n in CLASS_NAMES
}
GRID_COLS = 3 

# ---------------- PAGE + THEME ----------------
st.set_page_config(page_title="PCB Defect Detector", layout="wide", page_icon="🔬")

# Your Original Neon/Glass CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }
@keyframes gradientMove { 0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%} }
body {
  background: linear-gradient(135deg, #071A52, #0B2545, #0F172A);
  background-size: 400% 400%;
  animation: gradientMove 18s ease infinite;
  color: #e6eef8;
}
.glass-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
  border-radius: 14px;
  padding: 14px;
  border: 1px solid rgba(255,255,255,0.04);
  box-shadow: 0 8px 40px rgba(2,6,23,0.5);
  margin-bottom: 16px;
}
.title {
  text-align:center; font-size:36px; font-weight:700;
  background: linear-gradient(90deg, #60a5fa, #a78bfa, #fb7185);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.subtitle { text-align:center; color:#a3b3c6; margin-top:6px; margin-bottom:12px; }
.stButton>button {
  color: white; border-radius: 10px; padding: 10px 18px;
  border: 1px solid rgba(120,100,255,0.6);
  background: linear-gradient(90deg, rgba(120,100,255,0.12), rgba(96,165,250,0.08));
}
.status-badge { padding: 4px 10px; border-radius: 15px; font-weight: 600; font-size: 13px; text-align: center; display: inline-block; min-width: 80px; }
.status-GOOD { background-color: #16a34a; color: white; }
.status-DEFECTIVE { background-color: #ef4444; color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🔬 PCB DEFECT DETECTOR</div>', unsafe_allow_html=True) 
st.markdown('<div class="subtitle">FastAPI Engine Integrated | Full UI Restored</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Backend Connection")
try:
    requests.get("http://127.0.0.1:8000/", timeout=0.5)
    st.sidebar.success("🟢 API Engine Online")
except:
    st.sidebar.error("🔴 API Engine Offline")

conf_thres = st.sidebar.slider("Confidence threshold", 0.01, 0.9, 0.25, 0.01)
enable_pdf = st.sidebar.checkbox("Enable PDF export", value=True)
heatmap_bins = st.sidebar.slider("Heatmap bins", 32, 200, 100, 8)

# ---------------- SESSION STATE ----------------
if "results" not in st.session_state: st.session_state.results = []
if "last_uploaded" not in st.session_state: st.session_state.last_uploaded = []
if "_open_expander" not in st.session_state: st.session_state._open_expander = None
if "_show_grid" not in st.session_state: st.session_state._show_grid = False

# ---------------- HELPER UTILITIES (Original Logic Restored) ----------------
def image_to_bytes(img: Image.Image) -> bytes:
    buf = BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

def make_json_safe(report):
    return {k: v for k, v in report.items() if not isinstance(v, Image.Image)}

def report_to_txt(r):
    lines = [f"Filename: {r['filename']}", f"Status: {r['status']}", f"Detected: {len(r['defects'])}", ""]
    for i, d in enumerate(r['defects'], 1):
        lines.append(f"{i}. {d['Type']} | Conf: {d['Confidence']:.3f}")
    return "\n".join(lines)

def reports_to_pdf(reports):
    pdf = FPDF(unit="pt", format="A4")
    for r in reports:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 20, f"PCB Inspection: {r['filename']} - {r['status']}", ln=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            r["annotated"].save(tmp.name)
            pdf.image(tmp.name, x=50, w=500)
    return pdf.output(dest="S").encode("latin-1")

def create_zip(reports):
    mem = BytesIO()
    with zipfile.ZipFile(mem, "w") as zf:
        for r in reports:
            zf.writestr(f"annotated_{r['filename']}.png", image_to_bytes(r['annotated']))
            zf.writestr(f"report_{r['filename']}.txt", report_to_txt(r))
    mem.seek(0); return mem

def create_composite_grid_image(reports, grid_cols=GRID_COLS):
    if not reports: return Image.new('RGB', (10, 10))
    num_rows = math.ceil(len(reports) / grid_cols)
    thumb_w, thumb_h = 450, 450
    composite = Image.new('RGB', (grid_cols * thumb_w + 20, num_rows * thumb_h + 20), 'black')
    for i, r in enumerate(reports):
        thumb = r["annotated"].copy(); thumb.thumbnail((thumb_w, thumb_h))
        composite.paste(thumb, ((i % grid_cols) * thumb_w + 10, (i // grid_cols) * thumb_h + 10))
    return composite

# ---------------- UPLOAD & DETECT (MODIFIED FOR FASTAPI) ----------------
st.markdown('<div class="glass-card"><b>Upload & Detect</b></div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload many PCB images", accept_multiple_files=True, type=['png','jpg','jpeg'])

if st.button("🟢 Detect") and uploaded:
    st.session_state.results = []
    progress = st.progress(0)
    for i, f in enumerate(uploaded):
        # 🚀 Send to FastAPI
        files = {"file": (f.name, f.getvalue(), f.type)}
        try:
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                data = response.json()
                
                # Fetch result image from Backend
                annotated_res = requests.get(data["result_url"])
                annotated_img = Image.open(BytesIO(annotated_res.content))

                # Map data to your Original Format
                defects = []
                for d in data["detections"]:
                    defects.append({
                        "Type": d["defect"], "Confidence": d["confidence"],
                        "X": int(d["coordinates"][0]), "Y": int(d["coordinates"][1]),
                        "W": int(d["coordinates"][2] - d["coordinates"][0]),
                        "H": int(d["coordinates"][3] - d["coordinates"][1])
                    })

                st.session_state.results.append({
                    "filename": f.name, "annotated": annotated_img, "defects": defects,
                    "status": "DEFECTIVE" if data["defect_count"] > 0 else "GOOD",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "resolution": f"{annotated_img.width}x{annotated_img.height}",
                    "inference_time": f"{data['speed_ms']}ms"
                })
        except Exception as e:
            st.error(f"Error: {e}")
        progress.progress((i+1)/len(uploaded))
    progress.empty()

# ---------------- RESTORED UI COMPONENTS (Tables, Grid, Heatmaps) ----------------
if st.session_state.results:
    # --- Search Filter ---
    search_query = st.text_input("🔍 Filter by defect type or status").lower().strip()
    filtered = [r for r in st.session_state.results if search_query in r['status'].lower() or any(search_query in d['Type'].lower() for d in r['defects'])] if search_query else st.session_state.results

    # --- Results Table ---
    st.markdown('<div class="glass-card"><b>Analysis Results</b></div>', unsafe_allow_html=True)
    h0, h1, h2, h3, h4 = st.columns([0.5, 3.0, 1.5, 1.5, 1.5])
    h1.write("File Name"); h2.write("Status"); h3.write("Defects"); h4.write("Options")

    for idx, r in enumerate(filtered):
        c0, c1, c2, c3, c4 = st.columns([0.5, 3.0, 1.5, 1.5, 1.5])
        c0.write(idx+1)
        if c1.button(r["filename"], key=f"fbtn_{idx}", use_container_width=True):
            st.session_state._open_expander = idx
        
        c2.markdown(f'<div class="status-badge status-{r["status"]}">{r["status"]}</div>', unsafe_allow_html=True)
        c3.write(len(r["defects"]))
        
        with c4:
            with st.popover("⬇️"):
                st.download_button("🖼️ PNG", image_to_bytes(r["annotated"]), f"img_{idx}.png", use_container_width=True)
                st.download_button("📝 JSON", json.dumps(make_json_safe(r), indent=2), f"rep_{idx}.json", use_container_width=True)

        with st.expander(f"Details: {r['filename']}", expanded=(st.session_state._open_expander == idx)):
            st.image(r["annotated"], width=700)
            if r["defects"]:
                st.dataframe(pd.DataFrame(r["defects"]), use_container_width=True, hide_index=True)

    # --- Grid View Toggle ---
    st.markdown("---")
    if st.button("🖼️ Toggle Annotated Grid View", use_container_width=True):
        st.session_state._show_grid = not st.session_state._show_grid
    
    if st.session_state._show_grid:
        st.image(create_composite_grid_image(filtered), use_column_width=True)

    # --- Heatmaps & Analytics ---
    st.markdown("---")
    with st.expander("📊 Defect Analytics & Distribution"):
        c1, c2 = st.columns(2)
        with c1:
            all_types = [d["Type"] for r in st.session_state.results for d in r["defects"]]
            if all_types: st.bar_chart(pd.Series(all_types).value_counts())
        with c2:
            coords = [[d["X"], d["Y"]] for r in st.session_state.results for d in r["defects"]]
            if coords:
                fig, ax = plt.subplots()
                cdf = pd.DataFrame(coords, columns=['x','y'])
                ax.hist2d(cdf['x'], cdf['y'], bins=heatmap_bins, cmap='hot')
                st.pyplot(fig)

    # --- Bulk Downloads ---
    st.markdown("---")
    st.subheader("Bulk Export Tools")
    b1, b2, b3, b4 = st.columns(4)
    b1.download_button("📦 ZIP Reports", create_zip(st.session_state.results).getvalue(), "pcb_zip.zip", use_container_width=True)
    b2.download_button("🖼️ Stitched PNG", image_to_bytes(create_composite_grid_image(st.session_state.results)), "grid.png", use_container_width=True)
    b3.download_button("📄 Full PDF", reports_to_pdf(st.session_state.results), "report.pdf", use_container_width=True)
    b4.download_button("📝 Combined JSON", json.dumps([make_json_safe(r) for r in st.session_state.results]), "all.json", use_container_width=True)

else:
    st.info("Upload images and click Detect to start.")