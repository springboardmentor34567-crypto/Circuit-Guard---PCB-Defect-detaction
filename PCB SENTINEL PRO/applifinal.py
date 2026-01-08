''' the final version with separate download button '''
import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
import zipfile
import base64

# --- 1. CONFIGURATION ---
API_URL = "http://127.0.0.1:8000/detect/" # Pointing to local backend

st.set_page_config(
    page_title="PCB Sentinel Pro | Fast Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS (EXACTLY AS REQUESTED IN YOUR SNIPPET) ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    
    /* Header */
    .main-title {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF8C00;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Table Headers Centering */
    .centered-header {
        text-align: center !important;
        font-weight: 700;
        font-size: 1.1rem;
        color: #e0e0e0;
        padding-bottom: 10px;
    }
    
    /* Smooth Buttons */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        background-color: #1f2937;
        color: white;
        border: 1px solid #374151;
        transition: all 0.2s;
        height: 45px; /* Fixed height for consistency */
    }
    div.stButton > button:hover {
        background-color: #FF8C00;
        color: white;
        border-color: #FF8C00;
        transform: scale(1.02);
    }
    
    /* Criticality Badges */
    .badge-high { background-color: #7f1d1d; color: #fecaca; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
    .badge-med { background-color: #78350f; color: #fde68a; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
    .badge-low { background-color: #064e3b; color: #a7f3d0; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
    
    /* Center text in metrics/status */
    div[data-testid="stMarkdownContainer"] p {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
def get_criticality(defect_type):
    high_risk = ['open_circuit', 'short', 'missing_hole']
    medium_risk = ['mouse_bite', 'spur']
    if defect_type in high_risk: return "HIGH", "badge-high"
    elif defect_type in medium_risk: return "MEDIUM", "badge-med"
    return "LOW", "badge-low"

# --- 4. POPUPS (FUNCTIONALITY PRESERVED: Single Download) ---
@st.dialog("🔍 High-Resolution Inspection", width="large")
def show_image_dialog(image, name):
    st.image(image, use_container_width=True)
    st.markdown(f"<p style='text-align: center; color: #aaa;'>Filename: {name}</p>", unsafe_allow_html=True)

@st.dialog("🔢 Analysis Summary")
def show_count_dialog(count, filename):
    st.markdown(f"<h3 style='text-align: center;'>{filename}</h3>", unsafe_allow_html=True)
    if count == 0: st.success("✅ PASSED: No defects found.")
    else: st.error(f"❌ FAILED: {count} defects found.")

@st.dialog("📋 Defect Specifications")
def show_details_dialog(df, filename, image): # <-- Functionality: Image added for download
    st.subheader(f"Log: {filename}")
    
    # Show Defects
    if df.empty: st.write("No defects to report.")
    else:
        for _, row in df.iterrows():
            _, css = get_criticality(row['Type'])
            with st.container():
                c1, c2 = st.columns([1, 3], vertical_alignment="center")
                with c1: st.markdown(f"**{row['Type'].upper()}**")
                with c2: st.markdown(f"<span class='{css}'>{row['Criticality']}</span> | Conf: {row['Confidence']}", unsafe_allow_html=True)
                st.caption(f"📍 Box: {row['Box']}")
                st.divider()

    # --- FUNCTIONALITY: SINGLE DOWNLOAD INSIDE POPUP ---
    st.markdown("### 📥 Actions")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        report_text = f"PCB INSPECTION REPORT\nFile: {filename}\nDefects: {len(df)}\n\n"
        if not df.empty: report_text += df.to_string(index=False)
        else: report_text += "Status: PASS"
        zf.writestr(f"report_{filename}.txt", report_text)
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        zf.writestr(f"annotated_{filename}", img_byte_arr.getvalue())

    st.download_button(
        label="⬇️ Download This Report",
        data=zip_buffer.getvalue(),
        file_name=f"Result_{filename}.zip",
        mime="application/zip",
        use_container_width=True
    )

# --- 5. MAIN INTERFACE ---
st.markdown("<div class='main-title'>⚡ PCB Sentinel Pro</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    conf_t = st.slider("Confidence", 0.0, 1.0, 0.25)
    iou_t = st.slider("IoU", 0.0, 1.0, 0.45)
    st.info("Sidebar collapsed for better view.")

uploaded_files = st.file_uploader("📂 Drop PCB Images Here", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if 'results' not in st.session_state:
    st.session_state.results = []

# --- 6. PROCESSING LOOP (FUNCTIONALITY: API CALL) ---
if uploaded_files:
    if st.button(f"🚀 Analyze {len(uploaded_files)} Images via API", type="primary"):
        results_temp = []
        bar = st.progress(0)
        
        for i, f in enumerate(uploaded_files):
            try:
                files = {"file": (f.name, f.getvalue(), "image/jpeg")}
                data = {"conf": conf_t, "iou": iou_t}
                
                # CALL BACKEND
                response = requests.post(API_URL, files=files, data=data)
                
                if response.status_code == 200:
                    api_data = response.json()
                    
                    if "image_base64" in api_data: img_key = "image_base64"
                    elif "annotated_image_base64" in api_data: img_key = "annotated_image_base64"
                    else:
                        st.error(f"❌ Backend Key Error. Keys: {list(api_data.keys())}")
                        st.stop()
                    
                    img_bytes = base64.b64decode(api_data[img_key])
                    pil_plot = Image.open(io.BytesIO(img_bytes))
                    thumbnail = pil_plot.copy()
                    thumbnail.thumbnail((250, 250))
                    
                    defects = []
                    for d in api_data["defects"]:
                        crit, _ = get_criticality(d["Type"])
                        defects.append({
                            "Type": d["Type"],
                            "Confidence": f"{d['Confidence']}",
                            "Box": d["Box"],
                            "Criticality": crit
                        })
                    
                    results_temp.append({
                        "name": api_data["filename"],
                        "full_img": pil_plot,
                        "thumb": thumbnail,
                        "count": api_data["defect_count"],
                        "df": pd.DataFrame(defects)
                    })
                else:
                    st.error(f"Backend returned Error {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to Backend! Ensure 'backend.py' is running.")
                st.stop()
            
            bar.progress((i+1)/len(uploaded_files))
            
        st.session_state.results = results_temp
        st.rerun()

# --- 7. RESULTS VIEW (UI: EXACTLY AS REQUESTED) ---
if st.session_state.results:
    st.markdown("### 🗂️ Results View")
    
    search_query = st.text_input("🔍 Search Image by Filename", placeholder="Type a filename (e.g., 'board_01') to filter results...")
    
    filtered_results = [item for item in st.session_state.results if search_query.lower() in item['name'].lower()]
    
    with st.container(border=True, height=700):
        
        # HEADERS (Centered using Custom CSS Class)
        h1, h2, h3, h4 = st.columns([1.5, 1, 1.5, 1.5], vertical_alignment="center")
        with h1: st.markdown("<div class='centered-header'>📷 Annotated View</div>", unsafe_allow_html=True)
        with h2: st.markdown("<div class='centered-header'>🔢 Count</div>", unsafe_allow_html=True)
        with h3: st.markdown("<div class='centered-header'>📊 Status</div>", unsafe_allow_html=True)
        with h4: st.markdown("<div class='centered-header'>📄 Details</div>", unsafe_allow_html=True)
        st.divider()
        
        if not filtered_results:
            st.info("No matching files.")
        
        for i, item in enumerate(filtered_results):
            # ROWS (Centered Vertical Alignment)
            c1, c2, c3, c4 = st.columns([1.5, 1, 1.5, 1.5], vertical_alignment="center")
            
            with c1:
                st.image(item['thumb'], use_container_width=True)
                if st.button("🔍 Zoom", key=f"z_{i}_{item['name']}"):
                    show_image_dialog(item['full_img'], item['name'])
            
            with c2:
                btn_label = f"🔴 {item['count']}" if item['count'] > 0 else "🟢 0"
                if st.button(btn_label, key=f"c_{i}_{item['name']}"):
                    show_count_dialog(item['count'], item['name'])
            
            with c3:
                if item['count'] == 0:
                    st.success("PASS")
                else:
                    crit_found = any(d['Criticality'] == 'HIGH' for _, d in item['df'].iterrows()) if not item['df'].empty else False
                    if crit_found: st.error("FAIL (Critical)")
                    else: st.warning("FAIL")
            
            with c4:
                # Functionality: Pass 'full_img' for download
                if st.button("📄 Details", key=f"d_{i}_{item['name']}"):
                    show_details_dialog(item['df'], item['name'], item['full_img'])
            
            st.divider()

    # --- FUNCTIONALITY: BATCH DOWNLOAD ---
    if st.button("📦 Download Batch Report (ZIP)"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            summary = ["BATCH SUMMARY REPORT\n"]
            for item in st.session_state.results:
                # 1. Save Image
                ib = io.BytesIO()
                item['full_img'].save(ib, format="JPEG")
                zf.writestr(f"images/annotated_{item['name']}", ib.getvalue())
                
                # 2. Save Individual Text Report
                report_txt = f"FILE: {item['name']}\nCOUNT: {item['count']}\n"
                if not item['df'].empty: report_txt += item['df'].to_string(index=False)
                else: report_txt += "Status: CLEAN"
                zf.writestr(f"reports/report_{item['name']}.txt", report_txt)
                
                # 3. Add to Master Summary
                summary.append(f"FILE: {item['name']} | Count: {item['count']}")
                
            zf.writestr("Master_Summary.txt", "\n".join(summary))
            
        st.download_button("Download Full Batch ZIP", buf.getvalue(), "batch_results.zip", "application/zip")