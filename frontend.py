import streamlit as st
import requests
import pandas as pd
import base64
import io
import zipfile
from PIL import Image

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="CircuitGuard Client", page_icon="🖥️")
API_URL = "http://127.0.0.1:8000/inspect"

# --- HELPER FUNCTION ---
def base64_to_image(b64_str):
    img_data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_data))

# --- UI LAYOUT ---
st.title("🖥️ CircuitGuard: Industrial Inspection Client")
st.markdown("Batch Processing via **FastAPI Microservice**.")

with st.sidebar:
    st.header("System Status")
    st.success("Backend API: Connected")
    st.info("Model: YOLOv8-Classification")
    st.caption(f"Endpoint: {API_URL}")

# FILE UPLOADER
uploaded_files = st.file_uploader("Upload Test Images", type=['jpg', 'png'], accept_multiple_files=True)

if uploaded_files:
    if st.button(f"🚀 Process {len(uploaded_files)} Images"):
        
        all_defects = []
        processed_images = {} # For Zip
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- BATCH LOOP ---
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Sending {file.name} to server...")
            
            # Prepare Payload
            files = {"file": (file.name, file, "image/jpeg")}
            
            try:
                # CALL API
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if "error" in data:
                        st.error(f"❌ {file.name}: {data['error']}")
                    else:
                        # Success: Parse Data
                        defects = data["defects"]
                        img_b64 = data["annotated_image"]
                        
                        final_img = base64_to_image(img_b64)
                        
                        # Store
                        all_defects.extend(defects)
                        processed_images[file.name] = final_img
                        
                        # Display
                        with st.expander(f"✅ {file.name} ({len(defects)} Defects)", expanded=False):
                            st.image(final_img, caption="Processed Result")
                else:
                    st.error(f"Server Error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 CRITICAL: Cannot connect to Backend! Is 'backend.py' running?")
                st.stop()
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.text("Batch Complete!")
        
        # --- REPORTING & DOWNLOAD ---
        if all_defects:
            st.divider()
            
            # 1. Table
            df = pd.DataFrame(all_defects)
            st.subheader("📋 Inspection Log")
            st.dataframe(df, use_container_width=True)
            
            # 2. Zip File
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                # Add CSV
                zf.writestr("Report.csv", df.to_csv(index=False))
                # Add Images
                for fname, img in processed_images.items():
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    zf.writestr(f"annotated_{fname}", img_byte_arr.getvalue())
            
            st.download_button(
                label="📦 Download Results Package (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="CircuitGuard_Results.zip",
                mime="application/zip"
            )
        else:
            st.warning("No defects found in this batch.")