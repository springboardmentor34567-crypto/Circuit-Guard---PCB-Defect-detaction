import requests
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="HARI PCB AI Inspector", page_icon="🟩", layout="wide")


# ---------------- GLOBAL CSS (STRONG NEON FOR PER-DEFECT) ----------------
st.markdown("""
<style>
/* base */
.main { background: radial-gradient(circle at top left, #0f1724 0, #020617 50%); color:#e5e7eb; font-family:"Segoe UI", system-ui; }

/* floating arrow */
#scrollTopBtn {
    position: fixed; bottom: 28px; right: 24px;
    width: 50px; height: 50px; background: #39ff14; border-radius: 50%;
    display: none; justify-content: center; align-items: center; color: #000;
    font-size: 28px; font-weight: 900; cursor: pointer; z-index: 9999;
    box-shadow: 0 0 18px #39ff14;
}
#scrollTopBtn:hover { transform: scale(1.08); }

/* summary preview */
.summary-img img { width: 260px !important; height: auto !important; border-radius: 8px; box-shadow: 0 6px 18px rgba(0,0,0,0.5); }

/* big image */
.big-img-container img { width:100% !important; max-width:1150px !important; height:auto !important; border-radius:12px; box-shadow: 0 0 22px rgba(100,120,255,0.08); }

/* default button style */
.stButton>button { font-size:16px !important; padding:10px 20px !important; border-radius:12px !important; transition: transform .12s; }

/* floating panel */
.floating-panel { border-radius:12px; padding:1rem; background: linear-gradient(180deg, rgba(6,10,23,0.9), rgba(12,18,45,0.85)); border:1px solid rgba(255,255,255,0.03); box-shadow: 0 12px 40px rgba(2,6,23,0.6); margin-bottom:1.25rem; }

/* ------------------- NEON PER-DEFECT BLOCKS ------------------- */
.neon-defect {
    position: relative;
    padding: 14px;
    margin-bottom: 12px;
    border-radius: 12px;
    overflow: hidden;
    background: linear-gradient(180deg, rgba(8,12,25,0.6), rgba(6,10,23,0.8));
    border: 1px solid rgba(255,255,255,0.03);
    display: flex;
    gap: 12px;
    align-items: center;
} 
            

/* left accent bar */
.neon-defect:after {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    width: 6px;
    height: 100%;
    border-top-left-radius: 12px;
    border-bottom-left-radius: 12px;
    background: linear-gradient(180deg, rgba(96,165,250,1), rgba(168,85,247,1), rgba(34,197,94,1));
    box-shadow: 0 0 18px rgba(100,150,255,0.12), inset 0 0 12px rgba(0,0,0,0.15);
    z-index: 0;
}

/* soft neon glow outline */
.neon-defect:before {
    content: '';
    position: absolute;
    inset: -2px;
    border-radius: 14px;
    background: linear-gradient(90deg, rgba(96,165,250,0.06), rgba(168,85,247,0.06), rgba(34,197,94,0.06));
    filter: blur(14px);
    z-index: 0;
}

/* inner content above glow */
.neon-defect .nd-content { position: relative; z-index: 2; display:flex; gap:12px; width:100%; align-items:center; }

/* defect text block */
.nd-text { flex:1; color:#e5e7eb; }
.nd-title { font-weight:800; font-size:1rem; margin-bottom:6px; letter-spacing:0.01em; }
/* small metadata */
.nd-meta { color:#9ca3af; font-size:0.92rem; }

/* neon badges */
.badge {
    display:inline-block; padding:6px 10px; border-radius:999px; font-weight:800; font-size:0.85rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.4);
}

/* type badge gradient */
.badge-type { background: linear-gradient(90deg,#60a5fa,#a855f7); color:#021; }

/* severity badges */
.badge-sev-high { background: linear-gradient(90deg,#ff4d6d,#ff7a59); color:#fff; }
.badge-sev-medium { background: linear-gradient(90deg,#ffd166,#ffdd57); color:#081; }
.badge-sev-low { background: linear-gradient(90deg,#9be15d,#00e6b8); color:#021; }

/* right-side quick stats */
.nd-stats { min-width:160px; text-align:right; color:#e5e7eb; font-weight:700; }

/* subtle pulse animation on severity high */
@keyframes neonPulse {
    0% { box-shadow:0 0 8px rgba(255,77,109,0.35); transform:translateY(0px); }
    50% { box-shadow:0 0 18px rgba(255,77,109,0.55); transform:translateY(-2px); }
    100% { box-shadow:0 0 8px rgba(255,77,109,0.35); transform:translateY(0px); }
}
.pulse { animation: neonPulse 1.8s infinite; }

/* make badges crisp */
.badge { color: #061018; }

/* small screen tweaks */
@media (max-width: 900px) {
    .nd-stats { display:none; }
    .summary-img img { width: 200px !important; }
    .big-img-container img { max-width: 900px !important; }
}
</style>

<div id="scrollTopBtn" onclick="document.getElementById('upload_section').scrollIntoView({behavior:'smooth'})">↑</div>

<script>
document.addEventListener('scroll', function() {
    var b = document.getElementById("scrollTopBtn");
    if (!b) return;
    if (window.scrollY > 180) b.style.display = "flex"; else b.style.display = "none";
});
</script>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div style="font-size:2.4rem;font-weight:900;text-align:center;text-transform:uppercase;
background:linear-gradient(90deg,#60a5fa,#a855f7,#22c55e);-webkit-background-clip:text;-webkit-text-fill-color:transparent;padding:0.8rem 0;">
YOLOv8 PCB Defect Detection System
</div>
""", unsafe_allow_html=True)
st.markdown('<div style="text-align:center;opacity:0.8;margin-bottom:8px;">Real-time PCB defect detection using deep learning.</div>', unsafe_allow_html=True)


# ---------------- SESSION ----------------
if "pred_results" not in st.session_state:
    st.session_state["pred_results"] = []
if "open_panels" not in st.session_state:
    st.session_state["open_panels"] = []
if "scroll_to" not in st.session_state:
    st.session_state["scroll_to"] = None
if "search_value" not in st.session_state:
    st.session_state["search_value"] = ""

# ---------------- UPLOAD ----------------
st.markdown('<a id="upload_section"></a>', unsafe_allow_html=True)
uploaded_files = st.file_uploader("Upload PCB Images", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded_files:
    new_names = [f.name for f in uploaded_files]
    old_names = [p["name"] for p in st.session_state["pred_results"]]
    if new_names != old_names:
        st.session_state["pred_results"] = []
        st.session_state["open_panels"] = []

# ---------------- PREDICTION BUTTON ----------------
# ---------------- PREDICTION BUTTON ----------------
if uploaded_files:
    ca, cb, cc = st.columns([1, 1, 1])

    with cb:
        btn = st.button("🟩 RUN PREDICTION")

    # ---- STYLE AFTER BUTTON EXISTS ----
    st.markdown("""
    <script>
    (function(){
        const bts = Array.from(document.querySelectorAll('button'));
        for (const b of bts) {
            if (b.innerText && b.innerText.trim().startsWith('🟩 RUN PREDICTION')) {
                b.style.fontSize='20px';
                b.style.padding='12px 36px';
                b.style.borderRadius='12px';
                b.style.fontWeight='900';
                b.style.background='linear-gradient(90deg,#22c55e,#10b981)';
                b.style.color='#000';
                b.style.boxShadow='0 0 30px rgba(34,197,94,0.75)';
                b.style.border='none';
            }
        }
    })();
    </script>
    """, unsafe_allow_html=True)

    if btn:
        st.session_state["pred_results"] = []

        with st.spinner("Running inference..."):
            for f in uploaded_files:

                api_url = "http://127.0.0.1:8000/predict"
                files = {"file": (f.name, f.getvalue(), "image/jpeg")}

                response = requests.post(api_url, files=files)
                if response.status_code != 200:
                    st.error("❌ Backend prediction failed")
                    st.stop()

                data = response.json()
                
                # ---------------- DECODE YOLO OUTPUT IMAGE ----------------

                img_bytes = base64.b64decode(
                 data["image"]
                )

                result_img = Image.open(
                BytesIO(img_bytes)
                ).convert("RGB")


                rows = []
                for i, d in enumerate(data["boxes"]):
                    x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
                    w, h = x2 - x1, y2 - y1
                    cx, cy = x1 + w // 2, y1 + h // 2

                    sev = "High" if d["confidence"] > 0.85 else \
                          "Medium" if d["confidence"] > 0.6 else "Low"

                    rows.append({
                        "Index": i + 1,
                        "Type": d["type"],
                        "Confidence": round(d["confidence"], 3),
                        "Severity": sev,
                        "Location": f"{x1}, {y1} → {x2}, {y2}",
                        "Size": f"{w}×{h}",
                        "Center": f"{cx}, {cy}"
                    })

                type_counts = pd.Series([r["Type"] for r in rows]).value_counts().to_dict()
                
            st.session_state["pred_results"].append(
    {
        "name": f.name,

        # Original uploaded image
        "input_pil": Image.open(
            f
        ).convert(
            "RGB"
        ),

        # YOLO annotated image (decoded from backend)
        "result_pil": result_img,

        # Detection details
        "defect_rows": rows,
        "type_counts": type_counts
    }
)

      

# ---------------- SEARCH ----------------
# ---------------- IMPROVED SMART SEARCH ----------------
# ---------------- SEARCH (with icon button) ----------------
if st.session_state["pred_results"]:
    c1, c2 = st.columns([5, 1])

    with c1:
        search_text = st.text_input(
            "Search defects or file name",
            placeholder="mis, hole, circ, 01_",
            value=st.session_state["search_value"],
            label_visibility="collapsed"
        )

    with c2:
        search_btn = st.button("🔍", use_container_width=True)

    # Update search value when typing OR clicking icon
    if search_text != st.session_state["search_value"] or search_btn:
        st.session_state["search_value"] = search_text


search_raw = st.session_state.get("search_value", "").strip().lower()
results_to_show = st.session_state["pred_results"]  # default - all

if search_raw:
    keys = [k.strip() for k in search_raw.split(",") if k.strip()]
    filtered = []

    for r in st.session_state["pred_results"]:
        name_l = r["name"].lower()
        defect_types = [d["Type"].lower() for d in r["defect_rows"]]

        match_found = False

        # partial match in filename
        for key in keys:
            if key in name_l:
                match_found = True
                break

        # partial match in defect types
        if not match_found:
            for d in defect_types:
                for key in keys:
                    if key in d:
                        match_found = True
                        break
                if match_found:
                    break

        if match_found:
            filtered.append(r)

    if len(filtered) == 0:
        st.warning("No matching results found. Check the name or defect spelling.")
    else:
        results_to_show = filtered
# ---------------- SUMMARY ----------------
if st.session_state["pred_results"]:
    st.markdown("### Summary Table")

    for idx, r in enumerate(results_to_show):
        c0, c1, c2, c3 = st.columns([1, 2, 2, 1])

        # ----------- Image Preview ----------
        with c0:
            st.markdown('<div class="summary-img">', unsafe_allow_html=True)
            st.image(r["result_pil"])
            st.markdown('</div>', unsafe_allow_html=True)

        # ----------- File Info ----------
        with c1:
            st.markdown(f"**{r['name']}**")
            st.caption(f"{len(r['defect_rows'])} defects")

        # ----------- Defect Buttons ----------
        with c2:
            for j, (t, v) in enumerate(r["type_counts"].items()):
                if st.button(f"{t} ({v})", key=f"type_{idx}_{j}"):
                    if idx not in st.session_state["open_panels"]:
                        st.session_state["open_panels"].append(idx)

                    st.session_state["scroll_to"] = f"panel_{idx}"
                    st.rerun()

        # ----------- View Button (Only button now) ----------
        with c3:
            if st.button("View", key=f"view_{idx}"):
                if idx not in st.session_state["open_panels"]:
                    st.session_state["open_panels"].append(idx)

                st.session_state["scroll_to"] = f"panel_{idx}"
                st.rerun()
# ---------------- DETAIL PANELS (NEON DEFECTS) ----------------
for idx in st.session_state["open_panels"][:]:
    if idx >= len(st.session_state["pred_results"]): continue
    det = st.session_state["pred_results"][idx]
    st.markdown(f"<div id='panel_{idx}'></div>", unsafe_allow_html=True)
    st.markdown('<div class="floating-panel">', unsafe_allow_html=True)
    st.markdown(f"### Detailed Report — **{det['name']}**")
    colA, colB = st.columns([1,1])
    with colA:
        st.markdown("**Original Image**")
        st.markdown('<div class="big-img-container">', unsafe_allow_html=True)
        st.image(det["input_pil"])
        st.markdown('</div>', unsafe_allow_html=True)
    with colB:
        st.markdown("**Prediction Output**")
        st.markdown('<div class="big-img-container">', unsafe_allow_html=True)
        st.image(det["result_pil"])
        st.markdown('</div>', unsafe_allow_html=True)
    df = pd.DataFrame(det["defect_rows"])
    st.dataframe(df, use_container_width=True)
    st.markdown("### Per-Defect Details")
    for row in df.to_dict("records"):
        sev_class = "badge-sev-low"
        extra = ""
        if row['Severity'] == "High":
            sev_class = "badge-sev-high"; extra = "pulse"
        elif row['Severity'] == "Medium":
            sev_class = "badge-sev-medium"
        st.markdown(f"""
        <div class="neon-defect">
            <div class="nd-content">
                <div class="nd-text">
                    <div class="nd-title">{row['Type']} <span style="color:#9ca3af; font-weight:700; font-size:0.9rem;">#{row['Index']}</span></div>
                    <div style="margin-bottom:8px;">
                        <span class="badge badge-type">Type: {row['Type']}</span>
                        <span class="badge {sev_class} {extra}">Severity: {row['Severity']}</span>
                    </div>
                    <div class="nd-meta"><strong>Confidence:</strong> {row['Confidence']} &nbsp; • &nbsp; <strong>Location:</strong> {row['Location']} &nbsp; • &nbsp; <strong>Size:</strong> {row['Size']}</div>
                </div>
                <div class="nd-stats">
                    <div style="font-size:0.95rem; color:#9ca3af;">Center</div>
                    <div style="font-weight:800; font-size:0.95rem;">{row['Center']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("### Defect Graph Summary")
    if det["type_counts"]:
        g1,g2 = st.columns([1,1])
        with g1:
            fig, ax = plt.subplots(figsize=(5.5,3), dpi=130)
            fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#0f172a")
            keys = list(det["type_counts"].keys()); vals = list(det["type_counts"].values())
            colors = ["#60a5fa","#a855f7","#22c55e","#4ade80"]
            bars = ax.bar(keys, vals, color=colors[:len(keys)], edgecolor="white")
            ax.set_title("Defect Count", color="#e5e7eb")
            ax.tick_params(colors="#e5e7eb")
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, str(int(b.get_height())), ha="center", color="#e5e7eb")
            for s in ax.spines.values(): s.set_visible(False)
            st.pyplot(fig)
        with g2:
            st.write("**Total Defects:**", sum(vals))
            for k,v in zip(keys, vals):
                st.write(f"- **{k}:** {v}")
    else:
        st.info("No defects found.")
    if st.button(f"Close {det['name']}", key=f"close_{idx}"):
        st.session_state["open_panels"].remove(idx); st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
# ---------------- ZIP DOWNLOAD BUTTONS ----------------
import io
import zipfile
from reportlab.pdfgen import canvas

def generate_pdf(defects, filename):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica", 12)
    c.drawString(30, 800, f"Defect Report: {filename}")
    y = 770
    for d in defects:
        line = f"{d['Index']}. {d['Type']} | Conf: {d['Confidence']} | Sev: {d['Severity']} | Loc: {d['Location']}"
        c.drawString(30, y, line)
        y -= 18
        if y < 40:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = 800
    c.save()
    buffer.seek(0)
    return buffer.read()

def create_zip(result_list):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as z:
        for r in result_list:
            name = r["name"]

            # Add prediction image
            img_buffer = io.BytesIO()
            r["result_pil"].save(img_buffer, format="PNG")
            z.writestr(f"{name}/prediction.png", img_buffer.getvalue())

            # Add original image
            ori_buffer = io.BytesIO()
            r["input_pil"].save(ori_buffer, format="PNG")
            z.writestr(f"{name}/original.png", ori_buffer.getvalue())

            # Add text file
            txt = ""
            for d in r["defect_rows"]:
                txt += f"{d['Index']} - {d['Type']} | Conf:{d['Confidence']} | Sev:{d['Severity']} | Loc:{d['Location']}\n"
            z.writestr(f"{name}/details.txt", txt)

            # Add PDF
            pdf_bytes = generate_pdf(r["defect_rows"], name)
            z.writestr(f"{name}/report.pdf", pdf_bytes)

    zip_buffer.seek(0)
    return zip_buffer

# Buttons visible only when results exist
if st.session_state["pred_results"]:
    st.markdown("### Downloads")

    # BUTTON – Download your Search Results
    if search_raw and results_to_show:
        if st.button(" Download your Search Results (ZIP)"):
            zip_file = create_zip(results_to_show)
            st.download_button(
                label="⬇️ Click to Download Searched ZIP",
                data=zip_file,
                file_name="searched_results.zip",
                mime="application/zip"
            )

    # BUTTON – Download All Results ZIP
    if st.button(" Download All Results (ZIP)"):
        zip_file = create_zip(st.session_state["pred_results"])
        st.download_button(
            label="⬇️ Click to Download ALL Results ZIP",
            data=zip_file,
            file_name="all_results.zip",
            mime="application/zip"
        )

# ---------------- ROBUST SCROLL ----------------
if st.session_state.get("scroll_to"):
    anchor = st.session_state["scroll_to"]
    st.markdown(f"""
    <script>
    setTimeout(function() {{
        var el = document.getElementById("{anchor}");
        if (!el) {{
            var tries = 0;
            var interval = setInterval(function() {{
                el = document.getElementById("{anchor}");
                if (el) {{
                    clearInterval(interval);
                    setTimeout(function(){{ el.scrollIntoView({{behavior:'smooth', block:'start'}}); }}, 70);
                }}
                if (tries > 70) clearInterval(interval);
                tries++;
            }}, 180);
        }} else {{
            el.scrollIntoView({{behavior:'smooth', block:'start'}});
        }}
    }}, 420);
    </script>
    """, unsafe_allow_html=True)
    st.session_state["scroll_to"] = None
# ---------------- GLOBAL JS LISTENER — Scroll to TOP anchor ----------------
st.markdown("""
<script>
window.addEventListener('message', (event) => {
    if (!event.data || event.data.type !== "scrollToUpload") return;

    function scrollTry(times) {
        let el = document.getElementById("upload_section");
        if (el) {
            el.scrollIntoView({ behavior:"smooth", block:"start" });
        } else if (times > 0) {
            setTimeout(() => scrollTry(times - 1), 150);
        }
    }

    scrollTry(40);  // keep trying for ~6 seconds
});
</script>
""", unsafe_allow_html=True)
# ---------------- FLOATING ARROW BUTTON ----------------
st.markdown("""
<style>
#fixedScrollUpButton {
    position: fixed;
    bottom: 24px;
    right: 24px;
    width: 52px;
    height: 52px;
    border-radius: 50%;
    background: #10b981;
    color: #021;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 28px;
    font-weight: 900;
    cursor: pointer;
    z-index: 99999;
    box-shadow: 0 0 16px rgba(16,185,129,0.55);
}
#fixedScrollUpButton:hover { transform: scale(1.08); }
</style>

<div id="fixedScrollUpButton">⮝</div>

<script>
document.getElementById("fixedScrollUpButton").onclick = function() {
    window.postMessage({ type: "scrollToUpload" }, "*");
};
</script>
""", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown('<div style="text-align:center;opacity:0.75;padding:1rem;">Created by HARI</div>', unsafe_allow_html=True)
