import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("PCB Defect Detection")

# ---------------- CSS ----------------
st.markdown(
    """
    <style>
    /* Hide default uploader file list & pagination */
    div[data-testid="stFileUploader"] ul,
    div[data-testid="stFileUploader"] li,
    div[data-testid="stFileUploader"] small {
        display: none !important;
    }
    div[data-testid="stFileUploader"] button[aria-label="Next page"],
    div[data-testid="stFileUploader"] button[aria-label="Previous page"] {
        display: none !important;
    }

    /* Floating buttons */
    .float-btn {
        position: fixed;
        right: 30px;
        padding: 10px 14px;
        border-radius: 50%;
        border: none;
        font-size: 16px;
        cursor: pointer;
        z-index: 9999;
        display: none;
    }
    #downloadBtn {
        top: 90px;
        background-color: #ff4b4b;
        color: white;
    }
    #upBtn {
        bottom: 40px;
        background-color: #262730;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Session State ----------------
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "result" not in st.session_state:
    st.session_state.result = None

# ---------------- File Uploader ----------------
uploaded = st.file_uploader(
    "Upload PCB Images",
    type=["jpg", "png"],
    accept_multiple_files=True
)

if uploaded:
    for f in uploaded:
        if f.name not in [uf.name for uf in st.session_state.uploaded_files]:
            st.session_state.uploaded_files.append(f)

# ---------------- One-row uploaded files ----------------
if st.session_state.uploaded_files:
    st.subheader("Uploaded Files")
    cols = st.columns(len(st.session_state.uploaded_files))

    for idx, file in enumerate(st.session_state.uploaded_files):
        with cols[idx]:
            st.markdown(
                f"""
                <div style="border:1px solid #444;padding:8px;border-radius:6px;text-align:center">
                    <b>{file.name}</b><br>
                    <small>{round(file.size/1024,2)} KB</small>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("X", key=f"rm_{idx}", help="Remove file"):
                st.session_state.uploaded_files.pop(idx)
                st.rerun()

# ---------------- Analyze ----------------
if st.button("Analyze Defects") and st.session_state.uploaded_files:
    res = requests.post(
        "http://localhost:8000/analyze",
        files=[("files", f) for f in st.session_state.uploaded_files]
    )
    st.session_state.result = res.json()

# ---------------- Floating buttons (Download / Up) ----------------
st.markdown(
    """
    <button id="downloadBtn" class="float-btn"
        onclick="document.getElementById('download').scrollIntoView({behavior:'smooth'});">
        ⬇
    </button>

    <button id="upBtn" class="float-btn"
        onclick="window.scrollTo({top:0, behavior:'smooth'});">
        ⬆
    </button>

    <script>
    const downloadBtn = document.getElementById("downloadBtn");
    const upBtn = document.getElementById("upBtn");

    window.addEventListener("scroll", () => {
        if (window.scrollY > 200) {
            downloadBtn.style.display = "none";
            upBtn.style.display = "block";
        } else {
            downloadBtn.style.display = "block";
            upBtn.style.display = "none";
        }
    });
    </script>
    """,
    unsafe_allow_html=True
)

# ===================== RESULTS =====================
if st.session_state.result:
    result = st.session_state.result
    df = pd.DataFrame(result["data"])

    # ---------- 1. Annotated Images ----------
    st.subheader("Annotated Images")
    imgs = result.get("annotated_images", [])

    for i in range(0, len(imgs), 4):
        cols = st.columns(4)
        for col, img in zip(cols, imgs[i:i + 4]):
            with col:
                st.image(
                    f"http://localhost:8000/download?path={img}",
                    width=200
                )

    # ---------- 2. Defect Analysis (smaller graphs) ----------
    st.subheader("Defect Analysis")
    if not df.empty:
        c1, c2 = st.columns(2)

        with c1:
            fig1, ax1 = plt.subplots(figsize=(3, 2))
            df["class"].value_counts().plot(kind="bar", ax=ax1)
            ax1.set_title("Defect Count", fontsize=10)
            ax1.tick_params(axis="x", labelrotation=45)
            st.pyplot(fig1)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(3, 2))
            df.boxplot(column="confidence", by="class", ax=ax2)
            plt.suptitle("")
            ax2.set_title("Confidence", fontsize=10)
            st.pyplot(fig2)

    # ---------- 3. Detected Defects Table ----------
    st.subheader("Detected Defects")
    st.dataframe(df, use_container_width=True)

    # ===================== DOWNLOADS =====================
    st.markdown('<div id="download"></div>', unsafe_allow_html=True)
    st.subheader("Download Results")

    # -------- Horizontal pop-downs --------
    c1, c2, c3 = st.columns(3)

    with c1:
        with st.expander("📊 CSV"):
            st.download_button(
                "Combined CSV",
                requests.get(
                    f"http://localhost:8000/download?path={result['csv']}"
                ).content,
                "combined_results.csv"
            )
            for img, path in result.get("csv_separate", {}).items():
                st.download_button(
                    img,
                    requests.get(
                        f"http://localhost:8000/download?path={path}"
                    ).content,
                    f"{img}.csv"
                )

    with c2:
        with st.expander("📄 TXT"):
            st.download_button(
                "Combined TXT",
                requests.get(
                    f"http://localhost:8000/download?path={result['txt']}"
                ).content,
                "combined_results.txt"
            )
            for img, path in result.get("txt_separate", {}).items():
                st.download_button(
                    img,
                    requests.get(
                        f"http://localhost:8000/download?path={path}"
                    ).content,
                    f"{img}.txt"
                )

    with c3:
        with st.expander("📘 PDF"):
            st.download_button(
                "Combined PDF",
                requests.get(
                    f"http://localhost:8000/download?path={result['pdf']}"
                ).content,
                "combined_report.pdf"
            )
            for img, path in result.get("pdf_separate", {}).items():
                st.download_button(
                    img,
                    requests.get(
                        f"http://localhost:8000/download?path={path}"
                    ).content,
                    f"{img}.pdf"
                )

    # -------- ZIP (next line) --------
    if result.get("zip"):
        st.download_button(
            "Download ZIP (All Results)",
            requests.get(
                f"http://localhost:8000/download?path={result['zip']}"
            ).content,
            "all_results.zip"
        )

    # -------- Reset (last line) --------
    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()
