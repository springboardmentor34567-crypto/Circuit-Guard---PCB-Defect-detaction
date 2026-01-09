# interface.py
import gradio as gr
import pandas as pd
import zipfile
import os
import shutil
import time
import json
from datetime import datetime
from collections import Counter

from ultralytics import YOLO
from PIL import Image
import numpy as np

# PDF generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# Plotly
import plotly.graph_objects as go

# -------------------------
# Configuration
# -------------------------
MODEL_PATH = "/home/omen/Downloads/ML/pcb_detector/ML/runs/detect/yolov11m/weights/last.pt"
PROCESSED_FOLDER = "processed_results"
ZIP_NAME = "report_package.zip"
CONF_THRESHOLD = 0.25

model = YOLO(MODEL_PATH)

# -------------------------
# Save annotated result
# -------------------------
def _save_annotated(result, out_path):
    try:
        img = result.plot()
        Image.fromarray(img).save(out_path)
        return True
    except:
        return False

# -------------------------
# JSON Metadata
# -------------------------
def save_json(path, metadata):
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

# -------------------------
# PDF Report Generator
# -------------------------
def create_pdf(pdf_path, orig_path, annot_path, metadata):
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>PCB DEFECT REPORT — {metadata['file_name']}</b>", styles['Title']))
    story.append(Spacer(1, 10))

    story.append(Paragraph(f"Model: {metadata['model']}", styles['Normal']))
    story.append(Paragraph(f"Timestamp: {metadata['timestamp']}", styles['Normal']))
    story.append(Paragraph(f"Inference Time: {metadata['inference_ms']} ms", styles['Normal']))
    story.append(Paragraph(f"Total Defects: {metadata['defect_count']}", styles['Normal']))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Original Image</b>", styles['Heading3']))
    try:
        story.append(RLImage(orig_path, width=350, height=350))
    except:
        story.append(Paragraph("Original image unavailable", styles['Normal']))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Annotated Image</b>", styles['Heading3']))
    try:
        story.append(RLImage(annot_path, width=350, height=350))
    except:
        story.append(Paragraph("Annotated image unavailable", styles['Normal']))

    # Defect table
    table_data = [["Label", "Confidence", "BBox"]]
    for d in metadata.get("defects", []):
        table_data.append([d.get("label", "-"), round(d.get("confidence", 0), 3), str(d.get("bbox", "-"))])

    story.append(Table(table_data))
    doc.build(story)

# -------------------------
# PROCESSING
# -------------------------
def process_images_layout(files, progress=gr.Progress()):
    if not files:
        return (
            gr.update(visible=False), None,
            "No files uploaded.", "-", "-", "-",
            gr.update(visible=False)
        )

    # prepare output folder
    if os.path.exists(PROCESSED_FOLDER):
        shutil.rmtree(PROCESSED_FOLDER)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    rows = []
    files_state = []

    sum_conf = 0
    count_conf = 0
    total_defects = 0
    total_files = len(files)

    progress(0.05, desc="Processing images...")

    for idx, f in enumerate(files):
        orig_path = f.name if hasattr(f, "name") else str(f)
        base = os.path.basename(orig_path)
        name_noext = os.path.splitext(base)[0]

        pred_path = os.path.join(PROCESSED_FOLDER, f"{name_noext}_annotated.jpg")
        json_path = os.path.join(PROCESSED_FOLDER, f"{name_noext}.json")
        pdf_path = os.path.join(PROCESSED_FOLDER, f"{name_noext}.pdf")

        # YOLO inference
        t0 = time.time()
        results = model.predict(orig_path, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
        res = results[0]
        t1 = time.time()

        inference_ms = round((t1 - t0) * 1000, 2)
        defects_count = len(res.boxes) if hasattr(res, "boxes") else 0

        # annotated image
        if not _save_annotated(res, pred_path):
            try:
                shutil.copy(orig_path, pred_path)
            except:
                pass

        # defect list
        defects_list = []
        try:
            # res.boxes is iterable of Box objects
            for b in res.boxes:
                # get bbox xyxy
                try:
                    xy = b.xyxy[0].tolist()
                except:
                    # fallback if structure different
                    xy = []
                try:
                    conf = float(b.conf[0])
                except:
                    conf = 0.0
                try:
                    cls_id = int(b.cls[0])
                    label = model.names.get(cls_id, str(cls_id)) if hasattr(model, "names") else str(cls_id)
                except:
                    label = "unknown"

                defects_list.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": xy
                })
        except Exception:
            defects_list = []

        max_conf = max([d["confidence"] for d in defects_list], default=0)
        if max_conf > 0:
            sum_conf += max_conf
            count_conf += 1

        total_defects += defects_count

        metadata = {
            "file_name": base,
            "model": MODEL_PATH,
            "timestamp": datetime.now().isoformat(),
            "inference_ms": inference_ms,
            "defect_count": defects_count,
            "defects": defects_list,
        }

        # Save JSON + PDF
        try:
            save_json(json_path, metadata)
        except:
            pass
        try:
            create_pdf(pdf_path, orig_path, pred_path, metadata)
        except:
            pass

        rows.append([base, round(max_conf, 3), defects_count, "DEFECT" if defects_count else "OK", "View"])

        files_state.append({
            "orig_path": orig_path,
            "annot_path": pred_path,
            "conf": max_conf,
            "defects": defects_count,
            "time_ms": inference_ms,
            "defects_list": defects_list,
            "json_path": json_path,
            "pdf_path": pdf_path
        })

        progress((idx + 1) / total_files, desc=f"Processed {idx+1}/{total_files}")

    # ZIP creation (include all files in processed folder)
    if os.path.exists(ZIP_NAME):
        os.remove(ZIP_NAME)

    with zipfile.ZipFile(ZIP_NAME, "w") as zf:
        for file in os.listdir(PROCESSED_FOLDER):
            zf.write(os.path.join(PROCESSED_FOLDER, file), file)

    df = pd.DataFrame(rows, columns=["File", "Confidence", "Defects", "Status", "Action"])
    avg_conf = round(sum_conf / count_conf, 3) if count_conf else "-"

    logs = f"Processed {total_files} images\nTotal defects: {total_defects}\nAverage confidence: {avg_conf}"

    return (
        gr.update(visible=True, value=df),
        files_state,
        logs,
        total_files,
        avg_conf,
        total_defects,
        gr.update(visible=True, value=ZIP_NAME)
    )

# -------------------------
# DETAIL VIEW (returns charts + table)
# -------------------------
def _make_charts_and_table(defects_list):
    """
    Returns:
      - plotly bar figure (confidence per defect)
      - plotly pie figure (defect type distribution)
      - table rows for DataFrame: [idx, label, confidence, bbox]
    """
    # Build table rows
    table_rows = []
    labels_for_bar = []
    confs_for_bar = []
    # If no defects, return empty structures
    if not defects_list:
        return None, None, []

    # Build bars: label#i to separate duplicates
    for i, d in enumerate(defects_list, start=1):
        lab = d.get("label", "unknown")
        conf = float(d.get("confidence", 0.0))
        bbox = d.get("bbox", [])
        labels_for_bar.append(f"{lab} #{i}")
        confs_for_bar.append(conf)
        table_rows.append([i, lab, round(conf, 3), bbox])

    # Bar chart (responsive)
    fig_bar = go.Figure(
        data=[go.Bar(x=labels_for_bar, y=confs_for_bar, text=[f"{c:.3f}" for c in confs_for_bar], textposition='auto')],
    )
    fig_bar.update_layout(title_text="Confidence per Defect", autosize=True, margin=dict(l=20, r=20, t=40, b=20),
                          yaxis=dict(range=[0, 1]))

    # Pie chart: count per label
    label_counts = Counter([d.get("label", "unknown") for d in defects_list])
    pie_labels = list(label_counts.keys())
    pie_values = list(label_counts.values())
    fig_pie = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values, textinfo='label+percent')])
    fig_pie.update_layout(title_text="Defect Type Distribution", autosize=True, margin=dict(l=20, r=20, t=40, b=20))

    return fig_bar, fig_pie, table_rows

def open_details(evt: gr.SelectData, files_state, *args):
    idx = evt.index[0]
    item = files_state[idx]

    # build charts and table from defects_list
    defects_list = item.get("defects_list", [])
    fig_bar, fig_pie, table_rows = _make_charts_and_table(defects_list)

    # If no charts, return empty placeholders (Gradio handles None)
    return (
        gr.update(visible=False),     # hide main
        gr.update(visible=True),      # show details
        item["orig_path"],            # orig image
        item["annot_path"],           # annotated image
        item["conf"],                 # top confidence
        item["defects"],              # defect count
        item["time_ms"],              # inference time
        fig_bar,                      # bar chart
        fig_pie,                      # pie chart
        table_rows,                   # defects table rows
        "Logs available in ZIP."      # logs text
    )

def go_back(*args):
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        None, None, "-", "-", "-", None, None, [], "Back"
    )

# -------------------------
# UI
# -------------------------
with gr.Blocks(fill_height=True) as demo:

    file_state = gr.State([])

    with gr.Column(visible=True) as page_main:
        gr.Markdown("""
        <h1 style="
            text-align:center;
            font-weight:800;
            background: linear-gradient(90deg, #4cc9f0, #4361ee, #3a0ca3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 42px;
            margin-top: 10px;
        ">
        PCB Defect Inspection Dashboard
        </h1>
        """)


        upload_box = gr.Files(file_count="multiple", file_types=["image"])
        process_btn = gr.Button("Process Images")

        with gr.Row():
            total_files_box = gr.Number(label="Total Files", interactive=False)
            avg_conf_box = gr.Number(label="Average Confidence", interactive=False)
            total_def_box = gr.Number(label="Total Defects", interactive=False)

        results_table = gr.DataFrame(
            headers=["File", "Confidence", "Defects", "Status", "Action"],
            visible=False, interactive=False
        )

        zip_out = gr.File(label="Download Full Report Package (.zip)", visible=False)

        with gr.Accordion("Logs", open=False):
            logs_text = gr.Textbox(lines=6, interactive=False)

    # ------------------ DETAILS PAGE ------------------
    with gr.Column(visible=False) as page_details:
        back_btn = gr.Button("Back")

        # Top: original and annotated images side-by-side
        with gr.Row():
            orig_image = gr.Image(label="Original")
            annot_image = gr.Image(label="Annotated")

        # Middle: metrics row
        with gr.Row():
            det_conf = gr.Number(label="Top Confidence", interactive=False)
            det_def = gr.Number(label="Defect Count", interactive=False)
            det_time = gr.Number(label="Inference (ms)", interactive=False)

        # Charts row - always visible, left and right
        with gr.Row():
            bar_plot = gr.Plot(label="Confidence Bar Chart")
            pie_plot = gr.Plot(label="Defect Type Pie Chart")

        # Defect details table (full width)
        defects_table = gr.DataFrame(headers=["#", "Label", "Confidence", "BBox"], interactive=False)

        det_logs_text = gr.Textbox(lines=6, interactive=False)

    # -------------------------
    # Events
    # -------------------------
    process_btn.click(
        process_images_layout,
        [upload_box],
        [
            results_table, file_state, logs_text,
            total_files_box, avg_conf_box, total_def_box,
            zip_out,
        ]
    )

    results_table.select(
        open_details,
        [file_state],
        [
            page_main, page_details,
            orig_image, annot_image,
            det_conf, det_def, det_time,
            bar_plot, pie_plot, defects_table,
            det_logs_text
        ]
    )

    back_btn.click(
        go_back,
        [],
        [
            page_main, page_details,
            orig_image, annot_image,
            det_conf, det_def, det_time,
            bar_plot, pie_plot, defects_table,
            det_logs_text
        ]
    )

demo.launch(theme=gr.themes.Soft(primary_hue="blue"))
