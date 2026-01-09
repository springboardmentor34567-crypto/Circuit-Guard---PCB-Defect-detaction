from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

def create_pdf(pdf_path, original_img, annotated_img, metadata):
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>PCB DEFECT REPORT</b>", styles["Title"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(f"File: {metadata['file_name']}", styles["Normal"]))
    story.append(Paragraph(f"Inference Time: {metadata['inference_ms']} ms", styles["Normal"]))
    story.append(Paragraph(f"Defects: {metadata['defect_count']}", styles["Normal"]))

    story.append(Spacer(1, 10))
    story.append(RLImage(original_img, width=300, height=300))
    story.append(Spacer(1, 10))
    story.append(RLImage(annotated_img, width=300, height=300))

    table_data = [["Label", "Confidence", "BBox"]]
    for d in metadata["defects"]:
        table_data.append([d["label"], round(d["confidence"], 3), str(d["bbox"])])

    story.append(Table(table_data))
    doc.build(story)
