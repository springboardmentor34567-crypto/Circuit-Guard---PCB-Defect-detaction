import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import tempfile

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


def generate_detailed_pdf(data, originals, annotated, base_dir):
    """
    Creates a detailed PDF report.
    One image = one page.
    Order:
    1. Title
    2. Original vs Annotated
    3. Defect table
    4. Confidence distribution graph
    5. Defect count graph
    """

    pdf_path = os.path.join(base_dir, "detailed_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)

    width, height = A4
    df = pd.DataFrame(data)

    # Temporary directory for graphs
    with tempfile.TemporaryDirectory() as tmpdir:

        for orig, ann in zip(originals, annotated):
            image_name = os.path.basename(orig)
            image_id = os.path.splitext(image_name)[0]

            # ================= TITLE =================
            c.setFont("Helvetica-Bold", 14)
            c.drawString(
                40, height - 40,
                f"Detailed Report of Image : {image_name}"
            )

            # ================= IMAGES =================
            img_y = height - 300
            img_w, img_h = 240, 200

            c.drawImage(orig, 40, img_y, width=img_w, height=img_h,
                        preserveAspectRatio=True)
            c.drawImage(ann, 320, img_y, width=img_w, height=img_h,
                        preserveAspectRatio=True)

            c.setFont("Helvetica", 9)
            c.drawString(110, img_y - 12, "Original Image")
            c.drawString(395, img_y - 12, "Annotated Image")

            # ================= TABLE =================
            table_y = img_y - 40
            c.setFont("Helvetica-Bold", 10)
            c.drawString(
                40, table_y,
                "Class    Conf    x1    y1    x2    y2"
            )

            table_y -= 15
            c.setFont("Helvetica", 9)

            rows = df[df["image"] == image_name]

            for _, r in rows.iterrows():
                c.drawString(
                    40, table_y,
                    f'{r["class"]}    {r["confidence"]}    '
                    f'{r["x1"]}    {r["y1"]}    {r["x2"]}    {r["y2"]}'
                )
                table_y -= 12
                if table_y < 420:
                    break

            # ================= CONFIDENCE GRAPH =================
            if not rows.empty:
                conf_graph = os.path.join(tmpdir, f"{image_id}_conf.png")

                plt.figure(figsize=(3, 2))
                rows.boxplot(column="confidence", by="class")
                plt.title("Confidence Distribution")
                plt.suptitle("")
                plt.tight_layout()
                plt.savefig(conf_graph)
                plt.close()

                c.setFont("Helvetica-Bold", 10)
                c.drawString(40, 300, "Confidence Distribution")
                c.drawImage(conf_graph, 40, 160, width=220, height=130)

            # ================= DEFECT COUNT GRAPH =================
            if not rows.empty:
                count_graph = os.path.join(tmpdir, f"{image_id}_count.png")

                plt.figure(figsize=(3, 2))
                rows["class"].value_counts().plot(kind="bar")
                plt.title("Defect Count by Class")
                plt.tight_layout()
                plt.savefig(count_graph)
                plt.close()

                c.setFont("Helvetica-Bold", 10)
                c.drawString(320, 300, "Defect Count by Class")
                c.drawImage(count_graph, 320, 160, width=220, height=130)

            # ================= NEW PAGE =================
            c.showPage()

    c.save()
    return pdf_path


def generate_separate_pdfs(data, originals, annotated, base_dir):
    """
    Generates one PDF per image.
    Returns: { image_name : pdf_path }
    """
    pdfs = {}

    for orig, ann in zip(originals, annotated):
        image_name = os.path.basename(orig)
        image_id = os.path.splitext(image_name)[0]

        single_dir = os.path.join(base_dir, image_id)
        os.makedirs(single_dir, exist_ok=True)

        pdf_path = generate_detailed_pdf(
            data, [orig], [ann], single_dir
        )
        pdfs[image_name] = pdf_path

    return pdfs


def generate_zip(base_dir):
    """
    Creates a ZIP containing:
    - Combined CSV / TXT / PDF
    - Separate CSV / TXT / PDF
    - Annotated images
    """

    zip_path = os.path.join(base_dir, "all_results.zip")

    with zipfile.ZipFile(zip_path, "w") as z:
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.endswith(".zip"):
                    continue
                full_path = os.path.join(root, f)
                z.write(
                    full_path,
                    arcname=os.path.relpath(full_path, base_dir)
                )

    return zip_path
