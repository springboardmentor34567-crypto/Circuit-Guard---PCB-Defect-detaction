import io
import csv
import json
import zipfile
from services.inference import get_all_results

def generate_zip_report():
    results = get_all_results()
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        # 1. Generate CSV
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(['Filename', 'Status', 'Defect_Type', 'Confidence', 'BBox'])
        
        for res in results:
            if not res['defects']:
                writer.writerow([res['filename'], 'Pass', '', '', ''])
            else:
                for d in res['defects']:
                    writer.writerow([
                        res['filename'], 
                        'Fail', 
                        d['label'], 
                        d['confidence'], 
                        str(d['bbox'])
                    ])
        
        zip_file.writestr('report_summary.csv', csv_buffer.getvalue())

        # 2. Generate Annotations (TXT/JSON)
        for res in results:
            txt_content = []
            for d in res['defects']:
                # YOLO format: class x_center y_center width height (normalized)
                # Mocking simple format for now
                line = f"{d['class_id']} {d['confidence']} {d['bbox']}"
                txt_content.append(line)
            
            if txt_content:
                zip_file.writestr(f"annotations/{res['filename']}.txt", "\n".join(txt_content))

    zip_buffer.seek(0)
    return zip_buffer