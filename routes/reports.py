from flask import Blueprint, render_template, send_file
from services.reports import generate_zip_report
from services.inference import get_all_results

reports_bp = Blueprint('reports', __name__)

@reports_bp.route('/')
def index():
    results = get_all_results()
    # CHANGE: passing 'results' list instead of just 'count'
    return render_template('reports.html', results=results, count=len(results)) 

@reports_bp.route('/download')
def download():
    zip_buffer = generate_zip_report()
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='pcb_defects_report.zip'
    )