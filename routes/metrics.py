from flask import Blueprint, render_template
from services.analytics import get_confusion_matrix_data

metrics_bp = Blueprint('metrics', __name__)

@metrics_bp.route('/')
def index():
    cm_data = get_confusion_matrix_data()
    return render_template('metrics.html', cm=cm_data)