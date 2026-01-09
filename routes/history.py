from flask import Blueprint, render_template, jsonify, request
from database import get_all_scans, clear_all_scans # <--- Import the new function

history_bp = Blueprint('history', __name__)

@history_bp.route('/')
def index():
    scans = get_all_scans()
    return render_template('history.html', scans=scans)

# --- NEW ROUTE TO CLEAR DATABASE ---
@history_bp.route('/clear', methods=['POST'])
def clear():
    try:
        clear_all_scans()
        return jsonify({'success': True, 'message': 'History cleared!'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500