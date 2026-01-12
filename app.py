from flask import Flask, render_template
from config import Config
from models.model_loader import ModelLoader
# Import the new DB module
from database import init_db

# Initialize Blueprint imports
from routes.dashboard import dashboard_bp
from routes.inspection import inspection_bp
from routes.metrics import metrics_bp
from routes.reports import reports_bp
from routes.history import history_bp  # <--- NEW: Import History route

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize the Database (Creates pcb_history.db if missing)
    with app.app_context():
        init_db()

    # Register Blueprints
    app.register_blueprint(dashboard_bp) # Main Dashboard at /
    app.register_blueprint(inspection_bp, url_prefix='/inspection')
    app.register_blueprint(metrics_bp, url_prefix='/metrics')
    app.register_blueprint(reports_bp, url_prefix='/reports')
    app.register_blueprint(history_bp, url_prefix='/history') # <--- NEW: Register History

    # NEW: Route for Architecture Page 
    # (Kept simple inside app.py since it's just one static page)
    @app.route('/architecture')
    def architecture():
        return render_template('architecture.html')

    # Trigger Lazy Model Load
    ModelLoader.get_instance()

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)