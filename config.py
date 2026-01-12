import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev_key_very_secret'
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    
    # CHANGE THIS LINE: Increase limit to 100MB
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    CONF_THRESHOLD_LOW = 0.25
    CONF_THRESHOLD_HIGH = 1.0

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)