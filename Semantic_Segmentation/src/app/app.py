# app.py

import os
from flask import Flask
from src.app.routes import main as main_blueprint
from src.utils.logging_setup import logger

def create_app():
    """Application factory to create and configure the Flask app."""
    logger.info("Starting Flask application setup.")
    
    app = Flask(__name__)
    
    # Required for flashing messages in HTML templates
    app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key_for_dev')
    
    # Create the folder for temporary file uploads
    os.makedirs('uploads', exist_ok=True)
    
    # Register Blueprints
    # This automatically imports the module and triggers the PredictionService initialization
    app.register_blueprint(main_blueprint)

    logger.info("Flask application setup complete.")
    return app

if __name__ == '__main__':
    # Use environment variables for flexible deployment
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    app = create_app()
    logger.info(f"Starting Flask server on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=True)