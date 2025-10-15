
import os
from Semantic_Segmentation.src.app.app import create_app
from Semantic_Segmentation.src.utils.logging_setup import logger

if __name__ == '__main__':
    # Use environment variables for flexible deployment
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    app = create_app()
    logger.info(f"Starting Flask server on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=True)