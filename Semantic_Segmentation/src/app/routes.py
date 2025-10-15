import os
from flask import Blueprint, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from src.app.prediction_service import PredictionService
from src.app.utils import image_to_base64 

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize Blueprint
main = Blueprint('main', __name__)

# Initialize the Prediction Service. This is where the model loading occurs.
# It is essential for the service to be initialized once at app startup (on first import).
prediction_service = PredictionService()

def allowed_file(filename):
    """Checks if the uploaded file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    """Handles image upload and calls the prediction service."""
    if 'file' not in request.files:
        flash('No file part in the request.')
        return redirect(url_for('main.index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file.')
        return redirect(url_for('main.index'))
    
    if file and allowed_file(file.filename):
        try:
            # 1. Save the file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # 2. Get prediction from the service class
            original_img, mask_img = prediction_service.predict(filepath)
            
            # 3. Clean up the temporary file
            os.remove(filepath)
            
            if original_img and mask_img:
                # 4. Convert images to base64 for web display
                original_b64 = image_to_base64(original_img)
                mask_b64 = image_to_base64(mask_img)
                
                return render_template(
                    'index.html', 
                    original_img_b64=original_b64, 
                    mask_img_b64=mask_b64,
                    prediction_success=True
                )
            else:
                flash('Prediction failed. Please check the model checkpoint path and server logs.')
                return redirect(url_for('main.index'))
        
        except Exception as e:
            flash(f'An unexpected error occurred during processing: {e}')
            return redirect(url_for('main.index'))
            
    else:
        flash('Invalid file type. Only PNG, JPG, JPEG are allowed.')
        return redirect(url_for('main.index'))