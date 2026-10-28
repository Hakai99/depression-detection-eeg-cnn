# ============================================================================
# app.py - Flask Application for Depression Detection EEG
# ============================================================================

import os
import sys
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import preprocessing module
from preprocess import EEGProcessor, validate_edf_file

# ============================================================================
# FLASK APP CONFIGURATION
# ============================================================================

app = Flask(__name__)

# Get the base directory (where app.py is located)
BASE_DIR = Path(__file__).parent.absolute()
logger.info(f"Base directory: {BASE_DIR}")

# Configuration
app.config['UPLOAD_FOLDER'] = BASE_DIR / 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'edf'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================================
# MODEL LOADING (FIX FOR YOUR ISSUE)
# ============================================================================

# Define model path correctly
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "best_model_fixed.keras"

from tensorflow.keras.models import load_model
model = load_model(str(MODEL_PATH), compile=False)
logger.info(f"\n{'='*60}")
logger.info("MODEL LOADING CONFIGURATION")
logger.info(f"{'='*60}")
logger.info(f"Model directory: {MODEL_DIR}")
logger.info(f"Expected model path: {MODEL_PATH}")
logger.info(f"Model exists: {MODEL_PATH.exists()}")

# Global variable to store model
model = None

def load_model():
    """
    Load the trained Keras model with proper error handling.
    """
    global model

    try:
        logger.info("\nAttempting to load model...")

        # Check if model file exists
        if not MODEL_PATH.exists():
            logger.error(f"❌ Model file not found at: {MODEL_PATH}")
            logger.error("Please ensure 'final_model_compatible.h5' is in the 'model' folder")
            return False

        # Import TensorFlow/Keras
        try:
            from tensorflow.keras.models import load_model as keras_load_model
        except ImportError:
            logger.error("❌ TensorFlow/Keras not installed.")
            return False

        # Load the model
        logger.info(f"Loading from: {MODEL_PATH}")
        model = keras_load_model(str(MODEL_PATH))

        logger.info("✓ Model loaded successfully!")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Input shape: {model.input_shape}")
        logger.info(f"Output shape: {model.output_shape}")

        return True

    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def make_prediction(eeg_windows):
    """
    Make prediction on preprocessed EEG windows.
    """
    if model is None:
        return {
            'error': 'Model not loaded',
            'status': 'error'
        }
    
    try:
        # Ensure correct shape
        if eeg_windows.ndim != 4:
            return {
                'error': f'Invalid window shape: {eeg_windows.shape}. Expected (num_windows, 1024, 19, 1)',
                'status': 'error'
            }
        
        if eeg_windows.shape[1:] != (19, 1024, 1):
            return {
                'error': f'Invalid dimensions. Got {eeg_windows.shape}, expected (*, 1024, 19, 1)',
                'status': 'error'
            }
        
        logger.info(f"Making predictions on {len(eeg_windows)} windows...")
        
        # Make predictions
        predictions = model.predict(eeg_windows, verbose=0)
        
        # Extract results
        healthy_prob = float(predictions[:, 0].mean())
        mdd_prob = float(predictions[:, 1].mean())
        
        # Determine final classification
        final_class = 1 if mdd_prob > healthy_prob else 0
        
        class_names = {
            0: "Healthy",
            1: "Major Depressive Disorder (MDD)"
        }
        
        # Calculate confidence
        max_prob = max(healthy_prob, mdd_prob)
        
        result = {
            'status': 'success',
            'final_class': int(final_class),
            'diagnosis': class_names[final_class],
            'healthy_probability': round(healthy_prob, 4),
            'mdd_probability': round(mdd_prob, 4),
            'confidence': round(max_prob, 4),
            'confidence_percent': f"{max_prob*100:.2f}%",
            'total_windows': len(eeg_windows),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"✓ Prediction complete:")
        logger.info(f"  Diagnosis: {result['diagnosis']}")
        logger.info(f"  Confidence: {result['confidence_percent']}")
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return {
            'error': str(e),
            'status': 'error'
        }
# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Handle EEG file upload and prediction.
    """
    if request.method == 'GET':
        return render_template('predict.html')
    
    if request.method == 'POST':
        # Check if model is loaded
        if model is None:
            return jsonify({
                'status': 'error',
                'error': 'Model not loaded. Please check server logs.'
            }), 500
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'error': 'No file selected'
            }), 400
        
        # Validate file
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'error': 'File must be in .edf format'
            }), 400
        
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
            filename = timestamp + filename
            filepath = app.config['UPLOAD_FOLDER'] / filename
            file.save(str(filepath))
            
            logger.info(f"\nFile uploaded: {filename}")
            
            # Validate EDF file
            is_valid, validation_msg = validate_edf_file(str(filepath))
            if not is_valid:
                return jsonify({
                    'status': 'error',
                    'error': f'Invalid EEG file: {validation_msg}'
                }), 400
            
            logger.info(f"✓ File validation passed")
            
            # Preprocess
            logger.info(f"Starting preprocessing...")
            processor = EEGProcessor()
            eeg_windows, sfreq = processor.process_file(str(filepath))
            
            logger.info(f"✓ Preprocessing complete. Shape: {eeg_windows.shape}")
            
            # Make prediction
            result = make_prediction(eeg_windows)
            
            # Add file info
            result['filename'] = filename
            result['filepath'] = str(filepath)
            
            # Clean up uploaded file after processing
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up temporary file")
            except:
                pass
            
            return jsonify(result), 200
        
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            return jsonify({
                'status': 'error',
                'error': f'Processing failed: {str(e)}'
            }), 500


@app.route('/result', methods=['GET', 'POST'])
def result():
    """Display prediction results."""
    if request.method == 'POST':
        # Receive AJAX request with results
        data = request.get_json()
        return render_template('result.html', result=data)
    
    return render_template('result.html')


@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    }), 200


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        'status': 'error',
        'error': 'File size exceeds 100MB limit'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 error."""
    return render_template('index.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 error."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'status': 'error',
        'error': 'Internal server error'
    }), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    logger.info(f"\n{'='*60}")
    logger.info("DEPRESSION DETECTION EEG - FLASK SERVER")
    logger.info(f"{'='*60}")
    
    # Load model at startup
    logger.info("\n>>> LOADING MODEL AT STARTUP <<<\n")
    if load_model():
        logger.info("✓ Model loaded successfully! Server ready.")
    else:
        logger.error("❌ Failed to load model. Check configuration above.")
    
    logger.info(f"{'='*60}\n")
    
    # Run Flask app
    logger.info("Starting Flask server on http://127.0.0.1:5000/")
    logger.info("Press CTRL+C to stop the server\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)