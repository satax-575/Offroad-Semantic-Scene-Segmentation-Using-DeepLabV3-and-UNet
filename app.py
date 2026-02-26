"""
Web Interface for Segmentation Model
Upload images and see predictions in real-time!
"""

from flask import Flask, render_template, request, jsonify, send_file
import torch
from predict import predict, load_image
from config import NUM_CLASSES, DEVICE
from PIL import Image
import numpy as np
import io
import base64
from pathlib import Path
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder
UPLOAD_FOLDER = Path('static/uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)

# Color palette for visualization
COLOR_PALETTE = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)

CLASS_NAMES = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Landscape', 'Sky']


def colorize_mask(mask):
    """Convert class mask to RGB image"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        color_mask[mask == class_id] = COLOR_PALETTE[class_id]
    return color_mask


def image_to_base64(img_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = file.filename
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)
        
        print(f"Processing: {filename}")
        
        # Load original image for display
        original = Image.open(filepath).convert('RGB')
        original_size = original.size  # (width, height)
        original_array = np.array(original)
        
        # Make prediction (returns 320x320)
        prediction = predict(str(filepath), None)
        
        # Resize prediction back to original size
        prediction_resized = Image.fromarray(prediction.astype(np.uint8))
        prediction_resized = prediction_resized.resize(original_size, Image.NEAREST)
        prediction_resized = np.array(prediction_resized)
        
        # Colorize prediction
        colored_pred = colorize_mask(prediction_resized)
        
        # Create overlay (50% original + 50% prediction)
        overlay = (0.5 * original_array + 0.5 * colored_pred).astype(np.uint8)
        
        # Convert to base64
        original_b64 = image_to_base64(original_array)
        prediction_b64 = image_to_base64(colored_pred)
        overlay_b64 = image_to_base64(overlay)
        
        # Calculate class distribution
        unique, counts = np.unique(prediction_resized, return_counts=True)
        total_pixels = prediction_resized.size
        class_dist = {}
        for cls, count in zip(unique, counts):
            if cls < len(CLASS_NAMES):
                percentage = (count / total_pixels) * 100
                class_dist[CLASS_NAMES[cls]] = f"{percentage:.2f}%"
        
        return jsonify({
            'success': True,
            'original': original_b64,
            'prediction': prediction_b64,
            'overlay': overlay_b64,
            'class_distribution': class_dist
        })
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'device': str(DEVICE)})


if __name__ == '__main__':
    print("="*60)
    print("🚀 SEGMENTATION MODEL WEB INTERFACE")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Models loaded: UNet + DeepLabV3")
    print(f"\n📱 Open in browser: http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
