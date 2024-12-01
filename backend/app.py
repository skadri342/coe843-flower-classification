from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
from model import load_saved_model
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__, static_folder='static')
CORS(app, resources={
    r"/static/*": {"origins": "*"},
    r"/api_model/*": {"origins": "*"}
})

UPLOAD_FOLDER = 'static/uploads'
CARTOON_FOLDER = 'static/cartoon_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CARTOON_FOLDER'] = CARTOON_FOLDER

# Load the model when the app starts
model, class_names = load_saved_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    # Read image
    image = cv2.imread(image_path)
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize
    image_resized = cv2.resize(image, (224, 224))
    # Preprocess using ResNet50's preprocessing
    image_array = tf.keras.applications.resnet50.preprocess_input(
        image_resized.astype('float32')
    )
    # Expand dimensions
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def cartoonize_image(image_path, output_folder):
    """
    Cartoonizes a single image and saves it to the specified folder with a unique name.
    """
    # Load the original image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Unable to read image: {image_path}")
        return None

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Step 2: Apply median blur to reduce noise
    gray_blurred = cv2.medianBlur(gray, 5)
    # Step 3: Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(
        gray_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=9
    )
    # Step 4: Apply bilateral filter for smoothening while preserving edges
    color = cv2.bilateralFilter(img, d=9, sigmaColor=250, sigmaSpace=250)
    # Step 5: Combine edges with the smoothed color image
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate a unique output filename
    unique_filename = f"cartoon_{uuid.uuid4().hex[:8]}.jpg"
    output_path = os.path.join(output_folder, unique_filename)
    
    # Save the cartoonized image
    cv2.imwrite(output_path, cartoon)
    return output_path

@app.route('/api_model/cartoon_uploads/<filename>')
def serve_cartoon_image(filename):
    return send_from_directory(CARTOON_FOLDER, filename)

@app.route('/api_model/cartoonify', methods=['POST'])
def cartoonify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Cartoonify the image
            cartoon_path = cartoonize_image(filepath, app.config['CARTOON_FOLDER'])
            
            if cartoon_path is None:
                return jsonify({'error': 'Failed to cartoonify image'}), 500
            
            # Get the relative path from the static folder
            relative_path = os.path.relpath(cartoon_path, app.config['STATIC_FOLDER'])
            
            return jsonify({
                'cartoon_path': relative_path
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
        
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api_model/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess the image
            processed_image = preprocess_image(filepath)
            
            # Make prediction
            pred = model.predict(processed_image)
            predicted_class = class_names[np.argmax(pred)]
            confidence = float(np.max(pred)) * 100
            
            # Get all class probabilities
            class_probabilities = {}
            for class_name, probability in zip(class_names, pred[0]):
                class_probabilities[class_name] = float(probability * 100)
            
            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': class_probabilities,
                'filepath': filepath
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
        
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.config['STATIC_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CARTOON_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5001)