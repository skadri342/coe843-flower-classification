from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from model import load_saved_model
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    image_resized = cv2.resize(image, (180, 180))
    # Preprocess using ResNet50's preprocessing
    image_array = tf.keras.applications.resnet50.preprocess_input(
        image_resized.astype('float32')
    )
    # Expand dimensions
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/api/predict', methods=['POST'])
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
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5001)