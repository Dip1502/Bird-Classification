import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained MobileNetV2 model
MODEL_PATH = "bird_classification_model_1.h5"  # Ensure this file is in your project folder
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names (Ensure they match your dataset classes)
class_names = [
    'ABBOTTS BABBLER', 'ABBOTTS BOOBY', 'ABYSSINIAN GROUND HORNBILL',
    'AFRICAN CROWNED CRANE', 'AFRICAN EMERALD CUCKOO', 'AFRICAN FIREFINCH',
    'AFRICAN OYSTER CATCHER', 'AFRICAN PIED HORNBILL', 'AFRICAN PYGMY GOOSE',
    'ALBATROSS', 'ALBERTS TOWHEE', 'ALEXANDRINE PARAKEET', 'ALPINE CHOUGH',
    'ALTAMIRA YELLOWTHROAT', 'AMERICAN AVOCET', 'AMERICAN BITTERN',
    'AMERICAN COOT', 'AMERICAN FLAMINGO', 'AMERICAN GOLDFINCH',
    'AMERICAN KESTREL'
]

# Confidence threshold for classification
CONFIDENCE_THRESHOLD = 0.9

# Health check route
@app.route('/')
def health_check():
    return jsonify({"status": "API is running"})

# Route: Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Load and preprocess the image
            img = Image.open(file_path).resize((256, 256))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            # Predict
            predictions = model.predict(img_array)
            confidence = np.max(predictions)
            predicted_index = np.argmax(predictions)

            if confidence >= CONFIDENCE_THRESHOLD:
                predicted_label = class_names[predicted_index]
            else:
                predicted_label = "This species does not belong to the database."

            return jsonify({"result": predicted_label, "confidence": float(confidence)})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file upload"}), 400

# Run app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)