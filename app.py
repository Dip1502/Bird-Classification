from flask import Flask, request, render_template, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import os
import uuid

app = Flask(__name__)

# Set folder to temporarily store uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
MODEL_PATH = 'model/bird_classification_model_1.h5'
model = load_model(MODEL_PATH)

# Your 20 bird species
class_names = [
    'ABBOTTS BABBLER', 'ABBOTTS BOOBY', 'ABYSSINIAN GROUND HORNBILL',
    'AFRICAN CROWNED CRANE', 'AFRICAN EMERALD CUCKOO', 'AFRICAN FIREFINCH',
    'AFRICAN OYSTER CATCHER', 'AFRICAN PIED HORNBILL', 'AFRICAN PYGMY GOOSE',
    'ALBATROSS', 'ALBERTS TOWHEE', 'ALEXANDRINE PARAKEET', 'ALPINE CHOUGH',
    'ALTAMIRA YELLOWTHROAT', 'AMERICAN AVOCET', 'AMERICAN BITTERN',
    'AMERICAN COOT', 'AMERICAN FLAMINGO', 'AMERICAN GOLDFINCH',
    'AMERICAN KESTREL'
]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message='No file uploaded.')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='No image selected.')

    try:
        # Save uploaded image
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        image = Image.open(filepath).convert('RGB')
        image = image.resize((256, 256))
        image_array = np.array(image)
        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)

        # Make prediction
        predictions = model.predict(image_array)
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        if confidence < 0.9:
            result = "This species does not belong to the database."
        else:
            result = class_names[predicted_index]

        image_url = f"/{filepath.replace(os.sep, '/')}"
        return render_template('index.html', prediction=result, confidence=round(confidence * 100, 2), image_url=image_url)

    except Exception as e:
        return render_template('index.html', message=f"Error: {str(e)}")

# Serve uploaded images
@app.route('/static/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
