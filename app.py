from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import uuid

app = Flask(__name__)

# Set folder to temporarily store uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="bird_classification_model_quantized.tflite")
interpreter.allocate_tensors()

# Define class names (20 species)
class_names = [
    'ABBOTTS BABBLER', 'ABBOTTS BOOBY', 'ABYSSINIAN GROUND HORNBILL',
    'AFRICAN CROWNED CRANE', 'AFRICAN EMERALD CUCKOO', 'AFRICAN FIREFINCH',
    'AFRICAN OYSTER CATCHER', 'AFRICAN PIED HORNBILL', 'AFRICAN PYGMY GOOSE',
    'ALBATROSS', 'ALBERTS TOWHEE', 'ALEXANDRINE PARAKEET', 'ALPINE CHOUGH',
    'ALTAMIRA YELLOWTHROAT', 'AMERICAN AVOCET', 'AMERICAN BITTERN',
    'AMERICAN COOT', 'AMERICAN FLAMINGO', 'AMERICAN GOLDFINCH',
    'AMERICAN KESTREL'
]


# Function to preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 256))  # MobileNetV2 input size
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.array(image_array) / 255.0  # Normalize to [0,1]
    return image_array


# Route to render the upload page
@app.route('/')
def index():
    return render_template('index.html')


# Route to handle the image upload and prediction
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

        # Preprocess the image and make predictions
        image_array = preprocess_image(filepath)

        # Get input and output details from TFLite model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], image_array)

        # Run inference
        interpreter.invoke()

        # Get the result
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = np.argmax(output_data[0])
        confidence = float(np.max(output_data[0]))

        # Check if the confidence is above a threshold
        if confidence < 0.9:
            result = "This species does not belong to the database."
        else:
            result = class_names[predicted_index]

        image_url = f"/{filepath.replace(os.sep, '/')}"
        return render_template('index.html', prediction=result, confidence=round(confidence * 100, 2),
                               image_url=image_url)

    except Exception as e:
        return render_template('index.html', message=f"Error: {str(e)}")


# Serve uploaded images
@app.route('/static/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
