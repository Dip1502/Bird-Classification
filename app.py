import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained MobileNetV2 model
model = tf.keras.models.load_model("bird_classification_model_1.h5")

# Define class names (in the same order used during training)
class_names = ['ABBOTTS BABBLER', 'ABBOTTS BOOBY', 'ABYSSINIAN GROUND HORNBILL',
               'AFRICAN CROWNED CRANE', 'AFRICAN EMERALD CUCKOO', 'AFRICAN FIREFINCH',
               'AFRICAN OYSTER CATCHER', 'AFRICAN PIED HORNBILL', 'AFRICAN PYGMY GOOSE',
               'ALBATROSS', 'ALBERTS TOWHEE', 'ALEXANDRINE PARAKEET', 'ALPINE CHOUGH',
               'ALTAMIRA YELLOWTHROAT', 'AMERICAN AVOCET', 'AMERICAN BITTERN',
               'AMERICAN COOT', 'AMERICAN FLAMINGO', 'AMERICAN GOLDFINCH',
               'AMERICAN KESTREL']

# Prediction threshold
CONFIDENCE_THRESHOLD = 0.9

# Route: Home
@app.route('/')
def index():
    return render_template('index.html')

# Route: Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load and preprocess the image
        img = Image.open(file_path).resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Predict
        predictions = model.predict(img_array)
        confidence = np.max(predictions)
        predicted_index = np.argmax(predictions)

        if confidence >= CONFIDENCE_THRESHOLD:
            predicted_label = class_names[predicted_index]
        else:
            predicted_label = "This species does not belong to the database."

        return render_template('index.html', prediction=predicted_label, filename=filename)

    return redirect(url_for('index'))

# Run app
if __name__ == '__main__':
    app.run(debug=True)

