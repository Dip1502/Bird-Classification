import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('bird_classification_model_1.h5')

# Apply quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the quantized model as TFLite format
with open('bird_classification_model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
