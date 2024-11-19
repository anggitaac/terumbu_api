from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
model = tf.keras.models.load_model('karang.keras')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data)).resize((150, 150))
        image = np.array(image) / 255.0
        prediction = model.predict(np.expand_dims(image, axis=0))
        label = ["Bleaching", "Dead", "Healthy"][np.argmax(prediction)]
        return jsonify({'prediction': label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Handler function
def handler(event, context):
    from flask_lambda import FlaskLambda
    lambda_app = FlaskLambda(app)
    return lambda_app(event, context)
