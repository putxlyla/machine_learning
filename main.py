from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import base64
import io

app = Flask(__name__)

model = load_model('my_model.h5')


def preprocess_image(image_data):
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    image = image.convert('L')

    image = image.resize((28, 28))

    image_array = np.array(image) / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


@app.route('/', methods=['GET'])
def index():
    return "Welcome to the Digit Recognition API!"


@app.route('/detect_number', methods=['POST'])
def detect_number():
    data = request.json
    image_data = data['image']

    try:

        processed_image = preprocess_image(image_data)

        prediction = model.predict(processed_image)

        predicted_digit = np.argmax(prediction)

        return jsonify({'predicted_digit': str(predicted_digit)})

    except Exception as e:
        return jsonify({'error': 'Failed to process the image.', 'details': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
