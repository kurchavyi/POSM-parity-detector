from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


app = Flask(__name__)

# Load the trained model
model = load_model('src/classification_model/model_weights/model.h5')

# Class names (update this with your actual class names)
class_names = ['beeline', 'mts', 'tele2']

def preprocess_image(image_path):
    """Preprocess the image for model prediction."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_path = data.get('image_path')
    if not image_path:
        return jsonify({'error': 'No image_path provided'}), 400

    try:
        # Preprocess the image
        img_array = preprocess_image(image_path)

        # Make prediction
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        class_name = class_names[class_idx]
        
        print(f'Predicted class index: {class_idx}')
        print(f'Predicted class name: {class_name}')

        # Return the predicted class
        return jsonify({'class_index': float(class_idx), 'class_name': class_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
