from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow as tf
import cv2


app = Flask(__name__)

# Load the trained model
model = load_model('src/classification_model/model_weights/model.h5')

# Class names (update this with your actual class names)
class_names = ['beeline', 'mts', 'tele2']

def preprocess_image(image):
    """Preprocess the image for model prediction."""
    image.setflags(write=True)
    img = cv2.resize(image, (224, 224))
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img_array = img.numpy()
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image'].read()

    np_img = np.frombuffer(image_file, np.uint8)

    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    img_array = preprocess_image(image)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    class_name = class_names[class_idx]
    
    print(f'Predicted class index: {class_idx}')
    print(f'Predicted class name: {class_name}')

    return jsonify({'class_index': float(class_idx), 'class_name': class_name})


if __name__ == '__main__':
    app.run(port=5001, debug=True)
