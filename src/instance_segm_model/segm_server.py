import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model_weights', 'best.pt')
model = YOLO(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image'].read()

    # Convert the image file to a numpy array
    np_img = np.frombuffer(image_file, np.uint8)

        # Decode the image array to OpenCV format
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    # # Check if the image exists
    # if not os.path.isfile(image_path):
    #     return jsonify({'error': 'Image file does not exist'}), 400

    results = model.predict(source=image, conf=0.45)

    if not results[0]:
        return "[]"
    return results[0].tojson(), 200

if __name__ == '__main__':
    print("Starting the segmentation model server...")
    app.run(port=5000, debug=True)
