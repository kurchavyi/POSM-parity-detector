import os
from flask import Flask, request, jsonify
from ultralytics import YOLO

# Initialize Flask application
app = Flask(__name__)

# Set base directory relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define model path
model_path = os.path.join(BASE_DIR, 'model_data', 'best_2.pt')  # Update this path to your model's location

# Load the fine-tuned YOLOv8 model
model = YOLO(model_path)

# Define a route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input image path from the request
    image_path = request.json['image_path']
    
    # Check if the image exists
    if not os.path.isfile(image_path):
        return jsonify({'error': 'Image file does not exist'}), 400

    # Perform segmentation
    results = model.predict(source=image_path)

    # Return the prediction results
    return jsonify(results[0].tojson())

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
