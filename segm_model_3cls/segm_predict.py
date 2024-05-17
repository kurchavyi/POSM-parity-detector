import os
from ultralytics import YOLO

# Set base directory relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define relative paths
model_path = os.path.join('model_data', 'best_2.pt')  # Update this path to your model's location
image_path = os.path.join('test_dataset', '943895_2517_1.jpg')  # Update this path to your image's location

# Load the fine-tuned YOLOv8 model
model = YOLO(model_path)

# Perform segmentation
results = model.predict(source=image_path, save=True)