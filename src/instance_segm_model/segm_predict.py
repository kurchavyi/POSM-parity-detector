import os
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join('model_weights', 'best.pt')
image_path = os.path.join('test_dataset', 'image_name')

model = YOLO(model_path)

results = model.predict(source=image_path, save=True)