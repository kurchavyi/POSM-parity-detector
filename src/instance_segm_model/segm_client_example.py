import json
import numpy as np
import cv2
import requests

url = 'http://127.0.0.1:5000/predict'

path = r'C:\Users\vital\learning\projects\POSM-parity-detector\src\backend_server\images\7a6a3014-dfd8-416f-bbe5-6e3ac2f7f7b3.jpg'
image = cv2.imread(path)
_, image_encoded = cv2.imencode('.jpg', image)
image_bites = image_encoded.tobytes()

response = requests.post(url, files={'image': image_bites})

response_data = json.loads(response.text)
response_data = json.loads(response_data)

# class_areas = {}

# def calculate_area(x_coords, y_coords):
#     # Convert segment coordinates to a NumPy array of shape (N, 1, 2)
#     contour = np.array(list(zip(x_coords, y_coords)), dtype=np.float32).reshape((-1, 1, 2))
#     # Calculate the area using OpenCV
#     area = cv2.contourArea(contour)
#     return area

# for result in response_data:
#     class_name = result['name']
#     confidence = result['confidence']
#     x_coords = result['segments']['x']
#     y_coords = result['segments']['y']

#     # Calculate the area of the segmented object
#     pixel_area = calculate_area(x_coords, y_coords)

#     # Update class areas dictionary
#     if class_name in class_areas:
#         class_areas[class_name].append(pixel_area)
#     else:
#         class_areas[class_name] = [pixel_area]

# total_areas = {class_name: sum(areas) for class_name, areas in class_areas.items()}

# for class_name, total_area in total_areas.items():
#     print(f"Class: {class_name}, Total Area: {total_area}")
