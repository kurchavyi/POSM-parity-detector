import requests
import json

def send_image_path_to_server(image_path):
    url = 'http://127.0.0.1:5001/predict'  # URL of the server endpoint
    data = {'image_path': image_path}
    
    headers = {'Content-Type': 'application/json'}
    
    # Send the POST request to the server
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        class_index = result.get('class_index')
        class_name = result.get('class_name')
        print(f'Predicted class index: {class_index}')
        print(f'Predicted class name: {class_name}')
    else:
        print(f'Error: {response.status_code}, {response.text}')

if __name__ == '__main__':
    # Replace this with the actual path to your image
    image_path = 'image_path'
    send_image_path_to_server(image_path)
