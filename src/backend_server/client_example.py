import requests
data = {'name_of_file': 'new_data_pos.xlsx', 'name_of_sheet': 'Где оценивать'}

response = requests.post('http://127.0.0.1:5002/report', json=data)