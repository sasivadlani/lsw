import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Block':2, 'Month':9, 'Day':6,'Hour':10,'Minute':12})

print(r.json())