import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'SSC Percentage':80, 'HSC Percentage':90,
                            'Undergrad Percentage':60, 'MBA Percentage':60,
                            'Workex':0})

print(r.json())
