import requests

url = "http://127.0.0.1:8000/mask-aadhaar"

files = {"file": open(r"images\img2.jpeg", "rb")}

response = requests.post(url, files=files)

print(response.json())
