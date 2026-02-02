import requests

url = "http://127.0.0.1:8000/mask-aadhaar"

files = {"file": open(r"images\img1.jpeg", "rb")}

response = requests.post(url, files=files)

# Save the returned image
with open("masked_from_api.png", "wb") as f:
    f.write(response.content)

print("Masked image saved as masked_from_api.png")
