import requests

response = requests.post(
    "http://localhost:8000/rectify",
    json={"query": "TR Vibrations"}
)

print(response.status_code)
print(response.json())
