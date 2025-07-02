import requests

response = requests.post(
    "http://192.168.2.53:7000/analytics",
    json={
        "query": "TR Vibrations",
        "helicopter_type": "",
        "flight_hours": "",
        "event_type": "",
        "status": "",
        "raised_by": ""
    }
)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())

