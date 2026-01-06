import requests

response = requests.post(
    "http://192.168.2.53:8000/rectify",
    json={
        "query": "TR Vibrations",
        "helicopter_type": "",
        "flight_hours": {
            "lower": 0,
            "upper": 2000
            },
        "event_type": "",
        "status": "",
        "raised_by": ""
    }
)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())

