import requests

def test_health():
    response = requests.get("http://localhost:5000/health")
    assert response.status_code == 200

def test_prediction():
    payload = {
        "features": [0.1]*30
    }
    response = requests.post(
        "http://localhost:5000/predict",
        json=payload
    )
    assert response.status_code == 200
200
