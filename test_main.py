from fastapi.testclient import TestClient
from fast_api_app import app


client = TestClient(app)


def test_predict2_positive():
    response = client.post("/predict2/",
                           json={"text": "I like machine learning!"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'POSITIVE'


def test_predict2_negative():
    response = client.post("/predict2/",
                           json={"text": "I hate machine learning!"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'NEGATIVE'

def test_predict_correct_answer():
    response = client.post("/predict/",
                           json={"text": "I hate machine learning!"})
    assert response.status_code == 200

def test_predict_not_empty_answer():
    response = client.post("/predict/",
                           json={"text": "I hate machine learning!"})
    json_data = response.json()
    assert response.status_code == 200
    assert (len(json_data)) >= 1
    
