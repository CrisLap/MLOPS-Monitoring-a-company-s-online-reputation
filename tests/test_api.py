from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict():
    r = client.post("/predict", json={"text": "I love this product"})
    assert r.status_code == 200
    assert "label" in r.json()

