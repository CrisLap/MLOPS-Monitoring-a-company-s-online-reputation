from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict():
    """Test that the /predict endpoint returns a valid response with a label."""
    r = client.post("/predict", json={"text": "I love this product"})
    assert r.status_code == 200
    assert "label" in r.json()
