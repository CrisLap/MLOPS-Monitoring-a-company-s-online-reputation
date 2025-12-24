from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_prediction():
    """
    Tests the /predict endpoint with a valid text input.
    Verifies that the response contains a valid label, confidence score,
    and properly normalized scores dictionary.
    """
    response = client.post("/predict", json={"text": "I love this product"})
    assert response.status_code == 200
    data = response.json()

    assert "label" in data and data["label"] in {"negative", "neutral", "positive"}
    assert "confidence" in data and 0 <= data["confidence"] <= 1
    assert "scores" in data and set(data["scores"].keys()) == {
        "negative",
        "neutral",
        "positive",
    }
    assert abs(sum(data["scores"].values()) - 1.0) <= 1e-3


def test_empty_text():
    """
    Tests the /predict endpoint with an empty text input.
    Verifies that the endpoint returns a 422 Unprocessable Entity status.
    """
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
