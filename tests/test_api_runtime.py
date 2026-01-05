import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.metrics import REQUEST_COUNT, SENTIMENT_COUNTER

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset all Prometheus metrics to zero before and after tests."""
    # Cleaning metrics before each test
    REQUEST_COUNT._value.set(0)
    for label in ["positive", "negative", "neutral"]:
        SENTIMENT_COUNTER.labels(label)._value.set(0)
    yield
    # Post-test cleaning (optional)
    REQUEST_COUNT._value.set(0)
    for label in ["positive", "negative", "neutral"]:
        SENTIMENT_COUNTER.labels(label)._value.set(0)


def test_metrics_endpoint_and_predict_updates_counters():
    """Test that /predict updates Prometheus counters correctly."""
    payload = {"text": "I love this product!"}

    # Initial metrics check
    assert REQUEST_COUNT._value.get() == 0

    # Call predict
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Response structure validation
    assert "label" in data
    assert "score" in data

    # Check metrics update
    assert REQUEST_COUNT._value.get() == 1
    # check that the counter for the incremented label
    label = data["label"]
    assert SENTIMENT_COUNTER.labels(label)._value.get() == 1


def test_multiple_predict_requests():
    """Test that multiple /predict requests update Prometheus counters correctly."""
    payloads = [
        {"text": "I love this!"},
        {"text": "This is bad."},
        {"text": "Neutral text."},
    ]

    for payload in payloads:
        client.post("/predict", json=payload)

    # REQUEST_COUNT must be 3
    assert REQUEST_COUNT._value.get() == 3

    # SENTIMENT_COUNTER must be at least 1 for each label
    metrics_values = {
        label: SENTIMENT_COUNTER.labels(label)._value.get()
        for label in ["positive", "neutral", "negative"]
    }
    assert sum(metrics_values.values()) == 3
