import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.metrics import REQUEST_COUNT, SENTIMENT_COUNTER

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_metrics():
    # Pulizia delle metriche prima di ogni test
    REQUEST_COUNT._value.set(0)
    for label in SENTIMENT_COUNTER._metrics:
        SENTIMENT_COUNTER.labels(label).set(0)
    yield
    # Pulizia post-test (opzionale)
    REQUEST_COUNT._value.set(0)
    for label in SENTIMENT_COUNTER._metrics:
        SENTIMENT_COUNTER.labels(label).set(0)


def test_metrics_endpoint_and_predict_updates_counters():
    payload = {"text": "I love this product!"}

    # Controllo iniziale metrics
    assert REQUEST_COUNT._value.get() == 0

    # Chiamata predict
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Validazione struttura risposta
    assert "label" in data
    assert "confidence" in data
    assert "scores" in data

    # Verifica aggiornamento metrics
    assert REQUEST_COUNT._value.get() == 1
    # Controllo che il counter per la label incrementata
    label = data["label"]
    assert SENTIMENT_COUNTER.labels(label)._value.get() == 1


def test_multiple_predict_requests():
    payloads = [
        {"text": "I love this!"},
        {"text": "This is bad."},
        {"text": "Neutral text."},
    ]

    for payload in payloads:
        client.post("/predict", json=payload)

    # REQUEST_COUNT deve essere 3
    assert REQUEST_COUNT._value.get() == 3

    # SENTIMENT_COUNTER deve avere almeno 1 per ciascun label presente
    metrics_values = {
        label: SENTIMENT_COUNTER.labels(label)._value.get()
        for label in ["positive", "neutral", "negative"]
    }
    assert sum(metrics_values.values()) == 3
