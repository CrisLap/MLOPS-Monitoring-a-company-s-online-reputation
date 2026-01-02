import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_readiness_endpoint():
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "fallback_available" in data

def test_predict_endpoint():
    # Test payload
    payload = {"text": "I love this product!"}
    response = client.post("/predict", json=payload)
    
    # Validazione base della risposta
    assert response.status_code == 200
    data = response.json()
    
    # Tutti i campi richiesti devono essere presenti
    assert "label" in data
    assert "confidence" in data
    assert "scores" in data
    
    # Valori coerenti
    assert isinstance(data["label"], str)
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["scores"], dict)
    # Controllo che almeno la label principale sia nel dict scores
    assert data["label"] in data["scores"]

def test_predict_fallback_values():
    """Verifica che il fallback sia conforme a SentimentResponse"""
    payload = {"text": "Fallback test"}
    response = client.post("/predict", json=payload)
    data = response.json()
    
    # Controllo valori di fallback (opzionale)
    assert data["label"] == "neutral"
    assert abs(data["confidence"] - 0.5) < 1e-6
    assert "neutral" in data["scores"]
    assert "positive" in data["scores"]
    assert "negative" in data["scores"]
