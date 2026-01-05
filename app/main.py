from fastapi import FastAPI, Response, status
from fastapi.responses import RedirectResponse
from app.schemas import SentimentRequest, SentimentResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from app.inference import latest_file, predict
from huggingface_hub import HfApi
from app.metrics import REQUEST_COUNT, REQUEST_LATENCY, SENTIMENT_COUNTER
from typing import Dict
import numpy as np
import time
import os


app = FastAPI(
    title="Online Reputation API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


@app.get("/", include_in_schema=False)
def root():
    """Redirect to the API documentation (/docs)."""
    return RedirectResponse(url="/docs")


REPO_ID = os.getenv("HF_MODEL_REPO", "CrisLap/sentiment-model")
TOKEN = os.getenv("HF_TOKEN")


@app.get("/model")
def model_info():
    """Return info about the latest sentiment model in the repository."""
    api = HfApi()
    try:
        # List all files in the repository
        files = api.list_repo_files(repo_id=REPO_ID, token=TOKEN)

        # Filter only sentiment_ft*.bin template files
        model_files = sorted(
            [f for f in files if f.startswith("sentiment_ft") and f.endswith(".bin")]
        )

        # Get the latest (most recent) file
        latest_file = model_files[-1] if model_files else None

        return {
            "model_loaded": latest_file is not None,
            "model_id": REPO_ID,
            "latest_model_file": latest_file,
        }

    except Exception as e:
        return {"model_loaded": False, "error": str(e)}


@app.get("/metrics")
def metrics():
    """
    Expose Prometheus metrics.
    """
    data = generate_latest()  # genera output in formato Prometheus
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# --- READINESS ENDPOINT ---
@app.get("/ready")
def readiness():
    """
    Check if the app is ready to receive requests.
    """
    if latest_file:
        return {"ready": True, "message": "Model is loaded"}
    else:
        return Response(
            content='{"ready": false, "message": "Model not loaded"}',
            media_type="application/json",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@app.get("/health")
def health():
    """Return the health status of the API."""
    return {"status": "ok"}


@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(req: SentimentRequest):
    """Predict sentiment for input text and update Prometheus metrics."""
    start = time.time()
    label, score = predict(req.text)
    REQUEST_LATENCY.observe(time.time() - start)
    REQUEST_COUNT.inc()
    SENTIMENT_COUNTER.labels(sentiment=label).inc()
    return SentimentResponse(label=label, score=score)
