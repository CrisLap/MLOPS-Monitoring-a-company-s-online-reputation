from fastapi import FastAPI
from prometheus_client import make_asgi_app
from .metrics import REQUEST_COUNT, REQUEST_LATENCY, SENTIMENT_COUNTER
import time

from .schemas import SentimentRequest, SentimentResponse
from . import inference

app = FastAPI(title="Sentiment Analysis API")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.post("/predict", response_model=SentimentResponse)
def predict(req: SentimentRequest):
    """
    Exposes a prediction endpoint for sentiment inference.
    Accepts a text payload, runs sentiment prediction, updates request
    and sentiment metrics, and returns the prediction result.
    """
    start = time.time()
    REQUEST_COUNT.inc()

    result = inference.predict(req.text)
    SENTIMENT_COUNTER.labels(result["label"]).inc()

    REQUEST_LATENCY.observe(time.time() - start)
    return result
