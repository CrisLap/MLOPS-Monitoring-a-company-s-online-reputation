from fastapi import FastAPI
from app.schemas import SentimentRequest, SentimentResponse
from app.inference import predict
from app.metrics import REQUEST_COUNT, LATENCY

import time

app = FastAPI(title="Online Reputation API")

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(req: SentimentRequest):
    start = time.time()
    label, score = predict(req.text)
    LATENCY.observe(time.time() - start)
    REQUEST_COUNT.inc()
    return SentimentResponse(label=label, score=score)

@app.get("/health")
def health():
    return {"status": "ok"}
