from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.metrics import REQUEST_COUNT, REQUEST_LATENCY, SENTIMENT_COUNTER
from app.schemas import SentimentRequest, SentimentResponse
from app import inference
import time, uuid, logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Sentiment API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/ready")
def readiness_check():
    return {"model_loaded": getattr(inference, "_model_loaded", False)}

@app.post("/predict", response_model=SentimentResponse)
def predict_endpoint(req: SentimentRequest):
    start = time.time()
    REQUEST_COUNT.inc()
    try:
        result = inference.predict(req.text)
        SENTIMENT_COUNTER.labels(result["label"]).inc()
        REQUEST_LATENCY.observe(time.time() - start)
        return result
    except Exception as e:
        REQUEST_LATENCY.observe(time.time() - start)
        raise HTTPException(status_code=500, detail=str(e))

