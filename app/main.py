from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from .metrics import REQUEST_COUNT, REQUEST_LATENCY, SENTIMENT_COUNTER
import time
import logging
import uuid
from typing import Dict, Any

from .schemas import SentimentRequest, SentimentResponse
from . import inference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis with drift monitoring",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests for tracing."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={"request_id": request_id, "path": request.url.path},
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "message": "An unexpected error occurred",
        },
    )


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for load balancers and monitoring.
    Returns 200 if the service is running.
    """
    return {"status": "healthy", "service": "sentiment-analysis-api"}


@app.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint for Kubernetes and orchestration.
    Returns 200 if the service is ready to accept traffic.
    Note: Service is ready even without model (uses fallback predictor).
    """
    # Service is always ready (fallback predictor available)
    # Model loaded status is informational
    try:
        model_loaded = getattr(inference, "_model_loaded", False)
    except Exception:
        model_loaded = False

    return {
        "status": "ready",
        "service": "sentiment-analysis-api",
        "model_loaded": model_loaded,
        "fallback_available": True,
    }


@app.get("/")
async def root_status() -> Dict[str, Any]:
    """
    Root endpoint that returns model/training status.
    """
    try:
        model_loaded = getattr(inference, "_model_loaded", False)
    except Exception:
        model_loaded = False

    if model_loaded:
        return {"message": "App ready. The model is ready for predictions."}
    else:
        return {"message": "Model in training. Please try again later."}



@app.post("/predict", response_model=SentimentResponse)
def predict(req: SentimentRequest):
    """
    Exposes a prediction endpoint for sentiment inference.
    Accepts a text payload, runs sentiment prediction, updates request
    and sentiment metrics, and returns the prediction result.
    """
    start = time.time()
    REQUEST_COUNT.inc()

    try:
        result = inference.predict(req.text)
        SENTIMENT_COUNTER.labels(result["label"]).inc()
        REQUEST_LATENCY.observe(time.time() - start)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        REQUEST_LATENCY.observe(time.time() - start)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
