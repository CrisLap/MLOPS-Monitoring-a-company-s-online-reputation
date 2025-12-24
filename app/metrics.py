from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "sentiment_requests_total", "Total number of sentiment prediction requests"
)

REQUEST_LATENCY = Histogram(
    "sentiment_request_latency_seconds", "Latency of sentiment prediction requests"
)

SENTIMENT_COUNTER = Counter(
    "sentiment_predictions_total", "Number of predictions per sentiment", ["label"]
)
