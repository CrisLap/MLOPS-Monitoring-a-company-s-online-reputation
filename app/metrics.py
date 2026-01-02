from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("request_count", "Number of requests")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency")
SENTIMENT_COUNTER = Counter("sentiment_count", "Sentiment label counts", ["label"])
