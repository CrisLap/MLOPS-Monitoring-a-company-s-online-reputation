from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests"
)

LATENCY = Histogram(
    "api_latency_seconds", "API latency"
)

