from prometheus_client import Gauge

DRIFT_SCORE = Gauge("sentiment_drift_score", "Combined sentiment drift score")
