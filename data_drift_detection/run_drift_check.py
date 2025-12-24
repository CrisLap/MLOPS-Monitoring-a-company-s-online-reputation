import mlflow
import numpy as np
from detector import load_baseline, detect_label_drift, detect_embedding_drift
from metrics import DRIFT_SCORE

LABEL_THRESHOLD = 0.15
EMBED_THRESHOLD = 0.2


def run_drift(sentiment_dist, embeddings):
    """
    Evaluates concept drift using sentiment distribution and embeddings.
    Computes label and embedding drift against stored baselines, logs metrics
    to MLflow, updates the global drift score, and returns True if combined
    drift exceeds predefined thresholds.
    """
    baseline = load_baseline()

    label_drift = detect_label_drift(sentiment_dist, baseline["sentiment_dist"])
    embed_drift = detect_embedding_drift(
        embeddings, np.array(baseline["embedding_mean"])
    )

    drift_score = label_drift + embed_drift
    DRIFT_SCORE.set(drift_score)

    with mlflow.start_run(run_name="drift_monitoring"):
        mlflow.log_metric("label_drift", label_drift)
        mlflow.log_metric("embedding_drift", embed_drift)
        mlflow.log_metric("combined_drift", drift_score)

    return drift_score > (LABEL_THRESHOLD + EMBED_THRESHOLD)
