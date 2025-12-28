import mlflow
import numpy as np
import logging
from detector import load_baseline, detect_label_drift, detect_embedding_drift
from metrics import DRIFT_SCORE

logger = logging.getLogger(__name__)

LABEL_THRESHOLD = 0.15
EMBED_THRESHOLD = 0.2


def run_drift(sentiment_dist, embeddings):
    """
    Evaluates concept drift using sentiment distribution and embeddings.
    Computes label and embedding drift against stored baselines, logs metrics
    to MLflow, updates the global drift score, and returns True if combined
    drift exceeds predefined thresholds.
    
    Args:
        sentiment_dist: Current sentiment distribution array
        embeddings: Current embeddings array
        
    Returns:
        bool: True if drift detected, False otherwise
        
    Raises:
        FileNotFoundError: If baseline file doesn't exist
        ValueError: If inputs are invalid or baseline is malformed
    """
    try:
        baseline = load_baseline()
    except FileNotFoundError as e:
        logger.error(f"Baseline not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid baseline: {e}")
        raise
    
    # Validate inputs
    if sentiment_dist is None or embeddings is None:
        raise ValueError("sentiment_dist and embeddings must not be None")
    
    try:
        sentiment_dist = np.asarray(sentiment_dist)
        embeddings = np.asarray(embeddings)
        
        if sentiment_dist.ndim != 1:
            raise ValueError(f"sentiment_dist must be 1D, got {sentiment_dist.ndim}D")
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got {embeddings.ndim}D")
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        raise ValueError(f"Invalid input format: {e}")

    try:
        label_drift = detect_label_drift(sentiment_dist, baseline["sentiment_dist"])
        embed_drift = detect_embedding_drift(
            embeddings, np.array(baseline["embedding_mean"])
        )

        drift_score = label_drift + embed_drift
        DRIFT_SCORE.set(drift_score)

        # Log to MLflow if available
        try:
            with mlflow.start_run(run_name="drift_monitoring"):
                mlflow.log_metric("label_drift", label_drift)
                mlflow.log_metric("embedding_drift", embed_drift)
                mlflow.log_metric("combined_drift", drift_score)
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

        return drift_score > (LABEL_THRESHOLD + EMBED_THRESHOLD)
    except Exception as e:
        logger.error(f"Drift detection failed: {e}", exc_info=True)
        raise
