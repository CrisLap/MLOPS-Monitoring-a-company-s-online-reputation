import numpy as np
import json
import logging
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy.stats import entropy

logger = logging.getLogger(__name__)


def load_baseline(path="drift/baseline.json"):
    """
    Loads baseline statistics used for drift detection.
    Reads the baseline sentiment distribution and embedding
    reference from a JSON file.

    Raises:
        FileNotFoundError: If baseline file doesn't exist
        ValueError: If baseline file is invalid or missing required keys
    """
    baseline_path = Path(path)
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline file not found at {path}. "
            "Please create a baseline using data_drift_detection.baseline.save_baseline()"
        )

    try:
        with open(baseline_path) as f:
            baseline = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in baseline file {path}: {e}")

    # Validate baseline structure
    required_keys = ["sentiment_dist", "embedding_mean"]
    missing_keys = [key for key in required_keys if key not in baseline]
    if missing_keys:
        raise ValueError(
            f"Baseline file missing required keys: {missing_keys}. "
            f"Expected keys: {required_keys}"
        )

    return baseline


def detect_label_drift(current_dist, baseline_dist):
    """
    Computes label distribution drift against a baseline.
    Measures divergence between the current and baseline
    sentiment distributions.
    """
    return entropy(current_dist, baseline_dist)


def detect_embedding_drift(current_embeddings, baseline_embedding):
    """
    Computes embedding drift against a baseline representation.
    Compares the mean of current embeddings with the baseline
    embedding using cosine distance.
    """
    mean_embedding = np.mean(current_embeddings, axis=0)
    return cosine(mean_embedding, baseline_embedding)
