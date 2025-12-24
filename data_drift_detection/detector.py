import numpy as np
import json
from scipy.spatial.distance import cosine
from scipy.stats import entropy


def load_baseline(path="drift/baseline.json"):
    """
    Loads baseline statistics used for drift detection.
    Reads the baseline sentiment distribution and embedding
    reference from a JSON file.
    """
    with open(path) as f:
        return json.load(f)


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
