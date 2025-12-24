import numpy as np
import json


def save_baseline(sentiments, embeddings, path="drift/baseline.json"):
    """
    Computes and saves baseline statistics for drift detection.
    Stores the mean sentiment distribution and embedding vector
    to a JSON file for future drift comparisons.
    """
    baseline = {
        "sentiment_dist": np.mean(sentiments, axis=0).tolist(),
        "embedding_mean": np.mean(embeddings, axis=0).tolist(),
    }
    with open(path, "w") as f:
        json.dump(baseline, f)
