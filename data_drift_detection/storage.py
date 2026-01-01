import os
import tempfile
import logging
import numpy as np
import shutil

try:
    from mlflow.tracking import MlflowClient
except ImportError:
    MlflowClient = None

def _load_npz_file(path):
    """Load sentiment and embeddings from a .npz file."""
    data = np.load(path)
    return data["sentiment_dist"], data["embeddings"]

def load_from_mlflow():
    """
    Try to load recent.npz or recent.json from recent MLflow runs artifacts.
    Returns (sentiment_dist, embeddings) as numpy arrays or None if not found.
    """
    if MlflowClient is None:
        logging.info("Mlflow client not available")
        return None

    client = MlflowClient()
    temp_dir = None

    try:
        # Attempt to get recent runs sorted by start_time descending
        try:
            runs = client.search_runs(
                experiment_ids=None,
                filter_string="",
                max_results=100,
                order_by=["attributes.start_time desc"],
            )
        except Exception:
            runs = client.search_runs(
                experiment_ids=None, filter_string="", max_results=100
            )

        for r in runs:
            run_id = r.info.run_id
            try:
                arts = client.list_artifacts(run_id, path="drift")
            except Exception:
                arts = []

            for a in arts:
                if a.path.endswith("recent.npz") or a.path == "recent.npz":
                    if temp_dir is None:
                        temp_dir = tempfile.mkdtemp()
                    fp = client.download_artifacts(run_id, a.path, temp_dir)
                    result = _load_npz_file(fp)

                    # Clean up temp directory safely
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)

                    return result
    finally:
        # Fallback cleanup just in case
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    return None
