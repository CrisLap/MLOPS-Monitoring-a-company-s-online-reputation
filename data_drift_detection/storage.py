import os
import shutil
import tempfile
import logging
import json
import numpy as np
import sqlalchemy as sa

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

                    # Cleanup temp dir safely
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)

                    return result
    finally:
        # Fallback cleanup
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    return None


def load_from_db(uri="postgresql://test_user:test_pass@localhost:5432/test_db"):
    """
    Load the most recent sentiment_dist and embeddings from a database table.
    Returns (sentiment_dist, embeddings) as numpy arrays or None if not found.
    Compatible with SQLite and Postgres.
    """
    engine = sa.create_engine(uri)
    with engine.begin() as conn:
        result = conn.execute(
            sa.text("SELECT type, value FROM drift_data ORDER BY created_at DESC")
        ).fetchall()

    data_dict = {}
    for t, v in result:
        data_dict[t] = np.array(json.loads(v))

    if "sentiment_dist" in data_dict and "embeddings" in data_dict:
        return data_dict["sentiment_dist"], data_dict["embeddings"]
    else:
        return None
