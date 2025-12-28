import json
import logging
import numpy as np
import tempfile
import shutil
from pathlib import Path
import os

try:
    from mlflow.tracking import MlflowClient
except Exception:
    MlflowClient = None


def _load_npz_file(path):
    """
    Loads sentiment distribution and embeddings from a .npz file.
    Returns the arrays stored under 'sentiment_dist' and 'embeddings' keys.
    """
    with np.load(path) as d:
        sentiment_dist = d["sentiment_dist"]
        embeddings = d["embeddings"]
    return sentiment_dist, embeddings


def load_from_mlflow():
    """Try to load recent.npz or recent.json from recent MLflow runs artifacts.
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
                    # Clean up temp directory
                    import shutil
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    return result
                if a.path.endswith("recent.json") or a.path == "recent.json":
                    if temp_dir is None:
                        temp_dir = tempfile.mkdtemp()
                    fp = client.download_artifacts(run_id, a.path, temp_dir)
                    with open(fp) as f:
                        d = json.load(f)
                    result = np.array(d["sentiment_dist"]), np.array(d["embeddings"])
                    # Clean up temp directory
                    import shutil
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    return result
    finally:
        # Ensure cleanup even if an error occurs
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return None


def load_from_db(uri=None):
    """Load most recent `sentiment_dist` and `embeddings` from a DB table `drift_data`.
    If `uri` not provided, uses env `DRIFT_DB_URI` or Airflow conn `drift_db` if available.
    Returns (sentiment_dist, embeddings) as numpy arrays or None.
    """
    if uri is None:
        uri = os.environ.get("DRIFT_DB_URI")
        if not uri:
            try:
                from airflow.hooks.base import BaseHook

                conn = BaseHook.get_connection("drift_db")
                uri = conn.get_uri()
            except Exception as e:
                logging.info("No DB uri found via env or conn id: %s", e)
                return None

    try:
        import sqlalchemy as sa

        engine = sa.create_engine(uri)
        with engine.connect() as conn:
            s_row = conn.execute(
                sa.text(
                    "SELECT value FROM drift_data WHERE type='sentiment_dist' ORDER BY created_at DESC LIMIT 1"
                )
            ).fetchone()
            e_row = conn.execute(
                sa.text(
                    "SELECT value FROM drift_data WHERE type='embeddings' ORDER BY created_at DESC LIMIT 1"
                )
            ).fetchone()
            if s_row and e_row:
                s = np.array(json.loads(s_row[0]))
                e = np.array(json.loads(e_row[0]))
                return s, e
    except Exception as e:
        logging.info("DB fallback failed: %s", e)
    return None


def load_from_disk():
    """
    Loads the most recent sentiment and embedding data from disk.
    Tries to load from 'drift/recent.npz' first, then falls back to
    'drift/recent.json'. Returns None if no data is found.
    """
    p_npz = Path("drift/recent.npz")
    if p_npz.exists():
        return _load_npz_file(str(p_npz))

    p_json = Path("drift/recent.json")
    if p_json.exists():
        with open(p_json) as f:
            d = json.load(f)
        return np.array(d["sentiment_dist"]), np.array(d["embeddings"])

    return None


def load_from_storage():
    """
    Loads sentiment and embedding data from available storage.
    Attempts to load from MLflow first, then the database, and
    finally falls back to local disk if no other source is available.
    """
    # Try MLflow first
    m = None
    try:
        m = load_from_mlflow()
    except Exception as e:
        logging.info("MLflow loader error: %s", e)
    if m is not None:
        return m

    d = None
    try:
        d = load_from_db()
    except Exception as e:
        logging.info("DB loader error: %s", e)
    if d is not None:
        return d

    return load_from_disk()
