import json
import numpy as np
import sqlalchemy as sa

import data_drift_detection.storage as storage


def test_load_from_mlflow(monkeypatch, tmp_path):
    """
    Tests loading sentiment and embedding data from MLflow.
    Uses a fake MLflow client to simulate artifact retrieval and verifies
    that the loaded data matches the expected NPZ contents.
    """
    # create an NPZ file to be returned by fake MlflowClient
    npz_path = tmp_path / "recent.npz"
    np.savez(
        npz_path,
        sentiment_dist=np.array([0.1, 0.7, 0.2]),
        embeddings=np.array([[1.0, 2.0, 3.0]]),
    )

    class FakeInfo:
        def __init__(self, run_id):
            self.run_id = run_id

    class FakeRun:
        def __init__(self, run_id):
            self.info = FakeInfo(run_id)

    class FakeArtifact:
        def __init__(self, path):
            self.path = path

    class FakeClient:
        def search_runs(self, *args, **kwargs):
            return [FakeRun("r1")]

        def list_artifacts(self, run_id, path=None):
            return [FakeArtifact("recent.npz")]

        def download_artifacts(self, run_id, path, dst):
            # return path to our npz
            return str(npz_path)

    monkeypatch.setattr(storage, "MlflowClient", FakeClient)

    data = storage.load_from_mlflow()
    assert data is not None
    s, e = data
    assert np.allclose(s, np.array([0.1, 0.7, 0.2]))
    assert e.shape[0] == 1


def test_load_from_db(tmp_path):
    """
    Tests loading sentiment and embedding data from a database.
    Creates a temporary SQLite database, inserts test data, and verifies
    that the loader correctly retrieves and parses the stored arrays.
    """
    # create sqlite file
    db_file = tmp_path / "test.db"
    uri = f"sqlite:///{db_file}"

    engine = sa.create_engine(uri)
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "CREATE TABLE drift_data (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " type TEXT NOT NULL, value TEXT NOT NULL, created_at "
                "TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
        )
        conn.execute(
            sa.text("INSERT INTO drift_data(type, value) VALUES(:t, :v)"),
            [
                {"t": "sentiment_dist", "v": json.dumps([0.2, 0.5, 0.3])},
                {"t": "embeddings", "v": json.dumps([[0.1, 0.2, 0.3]])},
            ],
        )

    # call loader with explicit uri
    data = storage.load_from_db(uri=uri)
    assert data is not None
    s, e = data
    assert (s == np.array([0.2, 0.5, 0.3])).all()
    assert e.shape[0] == 1
