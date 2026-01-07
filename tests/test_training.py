import sys
from pathlib import Path
import training.train as train_module
import types

# make repo root visible
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))


class DummyDataset:
    """
    Dummy dataset for testing purposes.
    Provides minimal training and validation samples and supports
    item access via attribute names
    """

    def __init__(self):
        self.train = [{"text": "Hello world", "label": 2}]
        self.validation = [{"text": "Okay", "label": 1}]
        self.test = [{"label": 2, "text": "good"}]

    def __getitem__(self, k):
        return getattr(self, k)


class DummyMLflowRun:
    """
    Dummy MLflow run context manager for testing.
    Simulates MLflow's run context without performing any actual logging
    or operations.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_full_train_flow_creates_model_and_logs(monkeypatch, tmp_path):
    """
    Tests the full training flow including model creation and artifact logging.
    Uses dummy dataset, fake fastText training, and a dummy MLflow context
    to verify that a model file is created and logged correctly.
    """
    # replace load_data
    monkeypatch.setattr(train_module, "load_data", lambda: DummyDataset())

    # fake fasttext
    def fake_train_supervised(input, **kwargs):
        class M:
            def __init__(self):
                # Add the attributes your train.py expects
                self.epoch = 5
                self.lr = 0.1
                self.wordNgrams = 1
                self.dim = 100

            def save_model(self, path):
                # Ensure the file is actually created so downstream
                # artifact logging doesn't fail
                with open(path, "w") as f:
                    f.write("FAKE MODEL")

            def test(self, path):
                # Returns (number of samples, precision, recall)
                # These values can be dummy numbers
                return (100, 0.85, 0.85)

        return M()

    monkeypatch.setattr(
        train_module,
        "fasttext",
        types.SimpleNamespace(train_supervised=fake_train_supervised),
    )

    # fake mlflow
    logged = {}

    def fake_start_run():
        return DummyMLflowRun()

    def fake_log_artifact(path):
        logged["artifact"] = path

    monkeypatch.setattr(
        train_module,
        "mlflow",
        types.SimpleNamespace(
            start_run=fake_start_run,
            log_artifact=fake_log_artifact,
            log_params=lambda *args, **kwargs: None,
            log_metrics=lambda *args, **kwargs: None,
        ),
    )

    # ensure models dir is unique for test
    out = train_module.train(epoch=1)

    assert Path(out).exists()
    assert Path(out).read_text() == "FAKE MODEL"
    assert "artifact" in logged and logged["artifact"] == str(out)
