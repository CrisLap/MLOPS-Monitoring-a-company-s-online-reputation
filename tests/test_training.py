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
    item access via attribute names.
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
            def save_model(self, path):
                Path(path).write_text("FAKE MODEL")

            def test(self, path):
            # Returns (number_of_examples, precision, recall)
                return (1, 0.8, 0.8)

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
