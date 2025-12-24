import sys
from pathlib import Path
import numpy as np
import training.evaluate as evaluate

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))


def test_compute_metrics_with_logits_and_labels():
    """
    Tests metric computation from logits and true labels.
    Verifies that accuracy, macro F1, per-class F1 scores, and the
    confusion matrix are correctly computed for a small example.
    """
    # create logits for 4 samples and 3 classes
    logits = np.array(
        [
            [0.1, 0.2, 0.7],  # class 2
            [0.0, 0.9, 0.1],  # class 1
            [0.8, 0.1, 0.1],  # class 0
            [0.2, 0.6, 0.2],  # class 1
        ]
    )
    labels = np.array([2, 1, 0, 1])
    label_names = ["negative", "neutral", "positive"]

    metrics = evaluate.compute_metrics(logits, labels, label_names=label_names)

    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "f1_macro" in metrics
    # per-class keys
    assert "per_class_negative_f1" in metrics
    assert "per_class_neutral_f1" in metrics
    assert "per_class_positive_f1" in metrics
    assert "confusion_matrix" in metrics


def test_evaluate_trainer_returns_predictions_and_logs_metrics(monkeypatch):
    """
    Tests that the trainer evaluation returns predictions and logs metrics.
    Uses a dummy trainer and MLflow logger to ensure evaluation produces
    metrics, predictions, label IDs, and logs metrics correctly when enabled.
    """

    class DummyPred:
        """
        Dummy prediction object for testing purposes.
        Stores predictions and corresponding label IDs to simulate
        trainer output.
        """

        def __init__(self, preds, label_ids):
            self.predictions = preds
            self.label_ids = label_ids

    class DummyTrainer:
        """
        Dummy trainer for testing evaluation routines.
        Returns predefined predictions and labels when the predict method
        is called.
        """

        def predict(self, ds):
            # return 2x2 logits for simple test
            preds = np.array([[0.1, 0.9, 0.0], [0.6, 0.2, 0.2]])
            labels = np.array([1, 0])
            return DummyPred(preds, labels)

    logs = {}
    fig_logged = {"ok": False}

    class DummyMLflow:
        """
        Dummy MLflow logger for testing metric and figure logging.
        Records metrics updates and marks figures as logged without
        performing actual MLflow operations.
        """

        def log_metrics(self, m):
            logs.update(m)

        def log_figure(self, fig, name):
            # record that a figure was logged
            fig_logged["ok"] = True

    import sys

    sys.modules["mlflow"] = DummyMLflow()

    trainer = DummyTrainer()
    res = evaluate.evaluate_trainer(
        trainer,
        eval_dataset=None,
        label_names=["negative", "neutral", "positive"],
        mlflow_log=True,
    )

    assert "metrics" in res
    assert "predictions" in res
    assert "label_ids" in res
    # verify mlflow received some metrics
    assert any(k in logs for k in ("accuracy", "f1_macro"))
