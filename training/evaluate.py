"""Evaluation utilities for the training pipeline.

Functions provided:
- compute_metrics(preds, labels, label_names):
 compute accuracy, precision/recall/f1 (macro & per-class),
and confusion matrix.
- evaluate_trainer(trainer, eval_dataset, label_names=None,
 mlflow_log=True): run `trainer.predict` on `eval_dataset`,
 compute metrics, optionally log them to MLflow,
 and return a dict with results.

The functions avoid hard runtime dependencies where possible:
 sklearn and matplotlib are used if available; if not,
 basic metrics (accuracy, macro f1 via fallback) are computed.
"""

from typing import Dict, List, Optional, Any
import numpy as np


def _try_import_sklearn():
    """
    Attempts to import selected scikit-learn metrics functions.
    Returns accuracy_score, precision_recall_fscore_support,
    and confusion_matrix
    if available, otherwise returns None for each.
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix,
        )

        return accuracy_score, precision_recall_fscore_support, confusion_matrix
    except Exception:
        return None, None, None


def compute_metrics(
    preds: np.ndarray,
    labels: Optional[np.ndarray],
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute evaluation metrics from logits or predicted labels
    and ground-truth labels.
    preds: logits (N x C) or predicted label indices (N,)
    labels: ground-truth integer label ids (N,) or None
    label_names: list of label names to include in
    per-class metrics (optional)

    Returns a dictionary with keys:
      - accuracy (float)
      - f1_macro (float)
      - per_class_{label}_f1 (float) for each label
       if label_names provided
      - confusion_matrix (2D list) if sklearn is available
    """
    if preds is None:
        raise ValueError("preds must not be None")

    # Convert logits to predicted indices if needed
    if isinstance(preds, np.ndarray) and preds.ndim > 1:
        y_pred = np.argmax(preds, axis=1)
    else:
        y_pred = np.asarray(preds)

    out: Dict[str, Any] = {}

    if labels is None:
        # No labels available; return predictions only
        out["predictions"] = y_pred.tolist()
        return out

    y_true = np.asarray(labels)
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("predictions and labels must have" " the same length")

    # Try to use sklearn for robust metrics
    accuracy_score, precision_recall_fscore_support, confusion_matrix = (
        _try_import_sklearn()
    )

    if accuracy_score is not None:
        acc = float(accuracy_score(y_true, y_pred))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=sorted(set(y_true))
        )
        # Macro F1 average
        f1_macro = float(np.mean(f1))
        out.update({"accuracy": acc, "f1_macro": f1_macro})

        if label_names:
            for i, label in enumerate(label_names):
                if i < len(f1):
                    out[f"per_class_{label}_f1"] = float(f1[i])

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true))).tolist()
        out["confusion_matrix"] = cm
    else:
        # Fallback: compute accuracy and a simple per-class f1 approximation
        acc = float((y_pred == y_true).mean())
        out["accuracy"] = acc
        # compute per-class precision/recall/f1
        labels_set = sorted(set(y_true))
        f1s = []
        for l in labels_set:
            tp = int(((y_pred == l) & (y_true == l)).sum())
            fp = int(((y_pred == l) & (y_true != l)).sum())
            fn = int(((y_pred != l) & (y_true == l)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            f1s.append(f1)
            if label_names and l < len(label_names):
                out[f"per_class_{label_names[l]}_f1"] = float(f1)
        out["f1_macro"] = float(np.mean(f1s)) if f1s else 0.0

    return out


def evaluate_trainer(
    trainer: Any,
    eval_dataset: Any,
    label_names: Optional[List[str]] = None,
    mlflow_log: bool = True,
) -> Dict[str, Any]:
    """Run predictions with the provided Hugging Face `Trainer` and compute metrics.

    Parameters
    - trainer: Hugging Face Trainer instance (must implement `predict(dataset)`)
    - eval_dataset: dataset to evaluate on (compatible with trainer.predict)
    - label_names: optional list of label names for per-class metrics
    - mlflow_log: if True and mlflow is importable, log metrics to MLflow

    Returns a dict with keys: `metrics` (dict), `predictions` (list), `label_ids` (list), and optionally `mlflow_run` information.
    """
    # Run prediction
    pred_out = trainer.predict(eval_dataset)
    preds = pred_out.predictions
    labels = getattr(pred_out, "label_ids", None)

    metrics = compute_metrics(preds, labels, label_names=label_names)

    result = {"metrics": metrics}

    # Convert predictions/labels to list for JSON friendliness
    try:
        if preds is not None:
            if isinstance(preds, np.ndarray) and preds.ndim > 1:
                result["predictions"] = np.argmax(preds, axis=1).tolist()
            else:
                result["predictions"] = np.asarray(preds).tolist()
        if labels is not None:
            result["label_ids"] = np.asarray(labels).tolist()
    except Exception:
        # best-effort conversion
        pass

    # Optionally log to MLflow
    if mlflow_log:
        try:
            import mlflow

            # log flattened metrics
            mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            )
            # If sklearn & matplotlib available, log confusion matrix figure
            # Compute confusion matrix if not present and log it (matplotlib required)
            cm = metrics.get("confusion_matrix")
            if cm is None:
                # try sklearn if available
                try:
                    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

                    cm = sk_confusion_matrix(
                        result.get("label_ids", []), result.get("predictions", [])
                    )
                except Exception:
                    # fallback: compute confusion matrix with numpy
                    y_true = np.asarray(result.get("label_ids", []))
                    y_pred = np.asarray(result.get("predictions", []))
                    # if we have no labels/predictions, don't attempt
                    if y_true.size and y_pred.size:
                        labels_arr = np.unique(np.concatenate([y_true, y_pred]))
                        cm = np.zeros((labels_arr.size, labels_arr.size), dtype=int)
                        for i, lt in enumerate(labels_arr):
                            for j, lp in enumerate(labels_arr):
                                cm[i, j] = int(((y_true == lt) & (y_pred == lp)).sum())
                    else:
                        cm = None
            # try to plot and log the confusion matrix
            if cm is not None:
                try:
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots()
                    ax.imshow(cm, cmap="Blues")
                    ax.set_title("Confusion matrix")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    plt.close(fig)
                    mlflow.log_figure(fig, "confusion_matrix.png")
                except Exception:
                    # if plotting/logging fails, continue silently
                    pass
        except Exception:
            # mlflow not available or logging failed; ignore
            pass

    return result


__all__ = ["compute_metrics", "evaluate_trainer"]
