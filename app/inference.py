import logging
from typing import Dict
from pathlib import Path

try:
    import fasttext
except Exception:
    fasttext = None

labels = ["negative", "neutral", "positive"]
_model = None
_model_loaded = False
_model_path = Path("models") / "sentiment_ft.bin"

# Try to load fastText model if available
try:
    if fasttext is not None and _model_path.exists():
        _model = fasttext.load_model(str(_model_path))
        _model_loaded = True
    else:
        logging.warning(
            "FastText model not found or fasttext not installed; using fallback predictor."
        )
except Exception as e:
    logging.warning("Could not load FastText model: %s; using fallback predictor.", e)


def _fallback_predict(text: str):
    """
    Returns a default neutral prediction with uniform confidence scores.
    Used as a fallback when the primary prediction mechanism is unavailable,
    assigning equal probability to all labels.
    """
    scores_list = [1.0 / 3.0] * len(labels)
    scores_dict = {labels[i]: scores_list[i] for i in range(len(labels))}
    return {"label": "neutral", "confidence": scores_list[1], "scores": scores_dict}


def predict(text: str):
    """
    Predicts the sentiment label and confidence scores for the given text.
    Uses the loaded fastText model when available, normalizing label probabilities
    and returning the most likely label. Falls back to a default neutral prediction
    if the model is unavailable or inference fails.
    """
    if not _model_loaded:
        return _fallback_predict(text)

    # fastText predict returns labels like '__label__positive'
    try:
        labels_out, probs = _model.predict(text, k=len(labels))
        scores_dict = {l: 0.0 for l in labels}
        for lab, p in zip(labels_out, probs):
            lab_clean = lab.replace("__label__", "")
            if lab_clean in scores_dict:
                scores_dict[lab_clean] = float(p)
        # ensure sum to 1 normalization if needed
        s = sum(scores_dict.values())
        if s > 0:
            for k in scores_dict:
                scores_dict[k] = scores_dict[k] / s
        # pick best
        best_label = max(scores_dict.items(), key=lambda x: x[1])[0]
        return {
            "label": best_label,
            "confidence": scores_dict[best_label],
            "scores": scores_dict,
        }
    except Exception as e:
        logging.warning("FastText prediction failed (%s); using fallback.", e)
        return _fallback_predict(text)
