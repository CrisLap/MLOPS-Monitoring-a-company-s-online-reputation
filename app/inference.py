import os
import fasttext
from huggingface_hub import hf_hub_download

_model_loaded = False
_model = None

MODEL_TAG = os.environ.get("MODEL_TAG", "latest")
HF_TOKEN = os.environ.get("HF_TOKEN")

try:
    # Scarica modello dal HF Hub
    model_path = hf_hub_download(
        repo_id="crislap/sentiment-model",
        filename=f"sentiment_ft_{MODEL_TAG}.bin",
        token=HF_TOKEN,
    )
    # Carica FastText
    _model = fasttext.load_model(model_path)
    _model_loaded = True
    print(f"FastText model loaded from {model_path}")

except Exception as e:
    print(f"WARNING: Could not load FastText model ({e}), using fallback predictor.")
    _model_loaded = False


def predict(text: str) -> dict:
    """
    Restituisce la predizione di sentiment.
    Se il modello non è caricato, usa un fallback conforme a SentimentResponse.
    """
    if _model_loaded and _model:
        label, prob = _model.predict(text)
        # scores può essere un dict con tutte le classi previste
        return {
            "label": label[0],
            "confidence": float(prob[0]),
            "scores": {label[0]: float(prob[0])}  # struttura minima compatibile
        }
    else:
        # fallback completo per evitare errori di validazione
        return {
            "label": "neutral",
            "confidence": 0.5,
            "scores": {"neutral": 0.5, "positive": 0.25, "negative": 0.25}  # valori dummy
        }

