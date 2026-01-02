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
        token=HF_TOKEN
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
    Se il modello non è caricato, usa un fallback.
    """
    if _model_loaded and _model:
        label, prob = _model.predict(text)
        return {"label": label[0], "score": prob[0]}
    else:
        # fallback semplice: sentiment neutro
        return {"label": "neutral", "score": 0.5}

