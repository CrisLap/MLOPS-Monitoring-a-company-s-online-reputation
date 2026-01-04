import fasttext
import os
from huggingface_hub import HfApi, hf_hub_download

# Parametri
REPO_ID = "CrisLap/sentiment-model"
TOKEN = os.getenv("HF_TOKEN")


def get_latest_model_filename(repo_id, token):
    api = HfApi()
    # Recupera tutti i file nel repo
    files = api.list_repo_files(repo_id=repo_id, token=token)
    # Filtra quelli che iniziano con 'sentiment_ft' e finiscono con '.bin'
    model_files = [
        f for f in files if f.startswith("sentiment_ft") and f.endswith(".bin")
    ]
    # Ordina e prendi l'ultimo (Hugging Face li restituisce solitamente in ordine)
    return sorted(model_files)[-1] if model_files else None


# Trova il nome dell'ultimo file
latest_file = get_latest_model_filename(REPO_ID, TOKEN)

if latest_file:
    model_path = hf_hub_download(repo_id=REPO_ID, filename=latest_file, token=TOKEN)
else:
    raise FileNotFoundError("Nessun modello trovato nel repository!")


def predict(text: str):
    labels, scores = model.predict(text)
    return labels[0].replace("__label__", ""), float(scores[0])
