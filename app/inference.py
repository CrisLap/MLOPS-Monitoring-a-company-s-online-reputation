import fasttext
import os
from huggingface_hub import HfApi, hf_hub_download

# 1. Configurazione
REPO_ID = os.getenv("HF_MODEL_REPO", "CrisLap/sentiment-model")
TOKEN = os.getenv("HF_TOKEN")

# 2. Logica per trovare l'ultimo file (per gestire gli hash che abbiamo visto)
api = HfApi()
files = api.list_repo_files(repo_id=REPO_ID, token=TOKEN)
model_files = sorted(
    [f for f in files if f.startswith("sentiment_ft") and f.endswith(".bin")]
)
latest_file = model_files[-1] if model_files else None

# 3. Download e CARICAMENTO (Qui risolviamo l'errore)
if latest_file:
    model_path = hf_hub_download(repo_id=REPO_ID, filename=latest_file, token=TOKEN)

else:
    model = None  # O gestisci l'errore


def predict(text):
    # Riga 32: Ora 'model' esiste perché è stato definito sopra!
    if model is None:
        return "Modello non trovato"
    return model.predict(text)
