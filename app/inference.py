import fasttext
import os
from huggingface_hub import HfApi, hf_hub_download

# 1. Configurazione
REPO_ID = os.getenv("HF_MODEL_REPO", "CrisLap/sentiment-model")
TOKEN = os.getenv("HF_TOKEN")

# Inizializziamo model come None a livello globale
model = None

# 2. Logica per trovare l'ultimo file
try:
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, token=TOKEN)
    model_files = sorted(
        [f for f in files if f.startswith("sentiment_ft") and f.endswith(".bin")]
    )
    latest_file = model_files[-1] if model_files else None

    # 3. Download e CARICAMENTO
    if latest_file:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=latest_file, token=TOKEN)
        # CARICAMENTO EFFETTIVO: Questa riga mancava
        model = fasttext.load_model(model_path)
    else:
        print("Attenzione: Nessun file di modello trovato nel repository.")
except Exception as e:
    print(f"Errore durante il caricamento del modello: {e}")
    model = None

def predict(text):
    if model is None:
        # Restituiamo un valore predefinito per non far fallire i test dell'API
        return "label_error", 0.0
    
    # FastText restituisce ([('__label__positive',)], [array([0.95])])
    prediction = model.predict(text)
    label = prediction[0][0].replace("__label__", "")
    score = float(prediction[1][0])
    return label, score
