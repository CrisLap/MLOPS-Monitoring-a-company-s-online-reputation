import fasttext
import os
from huggingface_hub import hf_hub_download

MODEL_REPO = "CrisLap/sentiment-model"
MODEL_FILE = "sentiment_ft.bin"

token = os.getenv("HF_TOKEN")
model_path = hf_hub_download(
    repo_id="CrisLap/sentiment-model", 
    filename="sentiment_ft.bin",
    token=token
) 
model = fasttext.load_model(model_path)


def predict(text: str):
    labels, scores = model.predict(text)
    return labels[0].replace("__label__", ""), float(scores[0])
