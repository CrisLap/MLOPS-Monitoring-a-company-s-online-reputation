import fasttext
from huggingface_hub import hf_hub_download

MODEL_REPO = "CrisLap/sentiment-model"
MODEL_FILE = "sentiment_ft.bin"

model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
model = fasttext.load_model(model_path)


def predict(text: str):
    labels, scores = model.predict(text)
    return labels[0].replace("__label__", ""), float(scores[0])
