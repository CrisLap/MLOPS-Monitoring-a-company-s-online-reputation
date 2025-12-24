import mlflow

try:
    import fasttext
except Exception:
    fasttext = None
from .data_loader import load_data
import tempfile
from pathlib import Path

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

MODEL_OUT = Path("models") / "sentiment_ft.bin"


def _to_fasttext_format(dataset_split, path):
    """
    Converts a dataset split to FastText training format and writes to a file.
    Each line in the file contains a label prefixed with '__label__' followed
    by the corresponding text.
    """
    # dataset_split is a list/dict with 'text' and 'label' fields
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset_split:
            lbl = LABEL_MAP.get(int(item["label"]), "neutral")
            text = item["text"].replace("\\n", " ").strip()
            f.write(f"__label__{lbl} {text}\n")


def train(epoch=5, lr=0.1, wordNgrams=2, dim=100):
    """
    Trains a FastText supervised model on the TweetEval sentiment dataset.
    Converts the training split to FastText format, trains the model with
    specified hyperparameters, saves the model to disk, and logs it as an
    MLflow artifact.
    """
    dataset = load_data()
    train_ds = dataset["train"]

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, encoding="utf-8"
    ) as train_f:
        _to_fasttext_format(train_ds, train_f.name)
        train_path = train_f.name

    with mlflow.start_run():
        # train supervised fastText model
        model = fasttext.train_supervised(
            input=train_path, epoch=epoch, lr=lr, wordNgrams=wordNgrams, dim=dim
        )
        model.save_model(str(MODEL_OUT))
        # Log the model artifact to MLflow (user
        # must set MLFLOW_TRACKING_URI to track remotely)
        mlflow.log_artifact(str(MODEL_OUT))

    return MODEL_OUT


if __name__ == "__main__":
    train()
