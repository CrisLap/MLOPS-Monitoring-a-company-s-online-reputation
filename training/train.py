import mlflow
import logging
import os
import json
from pathlib import Path
import tempfile
import re

try:
    import fasttext
except ImportError:
    fasttext = None

from training.data_loader import load_data

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "models"))
MODEL_OUT = OUTPUT_DIR / "sentiment_ft.ftz"


def clean_text(text):
    """
    Cleans and normalizes text to improve FastText training performance.
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Isolate punctuation (e.g., "good!" becomes "good !")
    # This helps FastText treat the word and the punctuation as separate tokens
    text = re.sub(r"([.!?,'/()])", r" \1 ", text)

    # Remove "@user" tokens (specific to tweet_eval)
    text = text.replace("@user", "")

    # Remove the '#' symbol but keep the word (e.g., "#happy" -> "happy")
    text = text.replace("#", "")

    # Remove multiple spaces and newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text


def setup_mlflow():
    """Initialize MLflow tracking and configure the tracking URI."""
    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.start_run(run_name="healthcheck")
        mlflow.end_run()
    except Exception as e:
        print(f"MLflow unavailable, falling back to local mode: {e}")
        mlflow.set_tracking_uri("file:///tmp/mlruns")


setup_mlflow()


def _to_fasttext_format(dataset_split, path):
    """Convert a dataset split to the FastText specific text format."""
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset_split:
            lbl = LABEL_MAP.get(int(item["label"]), "neutral")
            text = clean_text(item["text"])
            f.write(f"__label__{lbl} {text}\n")


def train(epoch=10, lr=0.05, wordNgrams=2, dim=100):
    """Train the FastText model with Autotune, evaluate, and log to MLflow."""
    if fasttext is None:
        raise ImportError("fasttext is not installed.")

    dataset = load_data()
    train_ds = dataset["train"]
    test_ds = dataset["test"]
    val_ds = dataset["validation"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = None
    test_path = None
    val_path = None
    try:
        # Preparing files
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        ) as train_f:
            _to_fasttext_format(train_ds, train_f.name)
            train_path = train_f.name

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        ) as test_f:
            _to_fasttext_format(test_ds, test_f.name)
            test_path = test_f.name

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        ) as val_f:
            _to_fasttext_format(val_ds, val_f.name)
            val_path = val_f.name

        with mlflow.start_run():
            # --- AUTOTUNE IMPLEMENTATION ---
            # autotuneDuration is in seconds (e.g., 14400 = 240 minutes)
            # We use test_path as the validation set to optimize parameters
            model = fasttext.train_supervised(
                input=train_path,
                autotuneValidationFile=val_path,
                autotuneDuration=600,
                autotuneModelSize="50M",  # Force the model to weigh a maximum of 50MB
                loss="softmax",
                verbose=2,
            )

            # Retrieve the best parameters found by Autotune
            # Use getattr because these attributes might vary by version
            best_params = {
                "epoch": model.epoch,
                "lr": model.lr,
                "wordNgrams": model.wordNgrams,
                "dim": model.dim,
                "autotune_used": True,
            }

            # Log the discovered best parameters to MLflow
            mlflow.log_params(best_params)

            # --- METRIC CALCULATION ---
            samples, prec, rec = model.test(test_path)
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

            metrics = {
                "test_samples": samples,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
            }

            # Save JSON and Log Artifacts (Rest of your original code)
            metrics_file = OUTPUT_DIR / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)

            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(metrics_file))

            model.save_model(str(MODEL_OUT))
            mlflow.log_artifact(str(MODEL_OUT))

            logger.info(
                f"Autotune finished. Best Params: {best_params}. Metrics: {metrics}"
            )
            return MODEL_OUT

    finally:
        for p in [train_path, test_path, val_path]:
            if p and os.path.exists(p):
                os.unlink(p)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--wordNgrams", type=int, default=2)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--output", type=str, default=os.getenv("OUTPUT_DIR", "models"))
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output)
    MODEL_OUT = OUTPUT_DIR / "sentiment_ft.ftz"

    train(epoch=args.epoch, lr=args.lr, wordNgrams=args.wordNgrams, dim=args.dim)
