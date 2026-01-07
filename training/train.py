import mlflow
import logging
import os
import json
from pathlib import Path
import tempfile

try:
    import fasttext
except ImportError:
    fasttext = None

from training.data_loader import load_data

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "models"))
MODEL_OUT = OUTPUT_DIR / "sentiment_ft.bin"

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
            text = item["text"].replace("\n", " ").strip()
            f.write(f"__label__{lbl} {text}\n")

def train(epoch=5, lr=0.1, wordNgrams=2, dim=100):
    """Train the FastText model, evaluate on test data, and log metrics to MLflow."""
    if fasttext is None:
        raise ImportError("fasttext is not installed. Install it with: pip install fasttext")

    # Loading Dataset
    dataset = load_data()
    train_ds = dataset["train"]
    test_ds = dataset["test"] # use the native split 'test'

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = None
    test_path = None
    try:
        # Preparing files for training
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as train_f:
            _to_fasttext_format(train_ds, train_f.name)
            train_path = train_f.name

        # Preparing files for testing
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as test_f:
            _to_fasttext_format(test_ds, test_f.name)
            test_path = test_f.name

        with mlflow.start_run():
            mlflow.log_params({"epoch": epoch, "lr": lr, "wordNgrams": wordNgrams, "dim": dim})
            
            # training
            model = fasttext.train_supervised(
                input=train_path, epoch=epoch, lr=lr, wordNgrams=wordNgrams, dim=dim
            )

            # --- METRIC CALCULATION ---
            # model.test() returns: number of examples, precision, recall
            samples, prec, rec = model.test(test_path)
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            metrics = {
                "test_samples": samples,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4)
            }
            
            # Save JSON
            metrics_file = OUTPUT_DIR / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)

            # Logging MLflow
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(metrics_file))
            
            # Save model
            model.save_model(str(MODEL_OUT))
            mlflow.log_artifact(str(MODEL_OUT))
            
            logger.info(f"Training finished. Metrics: {metrics}")
            return MODEL_OUT

    finally:
        for p in [train_path, test_path]:
            if p and os.path.exists(p):
                os.unlink(p)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wordNgrams", type=int, default=2)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--output", type=str, default=os.getenv("OUTPUT_DIR", "models"))
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output)
    MODEL_OUT = OUTPUT_DIR / "sentiment_ft.bin"

    train(epoch=args.epoch, lr=args.lr, wordNgrams=args.wordNgrams, dim=args.dim)