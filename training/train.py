import mlflow
import logging
import os
from pathlib import Path
import tempfile

try:
    import fasttext
except ImportError:
    fasttext = None

# Usa import assoluto per evitare problemi con il -m
from training.data_loader import load_data

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# Ora la cartella di output è parametrizzabile via variabile d'ambiente
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "models"))
MODEL_OUT = OUTPUT_DIR / "sentiment_ft.bin"


def setup_mlflow():
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
    """Converts dataset split to FastText format and writes to a file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset_split:
            lbl = LABEL_MAP.get(int(item["label"]), "neutral")
            text = item["text"].replace("\\n", " ").strip()
            f.write(f"__label__{lbl} {text}\n")


def train(epoch=5, lr=0.1, wordNgrams=2, dim=100):
    if fasttext is None:
        raise ImportError(
            "fasttext is not installed. Install it with: pip install fasttext"
        )

    if epoch <= 0 or lr <= 0 or dim <= 0:
        raise ValueError("epoch, lr, and dim must be positive")

    try:
        dataset = load_data()
        train_ds = dataset["train"]
        if not train_ds:
            raise ValueError("Training dataset is empty")
        logger.info(f"Loaded dataset with {len(train_ds)} training samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise ValueError(f"Dataset loading failed: {e}")

    # Crea la cartella di output parametrizzabile
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, encoding="utf-8", suffix=".txt"
        ) as train_f:
            _to_fasttext_format(train_ds, train_f.name)
            train_path = train_f.name

        logger.info(f"Created training file: {train_path}")

        with mlflow.start_run():
            mlflow.log_params(
                {"epoch": epoch, "lr": lr, "wordNgrams": wordNgrams, "dim": dim}
            )
            logger.info("Starting model training...")
            model = fasttext.train_supervised(
                input=train_path, epoch=epoch, lr=lr, wordNgrams=wordNgrams, dim=dim
            )

            logger.info(f"Training completed, saving model to {MODEL_OUT}...")
            model.save_model(str(MODEL_OUT))
            mlflow.log_artifact(str(MODEL_OUT))
            logger.info("Model logged to MLflow successfully")

        return MODEL_OUT
    finally:
        if train_path and os.path.exists(train_path):
            try:
                os.unlink(train_path)
                logger.debug(f"Cleaned up temporary file: {train_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {train_path}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wordNgrams", type=int, default=2)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--output", type=str, default=os.getenv("OUTPUT_DIR", "models"))
    args = parser.parse_args()

    # Imposta OUTPUT_DIR dinamicamente da CLI
    OUTPUT_DIR = Path(args.output)
    MODEL_OUT = OUTPUT_DIR / "sentiment_ft.bin"

    train(epoch=args.epoch, lr=args.lr, wordNgrams=args.wordNgrams, dim=args.dim)
