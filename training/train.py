import mlflow
import logging
import os

try:
    import fasttext
except ImportError:
    fasttext = None

from .data_loader import load_data
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

MODEL_OUT = Path("models") / "sentiment_ft.bin"


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

    Args:
        epoch: Number of training epochs
        lr: Learning rate
        wordNgrams: Word n-gram size
        dim: Embedding dimension

    Returns:
        Path: Path to saved model

    Raises:
        ImportError: If fasttext is not available
        ValueError: If dataset loading fails or parameters are invalid
        RuntimeError: If training fails
    """
    if fasttext is None:
        raise ImportError(
            "fasttext is not installed. Install it with: pip install fasttext"
        )

    # Validate parameters
    if epoch <= 0 or lr <= 0 or dim <= 0:
        raise ValueError("epoch, lr, and dim must be positive")

    try:
        dataset = load_data()
        train_ds = dataset["train"]

        if not train_ds or len(train_ds) == 0:
            raise ValueError("Training dataset is empty")

        logger.info(f"Loaded dataset with {len(train_ds)} training samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise ValueError(f"Dataset loading failed: {e}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary file and ensure cleanup
    train_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, encoding="utf-8", suffix=".txt"
        ) as train_f:
            _to_fasttext_format(train_ds, train_f.name)
            train_path = train_f.name

        logger.info(f"Created training file: {train_path}")

        try:
            with mlflow.start_run():
                # Log hyperparameters
                mlflow.log_params(
                    {"epoch": epoch, "lr": lr, "wordNgrams": wordNgrams, "dim": dim}
                )

                # train supervised fastText model
                logger.info("Starting model training...")
                model = fasttext.train_supervised(
                    input=train_path, epoch=epoch, lr=lr, wordNgrams=wordNgrams, dim=dim
                )

                logger.info("Training completed, saving model...")
                model.save_model(str(MODEL_OUT))

                # Log the model artifact to MLflow
                mlflow.log_artifact(str(MODEL_OUT))
                logger.info(f"Model saved to {MODEL_OUT} and logged to MLflow")
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise RuntimeError(f"Model training failed: {e}")

        return MODEL_OUT
    finally:
        # Clean up temporary file
        if train_path and os.path.exists(train_path):
            try:
                os.unlink(train_path)
                logger.debug(f"Cleaned up temporary file: {train_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {train_path}: {e}")


if __name__ == "__main__":
    train()
