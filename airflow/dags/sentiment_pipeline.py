from airflow import DAG
from airflow.operators.python import ShortCircuitOperator, PythonOperator
from datetime import datetime, timedelta
import logging
import numpy as np

# --- Helpers to load recent data ---
from data_drift_detection.storage import load_from_storage

# Import the run_drift function
try:
    from data_drift_detection.run_drift_check import run_drift
except ImportError as e:
    run_drift = None
    logging.error(
        f"Failed to import run_drift from data_drift_detection.run_drift_check: {e}. "
        "Drift checking will not be available."
    )

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# --- DAG tasks ---


def prepare_data(**context):
    """Prepare data for the drift check.

    Strategy: try to pull from XCom (keys: 'sentiment_dist', 'embeddings').
    If not available, fall back to MLflow artifacts, then DB, then files on disk.

    The task will push the arrays to XCom (as lists) under the same keys.

    Raises:
        RuntimeError: If no data source is available
    """
    ti = context["ti"]

    s = ti.xcom_pull(key="sentiment_dist", task_ids=None)
    e = ti.xcom_pull(key="embeddings", task_ids=None)

    if s is not None and e is not None:
        sentiment_dist = np.array(s)
        embeddings = np.array(e)
        logging.info("Loaded data from XCom")
    else:
        logging.info("XCom data not available, trying storage fallback")
        result = load_from_storage()
        if result is None:
            raise RuntimeError(
                "No data source available. Tried XCom, MLflow, DB, and disk. "
                "Please ensure at least one data source is configured."
            )
        sentiment_dist, embeddings = result
        logging.info("Loaded data from storage fallback")

    # Validate loaded data
    if sentiment_dist is None or embeddings is None:
        raise RuntimeError("Loaded data contains None values")

    if len(sentiment_dist) == 0 or len(embeddings) == 0:
        raise RuntimeError("Loaded data is empty")

    # push canonical arrays (JSON-safe lists) to XCom for downstream
    ti.xcom_push(key="sentiment_dist", value=sentiment_dist.tolist())
    ti.xcom_push(key="embeddings", value=embeddings.tolist())
    return True


def _check_drift_callable(**context):
    """
    Retrieves sentiment distribution and embeddings from XComs and runs a drift check.

    Expects the upstream `prepare_data` task to provide `sentiment_dist` and
    `embeddings` via XComs, and delegates the drift evaluation to `run_drift`.
    Raises a RuntimeError if required inputs or the drift function are unavailable.
    """
    if run_drift is None:
        raise RuntimeError("run_drift not available; cannot perform drift check")

    ti = context["ti"]
    s = ti.xcom_pull(task_ids="prepare_data", key="sentiment_dist")
    e = ti.xcom_pull(task_ids="prepare_data", key="embeddings")

    if s is None or e is None:
        raise RuntimeError("prepare_data did not provide required XComs")

    sentiment_dist = np.array(s)
    embeddings = np.array(e)

    return run_drift(sentiment_dist, embeddings)


def _train_callable(**context):
    """
    Executes the model training routine and propagates any failure.

    Imports and runs the training entry point, logging and re-raising
    exceptions to ensure task failure is properly reported.
    """
    try:
        import training.train as train_module

        train_module.train()
    except Exception as e:
        logging.exception("Training failed: %s", e)
        raise


with DAG(
    "sentiment_pipeline",
    default_args=DEFAULT_ARGS,
    description="Drift monitoring and conditional retraining for sentiment",
    schedule="@daily",  # Use 'schedule' instead of deprecated 'schedule_interval'
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["sentiment", "drift", "mlops"],
) as dag:

    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data,
        provide_context=True,
    )

    check_drift = ShortCircuitOperator(
        task_id="check_drift",
        python_callable=_check_drift_callable,
        provide_context=True,
    )

    train = PythonOperator(
        task_id="train",
        python_callable=_train_callable,
        provide_context=True,
    )

    prepare >> check_drift >> train
