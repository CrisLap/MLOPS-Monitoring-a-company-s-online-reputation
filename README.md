# Monitoring a company's online reputation

## Project summary
MachineInnovators Inc. is a leader in developing scalable, production-ready machine learning applications. The main focus of the project is to integrate MLOps methodologies to facilitate the development, implementation, continuous monitoring, and retraining of sentiment analysis models. The goal is to enable the company to improve and monitor its reputation on social media through automatic sentiment analysis.

A compact sentiment analysis repository built with Hugging Face Transformers for training, MLflow for experiment tracking, and FastAPI for serving predictions. It includes utilities for data-drift detection, an Airflow DAG to schedule drift checks and conditional retraining, and Prometheus/Grafana monitoring assets to track request behavior and drift.

---

## Table of contents
- [Key components](#key-components)
- [Quick start](#quick-start)
- [Setup](#setup)
- [Usage](#usage)
- [Monitoring & Metrics](#monitoring--metrics)
- [Data drift detection](#data-drift-detection)
- [Airflow DAG & scheduling](#airflow-dag--scheduling)
- [Testing](#testing)
- [Development notes & housekeeping](#development-notes--housekeeping)
- [TODOs & Open questions](#todos--open-questions)

---

## Key components 🔧
- `training/` — training pipeline:
  - `train.py` — fine-tunes a model and saves to `models/` (and logs artifacts to MLflow).
  - `data_loader.py` — loads datasets.
  - `evaluate.py` — evaluation utilities (compute metrics, log to MLflow and save confusion matrix).
- `app/` — API & serving:
  - `main.py` — FastAPI app exposing `POST /predict` and mounting `/metrics` (Prometheus).
  - `inference.py` — loads model (fastText) and provides a fallback predictor if model unavailable.
  - `metrics.py` — Prometheus metric definitions.
  - `schemas.py` — Pydantic request/response models.
- `data_drift_detection/` — baseline creation and drift checks (`baseline.py`, `detector.py`, `run_drift_check.py`).
- `airflow/dags/sentiment_pipeline.py` — DAG that runs drift checks and triggers retraining if drift is detected (the DAG loads recent data from XCom, MLflow artifacts, a DB, or disk -- see below).
- `monitoring/` — `prometheus.yml` and `grafana_dashboard.json` (optional local monitoring assets).
- `docker/` — images for API and training.
- `docker-compose.yml` — local compose (note: training mounts `./models:/app/models`).
- `tests/` — unit tests for API, training, and evaluation utilities.

---

## Quick start 🚀
1. Clone the repo:

```bash
git clone <repo-url>
cd mlops_reputation
```

2. Create and activate a virtualenv (example):

- Windows
```powershell
python -m venv .venv
.venv\Scripts\activate
```
- macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install runtime dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install development deps (includes Airflow, SQLAlchemy and DB driver to test DAG/DB fallback):
```bash
pip install -r requirements-dev.txt
```

---

## Setup ⚙️
- **Recommended Python:** 3.10 (verify compatibility of optional deps like Airflow with your environment).
- **Environment variables / connections:**
  - `MLFLOW_TRACKING_URI` — if you use a remote MLflow server (default in `docker-compose` is `http://mlflow:5000`).
  - `DRIFT_DB_URI` — optional DB URI used by DAG fallback to retrieve recent drift data.
  - Airflow connection id `drift_db` — alternatively configure the DB connection inside Airflow.
- **Ports used:**
  - API: `8000`
  - MLflow UI: `5000` (when using `docker-compose`)
  - Prometheus / Grafana: run locally or add services to compose as needed.

---

## Usage 📌

### Train locally
```bash
python -m training.train
```
- A trained model is saved to `models/` and logged to MLflow (if configured).

**Important:** do not commit large binary models to the repo. Prefer storing released models as artifacts in MLflow, S3, or an artifact store. Add `models/` to `.gitignore` to avoid accidental commits.

### Retrieve model from MLflow (example)
You can download the latest model artifact `models/sentiment_ft.bin` from MLflow and place it in `models/` prior to starting the API. A small helper script (not included) can call `MlflowClient().download_artifacts(...)`.

### Run the API (local)
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- Predict:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"I love this product"}'
```
- Metrics (Prometheus format): `http://localhost:8000/metrics`.

### Train via Docker Compose (with MLflow server)
```bash
docker compose up --build
```
- This will start a local `mlflow` service and run the `training` service (as configured in `docker-compose.yml`).
- Artifacts and models persist to `./models` and `./mlflow/artifacts`.

---

## Monitoring & Metrics 📈
- `monitoring/prometheus.yml` is a simple example config that scrapes `api:8000` (useful with Docker Compose). Removing it **does not** affect the API; it only removes the convenience config for running Prometheus locally.
- Grafana dashboard JSON in `monitoring/grafana_dashboard.json` contains panels for request rate, latency, sentiment distribution, and drift scores.
- Key metrics exposed by the API:
  - `sentiment_requests_total` (counter)
  - `sentiment_request_latency_seconds` (histogram)
  - `sentiment_predictions_total{label}` (counter)
  - `sentiment_drift_score` (gauge from drift checks)

---

## Data drift detection 🔍
- Use `data_drift_detection/baseline.py` to save baseline statistics (`sentiment_dist`, `embedding_mean`) to a JSON baseline file.
- Run checks with `data_drift_detection/run_drift_check.py` to compute label/embedding drift and log metrics to MLflow.

### DAG behavior & fallbacks
The Airflow DAG (`airflow/dags/sentiment_pipeline.py`) prepares recent data and then runs `run_drift(sentiment_dist, embeddings)`. The data preparation strategy is (in order):
1. **XCom** — if a producer task pushed `sentiment_dist` and `embeddings` to XCom.
2. **MLflow artifacts** — searches recent MLflow runs and downloads `drift/recent.npz` or `drift/recent.json` if present.
3. **Database** — queries a table `drift_data` for the latest `sentiment_dist` and `embeddings` rows (expected JSON in the `value` column). Provide `DRIFT_DB_URI` or configure Airflow connection `drift_db`.
4. **Disk files** — fallback to `drift/recent.npz` or `drift/recent.json`.

If drift is detected (`run_drift` returns True), the DAG triggers the `train` task which calls the project's training code. In production you may want the `train` task to trigger a remote training job rather than run training inside Airflow.

### Example DB table (suggested)
```sql
CREATE TABLE drift_data (
  id SERIAL PRIMARY KEY,
  type TEXT NOT NULL, -- 'sentiment_dist' or 'embeddings'
  value JSONB NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
```

---

## Testing ✅

This project includes unit tests and higher-level tests for the API, training, evaluation and (optionally) Airflow DAGs. Below are concrete commands and quick examples to run and extend the test suite.

### Unit tests (fast)
- Run all tests locally:
```bash
pytest -q
```
- Run a single test file or function:
```bash
pytest tests/test_api.py -q
pytest tests/test_training.py::test_full_train_flow_creates_model_and_logs -q
```
- Run tests matching a pattern:
```bash
pytest -k train -q
```

### Coverage
- Produce a coverage report (requires `coverage`):
```bash
pip install coverage
coverage run -m pytest
coverage report -m
coverage html  # optional: opens a detailed HTML report
```

### Integration / runtime tests
Some tests interact with the running API or simulated external services:
- Start the API locally in one terminal:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- In another terminal run the runtime tests (or the specific test file):
```bash
pytest tests/test_api_runtime.py -q
```
Notes:
- Tests attempt to isolate external dependencies with mocks; `test_api_runtime.py` exercises the running API and may require the model artifact in `models/` or the API to run in fallback mode.

### Testing the Airflow DAG locally
To test `airflow/dags/sentiment_pipeline.py` you'll need a local Airflow dev install (see `requirements-dev.txt`):
1. Create sample recent data:
```python
python - <<PY
import numpy as np
np.savez('drift/recent.npz', sentiment_dist=np.array([0.2,0.6,0.2]), embeddings=np.random.randn(10, 768))
PY
```
2. Run task-level tests:
```bash
airflow tasks test sentiment_pipeline prepare_data 2025-12-21
airflow tasks test sentiment_pipeline check_drift 2025-12-21
```
This will exercise the XCom/MLflow/DB/disk fallback logic implemented in the DAG.

### Testing MLflow fallback
- Upload `drift/recent.npz` under `drift/` for a run in MLflow:
```python
import numpy as np, mlflow
np.savez('recent.npz', sentiment_dist=np.array([0.2,0.6,0.2]), embeddings=np.random.randn(10, 768))
with mlflow.start_run(run_name='local_test'):
    mlflow.log_artifact('recent.npz', artifact_path='drift')
```
- Then test `prepare_data` (via `airflow tasks test`) and verify XCom or logs to confirm the MLflow loader worked.

### Testing DB fallback
- Create a `drift_data` table (example for Postgres):
```sql
CREATE TABLE drift_data (
  id SERIAL PRIMARY KEY,
  type TEXT NOT NULL,
  value JSONB NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
```
- Insert test rows (JSON arrays for `value`) and set `DRIFT_DB_URI` env var or an Airflow connection `drift_db`.
- Run `airflow tasks test sentiment_pipeline prepare_data` to confirm the DB loader picks up the latest rows.

### Test design tips
- Prefer unit tests with mocks for network/MLflow/DB to keep CI fast and deterministic.
- Use small fixtures for embeddings (low dimensionality) to speed up tests.
- Add focused tests for the DAG helpers: e.g., unit-tests for `_load_from_mlflow()` (mock `MlflowClient`) and `_load_from_db()` (use SQLite in-memory or mock `sqlalchemy` calls).

### CI suggestions (GitHub Actions)
- Run `pytest -q` and optionally a linter/formatter (black/flake8) on PRs.
- If you want to test DAG/DB integration in CI, use an additional job that spins up a Postgres service and an MLflow docker service and runs the relevant integration tests.

---

## Development notes & housekeeping 💡
- **Dev deps added**: `SQLAlchemy`, `psycopg2-binary`, and `apache-airflow>=2.9,<3` (in `requirements-dev.txt`) to support local DAG testing and DB fallback.
- **models/**: keep out of git (add `models/` to `.gitignore`) and store production models in MLflow / S3 / artifact storage.
- **mlruns/**: local MLflow storage; safe to back up or delete if you want to clear local experiment history.
- **.pytest_cache/** and `__pycache__` directories: safe to delete (regenerable). Add them to `.gitignore` if you want.
- **Prometheus config**: `monitoring/prometheus.yml` is optional and can be removed without breaking the app.

---

## TODOs & Open questions ❓
- Decide on production MLflow backend (e.g., Postgres + S3/MinIO) and update `docker-compose.yml` with persistent stores.
- Add a small helper script to download the latest model from MLflow into `models/` at deployment time.
- Add a migration or creation script for the `drift_data` table used by the DAG DB fallback.
- Improve DAG to trigger remote training jobs and to include notifications/alerts on drift detection.

---

## Contributing
Contributions welcome! Please open a PR with a clear description and include tests for new behavior. Follow the existing code style.

---

## License
MIT
