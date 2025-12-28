# Monitoring a company's online reputation

[![CI/CD Pipeline](https://github.com/USERNAME/REPO_NAME/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/USERNAME/REPO_NAME/actions/workflows/ci_cd.yml)
[![Training Pipeline](https://github.com/USERNAME/REPO_NAME/actions/workflows/training.yml/badge.svg)](https://github.com/USERNAME/REPO_NAME/actions/workflows/training.yml)

## Project summary
MachineInnovators Inc. is a leader in developing scalable, production-ready machine learning applications. The main focus of the project is to integrate MLOps methodologies to facilitate the development, implementation, continuous monitoring, and retraining of sentiment analysis models. The goal is to enable the company to improve and monitor its reputation on social media through automatic sentiment analysis.

A compact sentiment analysis repository built with Hugging Face Transformers for training, MLflow for experiment tracking, and FastAPI for serving predictions. It includes utilities for data-drift detection, an Airflow DAG to schedule drift checks and conditional retraining, and Prometheus/Grafana monitoring assets to track request behavior and drift.

---

## Table of contents
- [Key components](#key-components)
- [Quick start](#quick-start)
- [Setup](#setup)
- [Usage](#usage)
- [Deployment](#deployment)
- [Monitoring & Metrics](#monitoring--metrics)
- [Data drift detection](#data-drift-detection)
- [Airflow DAG & scheduling](#airflow-dag--scheduling)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Development notes & housekeeping](#development-notes--housekeeping)
- [TODOs & Open questions](#todos--open-questions)

---

## Key components 🔧
- `training/` — training pipeline:
  - `train.py` — fine-tunes a model and saves to `models/` (and logs artifacts to MLflow).
  - `data_loader.py` — loads datasets.
  - `evaluate.py` — evaluation utilities (compute metrics, log to MLflow and save confusion matrix).
- `app/` — API & serving:
  - `main.py` — FastAPI app exposing `POST /predict`, `/health`, `/ready` endpoints and mounting `/metrics` (Prometheus).
  - `inference.py` — loads model (fastText) and provides a fallback predictor if model unavailable.
  - `metrics.py` — Prometheus metric definitions.
  - `schemas.py` — Pydantic request/response models with input validation.
- `data_drift_detection/` — baseline creation and drift checks (`baseline.py`, `detector.py`, `run_drift_check.py`).
- `airflow/dags/sentiment_pipeline.py` — DAG that runs drift checks and triggers retraining if drift is detected (the DAG loads recent data from XCom, MLflow artifacts, a DB, or disk -- see below).
- `monitoring/` — `prometheus.yml` and `grafana_dashboard.json` (optional local monitoring assets).
- `docker/` — images for API and training.
- `docker-compose.yml` — complete local stack with MLflow, API, and training services.
- `.github/workflows/` — CI/CD automation:
  - `ci_cd.yml` — main CI/CD pipeline (testing, validation, Docker builds, deployment).
  - `training.yml` — training workflow (manual dispatch and scheduled training).
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

**API Endpoints:**
- **Predict**: `POST /predict` — Sentiment analysis prediction
  ```bash
  curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"I love this product"}'
  ```
- **Health Check**: `GET /health` — Service health status
  ```bash
  curl http://localhost:8000/health
  ```
- **Readiness Check**: `GET /ready` — Service readiness (checks if model is loaded)
  ```bash
  curl http://localhost:8000/ready
  ```
- **Metrics**: `GET /metrics` — Prometheus metrics
  ```bash
  curl http://localhost:8000/metrics
  ```
- **API Documentation**: `GET /docs` — Interactive Swagger UI

**Input Validation:**
- Text input is limited to 10,000 characters to prevent DoS attacks
- All inputs are validated using Pydantic schemas

### Run Full Stack with Docker Compose
```bash
docker compose up --build
```

This starts the complete stack:
- **MLflow**: Experiment tracking server on port 5000
  - UI: http://localhost:5000
  - Backend: SQLite (persisted to `./mlflow/`)
- **API**: Sentiment analysis API on port 8000
  - Endpoints: http://localhost:8000
  - Health: http://localhost:8000/health
- **Training**: Run training jobs (one-time execution)

**Services:**
- All services are on a shared network (`mlops-network`)
- MLflow data persists to `./mlflow/`
- Models persist to `./models/`
- Services have health checks configured
- API depends on MLflow being healthy before starting

**Start specific services:**
```bash
# Start only MLflow and API
docker compose up mlflow api

# Run training once
docker compose run --rm training
```

---

## Deployment 🚀

This project includes automated CI/CD pipelines that handle building, testing, and deploying Docker images and the Hugging Face Space.

### Docker Images

Docker images are automatically built and pushed to GitHub Container Registry (GHCR) on pushes to `main`/`master` branch and version tags.

#### Image Locations

- **API Image**: `ghcr.io/{owner}/{repo}/api`
- **Training Image**: `ghcr.io/{owner}/{repo}/training`

#### Image Tags

Images are tagged with:
- `latest` — latest build from main branch
- `{sha}` — commit SHA for specific builds
- `{version}` — semantic version tags (e.g., `v1.0.0`)
- `{major}.{minor}` — major.minor version tags

#### Using Docker Images

**Pull and run the API image:**
```bash
docker pull ghcr.io/{owner}/{repo}/api:latest
docker run -p 8000:8000 ghcr.io/{owner}/{repo}/api:latest
```

**Pull and run the training image:**
```bash
docker pull ghcr.io/{owner}/{repo}/training:latest
docker run -e MLFLOW_TRACKING_URI="http://your-mlflow:5000" \
  -v $(pwd)/models:/app/models \
  ghcr.io/{owner}/{repo}/training:latest
```

**Note**: Replace `{owner}` and `{repo}` with your GitHub username/organization and repository name.

### Hugging Face Space

The API is automatically deployed to a Hugging Face Space on pushes to `main`/`master` branch and version tags.

#### Automatic Deployment

- Deployment is triggered automatically on:
  - Pushes to `main` or `master` branch
  - Version tags (e.g., `v1.0.0`, `v1.2.3`)
- The Space uses Docker SDK with the API Dockerfile
- Configuration is managed via `.huggingface.yaml`

#### Accessing the Deployed Space

Once deployed, the Space will be available at:
```
https://huggingface.co/spaces/{owner}/{repo-name}
```

The Space provides:
- Interactive API documentation at `/docs`
- Prediction endpoint at `/predict`
- Prometheus metrics at `/metrics`

#### Manual Deployment

If you need to manually deploy to Hugging Face Space:

1. Install Hugging Face CLI:
```bash
pip install huggingface_hub[cli]
```

2. Authenticate:
```bash
huggingface-cli login
```

3. Push files to your Space:
```bash
huggingface-cli upload {space-name} \
  docker/Dockerfile.api \
  .huggingface.yaml \
  app/ \
  requirements.txt \
  --repo-type space
```

### Required GitHub Secrets

For automated deployments, configure the following secret in your GitHub repository:

- **`HF_TOKEN`** — Hugging Face authentication token for Space deployment
  - Generate at: https://huggingface.co/settings/tokens
  - Required scope: `write` access

The `GITHUB_TOKEN` is automatically available for pushing to GHCR and doesn't need to be configured.

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

### Continuous Integration

This project uses GitHub Actions for automated testing and validation. The CI pipeline runs on every push and pull request, executing:

- **Code Quality**: Black formatting checks and Flake8 linting
- **Unit Tests**: Full test suite with coverage reporting
- **Integration Tests**: Tests with MLflow and PostgreSQL services
- **Airflow DAG Validation**: Syntax and dependency validation
- **Monitoring Config Validation**: Prometheus and Grafana config checks
- **Docker Builds**: Both API and training images are built and tested

See the [CI/CD Pipeline](#cicd-pipeline) section for detailed workflow information.

---

## CI/CD Pipeline 🔄

This project includes comprehensive CI/CD automation using GitHub Actions. The workflows ensure code quality, run tests, validate configurations, build Docker images, and handle deployments.

### Main CI/CD Workflow (`ci_cd.yml`)

The main workflow runs on:
- **Push** to `main`/`master` branch
- **Pull requests** to `main`/`master` branch
- **Version tags** (e.g., `v1.0.0`)
- **Manual dispatch** (workflow_dispatch)

#### Workflow Jobs

1. **Code Quality & Linting**
   - Black code formatter check
   - Flake8 linting
   - YAML file validation (docker-compose, Prometheus config)
   - JSON file validation (Grafana dashboard)
   - Hugging Face config validation

2. **Unit Tests**
   - Runs full pytest test suite
   - Generates coverage reports
   - Uploads coverage artifacts
   - Tests all modules: API, training, evaluation, metrics, storage, drift detection

3. **Integration Tests**
   - Starts MLflow server in background
   - Uses PostgreSQL service container for DB fallback testing
   - Tests API with running services
   - Tests data drift detection with MLflow integration
   - Tests storage fallback mechanisms (MLflow, DB, disk)

4. **Airflow DAG Validation**
   - Validates DAG syntax and imports
   - Verifies task dependencies
   - Tests DAG structure

5. **Monitoring Validation**
   - Validates Prometheus YAML configuration
   - Validates Grafana dashboard JSON schema
   - Verifies metric names match API metrics

6. **Docker Image Builds**
   - **Training Image**: Builds, tests, and tags training Docker image
   - **API Image**: Builds, tests, and performs health checks
   - Both images are tagged with multiple versions (latest, SHA, semantic versions)
   - Images are pushed to GHCR (on main/tags, not on PRs)

7. **Hugging Face Space Deployment** (conditional)
   - Deploys API to Hugging Face Space
   - Only runs on main branch and version tags
   - Creates/updates Space with Docker SDK configuration
   - Pushes necessary files (Dockerfile, app/, requirements.txt)

8. **Docker Image Push Summary**
   - Provides summary of pushed images to GHCR

### Training Workflow (`training.yml`)

A separate workflow handles model training with MLflow integration.

#### Triggers

- **Manual Dispatch**: Run training with custom parameters
  - Configurable epochs and learning rate
  - Accessible via GitHub Actions UI
- **Scheduled**: Weekly training on Sundays at 2 AM UTC

#### Workflow Steps

1. **Setup**: Installs dependencies and starts MLflow server
2. **Training**: Runs training script with MLflow tracking
3. **Validation**: Verifies model artifact creation
4. **Artifact Upload**: Uploads trained model as GitHub artifact (30-day retention)
5. **Docker Build**: Builds and pushes training image to GHCR (on main branch)

### Workflow Features

- **Parallel Execution**: Jobs run in parallel where possible for faster feedback
- **Conditional Deployment**: Deployments only occur on main branch and version tags
- **Docker Layer Caching**: Uses GitHub Actions cache for faster builds
- **Service Containers**: PostgreSQL service for integration testing
- **Artifact Management**: Coverage reports and trained models are stored as artifacts
- **Security**: Uses GitHub secrets for sensitive tokens

### Viewing Workflow Runs

- Navigate to the **Actions** tab in your GitHub repository
- View workflow runs, logs, and artifacts
- Re-run failed workflows or manually trigger workflows

### Required Configuration

#### GitHub Secrets

Configure the following secret in your repository settings:

- **`HF_TOKEN`**: Hugging Face authentication token
  - Required for Hugging Face Space deployment
  - Generate at: https://huggingface.co/settings/tokens
  - Required scope: `write` access

#### Environment Variables

The workflows automatically set:
- `MLFLOW_TRACKING_URI` — Points to local MLflow service in CI
- `DRIFT_DB_URI` — PostgreSQL connection string for integration tests

---

## Security & Production Features 🔒

### Security Improvements
- **Non-root Docker containers**: All Docker images run as non-root users
- **Input validation**: API enforces maximum input length (10,000 chars) to prevent DoS
- **Error handling**: Comprehensive error handling with proper logging
- **Health checks**: Docker health checks and Kubernetes-ready endpoints
- **Request ID tracking**: All API requests include request IDs for tracing

### Production-Ready Features
- **Health endpoints**: `/health` and `/ready` for orchestration systems
- **Global exception handling**: Catches and logs all unhandled exceptions
- **CORS support**: Configurable CORS middleware (configure `allow_origins` for production)
- **Structured logging**: Request context and error tracking
- **Temporary file cleanup**: Automatic cleanup of temporary files
- **Resource management**: Proper cleanup of MLflow artifacts and temp directories

### Docker Improvements
- **Optimized builds**: `.dockerignore` excludes unnecessary files
- **Layer caching**: Optimized Dockerfile layer ordering
- **Health checks**: Built-in health checks for containers
- **Security**: Non-root user execution
- **Multi-service stack**: Complete docker-compose setup with networking

## Development notes & housekeeping 💡
- **Dev deps added**: `SQLAlchemy`, `psycopg2-binary`, and `apache-airflow>=2.9,<3` (in `requirements-dev.txt`) to support local DAG testing and DB fallback.
- **models/**: keep out of git (add `models/` to `.gitignore`) and store production models in MLflow / S3 / artifact storage.
- **mlruns/**: local MLflow storage; safe to back up or delete if you want to clear local experiment history.
- **.pytest_cache/** and `__pycache__` directories: safe to delete (regenerable). Add them to `.gitignore` if you want.
- **Prometheus config**: `monitoring/prometheus.yml` is optional and can be removed without breaking the app.
- **Error handling**: All modules now include comprehensive error handling with informative messages.
- **Temporary files**: All temporary files are automatically cleaned up to prevent disk space issues.

---

## TODOs & Open questions ❓
- Decide on production MLflow backend (e.g., Postgres + S3/MinIO) and update `docker-compose.yml` with persistent stores.
- Add a small helper script to download the latest model from MLflow into `models/` at deployment time.
- Add a migration or creation script for the `drift_data` table used by the DAG DB fallback.
- Improve DAG to trigger remote training jobs and to include notifications/alerts on drift detection.
- ~~Set up CI/CD pipeline~~ ✅ **Completed**
- ~~Add Docker image builds and deployment~~ ✅ **Completed**
- ~~Add Hugging Face Space deployment~~ ✅ **Completed**
- ~~Add health check endpoints~~ ✅ **Completed**
- ~~Improve Docker security (non-root users)~~ ✅ **Completed**
- ~~Add comprehensive error handling~~ ✅ **Completed**
- ~~Fix temporary file cleanup~~ ✅ **Completed**
- ~~Complete docker-compose.yml with all services~~ ✅ **Completed**

---

## Contributing
Contributions welcome! Please open a PR with a clear description and include tests for new behavior. Follow the existing code style.

---

## License
MIT
