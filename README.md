---
title: Online Reputation API
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Monitoring a company's online reputation



## Project summary
MachineInnovators Inc. is a leader in developing scalable, production-ready machine learning applications. The main focus of the project is to integrate MLOps methodologies to facilitate the development, implementation, continuous monitoring, and retraining of sentiment analysis models. The goal is to enable the company to improve and monitor its reputation on social media through automatic sentiment analysis.

A compact sentiment analysis repository built with Hugging Face Transformers for training, MLflow for experiment tracking, and FastAPI for serving predictions.

---

## Table of contents
- [Key components](#key-components)
- [Quick start](#quick-start)
- [Setup](#setup)
- [Usage](#usage)
- [Deployment](#deployment)
- [Monitoring & Metrics](#monitoring--metrics)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Development notes & housekeeping](#development-notes--housekeeping)
- [TODOs & Open questions](#todos--open-questions)

---

## Key components 🔧
- `training/` — training pipeline:
  - `train.py` — trains a FastText model using autotune optimization, saves to `models/` as `.ftz` file, logs metrics to MLflow, and can upload to Hugging Face.
  - `data_loader.py` — loads TweetEval sentiment dataset from Hugging Face.
- `app/` — API & serving:
  - `main.py` — FastAPI app exposing `POST /predict`, `GET /health`, `GET /ready`, `GET /model`, and `GET /metrics` endpoints. Integrates with Hugging Face model repository.
  - `inference.py` — loads model from Hugging Face (`.ftz` files) or local files and provides fallback predictor.
  - `metrics.py` — Prometheus metric definitions.
  - `schemas.py` — Pydantic request/response models with input validation.
- `monitoring/` — `prometheus.yml` and `grafana_dashboard.json` (optional local monitoring assets).
- `Dockerfile` — API container image (runs on port 7860).
- `Dockerfile.training` — Training container image.
- `docker-compose.yml` — local development stack with API and training services.
- `.github/workflows/` — CI/CD automation:
  - `ci_cd.yml` — main CI/CD pipeline (validation, testing, Docker builds, model training with F1 threshold check, and Hugging Face deployment).
  - `cleanup_cache.yml` — cache cleanup workflow.
- `tests/` — unit tests for API, training, and metrics.

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

4. (Optional) Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

---

## Setup ⚙️
- **Recommended Python:** 3.10
- **Environment variables:**
  - `HF_MODEL_REPO` — Hugging Face model repository (default: `CrisLap/sentiment-model`).
  - `HF_TOKEN` — Hugging Face authentication token for model uploads and downloads.
  - `MLFLOW_TRACKING_URI` — optional MLflow tracking server URI for experiment logging (defaults to local file storage if not set).
  - `OUTPUT_DIR` — directory for model output (default: `models`).
- **Ports used:**
  - API: `7860` (default for Hugging Face Spaces)
  - Prometheus / Grafana: run locally or add services to compose as needed.

---

## Usage 📌

### Train locally
```bash
python -m training.train
```

Or with custom parameters:
```bash
python -m training.train --epoch 25 --lr 0.2 --wordNgrams 2 --dim 150
```

**Training Features:**
- Uses FastText **autotune** for automatic hyperparameter optimization (4-hour duration)
- Trains on TweetEval sentiment dataset
- Evaluates on test set and computes F1 score, precision, recall
- Saves model as `sentiment_ft.ftz` in `models/` directory
- Logs metrics to MLflow (if `MLFLOW_TRACKING_URI` is configured)
- Generates `metrics.json` with performance metrics

**Important:** Models are saved as `.ftz` files (FastText compressed format). Do not commit large binary models to the repo. Models are stored in Hugging Face model repository.

### Run the API (local)
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 7860
```

**API Endpoints:**
- **Root**: `GET /` — Redirects to API documentation (`/docs`)
- **Predict**: `POST /predict` — Sentiment analysis prediction
  ```bash
  curl -X POST http://localhost:7860/predict -H "Content-Type: application/json" -d '{"text":"I love this product"}'
  ```
  Response format:
  ```json
  {
    "label": "positive",
    "score": 0.95
  }
  ```
- **Model Info**: `GET /model` — Information about the loaded model
  ```bash
  curl http://localhost:7860/model
  ```
- **Health Check**: `GET /health` — Service health status
  ```bash
  curl http://localhost:7860/health
  ```
- **Readiness Check**: `GET /ready` — Service readiness (checks if model is loaded)
  ```bash
  curl http://localhost:7860/ready
  ```
- **Metrics**: `GET /metrics` — Prometheus metrics
  ```bash
  curl http://localhost:7860/metrics
  ```
- **API Documentation**: `GET /docs` — Interactive Swagger UI
  - Available at: http://localhost:7860/docs

**Model Loading:**
- The API automatically loads the latest `.ftz` model file from the Hugging Face repository specified by `HF_MODEL_REPO`
- Searches for files matching pattern `sentiment_ft*.ftz` and loads the most recent one
- Falls back gracefully if model download fails (returns error label)
- Models are cached locally by Hugging Face Hub

**Input Validation:**
- All inputs are validated using Pydantic schemas
- Text input is validated for proper format

### Run with Docker Compose
```bash
docker compose up --build
```

This starts the development stack:
- **API**: Sentiment analysis API on port 7860
  - Endpoints: http://localhost:7860
  - Documentation: http://localhost:7860/docs
- **Training**: Training service (runs training script)

**Services:**
- Models persist to `./models/` directory
- API service mounts local `app/` directory for development
- Training service can be run independently

**Start specific services:**
```bash
# Start only API
docker compose up api

# Run training once
docker compose run --rm training
```

---

## Deployment 🚀

This project includes automated CI/CD pipelines that handle building, testing, training models, and deploying to Hugging Face.

### Hugging Face Integration

The project integrates with Hugging Face for both model storage and API deployment:

#### Model Repository

- Trained models are automatically uploaded to a Hugging Face model repository
- Default repository: `CrisLap/sentiment-model` (configurable via `HF_MODEL_REPO`)
- Models are uploaded during CI/CD **only if F1 score meets threshold** (default: 0.60)
- Model files are saved as `.ftz` format (FastText compressed)
- The API automatically downloads the latest `.ftz` model file from the repository on startup
- Models are versioned in Hugging Face for reproducibility

#### Hugging Face Space

The API is automatically deployed to a Hugging Face Space on pushes to `main` branch.

**Automatic Deployment:**
- Deployment is triggered automatically on pushes to `main` branch
- The Space uses Docker SDK with the root `Dockerfile`
- Configuration is managed via `.huggingface.yaml`
- The Space runs on port 7860 (Hugging Face default)

**Accessing the Deployed Space:**
Once deployed, the Space will be available at:
```
https://huggingface.co/spaces/CrisLap/online-reputation-api
```

The Space provides:
- Interactive API documentation at `/docs`
- Prediction endpoint at `/predict`
- Prometheus metrics at `/metrics`
- Root redirect to documentation at `/`

**Manual Deployment:**

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
huggingface-cli upload CrisLap/online-reputation-api \
  Dockerfile \
  .huggingface.yaml \
  app/ \
  requirements.txt \
  --repo-type space
```

### Required GitHub Secrets

For automated deployments, configure the following secret in your GitHub repository:

- **`HF_TOKEN`** — Hugging Face authentication token
  - Required for model uploads and Space deployment
  - Generate at: https://huggingface.co/settings/tokens
  - Required scope: `write` access

---

## Monitoring & Metrics 📈
- `monitoring/prometheus.yml` is a simple example config that scrapes the API (useful with Docker Compose). **Note:** Update the port in `prometheus.yml` to `7860` if running locally. Removing it **does not** affect the API; it only removes the convenience config for running Prometheus locally.
- Grafana dashboard JSON in `monitoring/grafana_dashboard.json` contains panels for request rate, latency, and sentiment distribution.
- Key metrics exposed by the API:
  - `sentiment_requests_total` (counter) — Total number of prediction requests
  - `sentiment_request_latency_seconds` (histogram) — Request latency distribution
  - `sentiment_predictions_total{sentiment}` (counter) — Predictions per sentiment label (negative, neutral, positive)

---

## Testing ✅

This project includes unit tests and higher-level tests for the API, training, and metrics. Below are concrete commands and quick examples to run and extend the test suite.

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
Some tests interact with the running API:
- Start the API locally in one terminal:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 7860
```
- In another terminal run the runtime tests:
```bash
pytest tests/test_api_runtime.py -q
```
Notes:
- Tests attempt to isolate external dependencies with mocks
- `test_api_runtime.py` exercises the running API and may require the model artifact in `models/` or the API to run in fallback mode
- Tests verify Prometheus metrics endpoints

### Test design tips
- Prefer unit tests with mocks for network/Hugging Face API calls to keep CI fast and deterministic
- Use small fixtures for model files to speed up tests
- Mock Hugging Face API calls in tests that don't require actual model downloads

### Continuous Integration

This project uses GitHub Actions for automated testing and validation. The CI pipeline runs on every push, pull request, and on a daily schedule, executing:

- **Code Validation**: Python compilation checks
- **Code Quality**: Black formatting checks and Flake8 linting
- **Unit Tests**: Full pytest test suite
- **Docker Builds**: Both API and training images are built
- **Model Training**: Trains model and uploads to Hugging Face
- **Space Deployment**: Deploys API to Hugging Face Space

See the [CI/CD Pipeline](#cicd-pipeline) section for detailed workflow information.

---

## CI/CD Pipeline 🔄

This project includes automated CI/CD using GitHub Actions. The workflow ensures code quality, runs tests, builds Docker images, trains models, and deploys to Hugging Face.

### Main CI/CD Workflow (`ci_cd.yml`)

The workflow runs on:
- **Push** to `main` branch
- **Pull requests** to `main` branch
- **Manual dispatch** (workflow_dispatch)
- **Scheduled**: Daily at 1 AM UTC

#### Workflow Jobs

1. **Validate**
   - Installs dependencies
   - Compiles Python code to check for syntax errors

2. **Lint & Test**
   - Black code formatter check
   - Flake8 linting
   - Runs full pytest test suite

3. **Build**
   - Builds API Docker image (`Dockerfile`)
   - Builds training Docker image (`Dockerfile.training`)

4. **Train and Push Model**
   - Runs training script using FastText autotune optimization
   - Evaluates model and computes F1 score
   - **Conditionally uploads** trained model to Hugging Face repository only if F1 score ≥ 0.60
   - Model files are saved as `.ftz` format (FastText compressed)
   - Model is available at: `CrisLap/sentiment-model`

5. **Deploy Space**
   - Creates Hugging Face Space if it doesn't exist
   - Deploys API to Hugging Face Space
   - Updates Space with latest code and Dockerfile
   - Space is available at: `CrisLap/online-reputation-api`

### Workflow Features

- **Sequential Execution**: Jobs run in sequence to ensure proper dependencies
- **Model Versioning**: Models are versioned in Hugging Face repository
- **Automatic Deployment**: Space is automatically updated on main branch pushes
- **Docker Builds**: Both API and training images are built and validated

### Viewing Workflow Runs

- Navigate to the **Actions** tab in your GitHub repository
- View workflow runs, logs, and artifacts
- Re-run failed workflows or manually trigger workflows

### Required Configuration

#### GitHub Secrets

Configure the following secret in your repository settings:

- **`HF_TOKEN`**: Hugging Face authentication token
  - Required for model uploads and Space deployment
  - Generate at: https://huggingface.co/settings/tokens
  - Required scope: `write` access

#### Environment Variables (in workflow)

The workflow uses these environment variables:
- `HF_MODEL_REPO`: Hugging Face model repository (default: `CrisLap/sentiment-model`)
- `HF_SPACE_REPO`: Hugging Face Space repository (default: `CrisLap/online-reputation-api`)

---

## Security & Production Features 🔒

### Security Features
- **Input validation**: All API inputs are validated using Pydantic schemas
- **Error handling**: Comprehensive error handling with proper logging
- **Model caching**: Models are cached locally to reduce Hugging Face API calls
- **Fallback mechanisms**: API falls back to local models if Hugging Face download fails

### Production-Ready Features
- **Hugging Face Integration**: Automatic model loading from Hugging Face repositories
- **Model versioning**: Models are versioned in Hugging Face for reproducibility
- **API documentation**: Interactive Swagger UI at `/docs`
- **Metrics endpoint**: Prometheus-compatible metrics at `/metrics`
- **Docker support**: Containerized deployment for easy scaling

### Docker Features
- **Optimized builds**: `.dockerignore` excludes unnecessary files
- **Layer caching**: Optimized Dockerfile layer ordering for faster builds
- **Multi-stage support**: Separate Dockerfiles for API and training
- **Development mode**: Docker Compose supports volume mounting for development

## Development notes & housekeeping 💡
- **models/**: Keep out of git (already in `.gitignore`). Models are stored in Hugging Face model repository as `.ftz` files.
- **.pytest_cache/** and `__pycache__` directories: Safe to delete (regenerable). Already in `.gitignore`.
- **Prometheus config**: `monitoring/prometheus.yml` is optional and can be removed without breaking the app.
- **Hugging Face tokens**: Store `HF_TOKEN` securely. Never commit tokens to the repository.
- **Docker volumes**: Models directory is mounted in docker-compose for persistence between container runs.

---

## TODOs & Open questions ❓
- Implement data drift detection and monitoring
- Add model performance monitoring and alerting
- Add automated model rollback mechanism
- Implement A/B testing for model versions
- Add model evaluation metrics tracking over time

---

## Contributing
Contributions welcome! Please open a PR with a clear description and include tests for new behavior. Follow the existing code style.

---

## License
MIT
