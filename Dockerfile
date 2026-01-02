# Base image
FROM python:3.10-slim

# ARG per passare il modello da GH Actions
ARG MODEL_TAG=latest

# Environment variables
ENV MODEL_TAG=${MODEL_TAG}
ENV HF_TOKEN=${HF_TOKEN}

# Set workdir
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt \
    && pip install fasttext huggingface_hub uvicorn gradio

# Copy app source
COPY app ./app

# Optional: copy monitoring or extra files
COPY monitoring ./monitoring

# Download HF model (if available)
RUN python - <<EOF || echo "WARNING: Fallback predictor will be used"
import os
from huggingface_hub import hf_hub_download
import fasttext

_model_loaded = False
_model = None

repo_id = "crislap/sentiment-model"
tag = os.environ.get("MODEL_TAG", "latest")
hf_token = os.environ.get("HF_TOKEN")

try:
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"sentiment_ft_{tag}.bin",
        token=hf_token
    )
    _model = fasttext.load_model(model_path)
    _model_loaded = True
    print(f"Model loaded from {model_path}")
except Exception as e:
    print(f"WARNING: Could not load model ({e}), using fallback predictor")
EOF

# Expose port for Gradio
EXPOSE 7860
# Expose port for FastAPI
EXPOSE 8000

# Run both FastAPI and Gradio
# We will use uvicorn for FastAPI and launch Gradio separately
CMD ["python", "-m", "app.main"]



# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1
