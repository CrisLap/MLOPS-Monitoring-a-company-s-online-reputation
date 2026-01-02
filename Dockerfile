# Base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git curl build-essential && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt huggingface_hub fasttext

# ARG for versioned model
ARG MODEL_TAG=latest

# Download model from Hugging Face Hub
RUN python - <<EOF
from huggingface_hub import hf_hub_download
import os

repo_id = "crislap/sentiment-model"  # HF model repo
tag = os.environ.get("MODEL_TAG", "latest")

# Path where the model will be stored
model_path = hf_hub_download(repo_id=repo_id, filename=f"sentiment_ft_{tag}.bin")

print(f"Model downloaded at: {model_path}")
EOF

# Copy API code
COPY . /app
WORKDIR /app

# Expose default port for Gradio / FastAPI
EXPOSE 7860

# Run API
CMD ["python", "app.py"]


# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1
