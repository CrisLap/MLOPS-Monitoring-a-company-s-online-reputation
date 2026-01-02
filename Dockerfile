FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y git curl build-essential && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt huggingface_hub fasttext

# Copy app
COPY . /app
WORKDIR /app

# Runtime environment
ENV MODEL_TAG=latest
ENV HF_TOKEN=""

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]




# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1
