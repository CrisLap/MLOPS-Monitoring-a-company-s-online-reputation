FROM python:3.10-slim

WORKDIR /app
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

COPY app app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

