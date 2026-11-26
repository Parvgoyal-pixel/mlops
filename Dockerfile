# Dockerfile — MP126 Canteen Prediction API
# Image name/tag: mp126-api:1.0
# Author: G224 — Parv (S24CSEU1347)

FROM python:3.12-slim

WORKDIR /app

# Install dependencies BEFORE copying code (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artefacts
COPY app/ app/
COPY models/ models/
COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
