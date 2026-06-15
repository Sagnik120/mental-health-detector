# deployment/docker/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY outputs/checkpoints/best_model.pt ./outputs/checkpoints/best_model.pt

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

ENV MODEL_NAME="mental/mental-bert-base-uncased"
ENV CHECKPOINT_PATH="outputs/checkpoints/best_model.pt"
ENV MAX_LEN=256

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
