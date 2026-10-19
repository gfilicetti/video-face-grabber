# Cloud Run production container for Video Face Grab
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    INSIGHTFACE_HOME=/app/.insightface

# Install system libraries required by OpenCV and InsightFace build/runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the InsightFace buffalo_sc ONNX model during image build for fast Cloud Run cold starts
RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640, 640))"

COPY app ./app
COPY pyproject.toml .

# Run as non-root user on Cloud Run
RUN useradd --create-home appuser && \
    mkdir -p /app/local_gcs_bucket /app/tmp_uploads && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
