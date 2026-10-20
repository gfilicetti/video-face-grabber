# Unified Cloud Run & Vertex AI Agent Runtime production container for Video Face Grab
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    INSIGHTFACE_HOME=/app/.insightface

# Install system libraries required by OpenCV and InsightFace Cython build/runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runtime dependencies compatible with Linux x86_64 Python 3.11 and google-adk 2.x
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "fastapi>=0.133.0,<0.143.0" \
    "opentelemetry-api>=1.39.0,<=1.42.1" \
    "opentelemetry-sdk>=1.39.0,<=1.42.1" \
    "uvicorn[standard]>=0.34.0,<1.0.0" \
    "python-multipart>=0.0.9" \
    "python-dotenv>=1.0.0" \
    "numpy<2.0.0" \
    "Pillow>=10.0.0" \
    "google-cloud-storage>=2.16.0" \
    "google-adk>=2.7.0" \
    "google-cloud-aiplatform[adk,agent_engines]>=1.129.0" \
    "a2a-sdk[http-server]>=1.0.0,<2.0.0" \
    "pydantic>=2.12.0,<3.0.0" \
    "pydantic-settings>=2.2.0" \
    "onnxruntime>=1.17.0" \
    "insightface>=0.7.3" \
    "opencv-python-headless>=4.9.0,<4.11.0"

# Pre-download the InsightFace buffalo_sc ONNX model during image build for fast cold starts
RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_sc', root='/app/.insightface', providers=['CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640, 640))"

ARG AGENT_VERSION=1.0.0
ENV AGENT_VERSION=${AGENT_VERSION}

COPY app ./app
COPY pyproject.toml README.md agents-cli-manifest.yaml ./

# Run as non-root user on Cloud Run and Agent Runtime
RUN useradd --create-home appuser && \
    mkdir -p /app/local_gcs_bucket /app/tmp_uploads && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.fast_api_app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
