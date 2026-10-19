# Video Face Grab

A Cloud Run web application that accepts an uploaded video file, detects and clusters every discernable human face across the video using RetinaFace + ArcFace (`insightface`), selects the single sharpest and clearest frame for each person, and uploads both the **full video frame** and **cropped face preview** as lossless `.png` files to Google Cloud Storage under `<video-filename>-<unique-number-suffix>/`.

## Features
- **Identity Clustering (1 Person = 1 Best Frame Set)**: Uses 512-D ArcFace embeddings and cosine similarity clustering so each human appears once, even across thousands of frames.
- **Composite Clarity Scoring**: Evaluates every face sighting on Laplacian sharpness ($0.35$), 5-point landmark frontal symmetry ($0.30$), detector confidence ($0.20$), and relative face size ($0.15$).
- **Real-Time SSE UI**: Streams frame scanning progress, live face discoveries, clarity upgrades, and GCS upload completions over Server-Sent Events (`EventSource`).
- **Dual PNG Exports to GCS**: Saves both `person_XX_full_frame.png` and `person_XX_face_crop.png` to `gs://<GCS_BUCKET_NAME>/<video-stem>-<unique-number-suffix>/`.

## Local Development

```bash
# 1. Run tests
.venv/bin/pytest -q

# 2. Start the server (uses local_gcs_bucket/ fallback if GCS_BUCKET_NAME is unset)
.venv/bin/uvicorn app.main:app --reload --port 8080
```

## Deploy to GCP Cloud Run

```bash
chmod +x deploy_cloud_run.sh
GCS_BUCKET_NAME="your-target-gcs-bucket" GCP_PROJECT_ID="your-gcp-project" ./deploy_cloud_run.sh
```
