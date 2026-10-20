# Video Face Grab

**Video Face Grab** is a real-time computer vision web application built with **FastAPI**, **InsightFace (RetinaFace + ArcFace ONNX)**, **OpenCV**, and **Google Cloud Storage (GCS)**, packaged for stateless deployment on **Google Cloud Run**.

Upload any video file (`.mp4`, `.mov`, `.webm`, `.mkv`, `.avi`, `.m4v`) and the backend will:
1. Scan frames and detect every discernable human face/head (including individuals only ever seen in side-profile or turned at an angle).
2. Cluster sightings using 512-dimensional ArcFace identity embeddings so **each unique person is captured once**.
3. Continuously evaluate frontal pose and visual clarity (3D frontal symmetry, Laplacian sharpness, detector confidence, and face size) to lock onto the **single best, most frontal frame** per person.
4. Upload both the **full video frame** (`person_XX_full_frame.png`) and **padded face crop** (`person_XX_face_crop.png`) as lossless `.png` files to Google Cloud Storage under `<video-filename>-<unique-number-suffix>/`.
5. Stream live progress, face discoveries, clarity upgrades, and GCS object links to a responsive dark-mode web UI over **Server-Sent Events (SSE)**.

> For a deep dive into the 3D pose equations, composite clarity formula, clustering consolidation algorithm, and SSE lifecycle, see [**ARCHITECTURE.md**](./ARCHITECTURE.md).

---

## Key Features

- **Continuous 3D Frontal Pose Ranking**: Uses 5-point facial landmarks (eyes, nose tip, mouth corners) to score yaw asymmetry, vertical pitch ratio, and head roll angle on a `0–100` scale—ensuring **everyone in the video is captured** (even if only ever seen in profile) while guaranteeing that square-to-the-camera shots score highest and profile shots score lowest.
- **1 Person = 1 Best Frame Set**: Dual-anchor cosine similarity matching (against both the running cluster centroid and the best frame's embedding) plus an end-of-video cluster consolidation pass prevents duplicate identity splits across camera cuts and wide-angle profile shots.
- **Composite Clarity Scoring (`0–100`)**:
  $$\text{Score} = 0.45 \cdot \text{FrontalPose} + 0.30 \cdot \text{Sharpness} + 0.15 \cdot \text{Confidence} + 0.10 \cdot \text{RelativeSize}$$
- **Dual Lossless PNG Output**: Saves both the uncropped full video frame and a $35\%$-padded portrait crop per person to `gs://<GCS_BUCKET_NAME>/<video-stem>-<unique-number-suffix>/`.
- **Zero-Dependency Local Fallback**: Runs out-of-the-box without GCP credentials by automatically falling back to `./local_gcs_bucket/` when `GCS_BUCKET_NAME` is unset.

---

## Quickstart (Local Development)

### Prerequisites
- Python 3.11+
- `ffmpeg` / OpenCV runtime libraries (included automatically on macOS and in the provided `Dockerfile`)

### 1. Create a Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Automated Test Suite
```bash
pytest -q
```

### 3. Start the Local Server
```bash
# Runs in local storage mode (saves PNGs to ./local_gcs_bucket/)
uvicorn app.main:app --reload --port 8080
```
Open [http://localhost:8080](http://localhost:8080) in your browser and drag-and-drop a video file.

To test against a live Google Cloud Storage bucket locally:
```bash
export GCP_PROJECT_ID="your-gcp-project-id"
export GCS_BUCKET_NAME="your-gcs-bucket-name"
uvicorn app.main:app --reload --port 8080
```

---

## Configuration Reference

All settings in [`app/config.py`](./app/config.py) can be configured via environment variables or a `.env` file:

| Environment Variable | Default | Description |
| :--- | :---: | :--- |
| `GCP_PROJECT_ID` | `""` | Google Cloud project ID for GCS client initialization. |
| `GCS_BUCKET_NAME` | `""` | Target GCS bucket name. If empty, saves PNGs locally under `LOCAL_STORAGE_DIR`. |
| `LOCAL_STORAGE_DIR` | `local_gcs_bucket` | Local directory fallback when `GCS_BUCKET_NAME` is unset. |
| `FRAME_SAMPLE_FPS` | `3.0` | Number of frames per second of video to sample for face detection. |
| `SIMILARITY_THRESHOLD` | `0.40` | Cosine similarity threshold (`0..1`) for grouping two 512-D embeddings into the same identity cluster. |
| `MIN_FACE_CONFIDENCE` | `0.65` | Minimum RetinaFace detection score (`0..1`). |
| `MIN_FACE_SIZE_PX` | `44` | Minimum bounding-box width and height in pixels. |
| `REQUIRE_FORWARD_FACING` | `False` | When `False` (default), captures all detected faces/heads including profile-only individuals and ranks by frontalness. |
| `MIN_FRONTAL_SCORE` | `65.0` | Minimum combined 3D pose score (`0..100`) required to accept a face as forward-facing. |
| `MAX_YAW_ASYMMETRY` | `0.42` | Maximum horizontal nose/mouth asymmetry relative to inter-ocular half-width (~$20^\circ\text{–}25^\circ$ yaw tolerance). |
| `MAX_ROLL_DEGREES` | `28.0` | Maximum head tilt angle in degrees. |
| `MIN_NOSE_RATIO` | `0.36` | Minimum vertical nose-to-eye/mouth ratio (filters upward-looking faces). |
| `MAX_NOSE_RATIO` | `0.59` | Maximum vertical nose-to-eye/mouth ratio (filters downward-looking faces). |
| `FACE_CROP_PADDING` | `0.35` | Fractional padding added around the face bounding box for `person_XX_face_crop.png`. |
| `MAX_UPLOAD_MB` | `500` | Maximum uploaded video file size in megabytes. |

---

## Deploying to Google Cloud Run

The repository includes a production `Dockerfile` (which pre-downloads the `buffalo_sc` ONNX models during `docker build` to eliminate cold-start downloads) and a deployment helper script [`deploy_cloud_run.sh`](./deploy_cloud_run.sh).

### 1. Create a GCS Bucket (if you don't have one)
```bash
gcloud storage buckets create gs://YOUR_GCS_BUCKET_NAME \
  --project=YOUR_GCP_PROJECT_ID \
  --location=us-central1 \
  --uniform-bucket-level-access \
  --public-access-prevention
```

### 2. Deploy with One Command
```bash
chmod +x deploy_cloud_run.sh
GCP_PROJECT_ID="YOUR_GCP_PROJECT_ID" \
GCS_BUCKET_NAME="YOUR_GCS_BUCKET_NAME" \
REGION="us-central1" \
./deploy_cloud_run.sh
```

This builds the container via Cloud Build and deploys it to Cloud Run with `--cpu=2`, `--memory=4Gi`, and `--timeout=900` for streaming video jobs.

---

## Deploying to Gemini Enterprise (Gemini Agent)

The repository is also a full **Google Agent Development Kit (ADK)** project (`agents-cli-manifest.yaml`, `app/agent.py`, `app/tools.py`, `app/fast_api_app.py`) that can be deployed to **Vertex AI Agent Runtime** and registered with **Gemini Enterprise**.

### 1. Test Locally in ADK Playground
```bash
agents-cli playground
```

### 2. Deploy to Vertex AI Agent Runtime & Register with Gemini Enterprise
```bash
chmod +x deploy_gemini_enterprise.sh

# Deploy to Agent Runtime (and optionally pass GEMINI_ENTERPRISE_APP_ID to publish in one step)
GCP_PROJECT_ID="video-face-grab" \
GCS_BUCKET_NAME="video-face-grab-faces" \
REGION="us-central1" \
./deploy_gemini_enterprise.sh
```

Once you have created a Gemini Enterprise App in **Google Cloud Console → Gemini Enterprise → Apps**, register the deployed agent interactively or via flag:
```bash
agents-cli publish gemini-enterprise --interactive
# Or non-interactively:
agents-cli publish gemini-enterprise \
  --registration-type adk \
  --gemini-enterprise-app-id "projects/YOUR_PROJECT/locations/global/collections/default_collection/engines/YOUR_APP_ID" \
  --display-name "Video Face Grab"
```

---

## API Endpoints

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/` | Serves the responsive Studio Web UI. |
| `GET` | `/api/health` | Returns service health, active `gcs_bucket`, and `storage_mode` (`gcs` or `local`). |
| `POST` | `/api/jobs` | Accepts a `multipart/form-data` `file` upload, creates the `<video>-<suffix>` folder prefix, and starts background processing. |
| `GET` | `/api/jobs/{job_id}/stream` | Server-Sent Events (`text/event-stream`) feed of frame progress, face discoveries, clarity upgrades, and saved GCS URIs. |
| `GET` | `/api/jobs/{job_id}` | Returns a JSON snapshot of job status, discovered people, and completion summary. |
| `GET` | `/api/storage/{folder_name}/{file_name}` | Streams saved `.png` files from GCS (or local fallback) for in-browser full-frame inspection and download. |

---

## License & Third-Party Model Weights

- **Source Code**: Copyright 2026 Google LLC. Licensed under the [Apache License, Version 2.0](./LICENSE).
- **Third-Party Pretrained Models**: While the [`insightface`](https://github.com/deepinsight/insightface) Python package is MIT-licensed, the default `buffalo_sc` ONNX model weights downloaded at container build/runtime are distributed by InsightFace for **non-commercial research purposes only**. For commercial deployments, replace `INSIGHTFACE_MODEL_NAME` or the detector backend in [`app/vision/processor.py`](./app/vision/processor.py) with commercially licensed model weights.
- **Contributing & Security**: See [`CONTRIBUTING.md`](./CONTRIBUTING.md), [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md), and [`SECURITY.md`](./SECURITY.md).

---

## Disclaimer

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).
