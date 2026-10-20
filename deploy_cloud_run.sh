#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Deploy Video Face Grab to Google Cloud Run
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-video-face-grab}"
GCS_BUCKET_NAME="${GCS_BUCKET_NAME:-}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Error: GCP_PROJECT_ID is not set and no active gcloud project was found." >&2
  exit 1
fi

if [[ -z "${GCS_BUCKET_NAME}" ]]; then
  echo "Error: GCS_BUCKET_NAME must be set to your target Cloud Storage bucket." >&2
  echo "Usage: GCS_BUCKET_NAME=my-video-faces-bucket ./deploy_cloud_run.sh" >&2
  exit 1
fi

echo "Deploying ${SERVICE_NAME} to Cloud Run in project ${PROJECT_ID} (${REGION})..."
echo "Target GCS Bucket: gs://${GCS_BUCKET_NAME}"

CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }gcs-skills gcs-skills/1.0 (skill:google-cloud-storage-basics)" \
gcloud run deploy "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --source=. \
  --cpu=2 \
  --memory=4Gi \
  --cpu-boost \
  --timeout=900 \
  --concurrency=4 \
  --allow-unauthenticated \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCS_BUCKET_NAME=${GCS_BUCKET_NAME}"
