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

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-video-face-grab}"
REGION="${REGION:-us-central1}"
BUCKET_NAME="${GCS_BUCKET_NAME:-video-face-grab-faces}"
PUBLIC_URL="${PUBLIC_BASE_URL:-https://video-face-grab-994000852218.us-central1.run.app}"
GEMINI_APP_ID="${GEMINI_ENTERPRISE_APP_ID:-}"

echo "=================================================================="
echo " Deploying Video Face Grab Agent to Vertex AI Agent Runtime"
echo " Project: ${PROJECT_ID} | Region: ${REGION} | Bucket: gs://${BUCKET_NAME}"
echo "=================================================================="

agents-cli deploy \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --memory 4Gi \
  --cpu 2 \
  --no-confirm-project \
  --update-env-vars "GCS_BUCKET_NAME=${BUCKET_NAME},GCP_PROJECT_ID=${PROJECT_ID},PUBLIC_BASE_URL=${PUBLIC_URL}"

if [[ -n "${GEMINI_APP_ID}" ]]; then
  echo "=================================================================="
  echo " Registering Agent with Gemini Enterprise App: ${GEMINI_APP_ID}"
  echo "=================================================================="
  agents-cli publish gemini-enterprise \
    --registration-type adk \
    --gemini-enterprise-app-id "${GEMINI_APP_ID}" \
    --display-name "Video Face Grab" \
    --description "Extracts the clearest forward-facing full frame and cropped face PNG for each unique person in an uploaded video and displays them inline." \
    --tool-description "Detects and clusters forward-facing human faces in uploaded videos or gs:// URIs, saves lossless PNGs to GCS, and renders them inline."
else
  echo ""
  echo "ℹ️  Agent Runtime deployment complete (see deployment_metadata.json)."
  echo "To register with Gemini Enterprise once your Gemini Enterprise App is created in Cloud Console:"
  echo "  GEMINI_ENTERPRISE_APP_ID=\"projects/${PROJECT_ID}/locations/global/collections/default_collection/engines/YOUR_APP_ID\" ./deploy_gemini_enterprise.sh"
  echo "Or run interactively:"
  echo "  agents-cli publish gemini-enterprise --interactive"
fi
