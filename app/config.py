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

"""Application configuration and environment settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings for Video Face Grab."""

    app_name: str = "Video Face Grab"
    gcp_project_id: str = ""
    gcs_bucket_name: str = ""
    public_base_url: str = "https://video-face-grab-994000852218.us-central1.run.app"
    signed_url_ttl_seconds: int = 3600
    gemini_model: str = "gemini-2.5-flash"
    local_storage_dir: Path = Path("local_gcs_bucket")
    temp_upload_dir: Path = Path("tmp_uploads")

    # Video & Face Pipeline tuning
    frame_sample_fps: float = 3.0
    similarity_threshold: float = 0.40
    min_face_confidence: float = 0.65
    min_face_size_px: int = 44
    min_sharpness_score: float = 0.0
    require_forward_facing: bool = False
    min_frontal_score: float = 65.0
    max_yaw_asymmetry: float = 0.42
    max_roll_degrees: float = 28.0
    min_nose_ratio: float = 0.36
    max_nose_ratio: float = 0.59
    min_sightings: int = 1
    face_crop_padding: float = 0.35
    max_upload_mb: int = 500
    insightface_model_name: str = "buffalo_sc"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
