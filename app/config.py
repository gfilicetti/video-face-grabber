"""Application configuration and environment settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings for Video Face Grab."""

    app_name: str = "Video Face Grab"
    gcp_project_id: str = ""
    gcs_bucket_name: str = ""
    local_storage_dir: Path = Path("local_gcs_bucket")
    temp_upload_dir: Path = Path("tmp_uploads")

    # Video & Face Pipeline tuning
    frame_sample_fps: float = 3.0
    similarity_threshold: float = 0.42
    min_face_confidence: float = 0.60
    min_face_size_px: int = 44
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
