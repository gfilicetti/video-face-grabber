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

"""ADK tools for extracting forward-facing human faces from uploaded videos or GCS URIs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import uuid
from google.adk.tools import ToolContext
from google.genai import types
from starlette.concurrency import run_in_threadpool
from app.config import Settings, settings
from app.storage.gcs_store import StorageService, generate_run_folder_name
from app.vision.processor import InsightFaceDetector, VideoFaceProcessor


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

_shared_processor: VideoFaceProcessor | None = None


def get_or_create_processor(app_settings: Settings | None = None) -> VideoFaceProcessor:
    """Return a singleton VideoFaceProcessor configured from environment settings."""
    global _shared_processor
    if _shared_processor is not None and app_settings is None:
        return _shared_processor

    cfg = app_settings or settings
    cfg.local_storage_dir.mkdir(parents=True, exist_ok=True)
    cfg.temp_upload_dir.mkdir(parents=True, exist_ok=True)

    bucket_name = (
        cfg.gcs_bucket_name
        or os.environ.get("GCS_BUCKET_NAME", "")
        or os.environ.get("LOGS_BUCKET_NAME", "")
    ).strip()
    project_id = (
        cfg.gcp_project_id
        or os.environ.get("GCP_PROJECT_ID", "")
        or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    ).strip()

    storage_service = StorageService(
        bucket_name=bucket_name,
        project_id=project_id,
        local_storage_dir=cfg.local_storage_dir,
    )
    detector = InsightFaceDetector(model_name=cfg.insightface_model_name)
    processor = VideoFaceProcessor(
        detector=detector,
        storage_service=storage_service,
        sample_fps=cfg.frame_sample_fps,
        similarity_threshold=cfg.similarity_threshold,
        min_face_confidence=cfg.min_face_confidence,
        min_face_size_px=cfg.min_face_size_px,
        min_sharpness_score=cfg.min_sharpness_score,
        require_forward_facing=cfg.require_forward_facing,
        min_frontal_score=cfg.min_frontal_score,
        max_yaw_asymmetry=cfg.max_yaw_asymmetry,
        max_roll_degrees=cfg.max_roll_degrees,
        min_nose_ratio=cfg.min_nose_ratio,
        max_nose_ratio=cfg.max_nose_ratio,
        min_sightings=cfg.min_sightings,
        face_crop_padding=cfg.face_crop_padding,
    )
    if app_settings is None:
        _shared_processor = processor
    return processor


def set_processor_override(processor: VideoFaceProcessor | None) -> None:
    """Override the singleton processor (used by unit/integration tests)."""
    global _shared_processor
    _shared_processor = processor


async def _resolve_video_to_temp_file(
    video_source: str,
    tool_context: ToolContext,
    storage_service: StorageService,
    temp_upload_dir: Path,
) -> tuple[Path, str, bool]:
    """Resolve an artifact name, 'auto', gs:// URI, or local path into a local video file.

    Returns:
        Tuple of (local_video_path, original_filename, should_delete_after).
    """
    temp_upload_dir.mkdir(parents=True, exist_ok=True)
    source = (video_source or "").strip()

    # 1. Check if 'auto' or empty -> look up state["latest_video_source"] or session artifacts
    if not source or source.lower() in ("auto", "latest", "attached", "upload", "uploaded"):
        state_source = str(tool_context.state.get("latest_video_source", "")).strip()
        if state_source:
            source = state_source
        else:
            artifact_names = await tool_context.list_artifacts()
            video_artifacts = [
                name
                for name in artifact_names
                if Path(name).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS
                or "video" in name.lower()
            ]
            if video_artifacts:
                source = video_artifacts[-1]

    if not source:
        raise ValueError(
            "No video file was found. Please attach a video file (.mp4, .mov, .webm) "
            "in the chat or provide a gs:// Cloud Storage URI."
        )

    # 2. Check if source is a gs:// URI
    if source.startswith("gs://"):
        original_name = Path(source.split("/")[-1] or "video.mp4").name
        suffix = Path(original_name).suffix.lower() or ".mp4"
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            suffix = ".mp4"
        temp_path = temp_upload_dir / f"adk_gcs_{uuid.uuid4().hex}{suffix}"
        await run_in_threadpool(storage_service.download_gcs_uri_to_path, source, temp_path)
        return temp_path, original_name, True

    # 3. Try loading from ADK ArtifactService first
    try:
        artifact_part = await tool_context.load_artifact(source)
    except Exception:  # noqa: BLE001
        artifact_part = None

    if artifact_part is None:
        # Check if any artifact matches basename
        try:
            for cand in await tool_context.list_artifacts():
                if cand == source or Path(cand).name == Path(source).name:
                    artifact_part = await tool_context.load_artifact(cand)
                    source = cand
                    break
        except Exception:  # noqa: BLE001
            artifact_part = None

    if artifact_part is not None:
        if artifact_part.inline_data and artifact_part.inline_data.data:
            original_name = Path(source).name
            suffix = Path(original_name).suffix.lower() or ".mp4"
            if suffix not in ALLOWED_VIDEO_EXTENSIONS:
                suffix = ".mp4"
            temp_path = temp_upload_dir / f"adk_art_{uuid.uuid4().hex}{suffix}"
            temp_path.write_bytes(artifact_part.inline_data.data)
            return temp_path, original_name, True
        if artifact_part.file_data and artifact_part.file_data.file_uri:
            file_uri = artifact_part.file_data.file_uri
            if file_uri.startswith("gs://"):
                original_name = Path(file_uri.split("/")[-1] or source or "video.mp4").name
                suffix = Path(original_name).suffix.lower() or ".mp4"
                if suffix not in ALLOWED_VIDEO_EXTENSIONS:
                    suffix = ".mp4"
                temp_path = temp_upload_dir / f"adk_furi_{uuid.uuid4().hex}{suffix}"
                await run_in_threadpool(
                    storage_service.download_gcs_uri_to_path, file_uri, temp_path
                )
                return temp_path, original_name, True

    # 4. Fallback: check if source is a safe local file inside the workspace or temp_upload_dir
    candidate_path = Path(source).resolve()
    workspace_root = Path.cwd().resolve()
    temp_root = temp_upload_dir.resolve()
    if candidate_path.is_file() and (
        candidate_path.is_relative_to(workspace_root)
        or candidate_path.is_relative_to(temp_root)
    ):
        if candidate_path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video extension '{candidate_path.suffix}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
            )
        return candidate_path, candidate_path.name, False

    raise ValueError(
        f"Could not locate video source '{source}'. Attach a video in chat or pass a valid gs:// URI."
    )


def _run_processor_sync(
    processor: VideoFaceProcessor,
    video_path: Path,
    folder_name: str,
) -> dict[str, Any]:
    """Run VideoFaceProcessor.process_video synchronously and collect final summary."""
    summary: dict[str, Any] = {}
    for ev in processor.process_video(video_path, folder_name):
        if ev.get("event") == "job_complete":
            summary = ev
    return summary


async def list_uploaded_video_artifacts(tool_context: ToolContext) -> dict[str, Any]:
    """Lists any video files uploaded or attached by the user in the current conversation session.

    Returns:
        dict with 'status', 'latest_video_source', and 'artifacts' list.
    """
    try:
        artifacts = await tool_context.list_artifacts()
    except Exception:  # noqa: BLE001
        artifacts = []
    latest = str(tool_context.state.get("latest_video_source", "")).strip()
    return {
        "status": "success",
        "latest_video_source": latest or (artifacts[-1] if artifacts else ""),
        "artifacts": artifacts,
    }


async def extract_forward_facing_faces(
    video_source: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Detects and clusters all human faces/heads in a video (including profile-only individuals), ranks each person's frames by frontalness and clarity, uploads the best full-frame and face-crop PNGs per person to Google Cloud Storage, and saves inline PNG artifacts for the UI.

    Args:
        video_source: The video to process. Pass 'auto' to process the user's latest attached video in the session, an artifact filename (e.g. 'clip.mp4'), or a Cloud Storage URI ('gs://bucket/path.mp4').

    Returns:
        dict containing 'status', 'folder_name', 'gcs_folder_uri', 'total_people_saved', and 'people' (with inline image URLs, artifact names, and clarity scores).
    """
    processor = get_or_create_processor()
    storage_service = processor.storage_service
    public_base_url = (
        os.environ.get("PUBLIC_BASE_URL", "").strip() or settings.public_base_url
    )

    temp_video_path: Path | None = None
    should_delete = False
    try:
        temp_video_path, original_filename, should_delete = await _resolve_video_to_temp_file(
            video_source=video_source,
            tool_context=tool_context,
            storage_service=storage_service,
            temp_upload_dir=settings.temp_upload_dir,
        )
        folder_name = generate_run_folder_name(original_filename)
        job_summary = await run_in_threadpool(
            _run_processor_sync,
            processor,
            temp_video_path,
            folder_name,
        )

        people_out: list[dict[str, Any]] = []
        for person in job_summary.get("people", []):
            pid = int(person["person_id"])
            crop_filename = f"person_{pid:02d}_face_crop.png"
            full_filename = f"person_{pid:02d}_full_frame.png"

            crop_artifact_name = f"{folder_name}_{crop_filename}"
            full_artifact_name = f"{folder_name}_{full_filename}"

            # Read lossless PNG bytes and register both as ADK session artifacts for native inline rendering
            crop_bytes = storage_service.read_stored_png_bytes(folder_name, crop_filename)
            full_bytes = storage_service.read_stored_png_bytes(folder_name, full_filename)

            await tool_context.save_artifact(
                crop_artifact_name,
                types.Part.from_bytes(data=crop_bytes, mime_type="image/png"),
            )
            await tool_context.save_artifact(
                full_artifact_name,
                types.Part.from_bytes(data=full_bytes, mime_type="image/png"),
            )

            crop_render_url = storage_service.get_renderable_image_url(
                folder_name=folder_name,
                file_name=crop_filename,
                public_base_url=public_base_url,
                ttl_seconds=settings.signed_url_ttl_seconds,
            )
            full_render_url = storage_service.get_renderable_image_url(
                folder_name=folder_name,
                file_name=full_filename,
                public_base_url=public_base_url,
                ttl_seconds=settings.signed_url_ttl_seconds,
            )

            people_out.append(
                {
                    "person_id": pid,
                    "label": person["label"],
                    "sightings_count": person["sightings_count"],
                    "best_frame_index": person["best_frame_index"],
                    "best_timestamp_sec": person["best_timestamp_sec"],
                    "quality": person["quality"],
                    "face_crop_uri": person["face_crop_uri"],
                    "full_frame_uri": person["full_frame_uri"],
                    "face_crop_artifact": crop_artifact_name,
                    "full_frame_artifact": full_artifact_name,
                    "face_crop_render_url": crop_render_url,
                    "full_frame_render_url": full_render_url,
                }
            )

        bucket_name = job_summary.get("bucket_name", "local-bucket")
        result_payload = {
            "status": "success",
            "original_filename": original_filename,
            "folder_name": folder_name,
            "bucket_name": bucket_name,
            "gcs_folder_uri": f"gs://{bucket_name}/{folder_name}/",
            "total_frames_scanned": job_summary.get("total_frames_scanned", 0),
            "sampled_frames": job_summary.get("sampled_frames", 0),
            "total_people_saved": len(people_out),
            "people": people_out,
        }
        tool_context.state["last_extraction_result"] = result_payload
        return result_payload
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_message": str(exc),
        }
    finally:
        if should_delete and temp_video_path is not None:
            try:
                if temp_video_path.exists():
                    temp_video_path.unlink()
            except OSError:
                pass
