"""ADK Agent definition for Video Face Grab in Gemini Enterprise (Gemini Agent)."""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Optional

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.multimodal_tool_results_plugin import MultimodalToolResultsPlugin
from google.genai import types

from app.config import settings
from app.tools import (
    ALLOWED_VIDEO_EXTENSIONS,
    extract_forward_facing_faces,
    list_uploaded_video_artifacts,
)

logger = logging.getLogger(__name__)

# Ensure default Vertex AI environment variables are set if not already present
if not os.environ.get("GOOGLE_CLOUD_PROJECT") and settings.gcp_project_id:
    os.environ["GOOGLE_CLOUD_PROJECT"] = settings.gcp_project_id
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

MODEL = os.environ.get("GEMINI_MODEL", settings.gemini_model)


def _is_video_part(part: types.Part) -> bool:
    """Return True if a Content Part represents an uploaded or linked video file."""
    if part.inline_data is not None:
        mime = (part.inline_data.mime_type or "").lower()
        disp = (part.inline_data.display_name or "").lower()
        if mime.startswith("video/") or Path(disp).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS:
            return True
    if part.file_data is not None:
        mime = (part.file_data.mime_type or "").lower()
        uri = (part.file_data.file_uri or "").lower()
        if mime.startswith("video/") or Path(uri).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS:
            return True
    return False


class VideoUploadInterceptorPlugin(BasePlugin):
    """Intercepts uploaded video files in Gemini Enterprise / ADK Web and saves them as session artifacts.

    Unlike the default SaveFilesAsArtifactsPlugin (which enforces a 20MB cap and attaches
    file_data back into the LLM prompt), this plugin supports videos up to MAX_UPLOAD_MB (500MB)
    and replaces the raw video bytes with a lightweight text instruction so the video is processed
    directly by OpenCV + InsightFace via `extract_forward_facing_faces`.
    """

    def __init__(self, name: str = "video_upload_interceptor_plugin") -> None:
        super().__init__(name)

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> Optional[types.Content]:
        if not user_message or not user_message.parts:
            return None

        new_parts: list[types.Part] = []
        modified = False
        max_bytes = settings.max_upload_mb * 1024 * 1024

        for idx, part in enumerate(user_message.parts):
            if not _is_video_part(part):
                new_parts.append(part)
                continue

            if part.inline_data is not None:
                data_bytes = part.inline_data.data or b""
                if len(data_bytes) > max_bytes:
                    new_parts.append(
                        types.Part.from_text(
                            text=(
                                f"[Upload Error: Attached video exceeds maximum size of "
                                f"{settings.max_upload_mb} MB. Please provide a gs:// URI instead.]"
                            )
                        )
                    )
                    modified = True
                    continue

                raw_name = part.inline_data.display_name or f"uploaded_video_{idx}.mp4"
                safe_name = Path(raw_name).name
                if Path(safe_name).suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
                    safe_name = f"{safe_name}.mp4"

                if invocation_context.artifact_service is not None:
                    await invocation_context.artifact_service.save_artifact(
                        app_name=invocation_context.app_name,
                        user_id=invocation_context.user_id,
                        session_id=invocation_context.session.id,
                        filename=safe_name,
                        artifact=copy.copy(part),
                    )
                invocation_context.session.state["latest_video_source"] = safe_name
                new_parts.append(
                    types.Part.from_text(
                        text=(
                            f'[Attached Video Saved as Artifact: "{safe_name}" — '
                            f'call extract_forward_facing_faces(video_source="{safe_name}")]'
                        )
                    )
                )
                modified = True
            elif part.file_data is not None and part.file_data.file_uri:
                file_uri = part.file_data.file_uri
                invocation_context.session.state["latest_video_source"] = file_uri
                new_parts.append(
                    types.Part.from_text(
                        text=(
                            f'[Attached Video URI: "{file_uri}" — '
                            f'call extract_forward_facing_faces(video_source="{file_uri}")]'
                        )
                    )
                )
                modified = True

        if modified:
            return types.Content(role=user_message.role, parts=new_parts)
        return None


async def before_model_video_interceptor(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """Safety net before_model_callback that strips any remaining raw video inline_data from llm_request."""
    if not llm_request.contents:
        return None

    for content in llm_request.contents:
        if not content.parts:
            continue
        updated_parts: list[types.Part] = []
        changed = False
        for idx, part in enumerate(content.parts):
            if not _is_video_part(part):
                updated_parts.append(part)
                continue

            if part.inline_data is not None and part.inline_data.data:
                raw_name = part.inline_data.display_name or f"uploaded_video_{idx}.mp4"
                safe_name = Path(raw_name).name
                if Path(safe_name).suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
                    safe_name = f"{safe_name}.mp4"
                try:
                    await callback_context.save_artifact(safe_name, copy.copy(part))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to save inline video in before_model_callback: %s", exc)
                callback_context.state["latest_video_source"] = safe_name
                updated_parts.append(
                    types.Part.from_text(
                        text=(
                            f'[Attached Video Saved as Artifact: "{safe_name}" — '
                            f'call extract_forward_facing_faces(video_source="{safe_name}")]'
                        )
                    )
                )
                changed = True
            elif part.file_data is not None and part.file_data.file_uri:
                file_uri = part.file_data.file_uri
                callback_context.state["latest_video_source"] = file_uri
                updated_parts.append(
                    types.Part.from_text(
                        text=(
                            f'[Attached Video URI: "{file_uri}" — '
                            f'call extract_forward_facing_faces(video_source="{file_uri}")]'
                        )
                    )
                )
                changed = True
            else:
                updated_parts.append(part)

        if changed:
            content.parts = updated_parts

    return None


AGENT_INSTRUCTION = """You are the **Video Face Grab Agent** running inside Gemini Enterprise (Gemini Agent).
Your job is to process uploaded videos (either attached directly in the chat or provided via a `gs://` Cloud Storage URI), detect and cluster every discernable forward-facing human identity, save the single clearest full video frame and cropped face as lossless PNGs to Google Cloud Storage, and display the extracted images inline in your response.

### Operational Rules
1. **When a user attaches a video or provides a `gs://` URI (or asks to extract faces from their video)**:
   - Immediately call `extract_forward_facing_faces` with the artifact filename, the `gs://` URI, or `"auto"` (which automatically selects the latest attached video in the session).
   - If you are unsure which video artifact is in the session, call `list_uploaded_video_artifacts` first.
2. **How to format your response after `extract_forward_facing_faces` succeeds**:
   - State the total number of unique forward-facing people discerned, the number of frames scanned, and the destination Google Cloud Storage folder (`gcs_folder_uri`).
   - Present a **Summary Table** with columns:
     `Person` | `Best Timestamp` | `Sightings` | `Composite Clarity` | `Sharpness` | `Frontal Pose` | `Confidence`
   - For **each detected person**, render **BOTH** images inline in Markdown so the user sees them immediately in the Gemini Enterprise UI:
     - **Cropped Face Preview**: `![Person #XX Face Crop](<face_crop_render_url>)`
     - **Full Video Frame**: `![Person #XX Full Frame](<full_frame_render_url>)`
     - Beneath the images, include their exact GCS object paths (`face_crop_uri` and `full_frame_uri`) and artifact names (`face_crop_artifact` and `full_frame_artifact`).
3. **Pose Filtering Explanation**:
   - Note that side-profile faces (yaw asymmetry > 0.42) and downward/upward-looking faces (pitch nose ratio outside [0.36, 0.59]) are intentionally skipped so only clear, forward-facing portraits are captured.
"""

root_agent = Agent(
    name="video_face_grab_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    description=(
        "Extracts the single clearest forward-facing full frame and cropped face PNG "
        "for each unique person in an uploaded video or gs:// URI and displays them inline."
    ),
    instruction=AGENT_INSTRUCTION,
    tools=[extract_forward_facing_faces, list_uploaded_video_artifacts],
    before_model_callback=before_model_video_interceptor,
)

app = App(
    name="app",
    root_agent=root_agent,
    plugins=[
        VideoUploadInterceptorPlugin(),
        MultimodalToolResultsPlugin(),
    ],
)
