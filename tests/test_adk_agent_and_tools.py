"""Unit and integration tests for the Gemini Enterprise ADK agent, video upload interceptor, and tools."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import cv2
from fastapi.testclient import TestClient
from google.adk.artifacts import InMemoryArtifactService
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import numpy as np

from app.agent import (
    VideoUploadInterceptorPlugin,
    before_model_video_interceptor,
    root_agent,
)
from app.storage.gcs_store import StorageService
from app.tools import (
    extract_forward_facing_faces,
    list_uploaded_video_artifacts,
    set_processor_override,
)
from app.vision.processor import DetectedFace, VideoFaceProcessor


def _write_synthetic_video(video_path: Path, num_frames: int = 4) -> bytes:
    """Create a small synthetic MP4 file and return its raw bytes."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 4.0, (320, 240))
    assert writer.isOpened()
    try:
        for i in range(num_frames):
            frame = np.full((240, 320, 3), 35, dtype=np.uint8)
            step = 6 if i == 2 else 12
            for y in range(20, 200, step):
                for x in range(15, 280, step):
                    if ((x // step) + (y // step)) % 2 == 0:
                        frame[y : y + step, x : x + step] = (235, 195, 155)
            writer.write(frame)
    finally:
        writer.release()
    return video_path.read_bytes()


def _make_fake_processor(tmp_path: Path, gcs_client: MagicMock | None = None) -> VideoFaceProcessor:
    rng = np.random.default_rng(19)
    emb_1 = rng.standard_normal(512).astype(np.float32)
    emb_1 /= np.linalg.norm(emb_1)
    emb_2 = rng.standard_normal(512).astype(np.float32)
    emb_2 /= np.linalg.norm(emb_2)

    frontal_kps_1 = np.array(
        [[130.0, 100.0], [190.0, 100.0], [160.0, 125.0], [140.0, 155.0], [180.0, 155.0]],
        dtype=np.float32,
    )
    frontal_kps_2 = np.array(
        [[35.0, 65.0], [80.0, 65.0], [57.5, 85.0], [42.0, 110.0], [73.0, 110.0]],
        dtype=np.float32,
    )

    def fake_detector(frame_bgr: np.ndarray) -> list[DetectedFace]:
        return [
            DetectedFace(
                bbox=np.array([100.0, 60.0, 220.0, 180.0], dtype=np.float32),
                kps=frontal_kps_1,
                det_score=0.96,
                embedding=emb_1,
            ),
            DetectedFace(
                bbox=np.array([20.0, 40.0, 95.0, 130.0], dtype=np.float32),
                kps=frontal_kps_2,
                det_score=0.92,
                embedding=emb_2,
            ),
        ]

    storage_service = StorageService(
        bucket_name="video-face-grab-faces" if gcs_client is not None else "",
        project_id="video-face-grab",
        local_storage_dir=tmp_path / "storage",
        gcs_client=gcs_client,
    )
    return VideoFaceProcessor(
        detector=fake_detector,
        storage_service=storage_service,
        sample_fps=4.0,
    )


class _DummyToolContext:
    """Minimal ToolContext test double backed by ADK InMemoryArtifactService."""

    def __init__(self, artifact_service: InMemoryArtifactService) -> None:
        self.state: dict = {}
        self._artifact_service = artifact_service
        self._app_name = "app"
        self._user_id = "user_1"
        self._session_id = "sess_1"

    async def save_artifact(self, filename: str, artifact: types.Part) -> int:
        return await self._artifact_service.save_artifact(
            app_name=self._app_name,
            user_id=self._user_id,
            session_id=self._session_id,
            filename=filename,
            artifact=artifact,
        )

    async def load_artifact(self, filename: str, version: int | None = None) -> types.Part | None:
        return await self._artifact_service.load_artifact(
            app_name=self._app_name,
            user_id=self._user_id,
            session_id=self._session_id,
            filename=filename,
            version=version,
        )

    async def list_artifacts(self) -> list[str]:
        return await self._artifact_service.list_artifact_keys(
            app_name=self._app_name,
            user_id=self._user_id,
            session_id=self._session_id,
        )


def test_video_upload_interceptor_and_extract_faces_tool(tmp_path: Path) -> None:
    async def _run() -> None:
        video_path = tmp_path / "interview_upload.mp4"
        video_bytes = _write_synthetic_video(video_path)

        artifact_service = InMemoryArtifactService()
        tool_ctx = _DummyToolContext(artifact_service)

        # 1. Simulate before_model_video_interceptor stripping inline video bytes and saving artifact
        video_part = types.Part(
            inline_data=types.Blob(
                mime_type="video/mp4",
                data=video_bytes,
                display_name="interview_upload.mp4",
            )
        )
        llm_req = LlmRequest(
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text="Please grab the faces from this video"),
                        video_part,
                    ],
                )
            ]
        )
        await before_model_video_interceptor(tool_ctx, llm_req)  # type: ignore[arg-type]

        # Ensure raw inline_data was replaced by text pointer and saved into ArtifactService
        assert llm_req.contents[0].parts[1].inline_data is None
        assert "interview_upload.mp4" in (llm_req.contents[0].parts[1].text or "")
        assert tool_ctx.state["latest_video_source"] == "interview_upload.mp4"

        listed = await list_uploaded_video_artifacts(tool_ctx)  # type: ignore[arg-type]
        assert listed["latest_video_source"] == "interview_upload.mp4"
        assert "interview_upload.mp4" in listed["artifacts"]

        # 2. Run extract_forward_facing_faces with 'auto' resolution
        processor = _make_fake_processor(tmp_path)
        set_processor_override(processor)
        try:
            res = await extract_forward_facing_faces("auto", tool_ctx)  # type: ignore[arg-type]
        finally:
            set_processor_override(None)

        assert res["status"] == "success"
        assert res["total_people_saved"] == 2
        assert len(res["people"]) == 2

        p1 = res["people"][0]
        p2 = res["people"][1]
        assert p1["face_crop_render_url"].endswith("/person_01_face_crop.png")
        assert p1["full_frame_render_url"].endswith("/person_01_full_frame.png")
        assert p2["face_crop_render_url"].endswith("/person_02_face_crop.png")
        assert p2["full_frame_render_url"].endswith("/person_02_full_frame.png")

        # Verify all 4 PNGs were saved as ADK session artifacts for inline rendering
        saved_artifacts = await tool_ctx.list_artifacts()
        assert p1["face_crop_artifact"] in saved_artifacts
        assert p1["full_frame_artifact"] in saved_artifacts
        assert p2["face_crop_artifact"] in saved_artifacts
        assert p2["full_frame_artifact"] in saved_artifacts

        crop_part = await tool_ctx.load_artifact(p1["face_crop_artifact"])
        assert crop_part is not None
        assert crop_part.inline_data is not None
        assert crop_part.inline_data.mime_type == "image/png"
        assert crop_part.inline_data.data.startswith(b"\x89PNG\r\n\x1a\n")

    asyncio.run(_run())


def test_extract_faces_from_gcs_uri(tmp_path: Path) -> None:
    async def _run() -> None:
        video_path = tmp_path / "source_clip.mp4"
        video_bytes = _write_synthetic_video(video_path)

        mock_gcs = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_gcs.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.generate_signed_url.return_value = "https://storage.googleapis.com/signed-png-url"

        def _fake_download(dest_filename: str) -> None:
            Path(dest_filename).write_bytes(video_bytes)

        mock_blob.download_to_filename.side_effect = _fake_download

        artifact_service = InMemoryArtifactService()
        tool_ctx = _DummyToolContext(artifact_service)
        processor = _make_fake_processor(tmp_path, gcs_client=mock_gcs)
        set_processor_override(processor)
        try:
            res = await extract_forward_facing_faces(
                "gs://video-face-grab-faces/uploads/keynote.mp4",
                tool_ctx,  # type: ignore[arg-type]
            )
        finally:
            set_processor_override(None)

        assert res["status"] == "success"
        assert res["total_people_saved"] == 2
        assert res["people"][0]["face_crop_render_url"] == "https://storage.googleapis.com/signed-png-url"
        assert res["people"][0]["full_frame_render_url"] == "https://storage.googleapis.com/signed-png-url"

    asyncio.run(_run())


def test_fast_api_app_exposes_adk_a2a_reasoning_engine_and_studio_routes() -> None:
    from app.fast_api_app import app as unified_app

    assert root_agent.name == "video_face_grab_agent"
    assert isinstance(VideoUploadInterceptorPlugin(), VideoUploadInterceptorPlugin)

    with TestClient(unified_app) as client:
        # 1. Studio health route
        health_resp = client.get("/api/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "ok"

        # 2. A2A agent card route
        card_resp = client.get("/a2a/app/.well-known/agent-card.json")
        assert card_resp.status_code == 200
        card = card_resp.json()
        assert card["name"] == "video_face_grab_agent"

        # 3. Studio root UI route
        root_resp = client.get("/")
        assert root_resp.status_code == 200
        assert "Video Face Grab" in root_resp.text
