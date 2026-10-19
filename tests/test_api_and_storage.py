"""Integration and unit tests for GCS storage naming, PNG encoding, and FastAPI SSE endpoints."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import cv2
from fastapi.testclient import TestClient
import numpy as np
from app.config import Settings
from app.main import create_app
from app.storage.gcs_store import (
    StorageService,
    encode_png_bytes,
    generate_run_folder_name,
    sanitize_video_stem,
)
from app.vision.processor import DetectedFace, VideoFaceProcessor


def _create_synthetic_video(video_path: Path, num_frames: int = 6) -> None:
    """Write a small synthetic MP4 video for testing the full pipeline."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 6.0, (320, 240))
    assert writer.isOpened(), "Failed to open cv2.VideoWriter"
    try:
        for i in range(num_frames):
            frame = np.full((240, 320, 3), 30, dtype=np.uint8)
            # Draw high-contrast pattern that gets sharper on frame 3
            step = 8 if i == 3 else 32
            for y in range(60, 180, step):
                for x in range(100, 220, step):
                    if ((x // step) + (y // step)) % 2 == 0:
                        frame[y : y + step, x : x + step] = (240, 200, 160)
            writer.write(frame)
    finally:
        writer.release()


def test_generate_run_folder_name_has_video_stem_and_unique_number() -> None:
    assert sanitize_video_stem("My Family_Vacation (2026).mp4") == "my-family-vacation-2026"
    folder1 = generate_run_folder_name("Keynote_Speech.MP4")
    folder2 = generate_run_folder_name("Keynote_Speech.MP4")

    assert folder1.startswith("keynote-speech-")
    suffix1 = folder1.removeprefix("keynote-speech-")
    assert suffix1.isdigit()
    assert len(suffix1) >= 6
    assert folder1 != folder2


def test_storage_service_uploads_valid_png_to_mock_gcs(tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob_full = MagicMock()
    mock_blob_crop = MagicMock()

    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.side_effect = [mock_blob_full, mock_blob_crop]

    service = StorageService(
        bucket_name="test-face-bucket",
        project_id="demo-project",
        local_storage_dir=tmp_path / "local_gcs",
        gcs_client=mock_client,
    )

    full_img = np.zeros((200, 300, 3), dtype=np.uint8)
    crop_img = np.ones((100, 100, 3), dtype=np.uint8) * 128

    assets = service.save_person_pngs(
        folder_name="sample-clip-123456",
        person_id=1,
        full_frame_bgr=full_img,
        face_crop_bgr=crop_img,
    )

    assert assets.storage_backend == "gcs"
    assert assets.full_frame_uri == "gs://test-face-bucket/sample-clip-123456/person_01_full_frame.png"
    assert assets.face_crop_uri == "gs://test-face-bucket/sample-clip-123456/person_01_face_crop.png"

    mock_client.bucket.assert_called_once_with("test-face-bucket")
    assert mock_blob_full.upload_from_string.called
    uploaded_bytes = mock_blob_full.upload_from_string.call_args[0][0]
    assert uploaded_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert mock_blob_full.upload_from_string.call_args[1]["content_type"] == "image/png"


def test_fastapi_upload_and_sse_stream_end_to_end(tmp_path: Path) -> None:
    video_file = tmp_path / "team_meeting.mp4"
    _create_synthetic_video(video_file, num_frames=6)

    rng = np.random.default_rng(7)
    emb_person_1 = rng.standard_normal(512).astype(np.float32)
    emb_person_1 /= np.linalg.norm(emb_person_1)
    emb_person_2 = rng.standard_normal(512).astype(np.float32)
    emb_person_2 /= np.linalg.norm(emb_person_2)

    call_counter = {"idx": 0}

    def fake_detector(frame_bgr: np.ndarray) -> list[DetectedFace]:
        idx = call_counter["idx"]
        call_counter["idx"] += 1
        kps = np.array(
            [[130.0, 100.0], [190.0, 100.0], [160.0, 125.0], [140.0, 155.0], [180.0, 155.0]],
            dtype=np.float32,
        )
        # Return Person 1 on every frame, and Person 2 on frame index 1
        faces = [
            DetectedFace(
                bbox=np.array([100.0, 60.0, 220.0, 180.0], dtype=np.float32),
                kps=kps,
                det_score=0.95 if idx == 1 else 0.80,
                embedding=emb_person_1,
            )
        ]
        if idx == 1:
            faces.append(
                DetectedFace(
                    bbox=np.array([20.0, 40.0, 95.0, 130.0], dtype=np.float32),
                    kps=kps,
                    det_score=0.91,
                    embedding=emb_person_2,
                )
            )
        return faces

    cfg = Settings(
        local_storage_dir=tmp_path / "storage",
        temp_upload_dir=tmp_path / "uploads",
        frame_sample_fps=3.0,
    )
    storage_service = StorageService(
        bucket_name="",
        local_storage_dir=cfg.local_storage_dir,
    )
    processor = VideoFaceProcessor(
        detector=fake_detector,
        storage_service=storage_service,
        sample_fps=3.0,
    )
    app = create_app(app_settings=cfg, processor_override=processor)
    client = TestClient(app)

    # 1. Upload video
    with video_file.open("rb") as vf:
        resp = client.post(
            "/api/jobs",
            files={"file": ("team_meeting.mp4", vf, "video/mp4")},
        )
    assert resp.status_code == 200
    job_info = resp.json()
    job_id = job_info["job_id"]
    folder_name = job_info["folder_name"]
    assert folder_name.startswith("team-meeting-")

    # 2. Consume SSE stream
    events: list[dict[str, Any]] = []
    with client.stream("GET", f"/api/jobs/{job_id}/stream") as stream_resp:
        assert stream_resp.status_code == 200
        for line in stream_resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))

    event_types = [e["event"] for e in events]
    assert "video_loaded" in event_types
    assert "face_discovered" in event_types
    assert "face_upgraded" in event_types
    assert "face_saved" in event_types
    assert "job_complete" in event_types

    complete_ev = next(e for e in events if e["event"] == "job_complete")
    assert complete_ev["total_people_saved"] == 2

    # Verify PNG files exist on disk in `<video-filename>-<unique-suffix>/`
    saved_dir = cfg.local_storage_dir / folder_name
    assert (saved_dir / "person_01_full_frame.png").exists()
    assert (saved_dir / "person_01_face_crop.png").exists()
    assert (saved_dir / "person_02_full_frame.png").exists()
    assert (saved_dir / "person_02_face_crop.png").exists()

    # Verify frontend index.html and static assets are served
    index_resp = client.get("/")
    assert index_resp.status_code == 200
    assert "Video Face Grab" in index_resp.text

