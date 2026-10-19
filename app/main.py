"""FastAPI application for Video Face Grab with real-time SSE progress streaming."""

import asyncio
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import uuid
from typing import Any, AsyncIterator
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from app.config import Settings, settings
from app.storage.gcs_store import StorageService, generate_run_folder_name
from app.vision.processor import InsightFaceDetector, VideoFaceProcessor


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass
class JobRecord:
    """In-memory state and event log for an active or completed video processing job."""

    job_id: str
    original_filename: str
    folder_name: str
    status: str = "queued"
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    people: dict[int, dict[str, Any]] = field(default_factory=dict)
    summary: dict[str, Any] | None = None
    done: bool = False


def create_app(
    app_settings: Settings | None = None,
    processor_override: VideoFaceProcessor | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    cfg = app_settings or settings
    cfg.local_storage_dir.mkdir(parents=True, exist_ok=True)
    cfg.temp_upload_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title=cfg.app_name,
        description="Extracts the clearest single PNG frame for each discernable human face in a video and saves it to GCS.",
        version="1.0.0",
    )

    storage_service = StorageService(
        bucket_name=cfg.gcs_bucket_name,
        project_id=cfg.gcp_project_id,
        local_storage_dir=cfg.local_storage_dir,
    )
    detector = InsightFaceDetector(model_name=cfg.insightface_model_name)
    processor = processor_override or VideoFaceProcessor(
        detector=detector,
        storage_service=storage_service,
        sample_fps=cfg.frame_sample_fps,
        similarity_threshold=cfg.similarity_threshold,
        min_face_confidence=cfg.min_face_confidence,
        min_face_size_px=cfg.min_face_size_px,
        min_sightings=cfg.min_sightings,
        face_crop_padding=cfg.face_crop_padding,
    )

    jobs: dict[str, JobRecord] = {}
    jobs_lock = threading.Lock()

    def _run_job_worker(job: JobRecord, temp_video_path: Path) -> None:
        with jobs_lock:
            job.status = "processing"
        try:
            for ev in processor.process_video(temp_video_path, job.folder_name):
                with jobs_lock:
                    job.events.append(ev)
                    ev_type = ev.get("event")
                    if ev_type in ("face_discovered", "face_upgraded", "face_saved"):
                        pid = int(ev["person_id"])
                        existing = job.people.get(pid, {})
                        job.people[pid] = {**existing, **ev}
                    elif ev_type == "job_complete":
                        job.summary = ev
                        job.status = "completed"
        except Exception as exc:  # noqa: BLE001
            with jobs_lock:
                job.status = "failed"
                job.error = str(exc)
                job.events.append(
                    {
                        "event": "job_error",
                        "message": str(exc),
                    }
                )
        finally:
            with jobs_lock:
                job.done = True
            try:
                if temp_video_path.exists():
                    temp_video_path.unlink()
            except OSError:
                pass

    @app.get("/api/health")
    async def health_check() -> dict[str, Any]:
        return {
            "status": "ok",
            "app": cfg.app_name,
            "gcs_bucket": cfg.gcs_bucket_name or "local-bucket",
            "storage_mode": "gcs" if cfg.gcs_bucket_name else "local",
        }

    @app.post("/api/jobs")
    async def create_video_job(file: UploadFile = File(...)) -> dict[str, Any]:
        filename = file.filename or "upload.mp4"
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
            )

        job_id = uuid.uuid4().hex[:12]
        folder_name = generate_run_folder_name(filename)
        temp_video_path = (cfg.temp_upload_dir / f"{job_id}{ext}").resolve()

        max_bytes = cfg.max_upload_mb * 1024 * 1024
        written = 0
        try:
            with temp_video_path.open("wb") as out_f:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Uploaded video exceeds {cfg.max_upload_mb} MB limit.",
                        )
                    out_f.write(chunk)
        except HTTPException:
            if temp_video_path.exists():
                temp_video_path.unlink()
            raise

        if written == 0:
            if temp_video_path.exists():
                temp_video_path.unlink()
            raise HTTPException(status_code=400, detail="Uploaded video file is empty.")

        job = JobRecord(
            job_id=job_id,
            original_filename=filename,
            folder_name=folder_name,
        )
        with jobs_lock:
            jobs[job_id] = job

        worker = threading.Thread(
            target=_run_job_worker,
            args=(job, temp_video_path),
            daemon=True,
        )
        worker.start()

        return {
            "job_id": job_id,
            "original_filename": filename,
            "folder_name": folder_name,
            "bucket_name": cfg.gcs_bucket_name or "local-bucket",
            "stream_url": f"/api/jobs/{job_id}/stream",
            "status_url": f"/api/jobs/{job_id}",
        }

    @app.get("/api/jobs/{job_id}")
    async def get_job_status(job_id: str) -> dict[str, Any]:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found.")
            return {
                "job_id": job.job_id,
                "original_filename": job.original_filename,
                "folder_name": job.folder_name,
                "status": job.status,
                "done": job.done,
                "error": job.error,
                "people": list(job.people.values()),
                "summary": job.summary,
            }

    @app.get("/api/jobs/{job_id}/stream")
    async def stream_job_events(job_id: str) -> StreamingResponse:
        with jobs_lock:
            if job_id not in jobs:
                raise HTTPException(status_code=404, detail="Job not found.")

        async def _event_generator() -> AsyncIterator[str]:
            cursor = 0
            while True:
                with jobs_lock:
                    job = jobs[job_id]
                    pending = job.events[cursor:]
                    cursor += len(pending)
                    is_done = job.done

                for ev in pending:
                    ev_name = ev.get("event", "message")
                    payload = json.dumps(ev)
                    yield f"event: {ev_name}\ndata: {payload}\n\n"

                if is_done and not pending:
                    break
                await asyncio.sleep(0.08)

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    app.mount(
        "/files",
        StaticFiles(directory=str(cfg.local_storage_dir)),
        name="files",
    )

    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def serve_index() -> FileResponse:
        index_file = static_dir / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="Frontend index.html not found.")
        return FileResponse(index_file)

    return app


app = create_app()
