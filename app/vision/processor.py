"""Video frame face detection, identity clustering, and progressive event generator."""

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator
import cv2
import numpy as np
from app.storage.gcs_store import StorageService, encode_png_bytes
from app.vision.clustering import FaceClusterTracker
from app.vision.quality import compute_composite_quality, extract_padded_face_crop


@dataclass(frozen=True)
class DetectedFace:
    """Normalized representation of a single face detected in a video frame."""

    bbox: np.ndarray
    kps: np.ndarray | None
    det_score: float
    embedding: np.ndarray


class InsightFaceDetector:
    """Lazy-initialized wrapper around InsightFace FaceAnalysis (RetinaFace + ArcFace)."""

    def __init__(self, model_name: str = "buffalo_sc") -> None:
        self.model_name = model_name
        self._app: Any = None

    def _ensure_loaded(self) -> Any:
        if self._app is None:
            import os
            from insightface.app import FaceAnalysis

            insightface_root = os.environ.get("INSIGHTFACE_HOME", "~/.insightface")
            app = FaceAnalysis(
                name=self.model_name,
                root=insightface_root,
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._app = app
        return self._app

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        """Run face detection and 512-D ArcFace embedding extraction on a BGR frame."""
        app = self._ensure_loaded()
        raw_faces = app.get(frame_bgr)
        results: list[DetectedFace] = []
        for f in raw_faces:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = getattr(f, "embedding", None)
            if emb is None:
                continue
            results.append(
                DetectedFace(
                    bbox=np.asarray(f.bbox, dtype=np.float32),
                    kps=np.asarray(f.kps, dtype=np.float32) if getattr(f, "kps", None) is not None else None,
                    det_score=float(getattr(f, "det_score", 0.0)),
                    embedding=np.asarray(emb, dtype=np.float32),
                )
            )
        return results


def _make_preview_data_uri(crop_bgr: np.ndarray, max_dim: int = 220) -> str:
    """Encode a resized PNG thumbnail as a data URI for instant SSE browser rendering."""
    h, w = crop_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        crop_bgr = cv2.resize(
            crop_bgr,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    png_bytes = encode_png_bytes(crop_bgr)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


class VideoFaceProcessor:
    """Processes a video file frame-by-frame, clustering faces and saving best PNGs to GCS."""

    def __init__(
        self,
        detector: InsightFaceDetector | Callable[[np.ndarray], list[DetectedFace]],
        storage_service: StorageService,
        sample_fps: float = 3.0,
        similarity_threshold: float = 0.40,
        min_face_confidence: float = 0.65,
        min_face_size_px: int = 44,
        min_sharpness_score: float = 0.0,
        require_forward_facing: bool = False,
        min_frontal_score: float = 65.0,
        max_yaw_asymmetry: float = 0.42,
        max_roll_degrees: float = 28.0,
        min_nose_ratio: float = 0.36,
        max_nose_ratio: float = 0.59,
        min_sightings: int = 1,
        face_crop_padding: float = 0.35,
    ) -> None:
        self.detector = detector
        self.storage_service = storage_service
        self.sample_fps = max(0.5, sample_fps)
        self.similarity_threshold = similarity_threshold
        self.min_face_confidence = min_face_confidence
        self.min_face_size_px = min_face_size_px
        self.min_sharpness_score = min_sharpness_score
        self.require_forward_facing = require_forward_facing
        self.min_frontal_score = min_frontal_score
        self.max_yaw_asymmetry = max_yaw_asymmetry
        self.max_roll_degrees = max_roll_degrees
        self.min_nose_ratio = min_nose_ratio
        self.max_nose_ratio = max_nose_ratio
        self.min_sightings = min_sightings
        self.face_crop_padding = face_crop_padding

    def _detect_faces(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        if callable(self.detector):
            return self.detector(frame_bgr)
        return self.detector.detect(frame_bgr)

    def process_video(
        self,
        video_path: Path,
        folder_name: str,
    ) -> Iterator[dict[str, Any]]:
        """Generator yielding real-time structured progress events while scanning the video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path.name}")

        try:
            native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if native_fps <= 0.1 or native_fps > 240.0:
                native_fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            stride = max(1, int(round(native_fps / self.sample_fps)))

            yield {
                "event": "video_loaded",
                "folder_name": folder_name,
                "native_fps": round(native_fps, 2),
                "total_frames": total_frames,
                "sample_stride": stride,
                "resolution": f"{width}x{height}",
            }

            tracker = FaceClusterTracker(
                similarity_threshold=self.similarity_threshold,
                min_sightings=self.min_sightings,
            )

            frame_idx = 0
            sampled_frames = 0

            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break

                if frame_idx % stride == 0:
                    sampled_frames += 1
                    timestamp_sec = frame_idx / native_fps
                    detected = self._detect_faces(frame_bgr)

                    for face in detected:
                        if face.det_score < self.min_face_confidence:
                            continue
                        x1, y1, x2, y2 = [float(v) for v in face.bbox[:4]]
                        if (x2 - x1) < self.min_face_size_px or (y2 - y1) < self.min_face_size_px:
                            continue

                        crop_bgr = extract_padded_face_crop(
                            frame_bgr,
                            face.bbox,
                            padding_ratio=self.face_crop_padding,
                        )
                        quality = compute_composite_quality(
                            face_bgr=crop_bgr,
                            bbox=face.bbox,
                            kps=face.kps,
                            det_score=face.det_score,
                            frame_shape=frame_bgr.shape[:2],
                            max_yaw_asymmetry=self.max_yaw_asymmetry,
                            max_roll_degrees=self.max_roll_degrees,
                            min_frontal_score=self.min_frontal_score,
                            min_nose_ratio=self.min_nose_ratio,
                            max_nose_ratio=self.max_nose_ratio,
                        )
                        # Keep all detected faces/heads (even profile or angled) unless strict mode or sharpness floor is configured
                        if self.require_forward_facing and not quality.is_forward_facing:
                            continue
                        if quality.sharpness_score < self.min_sharpness_score:
                            continue

                        update_res = tracker.update(
                            embedding=face.embedding,
                            quality=quality,
                            frame_bgr=frame_bgr,
                            crop_bgr=crop_bgr,
                            frame_index=frame_idx,
                            timestamp_sec=timestamp_sec,
                            bbox=face.bbox,
                        )

                        cluster = update_res.cluster
                        if update_res.status in ("discovered", "upgraded"):
                            yield {
                                "event": f"face_{update_res.status}",
                                "person_id": cluster.person_id,
                                "label": cluster.label,
                                "sightings_count": cluster.sightings_count,
                                "frame_index": cluster.best_frame_index,
                                "timestamp_sec": cluster.best_timestamp_sec,
                                "quality": {
                                    "composite": cluster.best_quality.composite_score,
                                    "sharpness": cluster.best_quality.sharpness_score,
                                    "frontal": cluster.best_quality.frontal_score,
                                    "confidence": cluster.best_quality.confidence_score,
                                    "size": cluster.best_quality.size_score,
                                },
                                "previous_best_score": update_res.previous_best_score,
                                "preview_data_uri": _make_preview_data_uri(cluster.best_crop_bgr),
                            }

                    pct = (
                        round(min(99.0, ((frame_idx + 1) / total_frames) * 100.0), 1)
                        if total_frames > 0
                        else 50.0
                    )
                    yield {
                        "event": "frame_progress",
                        "frame_index": frame_idx + 1,
                        "total_frames": total_frames,
                        "sampled_frames": sampled_frames,
                        "timestamp_sec": round(timestamp_sec, 2),
                        "percent": pct,
                        "discerned_faces_count": len(tracker.clusters),
                    }

                frame_idx += 1

            # Finalize and upload each verified person's best frame & face crop as PNGs
            verified_clusters = tracker.get_verified_clusters()
            saved_people: list[dict[str, Any]] = []

            for cluster in verified_clusters:
                yield {
                    "event": "uploading_face",
                    "person_id": cluster.person_id,
                    "label": cluster.label,
                    "folder_name": folder_name,
                }
                assets = self.storage_service.save_person_pngs(
                    folder_name=folder_name,
                    person_id=cluster.person_id,
                    full_frame_bgr=cluster.best_frame_bgr,
                    face_crop_bgr=cluster.best_crop_bgr,
                )
                person_summary = {
                    "person_id": cluster.person_id,
                    "label": cluster.label,
                    "sightings_count": cluster.sightings_count,
                    "best_frame_index": cluster.best_frame_index,
                    "best_timestamp_sec": cluster.best_timestamp_sec,
                    "quality": {
                        "composite": cluster.best_quality.composite_score,
                        "sharpness": cluster.best_quality.sharpness_score,
                        "frontal": cluster.best_quality.frontal_score,
                        "confidence": cluster.best_quality.confidence_score,
                        "size": cluster.best_quality.size_score,
                    },
                    "full_frame_uri": assets.full_frame_uri,
                    "face_crop_uri": assets.face_crop_uri,
                    "full_frame_url": assets.full_frame_url,
                    "face_crop_url": assets.face_crop_url,
                    "storage_backend": assets.storage_backend,
                }
                saved_people.append(person_summary)
                yield {
                    "event": "face_saved",
                    **person_summary,
                }

            yield {
                "event": "job_complete",
                "folder_name": folder_name,
                "bucket_name": self.storage_service.bucket_name or "local-bucket",
                "storage_backend": "gcs" if self.storage_service.bucket_name else "local",
                "total_frames_scanned": frame_idx,
                "sampled_frames": sampled_frames,
                "total_people_saved": len(saved_people),
                "people": saved_people,
            }
        finally:
            cap.release()
