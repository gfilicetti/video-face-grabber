"""GCS and local storage service for saving best human face frames as PNGs."""

from dataclasses import dataclass
from pathlib import Path
import re
import secrets
import time
import cv2
import numpy as np
from google.api_core.client_info import ClientInfo
from google.cloud import storage


def sanitize_video_stem(filename: str) -> str:
    """Sanitize a video filename into a GCS-safe lowercase prefix."""
    raw_stem = Path(filename or "video").stem.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", raw_stem).strip("-")
    if not cleaned:
        cleaned = "video"
    return cleaned[:45].rstrip("-")


def generate_run_folder_name(filename: str, unique_suffix: str | None = None) -> str:
    """Create a folder name matching `<video-filename>-<unique-number-suffix>`."""
    stem = sanitize_video_stem(filename)
    if unique_suffix is None:
        millis = int(time.time() * 1000) % 1_000_000
        rand_digits = secrets.randbelow(9000) + 1000
        unique_suffix = f"{millis:06d}{rand_digits:04d}"
    numeric_only = re.sub(r"[^0-9]", "", str(unique_suffix))
    if not numeric_only:
        numeric_only = f"{secrets.randbelow(900000) + 100000}"
    return f"{stem}-{numeric_only}"


def encode_png_bytes(image_bgr: np.ndarray) -> bytes:
    """Losslessly encode a BGR numpy image into PNG bytes."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot encode an empty image to PNG.")
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("OpenCV failed to encode image as PNG.")
    data = buf.tobytes()
    # Verify PNG magic header (\x89PNG\r\n\x1a\n)
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("Encoded buffer is not a valid PNG.")
    return data


@dataclass(frozen=True)
class SavedPersonAssets:
    """Metadata and URIs for a single saved human identity."""

    person_id: int
    folder_name: str
    full_frame_object: str
    face_crop_object: str
    full_frame_uri: str
    face_crop_uri: str
    full_frame_url: str
    face_crop_url: str
    storage_backend: str


class StorageService:
    """Handles uploading PNG frames to GCS (or local fallback directory when unconfigured)."""

    def __init__(
        self,
        bucket_name: str = "",
        project_id: str = "",
        local_storage_dir: Path = Path("local_gcs_bucket"),
        gcs_client: storage.Client | None = None,
    ) -> None:
        self.bucket_name = (bucket_name or "").strip()
        self.project_id = (project_id or "").strip()
        self.local_storage_dir = Path(local_storage_dir)
        self._gcs_client = gcs_client

    def _get_gcs_client(self) -> storage.Client:
        if self._gcs_client is None:
            client_info = ClientInfo(
                user_agent="gcs-skills/1.0 (skill:google-cloud-storage-basics)"
            )
            self._gcs_client = storage.Client(
                project=self.project_id or None,
                client_info=client_info,
            )
        return self._gcs_client

    def save_person_pngs(
        self,
        folder_name: str,
        person_id: int,
        full_frame_bgr: np.ndarray,
        face_crop_bgr: np.ndarray,
    ) -> SavedPersonAssets:
        """Encode and upload both the full frame and cropped face as PNG files."""
        safe_folder = re.sub(r"[^a-z0-9\-_]", "", folder_name.lower())
        if not safe_folder:
            raise ValueError("Invalid folder_name.")

        full_filename = f"person_{person_id:02d}_full_frame.png"
        crop_filename = f"person_{person_id:02d}_face_crop.png"

        full_object_path = f"{safe_folder}/{full_filename}"
        crop_object_path = f"{safe_folder}/{crop_filename}"

        full_png = encode_png_bytes(full_frame_bgr)
        crop_png = encode_png_bytes(face_crop_bgr)

        # Always cache locally under local_storage_dir so the web UI can display high-res PNGs directly
        target_local_dir = (self.local_storage_dir / safe_folder).resolve()
        base_local_dir = self.local_storage_dir.resolve()
        if not target_local_dir.is_relative_to(base_local_dir):
            raise ValueError("Path traversal detected in folder_name.")
        target_local_dir.mkdir(parents=True, exist_ok=True)
        (target_local_dir / full_filename).write_bytes(full_png)
        (target_local_dir / crop_filename).write_bytes(crop_png)

        local_full_url = f"/files/{safe_folder}/{full_filename}"
        local_crop_url = f"/files/{safe_folder}/{crop_filename}"

        if self.bucket_name:
            client = self._get_gcs_client()
            bucket = client.bucket(self.bucket_name)

            full_blob = bucket.blob(full_object_path)
            full_blob.upload_from_string(full_png, content_type="image/png")

            crop_blob = bucket.blob(crop_object_path)
            crop_blob.upload_from_string(crop_png, content_type="image/png")

            return SavedPersonAssets(
                person_id=person_id,
                folder_name=safe_folder,
                full_frame_object=full_object_path,
                face_crop_object=crop_object_path,
                full_frame_uri=f"gs://{self.bucket_name}/{full_object_path}",
                face_crop_uri=f"gs://{self.bucket_name}/{crop_object_path}",
                full_frame_url=local_full_url,
                face_crop_url=local_crop_url,
                storage_backend="gcs",
            )

        return SavedPersonAssets(
            person_id=person_id,
            folder_name=safe_folder,
            full_frame_object=full_object_path,
            face_crop_object=crop_object_path,
            full_frame_uri=f"gs://local-bucket/{full_object_path}",
            face_crop_uri=f"gs://local-bucket/{crop_object_path}",
            full_frame_url=local_full_url,
            face_crop_url=local_crop_url,
            storage_backend="local",
        )
