"""Face clarity, sharpness, frontal pose, and quality scoring functions."""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class FaceQualityMetrics:
    """Breakdown of a single detected face's visual clarity and quality."""

    composite_score: float
    sharpness_score: float
    frontal_score: float
    confidence_score: float
    size_score: float
    raw_laplacian_var: float


def compute_laplacian_sharpness(face_bgr: np.ndarray) -> tuple[float, float]:
    """Compute variance of the Laplacian on a normalized grayscale face crop.

    Returns:
        tuple[float, float]: (normalized_score_0_to_100, raw_laplacian_variance)
    """
    if face_bgr is None or face_bgr.size == 0:
        return 0.0, 0.0

    h, w = face_bgr.shape[:2]
    if h < 8 or w < 8:
        return 0.0, 0.0

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.ndim == 3 else face_bgr
    # Resize to a canonical resolution so Laplacian variance is scale-invariant
    resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    lap_var = float(cv2.Laplacian(resized, cv2.CV_64F).var())

    # Map raw variance (typical sharp faces: 120 - 600+) smoothly to 0..100
    normalized = float(np.clip((lap_var / 350.0) * 100.0, 0.0, 100.0))
    return round(normalized, 2), round(lap_var, 2)


def compute_frontal_symmetry(kps: np.ndarray | None) -> float:
    """Estimate frontal pose quality (0..100) from 5 facial landmarks.

    Landmarks order in InsightFace / RetinaFace:
        0: left_eye, 1: right_eye, 2: nose, 3: left_mouth, 4: right_mouth
    """
    if kps is None:
        return 70.0

    pts = np.asarray(kps, dtype=np.float64)
    if pts.shape != (5, 2):
        return 70.0

    left_eye, right_eye, nose, left_mouth, right_mouth = pts
    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    interocular = float(np.hypot(eye_dx, eye_dy))
    if interocular < 4.0:
        return 10.0

    # 1. Roll penalty: tilt angle between eyes
    roll_deg = abs(float(np.degrees(np.arctan2(eye_dy, eye_dx))))
    roll_score = float(np.clip(1.0 - (roll_deg / 45.0), 0.0, 1.0))

    # 2. Yaw penalty: horizontal symmetry of nose relative to eye midpoint
    eye_mid_x = 0.5 * (left_eye[0] + right_eye[0])
    yaw_offset = abs(nose[0] - eye_mid_x) / (0.5 * interocular)
    yaw_score = float(np.clip(1.0 - yaw_offset, 0.0, 1.0))

    # 3. Pitch penalty: vertical placement of nose between eye line and mouth line
    eye_mid_y = 0.5 * (left_eye[1] + right_eye[1])
    mouth_mid_y = 0.5 * (left_mouth[1] + right_mouth[1])
    eye_to_mouth = max(mouth_mid_y - eye_mid_y, 4.0)
    nose_ratio = (nose[1] - eye_mid_y) / eye_to_mouth
    # Ideal nose vertical ratio is ~0.55 between eyes and mouth
    pitch_deviation = abs(nose_ratio - 0.55) / 0.45
    pitch_score = float(np.clip(1.0 - pitch_deviation, 0.0, 1.0))

    combined = (0.50 * yaw_score + 0.25 * roll_score + 0.25 * pitch_score) * 100.0
    return round(float(np.clip(combined, 0.0, 100.0)), 2)


def compute_composite_quality(
    face_bgr: np.ndarray,
    bbox: np.ndarray,
    kps: np.ndarray | None,
    det_score: float,
    frame_shape: tuple[int, int],
) -> FaceQualityMetrics:
    """Compute overall face quality (0..100) combining sharpness, frontal pose, confidence, and size."""
    sharpness_score, raw_lap = compute_laplacian_sharpness(face_bgr)
    frontal_score = compute_frontal_symmetry(kps)
    confidence_score = round(float(np.clip(det_score * 100.0, 0.0, 100.0)), 2)

    frame_h, frame_w = frame_shape[:2]
    frame_area = max(float(frame_h * frame_w), 1.0)
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    face_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    # A face covering >= 6% of the frame gets maximum size score (100)
    relative_area = face_area / frame_area
    size_score = round(float(np.clip((relative_area / 0.06) * 100.0, 0.0, 100.0)), 2)

    composite = (
        0.35 * sharpness_score
        + 0.30 * frontal_score
        + 0.20 * confidence_score
        + 0.15 * size_score
    )
    return FaceQualityMetrics(
        composite_score=round(composite, 2),
        sharpness_score=sharpness_score,
        frontal_score=frontal_score,
        confidence_score=confidence_score,
        size_score=size_score,
        raw_laplacian_var=raw_lap,
    )


def extract_padded_face_crop(
    frame_bgr: np.ndarray,
    bbox: np.ndarray,
    padding_ratio: float = 0.35,
) -> np.ndarray:
    """Extract a padded square-ish portrait crop around a face bounding box."""
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Cannot extract face crop from an empty frame.")

    frame_h, frame_w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    pad_x = w * padding_ratio
    pad_y = h * padding_ratio

    cx1 = max(0, int(round(x1 - pad_x)))
    cy1 = max(0, int(round(y1 - pad_y)))
    cx2 = min(frame_w, int(round(x2 + pad_x)))
    cy2 = min(frame_h, int(round(y2 + pad_y)))

    if cx2 <= cx1 or cy2 <= cy1:
        return frame_bgr.copy()

    return frame_bgr[cy1:cy2, cx1:cx2].copy()
