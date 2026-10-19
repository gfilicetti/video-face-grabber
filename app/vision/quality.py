"""Face clarity, sharpness, frontal pose, and quality scoring functions."""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class FacePoseAssessment:
    """Geometric pose assessment from 5 facial landmarks (eyes, nose, mouth corners)."""

    frontal_score: float
    yaw_asymmetry: float
    roll_degrees: float
    pitch_deviation: float
    is_forward_facing: bool


@dataclass(frozen=True)
class FaceQualityMetrics:
    """Breakdown of a single detected face's visual clarity and quality."""

    composite_score: float
    sharpness_score: float
    frontal_score: float
    confidence_score: float
    size_score: float
    raw_laplacian_var: float
    yaw_asymmetry: float = 0.0
    roll_degrees: float = 0.0
    is_forward_facing: bool = True


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


def assess_frontal_pose(
    kps: np.ndarray | None,
    max_yaw_asymmetry: float = 0.42,
    max_roll_degrees: float = 28.0,
    min_frontal_score: float = 65.0,
) -> FacePoseAssessment:
    """Evaluate 3D head orientation from 5 landmarks and determine if the face is forward-facing.

    Landmarks order in InsightFace / RetinaFace:
        0: left_eye, 1: right_eye, 2: nose, 3: left_mouth, 4: right_mouth
    """
    if kps is None:
        return FacePoseAssessment(
            frontal_score=0.0,
            yaw_asymmetry=1.0,
            roll_degrees=90.0,
            pitch_deviation=1.0,
            is_forward_facing=False,
        )

    pts = np.asarray(kps, dtype=np.float64)
    if pts.shape != (5, 2):
        return FacePoseAssessment(
            frontal_score=0.0,
            yaw_asymmetry=1.0,
            roll_degrees=90.0,
            pitch_deviation=1.0,
            is_forward_facing=False,
        )

    left_eye, right_eye, nose, left_mouth, right_mouth = pts
    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    interocular = float(np.hypot(eye_dx, eye_dy))
    # Eyes must be horizontally separated in anatomical order (not profile-collapsed)
    if interocular < 4.0 or eye_dx <= 2.0:
        return FacePoseAssessment(
            frontal_score=0.0,
            yaw_asymmetry=1.0,
            roll_degrees=90.0,
            pitch_deviation=1.0,
            is_forward_facing=False,
        )

    # 1. Roll penalty: head tilt angle between eyes
    roll_deg = abs(float(np.degrees(np.arctan2(eye_dy, eye_dx))))
    roll_score = float(np.clip(1.0 - (roll_deg / 45.0), 0.0, 1.0))

    # 2. Yaw penalty: horizontal symmetry of nose AND mouth midpoint relative to eye span
    eye_mid_x = 0.5 * (left_eye[0] + right_eye[0])
    half_eye_span = 0.5 * eye_dx
    nose_yaw_offset = abs(nose[0] - eye_mid_x) / half_eye_span

    mouth_mid_x = 0.5 * (left_mouth[0] + right_mouth[0])
    mouth_yaw_offset = abs(mouth_mid_x - eye_mid_x) / half_eye_span

    # Combine nose and mouth horizontal shift so turning to the side is caught reliably
    yaw_asymmetry = float(0.75 * nose_yaw_offset + 0.25 * mouth_yaw_offset)
    yaw_score = float(np.clip(1.0 - yaw_asymmetry, 0.0, 1.0))

    # 3. Pitch penalty: vertical placement of nose between eye line and mouth line
    eye_mid_y = 0.5 * (left_eye[1] + right_eye[1])
    mouth_mid_y = 0.5 * (left_mouth[1] + right_mouth[1])
    eye_to_mouth = mouth_mid_y - eye_mid_y
    if eye_to_mouth < 4.0:
        return FacePoseAssessment(
            frontal_score=0.0,
            yaw_asymmetry=round(yaw_asymmetry, 3),
            roll_degrees=round(roll_deg, 2),
            pitch_deviation=1.0,
            is_forward_facing=False,
        )

    nose_ratio = (nose[1] - eye_mid_y) / eye_to_mouth
    # Ideal nose vertical ratio is ~0.55 between eyes and mouth
    pitch_deviation = abs(nose_ratio - 0.55) / 0.45
    pitch_score = float(np.clip(1.0 - pitch_deviation, 0.0, 1.0))

    combined = round(
        float(np.clip((0.55 * yaw_score + 0.25 * roll_score + 0.20 * pitch_score) * 100.0, 0.0, 100.0)),
        2,
    )

    # Nose must remain strictly between the left and right eyes and within yaw/roll/pitch tolerances
    nose_between_eyes = left_eye[0] < nose[0] < right_eye[0]
    is_forward = bool(
        nose_between_eyes
        and yaw_asymmetry <= max_yaw_asymmetry
        and roll_deg <= max_roll_degrees
        and pitch_deviation <= 0.75
        and combined >= min_frontal_score
    )

    return FacePoseAssessment(
        frontal_score=combined,
        yaw_asymmetry=round(yaw_asymmetry, 3),
        roll_degrees=round(roll_deg, 2),
        pitch_deviation=round(float(pitch_deviation), 3),
        is_forward_facing=is_forward,
    )


def compute_frontal_symmetry(kps: np.ndarray | None) -> float:
    """Estimate frontal pose quality (0..100) from 5 facial landmarks."""
    return assess_frontal_pose(kps).frontal_score


def compute_composite_quality(
    face_bgr: np.ndarray,
    bbox: np.ndarray,
    kps: np.ndarray | None,
    det_score: float,
    frame_shape: tuple[int, int],
    max_yaw_asymmetry: float = 0.42,
    max_roll_degrees: float = 28.0,
    min_frontal_score: float = 65.0,
) -> FaceQualityMetrics:
    """Compute overall face quality (0..100) combining sharpness, frontal pose, confidence, and size."""
    sharpness_score, raw_lap = compute_laplacian_sharpness(face_bgr)
    pose = assess_frontal_pose(
        kps,
        max_yaw_asymmetry=max_yaw_asymmetry,
        max_roll_degrees=max_roll_degrees,
        min_frontal_score=min_frontal_score,
    )
    frontal_score = pose.frontal_score
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
        yaw_asymmetry=pose.yaw_asymmetry,
        roll_degrees=pose.roll_degrees,
        is_forward_facing=pose.is_forward_facing,
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
