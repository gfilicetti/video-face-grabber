"""Unit tests for face quality metrics and identity clustering."""

import cv2
import numpy as np
from app.vision.quality import (
    assess_frontal_pose,
    compute_composite_quality,
    compute_laplacian_sharpness,
    extract_padded_face_crop,
)
from app.vision.clustering import FaceClusterTracker, normalize_embedding


def _make_checkerboard(size: int = 128, square: int = 8) -> np.ndarray:
    """Create a high-frequency sharp synthetic BGR image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(0, size, square):
        for x in range(0, size, square):
            if ((x // square) + (y // square)) % 2 == 0:
                img[y : y + square, x : x + square] = 255
    return img


def test_sharpness_distinguishes_sharp_from_blurry() -> None:
    sharp_img = _make_checkerboard()
    blurry_img = cv2.GaussianBlur(sharp_img, (31, 31), 10.0)

    sharp_score, sharp_raw = compute_laplacian_sharpness(sharp_img)
    blur_score, blur_raw = compute_laplacian_sharpness(blurry_img)

    assert sharp_raw > blur_raw * 5
    assert sharp_score > blur_score
    assert sharp_score == 100.0
    assert blur_score < 25.0


def test_frontal_symmetry_prefers_centered_landmarks() -> None:
    frontal_kps = np.array(
        [
            [160.0, 170.0],  # left eye
            [240.0, 170.0],  # right eye
            [200.0, 210.0],  # nose centered
            [170.0, 250.0],  # left mouth
            [230.0, 250.0],  # right mouth
        ]
    )
    marginally_turned_kps = np.array(
        [
            [160.0, 170.0],  # left eye
            [240.0, 172.0],  # right eye
            [212.0, 210.0],  # nose shifted slightly right (yaw_asymmetry ~ 0.275 <= 0.42)
            [178.0, 250.0],  # left mouth
            [238.0, 250.0],  # right mouth
        ]
    )
    profile_kps = np.array(
        [
            [160.0, 170.0],
            [240.0, 195.0],  # tilted roll
            [255.0, 210.0],  # nose shifted far right past eye center (yaw)
            [195.0, 250.0],
            [250.0, 250.0],
        ]
    )

    frontal_pose = assess_frontal_pose(frontal_kps)
    marginal_pose = assess_frontal_pose(marginally_turned_kps)
    profile_pose = assess_frontal_pose(profile_kps)
    missing_pose = assess_frontal_pose(None)

    assert frontal_pose.is_forward_facing is True
    assert frontal_pose.frontal_score > 90.0

    assert marginal_pose.is_forward_facing is True
    assert marginal_pose.frontal_score >= 65.0
    assert marginal_pose.yaw_asymmetry <= 0.42

    assert profile_pose.is_forward_facing is False
    assert profile_pose.frontal_score < 50.0
    assert missing_pose.is_forward_facing is False


def test_extract_padded_face_crop_clamps_to_frame_bounds() -> None:
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    bbox = np.array([5, 10, 85, 90], dtype=np.float32)
    crop = extract_padded_face_crop(frame, bbox, padding_ratio=0.5)

    assert crop.shape[0] <= 200
    assert crop.shape[1] <= 300
    assert crop.shape[0] > 80
    assert crop.shape[1] > 80


def test_cluster_tracker_discovers_and_upgrades_best_frame() -> None:
    tracker = FaceClusterTracker(similarity_threshold=0.5, min_sightings=1)

    rng = np.random.default_rng(42)
    person_a_emb = normalize_embedding(rng.standard_normal(512))
    person_b_emb = normalize_embedding(rng.standard_normal(512))

    frame_blurry = np.full((240, 320, 3), 120, dtype=np.uint8)
    frame_sharp = _make_checkerboard(240, 12)
    bbox = np.array([40, 40, 180, 180], dtype=np.float32)
    frontal_kps = np.array(
        [[75.0, 85.0], [145.0, 85.0], [110.0, 115.0], [85.0, 150.0], [135.0, 150.0]],
        dtype=np.float32,
    )

    q_low = compute_composite_quality(frame_blurry, bbox, frontal_kps, 0.75, (240, 320))
    q_high = compute_composite_quality(frame_sharp, bbox, frontal_kps, 0.96, (240, 320))

    # 1. First sighting of Person A (blurry)
    res1 = tracker.update(person_a_emb, q_low, frame_blurry, frame_blurry, 10, 0.33, bbox)
    assert res1.status == "discovered"
    assert res1.cluster.person_id == 1
    assert res1.cluster.best_frame_index == 10

    # 2. First sighting of Person B
    res2 = tracker.update(person_b_emb, q_low, frame_blurry, frame_blurry, 15, 0.50, bbox)
    assert res2.status == "discovered"
    assert res2.cluster.person_id == 2

    # 3. Second sighting of Person A with a much clearer frame -> upgraded
    noisy_a_emb = normalize_embedding(person_a_emb + 0.05 * rng.standard_normal(512))
    res3 = tracker.update(noisy_a_emb, q_high, frame_sharp, frame_sharp, 25, 0.83, bbox)
    assert res3.status == "upgraded"
    assert res3.cluster.person_id == 1
    assert res3.cluster.sightings_count == 2
    assert res3.cluster.best_frame_index == 25
    assert res3.cluster.best_quality.composite_score == q_high.composite_score
    assert res3.previous_best_score == q_low.composite_score

    # 4. Third sighting of Person A with lower quality -> seen (keeps frame 25)
    res4 = tracker.update(noisy_a_emb, q_low, frame_blurry, frame_blurry, 30, 1.00, bbox)
    assert res4.status == "seen"
    assert res4.cluster.best_frame_index == 25
    assert len(tracker.get_verified_clusters()) == 2
