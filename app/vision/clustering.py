"""Identity clustering and best-frame tracking across video frames."""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from app.vision.quality import FaceQualityMetrics


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2-normalize a face embedding vector."""
    vec = np.asarray(embedding, dtype=np.float32).flatten()
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec
    return vec / norm


@dataclass
class IdentityCluster:
    """Represents a single discerned human identity across a video."""

    person_id: int
    centroid_embedding: np.ndarray
    best_embedding: np.ndarray
    sightings_count: int
    best_quality: FaceQualityMetrics
    best_frame_bgr: np.ndarray
    best_crop_bgr: np.ndarray
    best_frame_index: int
    best_timestamp_sec: float
    best_bbox: list[float] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"Person #{self.person_id:02d}"


@dataclass(frozen=True)
class ClusterUpdateResult:
    """Outcome of matching a detected face against tracked clusters."""

    status: Literal["discovered", "upgraded", "seen"]
    cluster: IdentityCluster
    similarity: float
    previous_best_score: float | None = None


class FaceClusterTracker:
    """Clusters face embeddings by cosine similarity and retains the clearest frame per human."""

    def __init__(self, similarity_threshold: float = 0.42, min_sightings: int = 1) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1].")
        self.similarity_threshold = similarity_threshold
        self.min_sightings = max(1, min_sightings)
        self._clusters: list[IdentityCluster] = []

    @property
    def clusters(self) -> list[IdentityCluster]:
        """Return all tracked clusters ordered by person_id."""
        return list(self._clusters)

    def get_verified_clusters(self) -> list[IdentityCluster]:
        """Consolidate any split clusters whose best embeddings match and return verified clusters."""
        self._consolidate_clusters()
        return [c for c in self._clusters if c.sightings_count >= self.min_sightings]

    def _consolidate_clusters(self) -> None:
        """Merge any secondary cluster into an earlier cluster if their embeddings exceed similarity_threshold."""
        merged_any = True
        while merged_any and len(self._clusters) > 1:
            merged_any = False
            for i in range(len(self._clusters)):
                for j in range(i + 1, len(self._clusters)):
                    ci = self._clusters[i]
                    cj = self._clusters[j]
                    sim = max(
                        float(np.dot(ci.best_embedding, cj.best_embedding)),
                        float(np.dot(ci.centroid_embedding, cj.best_embedding)),
                        float(np.dot(ci.best_embedding, cj.centroid_embedding)),
                        float(np.dot(ci.centroid_embedding, cj.centroid_embedding)),
                    )
                    if sim >= self.similarity_threshold:
                        total_sightings = ci.sightings_count + cj.sightings_count
                        ci.centroid_embedding = normalize_embedding(
                            ci.centroid_embedding * ci.sightings_count
                            + cj.centroid_embedding * cj.sightings_count
                        )
                        ci.sightings_count = total_sightings
                        if cj.best_quality.composite_score > ci.best_quality.composite_score:
                            ci.best_quality = cj.best_quality
                            ci.best_embedding = cj.best_embedding.copy()
                            ci.best_frame_bgr = cj.best_frame_bgr.copy()
                            ci.best_crop_bgr = cj.best_crop_bgr.copy()
                            ci.best_frame_index = cj.best_frame_index
                            ci.best_timestamp_sec = cj.best_timestamp_sec
                            ci.best_bbox = list(cj.best_bbox)
                        del self._clusters[j]
                        for idx, c in enumerate(self._clusters, start=1):
                            c.person_id = idx
                        merged_any = True
                        break
                if merged_any:
                    break

    def update(
        self,
        embedding: np.ndarray,
        quality: FaceQualityMetrics,
        frame_bgr: np.ndarray,
        crop_bgr: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
        bbox: np.ndarray | list[float],
    ) -> ClusterUpdateResult:
        """Match a face observation to an existing cluster or register a new human identity."""
        norm_emb = normalize_embedding(embedding)
        bbox_list = [round(float(v), 1) for v in np.asarray(bbox).flatten()[:4]]

        best_match: IdentityCluster | None = None
        best_sim = -1.0

        for cluster in self._clusters:
            sim = max(
                float(np.dot(norm_emb, cluster.centroid_embedding)),
                float(np.dot(norm_emb, cluster.best_embedding)),
            )
            if sim > best_sim:
                best_sim = sim
                best_match = cluster

        if best_match is None or best_sim < self.similarity_threshold:
            new_id = len(self._clusters) + 1
            new_cluster = IdentityCluster(
                person_id=new_id,
                centroid_embedding=norm_emb.copy(),
                best_embedding=norm_emb.copy(),
                sightings_count=1,
                best_quality=quality,
                best_frame_bgr=frame_bgr.copy(),
                best_crop_bgr=crop_bgr.copy(),
                best_frame_index=frame_index,
                best_timestamp_sec=round(timestamp_sec, 2),
                best_bbox=bbox_list,
            )
            self._clusters.append(new_cluster)
            return ClusterUpdateResult(
                status="discovered",
                cluster=new_cluster,
                similarity=1.0,
                previous_best_score=None,
            )

        # Update running centroid using bounded exponential moving average
        best_match.sightings_count += 1
        alpha = min(0.35, 1.0 / best_match.sightings_count)
        updated_centroid = (1.0 - alpha) * best_match.centroid_embedding + alpha * norm_emb
        best_match.centroid_embedding = normalize_embedding(updated_centroid)

        prev_score = best_match.best_quality.composite_score
        if quality.composite_score > prev_score:
            best_match.best_quality = quality
            best_match.best_embedding = norm_emb.copy()
            best_match.best_frame_bgr = frame_bgr.copy()
            best_match.best_crop_bgr = crop_bgr.copy()
            best_match.best_frame_index = frame_index
            best_match.best_timestamp_sec = round(timestamp_sec, 2)
            best_match.best_bbox = bbox_list
            return ClusterUpdateResult(
                status="upgraded",
                cluster=best_match,
                similarity=round(best_sim, 4),
                previous_best_score=prev_score,
            )

        return ClusterUpdateResult(
            status="seen",
            cluster=best_match,
            similarity=round(best_sim, 4),
            previous_best_score=prev_score,
        )
