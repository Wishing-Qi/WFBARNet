from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.preprocess.track import TrackPreprocessMeta
from src.utils.structures import TrackResult


@dataclass(slots=True)
class _HeatmapCandidate:
    center: tuple[float, float]
    score: float
    rank: tuple[float, float, float, float]


def _extract_ball_candidate(heatmap: np.ndarray, mask: np.ndarray) -> _HeatmapCandidate | None:
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if labels_count <= 1:
        return None

    best: _HeatmapCandidate | None = None
    for label_id in range(1, labels_count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area <= 0:
            continue

        component_mask = labels == label_id
        values = heatmap[component_mask].astype(np.float64, copy=False)
        peak = float(values.max(initial=0.0))
        total = float(values.sum())
        mean = total / float(area)

        ys, xs = np.nonzero(component_mask)
        if total > 1e-8:
            weights = values / total
            center = (float(np.sum(xs * weights)), float(np.sum(ys * weights)))
        else:
            cx, cy = centroids[label_id]
            center = (float(cx), float(cy))

        width = max(int(stats[label_id, cv2.CC_STAT_WIDTH]), 1)
        height = max(int(stats[label_id, cv2.CC_STAT_HEIGHT]), 1)
        compactness = float(area) / float(width * height)
        rank = (peak, mean, min(float(area), 24.0), compactness)
        candidate = _HeatmapCandidate(center=center, score=peak, rank=rank)
        if best is None or candidate.rank > best.rank:
            best = candidate

    return best


def _decode_single_heatmap(
    heatmap: np.ndarray,
    meta: TrackPreprocessMeta,
    score_thr: float,
) -> TrackResult:
    score = float(np.max(heatmap))
    binary_mask = (heatmap > score_thr).astype(np.uint8) * 255
    candidate = _extract_ball_candidate(heatmap, binary_mask)
    if candidate is None:
        return TrackResult(
            ball_xy=[-1.0, -1.0],
            visible=0,
            score=score,
            heatmap_shape=list(heatmap.shape),
        )

    x, y = candidate.center
    ball_xy = [x * meta.scale_x, y * meta.scale_y]
    return TrackResult(
        ball_xy=ball_xy,
        visible=1,
        score=candidate.score,
        heatmap_shape=list(heatmap.shape),
    )


def decode_track_heatmap(
    heatmaps: np.ndarray,
    meta: TrackPreprocessMeta,
    score_thr: float,
) -> TrackResult:
    if heatmaps.ndim == 4:
        heatmap = heatmaps[0, 1]
    elif heatmaps.ndim == 3:
        heatmap = heatmaps[1]
    elif heatmaps.ndim == 2:
        heatmap = heatmaps
    else:
        raise ValueError(f"Unexpected heatmap shape: {heatmaps.shape}")

    return _decode_single_heatmap(heatmap, meta, score_thr)


def decode_track_heatmap_batch(
    batch_heatmaps: np.ndarray,
    metas: list[TrackPreprocessMeta],
    score_thr: float,
) -> list[TrackResult]:
    if batch_heatmaps.ndim == 4:
        heatmap_planes = batch_heatmaps[:, 1]
    elif batch_heatmaps.ndim == 3:
        heatmap_planes = batch_heatmaps
    else:
        raise ValueError(f"Batch heatmaps must be 3D or 4D, got {batch_heatmaps.ndim}D")

    batch_size = heatmap_planes.shape[0]
    if batch_size != len(metas):
        raise ValueError(f"Heatmap batch size {batch_size} doesn't match metas length {len(metas)}")

    results = []
    for i in range(batch_size):
        heatmap = heatmap_planes[i]
        meta = metas[i]

        results.append(_decode_single_heatmap(heatmap, meta, score_thr))

    return results
