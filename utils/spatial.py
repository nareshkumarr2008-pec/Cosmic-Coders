"""
Spatial uniformity enforcement.

Without this, matches tend to cluster on high-texture regions (crater rims,
boulder fields) while the rest of the image is left unconstrained by any
correspondence — which weakens the global transform fit and is also an
explicitly graded deliverable. This module grids the reference image and
caps matches per tile, then reports a coverage/density metric so the
result is checkable, not just an internal QA step.
"""

from __future__ import annotations

import numpy as np

from matchers.base import MatchResult


def grid_cap_filter(
    match: MatchResult,
    img_shape: tuple[int, int],
    grid_size: tuple[int, int] = (8, 8),
    max_per_tile: int = 15,
) -> MatchResult:
    """Cap the number of retained matches per grid tile (indexed by pts1's
    location in the reference image), keeping the highest-scoring matches
    within each tile.
    """
    if len(match) == 0:
        return match

    h, w = img_shape[:2]
    rows, cols = grid_size
    tile_h, tile_w = h / rows, w / cols

    tile_ids = np.zeros(len(match), dtype=np.int64)
    for i, (x, y) in enumerate(match.pts1):
        col = min(int(x // tile_w), cols - 1)
        row = min(int(y // tile_h), rows - 1)
        tile_ids[i] = row * cols + col

    keep_indices = []
    for tile in np.unique(tile_ids):
        idx_in_tile = np.where(tile_ids == tile)[0]
        if len(idx_in_tile) <= max_per_tile:
            keep_indices.extend(idx_in_tile.tolist())
        else:
            # Keep the highest-confidence matches in this tile.
            top = idx_in_tile[np.argsort(-match.scores[idx_in_tile])[:max_per_tile]]
            keep_indices.extend(top.tolist())

    keep_indices = np.array(sorted(keep_indices))

    return MatchResult(
        pts1=match.pts1[keep_indices],
        pts2=match.pts2[keep_indices],
        scores=match.scores[keep_indices],
        name=match.name,
    )


def min_distance_filter(match: MatchResult, min_dist: float = 15.0) -> MatchResult:
    """Greedy non-max suppression: process matches highest-score first,
    drop any subsequent match within min_dist pixels (in image1 space) of
    an already-kept match. Alternative/complement to grid capping for
    enforcing spread.
    """
    if len(match) == 0:
        return match

    order = np.argsort(-match.scores)
    kept = []
    kept_pts = []

    for idx in order:
        p = match.pts1[idx]
        if not kept_pts:
            kept.append(idx)
            kept_pts.append(p)
            continue
        dists = np.linalg.norm(np.array(kept_pts) - p, axis=1)
        if np.min(dists) >= min_dist:
            kept.append(idx)
            kept_pts.append(p)

    kept = np.array(sorted(kept))
    return MatchResult(
        pts1=match.pts1[kept],
        pts2=match.pts2[kept],
        scores=match.scores[kept],
        name=match.name,
    )


def distribution_report(
    match: MatchResult,
    img_shape: tuple[int, int],
    grid_size: tuple[int, int] = (8, 8),
) -> dict:
    """Quantify how evenly matches are spread across the reference image.

    Returns tile counts, a coverage percentage (fraction of tiles with
    >=1 match), and the standard deviation of per-tile counts (lower =
    more uniform).
    """
    rows, cols = grid_size
    counts = np.zeros((rows, cols), dtype=np.int64)

    if len(match) > 0:
        h, w = img_shape[:2]
        tile_h, tile_w = h / rows, w / cols
        for x, y in match.pts1:
            col = min(int(x // tile_w), cols - 1)
            row = min(int(y // tile_h), rows - 1)
            counts[row, col] += 1

    occupied = int(np.count_nonzero(counts))
    total_tiles = rows * cols

    return {
        "grid_size": [rows, cols],
        "tile_counts": counts.tolist(),
        "coverage_pct": round(100.0 * occupied / total_tiles, 2),
        "count_std": round(float(np.std(counts)), 3),
        "count_mean": round(float(np.mean(counts)), 3),
    }
