"""
Evaluation metrics — the explicitly requested deliverable, not an
afterthought. Everything here is computed from real match data, never
placeholder numbers, once wired into the pipeline.
"""

from __future__ import annotations

import numpy as np

from matchers.base import MatchResult
from utils.spatial import distribution_report


def per_point_error(match: MatchResult, H: np.ndarray) -> np.ndarray:
    """Per-match reprojection error in pixels: ||H @ pts1 - pts2||."""
    if len(match) == 0:
        return np.zeros((0,), dtype=np.float32)

    pts1_h = np.hstack([match.pts1, np.ones((len(match), 1))])
    projected = (H @ pts1_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]

    return np.linalg.norm(projected - match.pts2, axis=1)


def rmse(match: MatchResult, H: np.ndarray) -> float:
    errors = per_point_error(match, H)
    if len(errors) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(errors ** 2)))


def inlier_stats(n_raw_matches: int, n_inliers: int) -> dict:
    ratio = n_inliers / n_raw_matches if n_raw_matches > 0 else 0.0
    return {
        "raw_match_count": n_raw_matches,
        "inlier_count": n_inliers,
        "inlier_ratio": round(ratio, 4),
    }


def rmse_to_ground_distance(rmse_px: float, gsd_m_per_px: float) -> float:
    """Convert pixel RMSE to ground distance using ground sample distance
    (GSD), when available from image metadata.
    """
    if np.isnan(rmse_px):
        return float("nan")
    return round(rmse_px * gsd_m_per_px, 4)


def full_report(
    raw_match: MatchResult,
    refined_match: MatchResult,
    H: np.ndarray,
    ref_img_shape: tuple[int, int],
    grid_size: tuple[int, int] = (8, 8),
    gsd_m_per_px: float | None = None,
) -> dict:
    """Assemble the full Stage 7 evaluation report for one image pair."""
    err_px = per_point_error(refined_match, H)
    rmse_px = rmse(refined_match, H)

    report = {
        "matcher": refined_match.name,
        "rmse_px": round(rmse_px, 4) if not np.isnan(rmse_px) else None,
        "mean_error_px": round(float(np.mean(err_px)), 4) if len(err_px) else None,
        "max_error_px": round(float(np.max(err_px)), 4) if len(err_px) else None,
        **inlier_stats(len(raw_match), len(refined_match)),
        "spatial_distribution": distribution_report(refined_match, ref_img_shape, grid_size),
    }

    if gsd_m_per_px is not None:
        report["rmse_ground_m"] = rmse_to_ground_distance(rmse_px, gsd_m_per_px)

    return report


def bakeoff_table(reports: list[dict]) -> list[dict]:
    """Sort multiple matcher reports for side-by-side comparison — lower
    RMSE and higher inlier ratio and coverage are better.
    """
    return sorted(
        reports,
        key=lambda r: (
            r["rmse_px"] if r["rmse_px"] is not None else float("inf"),
            -r["inlier_ratio"],
        ),
    )
