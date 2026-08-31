"""
Outlier rejection + sub-pixel refinement.

This is a two-step stage, deliberately kept separate:
  1. Fit a robust transform (MAGSAC++, falling back to RANSAC) to discard
     outlier matches.
  2. Refine each surviving inlier's coordinates to sub-pixel accuracy.
     RANSAC inliers are still pixel-level (or worse — detector-level)
     locations; sub-pixel refinement is a distinct, required step, not a
     side-effect of RANSAC.
"""

from __future__ import annotations

import cv2
import numpy as np

from matchers.base import MatchResult


def estimate_robust_homography(
    match: MatchResult,
    reproj_thresh: float = 5.0,
    confidence: float = 0.999,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Fit H mapping pts1 -> pts2 (or pts2 -> pts1, caller decides
    direction by which points are passed as src). Returns (H, inlier_mask).

    Uses MAGSAC++ when available (cv2.USAC_MAGSAC, OpenCV >= 4.5.4) since
    it tolerates a much higher outlier ratio than vanilla RANSAC — useful
    given how noisy raw lunar matches can be. Falls back to RANSAC on
    older OpenCV builds.
    """
    if len(match) < 4:
        return None, np.zeros((0,), dtype=bool)

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)

    H, mask = cv2.findHomography(
        match.pts1,
        match.pts2,
        method=method,
        ransacReprojThreshold=reproj_thresh,
        confidence=confidence,
        maxIters=5000,
    )

    if H is None:
        return None, np.zeros((len(match),), dtype=bool)

    inlier_mask = mask.ravel().astype(bool)
    return H, inlier_mask


def subpixel_refine(
    img: np.ndarray,
    pts: np.ndarray,
    win_size: tuple[int, int] = (5, 5),
    zero_zone: tuple[int, int] = (-1, -1),
    max_iters: int = 40,
    epsilon: float = 0.001,
) -> np.ndarray:
    """Refine integer/float keypoint locations to sub-pixel accuracy using
    corner sub-pixel refinement around each point.

    Points that cornerSubPix cannot refine (e.g. too close to the image
    border, or in a flat region with no corner structure) are left at
    their original location rather than dropped, so the array length
    always matches the input.
    """
    if len(pts) == 0:
        return pts

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) if gray.dtype != np.float32 else gray

    pts32 = pts.astype(np.float32).reshape(-1, 1, 2).copy()

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iters,
        epsilon,
    )

    refined = cv2.cornerSubPix(
        gray.astype(np.uint8) if gray.dtype == np.float32 else gray,
        pts32,
        win_size,
        zero_zone,
        criteria,
    )

    return refined.reshape(-1, 2)


def refine_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    match: MatchResult,
    reproj_thresh: float = 5.0,
) -> tuple[MatchResult, np.ndarray]:
    """Full Stage 4: robust homography -> inlier selection -> sub-pixel
    refinement of the surviving points on both images.

    Returns (refined_inlier_match, homography).
    """
    H, inlier_mask = estimate_robust_homography(match, reproj_thresh=reproj_thresh)

    if H is None or inlier_mask.sum() == 0:
        empty = MatchResult(
            pts1=np.zeros((0, 2), dtype=np.float32),
            pts2=np.zeros((0, 2), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            name=match.name,
        )
        return empty, np.eye(3, dtype=np.float64)

    inlier_pts1 = match.pts1[inlier_mask]
    inlier_pts2 = match.pts2[inlier_mask]
    inlier_scores = match.scores[inlier_mask]

    refined_pts1 = subpixel_refine(img1, inlier_pts1)
    refined_pts2 = subpixel_refine(img2, inlier_pts2)

    refined = MatchResult(
        pts1=refined_pts1.astype(np.float32),
        pts2=refined_pts2.astype(np.float32),
        scores=inlier_scores,
        name=match.name,
    )
    return refined, H
