"""
Final registration: transform fit + warp + visual QA outputs.

Supports both a single global homography and a piecewise/TPS-style local
warp, since lunar image pairs from different altitudes/orbits are often
not well modeled by one global homography alone.
"""

from __future__ import annotations

import cv2
import numpy as np

from matchers.base import MatchResult


def fit_homography(match: MatchResult) -> np.ndarray | None:
    if len(match) < 4:
        return None
    H, _ = cv2.findHomography(match.pts1, match.pts2, method=0)  # already-refined inliers
    return H


def warp_with_homography(
    src_img: np.ndarray, H: np.ndarray, ref_shape: tuple[int, int]
) -> np.ndarray:
    h, w = ref_shape[:2]
    return cv2.warpPerspective(src_img, H, (w, h))


def fit_thin_plate_spline(match: MatchResult) -> cv2.ThinPlateSplineShapeTransformer:
    """Local, non-rigid warp for cases where a single global homography
    visibly fails (residuals concentrated in one region rather than
    spread evenly — check with metrics.per_point_error before choosing
    this over a plain homography).
    """
    if len(match) < 4:
        raise ValueError("Need at least 4 matches to fit a TPS transform")

    tps = cv2.createThinPlateSplineShapeTransformer()
    pts1 = match.pts1.reshape(1, -1, 2)
    pts2 = match.pts2.reshape(1, -1, 2)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(match))]
    tps.estimateTransformation(pts2, pts1, matches)  # dest, src convention
    return tps


def warp_with_tps(
    src_img: np.ndarray, tps: cv2.ThinPlateSplineShapeTransformer
) -> np.ndarray:
    return tps.warpImage(src_img)


def make_checkerboard(
    ref_img: np.ndarray, warped_img: np.ndarray, tile_size: int = 40
) -> np.ndarray:
    """Alternating-tile overlay of reference vs warped-registered image —
    the single most convincing visual for a demo: misregistration shows
    up as discontinuities across tile boundaries.
    """
    ref = ref_img if ref_img.ndim == 3 else cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
    warped = warped_img if warped_img.ndim == 3 else cv2.cvtColor(warped_img, cv2.COLOR_GRAY2BGR)

    h, w = ref.shape[:2]
    warped = cv2.resize(warped, (w, h))

    board = np.zeros_like(ref)
    for row in range(0, h, tile_size):
        for col in range(0, w, tile_size):
            tile_idx = (row // tile_size) + (col // tile_size)
            src = ref if tile_idx % 2 == 0 else warped
            board[row:row + tile_size, col:col + tile_size] = src[row:row + tile_size, col:col + tile_size]
    return board


def make_diff_map(ref_img: np.ndarray, warped_img: np.ndarray) -> np.ndarray:
    """Absolute-difference heatmap between reference and warped image,
    useful alongside the checkerboard for a quick misregistration check.
    """
    ref_gray = ref_img if ref_img.ndim == 2 else cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    warped_gray = warped_img if warped_img.ndim == 2 else cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.resize(warped_gray, (ref_gray.shape[1], ref_gray.shape[0]))

    diff = cv2.absdiff(ref_gray, warped_gray)
    return cv2.applyColorMap(diff, cv2.COLORMAP_JET)
