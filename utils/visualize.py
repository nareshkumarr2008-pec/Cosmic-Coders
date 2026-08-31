"""
Visualization helpers for the web UI.

Kept separate from register.py / spatial.py so the core pipeline stays
free of anything display-only. Everything here takes the same
MatchResult / numpy conventions as the rest of the codebase.
"""
from __future__ import annotations

import cv2
import numpy as np

from matchers.base import MatchResult


def draw_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    match: MatchResult,
    max_lines: int = 250,
    line_thickness: int = 1,
    point_radius: int = 3,
) -> np.ndarray:
    """Side-by-side panel with lines drawn between corresponding points.

    Colors lines by a green->red ramp keyed on match score (if the
    matcher provides meaningful scores) so weak matches are visually
    distinguishable from strong ones even before outlier rejection.
    """
    im1 = img1 if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    im2 = img2 if img2.ndim == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    h = max(im1.shape[0], im2.shape[0])
    w1, w2 = im1.shape[1], im2.shape[1]
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[: im1.shape[0], :w1] = im1
    canvas[: im2.shape[0], w1 : w1 + w2] = im2

    n = len(match)
    if n == 0:
        return canvas

    idx = np.arange(n)
    if n > max_lines:
        # Sample by score so the drawing stays representative, not just
        # the first N in array order.
        idx = np.argsort(-match.scores)[:max_lines]

    scores = match.scores[idx]
    s_min, s_max = float(scores.min()), float(scores.max())
    s_range = (s_max - s_min) or 1.0

    for i in idx:
        p1 = tuple(np.round(match.pts1[i]).astype(int))
        p2 = tuple(np.round(match.pts2[i]).astype(int) + np.array([w1, 0]))

        norm_score = (match.scores[i] - s_min) / s_range
        color = (
            int(60 + (1 - norm_score) * 120),   # B
            int(80 + norm_score * 140),          # G
            int(70 + (1 - norm_score) * 60),     # R
        )

        cv2.line(canvas, p1, p2, color, line_thickness, cv2.LINE_AA)
        cv2.circle(canvas, p1, point_radius, (255, 200, 80), -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, point_radius, (255, 200, 80), -1, cv2.LINE_AA)

    return canvas


def heatmap_overlay(
    ref_img: np.ndarray,
    tile_counts: list,
    grid_size: tuple,
    alpha: float = 0.45,
) -> np.ndarray:
    """Render the spatial-distribution grid as a translucent heatmap over
    the reference image — the coverage/uniformity metric made visible.
    """
    ref = ref_img if ref_img.ndim == 3 else cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
    h, w = ref.shape[:2]
    rows, cols = grid_size

    counts = np.array(tile_counts, dtype=np.float32)
    small = counts  # already (rows, cols)
    heat = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    if heat.max() > 0:
        heat_norm = (heat / heat.max() * 255).astype(np.uint8)
    else:
        heat_norm = heat.astype(np.uint8)

    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_TURBO)
    blended = cv2.addWeighted(ref, 1 - alpha, heat_color, alpha, 0)

    # Draw grid lines so tile boundaries are legible.
    tile_h, tile_w = h / rows, w / cols
    for r in range(1, rows):
        y = int(r * tile_h)
        cv2.line(blended, (0, y), (w, y), (255, 255, 255), 1, cv2.LINE_AA)
    for c in range(1, cols):
        x = int(c * tile_w)
        cv2.line(blended, (x, 0), (x, h), (255, 255, 255), 1, cv2.LINE_AA)

    return blended


def side_by_side(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    im1 = img1 if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    im2 = img2 if img2.ndim == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    h = max(im1.shape[0], im2.shape[0])
    w1, w2 = im1.shape[1], im2.shape[1]
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[: im1.shape[0], :w1] = im1
    canvas[: im2.shape[0], w1 : w1 + w2] = im2
    return canvas
