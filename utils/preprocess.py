"""
Illumination normalization for lunar image pairs.

Chandrayaan-2 OHRC/TMC images and the lunar reference image are typically
captured under different sun-incidence angles, so raw-intensity matching
fails even when the underlying terrain is identical. This module flattens
illumination independently on each image (never sharing statistics between
the two) and produces a shadow mask so downstream matchers can down-weight
or exclude keypoints in deep-shadow regions, which shift completely with
sun angle and otherwise produce confident-but-wrong matches.
"""

from __future__ import annotations

import cv2
import numpy as np


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Normalize any decoded image to 3-channel uint8 BGR.

    cv2.imread(..., IMREAD_UNCHANGED) preserves an alpha channel when the
    source PNG has one, producing a 4-channel BGRA array. Every downstream
    function (draw_matches, heatmap_overlay, checkerboard, etc.) assumes
    3-channel BGR, and mixing a 4-channel array with a 3-channel one in a
    per-pixel op (e.g. cv2.addWeighted) raises a cryptic arithm_op size
    error. Call this once, right where images enter the pipeline, so
    nothing downstream has to special-case channel count.
    """
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError(f"Unsupported image shape for ensure_bgr: {img.shape}")


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Ensure single-channel uint8 grayscale."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def clahe_normalize(
    img: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization.

    Applied independently per image (call once per image, never on a
    shared/stacked array) since the two images' lighting differs by
    definition.
    """
    gray = to_grayscale(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def shadow_mask(
    img: np.ndarray,
    morph_kernel: int = 5,
    min_region_area: int = 64,
) -> np.ndarray:
    """Otsu threshold + morphological cleanup to flag deep-shadow regions.

    Returns a uint8 mask (255 = usable, 0 = shadow/excluded).
    """
    gray = to_grayscale(img)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu picks the split point; shadows are the low-intensity side.
    thresh_val, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Drop tiny speckle regions left over from morphology.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    mask = np.zeros_like(cleaned)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_region_area:
            mask[labels == label] = 255

    return mask


def phase_congruency_edges(img: np.ndarray) -> np.ndarray:
    """Gradient-orientation / structure map as an illumination-insensitive
    proxy for full phase congruency. Used to bias matchers (e.g. RIFT-style)
    toward structure rather than raw intensity.
    """
    gray = to_grayscale(img)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def preprocess_pair(img1: np.ndarray, img2: np.ndarray) -> dict:
    """Run the full normalization stage on both images independently.

    Returns a dict with normalized grayscale images, shadow masks, and
    edge/structure maps for both.
    """
    return {
        "norm1": clahe_normalize(img1),
        "norm2": clahe_normalize(img2),
        "mask1": shadow_mask(img1),
        "mask2": shadow_mask(img2),
        "edges1": phase_congruency_edges(img1),
        "edges2": phase_congruency_edges(img2),
    }
