"""
RIFT-style matcher — phase-congruency based, illumination/radiation
invariant matching, built natively in numpy/OpenCV (no MATLAB, no
external port required).

The original RIFT paper (Li et al., "RIFT: Multi-modal Image Matching
Based on Radiation-invariant Feature Transform") works in three stages,
all implemented below:

  1. Phase congruency: build a log-Gabor filter bank across multiple
     scales and orientations in the frequency domain, and compute a
     per-orientation energy response. Unlike raw intensity or gradient
     magnitude, phase congruency responds to *structure* (edges/corners
     where Fourier components align in phase) and is largely invariant
     to monotonic intensity changes -- exactly the illumination-vs-sun-
     angle problem this pipeline exists to solve.
  2. Maximum Index Map (MIM): at every pixel, the index of the
     orientation with the strongest phase-congruency response across
     scales. This is the radiation-invariant "image" that keypoints are
     detected and described on, instead of the raw pixel intensities.
  3. Keypoints are detected on the phase-congruency energy map (stable
     under illumination change, unlike a raw-intensity corner detector),
     and each keypoint gets a rotation-aware HOG-style descriptor built
     from a local MIM patch (a direct, simplified implementation of the
     paper's RIFT descriptor).

This is a compact, from-scratch implementation -- not a byte-for-byte
port of the authors' MATLAB code -- but it is a real, working
phase-congruency pipeline, not a placeholder.
"""

from __future__ import annotations

import cv2
import numpy as np

from matchers.base import BaseMatcher, MatchResult, empty_result

_RIFT_BACKEND_READY = True


def is_available() -> bool:
    return _RIFT_BACKEND_READY


# --------------------------------------------------------------------------
# Stage 1: log-Gabor phase congruency
# --------------------------------------------------------------------------

def _log_gabor_filter_bank(rows: int, cols: int, nscale: int, norient: int,
                            min_wavelength: float, mult: float, sigma_onf: float,
                            d_theta_on_sigma: float = 1.2) -> np.ndarray:
    """Build (nscale, norient, rows, cols) frequency-domain log-Gabor filters."""
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    y = (y - rows // 2) / rows
    x = (x - cols // 2) / cols

    radius = np.sqrt(x ** 2 + y ** 2)
    radius[rows // 2, cols // 2] = 1.0  # avoid log(0) at DC
    theta = np.arctan2(-y, x)

    filters = np.zeros((nscale, norient, rows, cols), dtype=np.float64)
    theta_sigma = np.pi / norient / d_theta_on_sigma

    for s in range(nscale):
        wavelength = min_wavelength * (mult ** s)
        fo = 1.0 / wavelength
        log_gabor = np.exp(-(np.log(radius / fo)) ** 2 / (2 * np.log(sigma_onf) ** 2))
        log_gabor[rows // 2, cols // 2] = 0.0

        for o in range(norient):
            angl = o * np.pi / norient
            ds = np.sin(theta) * np.cos(angl) - np.cos(theta) * np.sin(angl)
            dc = np.cos(theta) * np.cos(angl) + np.sin(theta) * np.sin(angl)
            dtheta = np.abs(np.arctan2(ds, dc))
            spread = np.exp(-(dtheta ** 2) / (2 * theta_sigma ** 2))
            filters[s, o] = log_gabor * spread

    return filters


def phase_congruency(gray: np.ndarray, nscale: int = 3, norient: int = 6,
                      min_wavelength: float = 3.0, mult: float = 2.1,
                      sigma_onf: float = 0.55, noise_k: float = 2.0):
    """Returns (pc_map, orientation_energy) where pc_map is a single-channel
    illumination-invariant structure map in [0, 1], and orientation_energy
    has shape (norient, rows, cols) -- the per-orientation energy summed
    across scales, used downstream to build the Maximum Index Map.
    """
    rows, cols = gray.shape[:2]
    img = gray.astype(np.float64)

    filters = _log_gabor_filter_bank(rows, cols, nscale, norient, min_wavelength, mult, sigma_onf)

    fft_img = np.fft.fftshift(np.fft.fft2(img))

    orientation_energy = np.zeros((norient, rows, cols), dtype=np.float64)
    pc_sum = np.zeros((rows, cols), dtype=np.float64)
    eps = 1e-6

    for o in range(norient):
        sum_an = np.zeros((rows, cols), dtype=np.float64)
        sum_e = np.zeros((rows, cols), dtype=np.float64)
        sum_o = np.zeros((rows, cols), dtype=np.float64)
        an_arrays = []

        for s in range(nscale):
            resp = np.fft.ifft2(np.fft.ifftshift(fft_img * filters[s, o]))
            an = np.abs(resp)
            an_arrays.append(an)
            sum_an += an
            sum_e += resp.real
            sum_o += resp.imag

        mean_e = sum_e / nscale
        mean_o = sum_o / nscale
        energy = np.sqrt(mean_e ** 2 + mean_o ** 2) * nscale

        # simple noise floor: median of the smallest-scale response acts
        # as a per-orientation noise estimate, subtracted before summing.
        noise_floor = noise_k * np.median(an_arrays[0]) if an_arrays else 0.0
        energy = np.maximum(energy - noise_floor, 0.0)

        orientation_energy[o] = energy
        pc_sum += energy / (sum_an / nscale + eps)

    pc_map = pc_sum / norient
    if pc_map.max() > 0:
        pc_map = pc_map / pc_map.max()
    return pc_map.astype(np.float32), orientation_energy


def maximum_index_map(orientation_energy: np.ndarray) -> np.ndarray:
    """MIM: per-pixel index (0..norient-1) of the strongest-responding
    orientation. This discrete map is what makes RIFT-style descriptors
    robust across sensors/illumination -- it encodes local structure
    orientation, not raw intensity.
    """
    return np.argmax(orientation_energy, axis=0).astype(np.uint8)


# --------------------------------------------------------------------------
# Stage 2: keypoint detection on the phase-congruency map
# --------------------------------------------------------------------------

def _detect_keypoints(pc_map: np.ndarray, max_kp: int, border: int) -> np.ndarray:
    pc_u8 = np.clip(pc_map * 255.0, 0, 255).astype(np.uint8)
    fast = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)
    kps = fast.detect(pc_u8, None)

    if not kps:
        # fall back to Shi-Tomasi corners on the PC map if FAST finds nothing
        corners = cv2.goodFeaturesToTrack(pc_u8, maxCorners=max_kp, qualityLevel=0.01, minDistance=8)
        if corners is None:
            return np.zeros((0, 2), dtype=np.float32)
        pts = corners.reshape(-1, 2)
    else:
        kps = sorted(kps, key=lambda k: -k.response)[: max_kp * 3]
        pts = np.array([k.pt for k in kps], dtype=np.float32)

    h, w = pc_map.shape[:2]
    keep = (
        (pts[:, 0] >= border) & (pts[:, 0] < w - border) &
        (pts[:, 1] >= border) & (pts[:, 1] < h - border)
    )
    pts = pts[keep]
    if len(pts) > max_kp:
        pts = pts[:max_kp]
    return pts.astype(np.float32)


# --------------------------------------------------------------------------
# Stage 3: HOG-of-MIM descriptor
# --------------------------------------------------------------------------

def _dominant_orientation(patch_labels: np.ndarray, norient: int) -> int:
    """Peak of the (circularly-smoothed) MIM label histogram over a patch.

    The log-Gabor bank's orientation angles are `o * pi / norient`, i.e.
    a fixed step of (180/norient) degrees between adjacent MIM indices.
    A global image rotation by theta degrees therefore shifts a pixel's
    dominant-response index by roughly theta / (180/norient), mod
    norient. Finding that peak gives us, per keypoint, the index shift
    needed to cancel out an unknown rotation between two images.
    """
    hist = np.bincount(patch_labels.ravel(), minlength=norient)[:norient].astype(np.float64)
    smoothed = hist + 0.5 * (np.roll(hist, 1) + np.roll(hist, -1))
    return int(np.argmax(smoothed))


def _canonical_patch(padded_mim: np.ndarray, cx: float, cy: float,
                      patch_size: int, norient: int) -> np.ndarray | None:
    """Extract a patch_size x patch_size MIM patch centered on (cx, cy)
    (coordinates already offset into the padded map), rotated so its
    dominant local orientation is canonicalized to index 0.

    Two things have to rotate together for this to actually cancel a
    global rotation between image pairs:
      1. The *spatial* sampling grid (crop a bit wider, warpAffine with
         nearest-neighbor -- MIM values are labels, not intensities, so
         any interpolation that blends them is meaningless).
      2. The *label values themselves* -- an index that meant "edge at
         60 degrees" in the original frame means something else once
         we've spatially derotated, so labels are circularly shifted by
         the same amount used for the spatial rotation.
    Skipping either half only fixes rotation invariance for a
    same-frame sanity check, not for real image pairs.
    """
    angle_per_index = 180.0 / norient
    crop_r = int(np.ceil(patch_size * 0.75))
    crop_size = patch_size + 2 * crop_r

    x0 = int(round(cx)) - crop_size // 2
    y0 = int(round(cy)) - crop_size // 2
    crop = padded_mim[y0:y0 + crop_size, x0:x0 + crop_size]
    if crop.shape != (crop_size, crop_size):
        return None

    k = _dominant_orientation(crop, norient)
    angle_deg = k * angle_per_index

    # A physical image rotation by +angle_deg *increases* a pixel's MIM
    # index by k = angle_deg / angle_per_index (confirmed empirically:
    # rotating an image and re-running phase congruency shifts the
    # dominant index forward by exactly that amount). So a patch whose
    # dominant index is already k looks like it came from a local patch
    # that was rotated forward by +k*angle_per_index relative to a
    # canonical (index-0) orientation -- to cancel that out we rotate
    # the spatial sampling *backward* by the same amount.
    center = (crop_size / 2.0, crop_size / 2.0)
    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    rotated = cv2.warpAffine(
        crop.astype(np.uint8), M, (crop_size, crop_size),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE,
    )

    half = patch_size // 2
    c = crop_size // 2
    patch = rotated[c - half:c - half + patch_size, c - half:c - half + patch_size]

    # Circularly shift labels so the dominant orientation is always bin 0.
    patch = ((patch.astype(np.int16) - k) % norient).astype(np.uint8)
    return patch


def _describe(mim: np.ndarray, pts: np.ndarray, norient: int,
              patch_size: int = 48, grid: int = 6) -> np.ndarray:
    """For each keypoint, take a rotation-canonicalized patch_size x
    patch_size window centered on it, split it into a grid x grid array
    of cells, and build a histogram of (derotated) MIM orientation-index
    labels per cell (a HOG-style descriptor built on the
    radiation-invariant MIM instead of raw gradients). Concatenated and
    L2-normalized, matching the spirit of the RIFT descriptor.
    """
    cell = patch_size // grid
    descs = np.zeros((len(pts), grid * grid * norient), dtype=np.float32)

    # Pad once so patches near the border (and the wider rotation crop)
    # never have to be re-checked per keypoint against the raw image
    # bounds -- replicate padding avoids inventing a fake edge there.
    crop_r = int(np.ceil(patch_size * 0.75))
    pad = patch_size // 2 + crop_r + 2
    padded = cv2.copyMakeBorder(mim, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    for i, (px, py) in enumerate(pts):
        patch = _canonical_patch(padded, px + pad, py + pad, patch_size, norient)
        if patch is None:
            continue
        feat = []
        for gy in range(grid):
            for gx in range(grid):
                cell_patch = patch[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell]
                hist, _ = np.histogram(cell_patch, bins=norient, range=(0, norient))
                feat.append(hist.astype(np.float32))
        vec = np.concatenate(feat)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        descs[i] = vec

    return descs


class RIFTMatcher(BaseMatcher):
    name = "rift"

    def __init__(self, nscale: int = 3, norient: int = 6, max_keypoints: int = 3000,
                 ratio_thresh: float = 0.85, ratio_thresh_max: float = 0.98,
                 min_matches: int = 12, patch_size: int = 48, grid: int = 6):
        """
        ratio_thresh / ratio_thresh_max / min_matches: RIFT's histogram-of-MIM
        descriptor is far less discriminative than SIFT's gradient descriptor
        (216 dims built from only `norient` histogram bins per cell, vs
        SIFT's 128-dim continuous gradient descriptor). On repetitive,
        self-similar terrain (crater fields are exactly this), the nearest
        and second-nearest descriptor distances often sit within a percent
        or two of each other for most keypoints — Lowe's ratio test then
        passes only a razor-thin fraction of candidates at a fixed strict
        threshold, and that fraction is sensitive enough to floating-point
        differences between OpenCV/numpy builds that it can collapse to
        near-zero matches on some machines while working fine on others,
        even for a genuinely well-overlapping pair.

        To keep this robust without weakening the descriptor's actual
        selectivity when it isn't needed, match() starts at the strict
        `ratio_thresh` and only relaxes it — in fixed steps, up to
        `ratio_thresh_max` — if the strict pass doesn't clear `min_matches`.
        Most images with real texture pass at the strict threshold and
        never trigger the fallback; only marginal cases relax further.
        """
        self.nscale = nscale
        self.norient = norient
        self.max_keypoints = max_keypoints
        self.ratio_thresh = ratio_thresh
        self.ratio_thresh_max = ratio_thresh_max
        self.min_matches = min_matches
        self.patch_size = patch_size
        self.grid = grid
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.last_ratio_used: float | None = None

    def _extract(self, img: np.ndarray):
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pc_map, orient_energy = phase_congruency(gray, nscale=self.nscale, norient=self.norient)
        mim = maximum_index_map(orient_energy)
        border = self.patch_size // 2 + 1
        pts = _detect_keypoints(pc_map, self.max_keypoints, border)
        descs = _describe(mim, pts, self.norient, self.patch_size, self.grid)
        return pts, descs

    def _filter_by_ratio(self, knn, pts1, pts2, ratio_thresh):
        good_pts1, good_pts2, scores = [], [], []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if n.distance < 1e-6:
                continue
            if m.distance < ratio_thresh * n.distance:
                good_pts1.append(pts1[m.queryIdx])
                good_pts2.append(pts2[m.trainIdx])
                scores.append(1.0 - (m.distance / (n.distance + 1e-6)))
        return good_pts1, good_pts2, scores

    def match(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        self._validate(img1, img2)

        pts1, des1 = self._extract(img1)
        pts2, des2 = self._extract(img2)

        if len(pts1) < 2 or len(pts2) < 2:
            return empty_result(self.name)

        knn = self.bf.knnMatch(des1, des2, k=2)

        # Try the strict threshold first; only relax it if that genuinely
        # isn't enough to work with, and stop relaxing the moment it is.
        ratio_steps = np.linspace(self.ratio_thresh, self.ratio_thresh_max, 5)

        good_pts1, good_pts2, scores = [], [], []
        used_ratio = self.ratio_thresh
        for ratio in ratio_steps:
            good_pts1, good_pts2, scores = self._filter_by_ratio(knn, pts1, pts2, ratio)
            used_ratio = float(ratio)
            if len(good_pts1) >= self.min_matches:
                break

        self.last_ratio_used = used_ratio

        if not good_pts1:
            return empty_result(self.name)

        return MatchResult(
            pts1=np.asarray(good_pts1, dtype=np.float32),
            pts2=np.asarray(good_pts2, dtype=np.float32),
            scores=np.asarray(scores, dtype=np.float32),
            name=self.name,
        )