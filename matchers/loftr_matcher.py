"""
LoFTR matcher (learned, dense, illumination-robust) via kornia.

This is the primary candidate for the SIFT/LoFTR/RIFT bake-off: no MATLAB
dependency risk, works directly on grayscale image pairs, and is
substantially more robust to the illumination differences between
Chandrayaan-2 pairs and the lunar reference than classical descriptors.

If torch/kornia are not installed, `is_available()` returns False and the
pipeline should fall back to SIFT rather than crash — this keeps the whole
system runnable in a lightweight environment.
"""

from __future__ import annotations

import numpy as np

from matchers.base import BaseMatcher, MatchResult, empty_result

try:
    import torch
    import kornia as K
    import kornia.feature as KF

    _KORNIA_AVAILABLE = True
except ImportError:
    _KORNIA_AVAILABLE = False


def is_available() -> bool:
    return _KORNIA_AVAILABLE


class LoFTRMatcher(BaseMatcher):
    name = "loftr"

    def __init__(self, pretrained: str = "outdoor", confidence_thresh: float = 0.5,
                 resize_long_side: int = 1024):
        if not _KORNIA_AVAILABLE:
            raise RuntimeError(
                "kornia/torch not installed. Install via `pip install torch kornia` "
                "or use the SIFT matcher instead."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = KF.LoFTR(pretrained=pretrained).to(self.device).eval()
        self.confidence_thresh = confidence_thresh
        self.resize_long_side = resize_long_side

    def _to_tensor(self, img: np.ndarray, scale: float):
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        import cv2
        resized = cv2.resize(img, (new_w, new_h))
        t = torch.from_numpy(resized).float()[None, None] / 255.0
        return t.to(self.device)

    def _resize_scale(self, img: np.ndarray) -> float:
        long_side = max(img.shape[:2])
        if long_side <= self.resize_long_side:
            return 1.0
        return self.resize_long_side / long_side

    def match(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        self._validate(img1, img2)

        # LoFTR is memory-hungry at full resolution; downscale then rescale
        # match coordinates back to original pixel space.
        scale1 = self._resize_scale(img1)
        scale2 = self._resize_scale(img2)

        t1 = self._to_tensor(img1, scale1)
        t2 = self._to_tensor(img2, scale2)

        with torch.no_grad():
            correspondences = self.model({"image0": t1, "image1": t2})

        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        conf = correspondences["confidence"].cpu().numpy()

        if len(mkpts0) == 0:
            return empty_result(self.name)

        keep = conf >= self.confidence_thresh
        if not np.any(keep):
            return empty_result(self.name)

        pts1 = mkpts0[keep] / scale1
        pts2 = mkpts1[keep] / scale2
        scores = conf[keep]

        return MatchResult(
            pts1=pts1.astype(np.float32),
            pts2=pts2.astype(np.float32),
            scores=scores.astype(np.float32),
            name=self.name,
        )
