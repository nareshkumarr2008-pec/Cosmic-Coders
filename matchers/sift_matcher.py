"""
SIFT baseline matcher.

Always available (pure OpenCV, no GPU/torch dependency), so this is both
the pipeline's scaffolding matcher during development and the "here's why
naive intensity-based matching fails under illumination change" opener in
the demo.
"""

from __future__ import annotations

import cv2
import numpy as np

from matchers.base import BaseMatcher, MatchResult, empty_result


class SIFTMatcher(BaseMatcher):
    name = "sift"

    def __init__(
        self,
        n_features: int = 8000,
        ratio_thresh: float = 0.75,
        contrast_threshold: float = 0.02,
    ):
        self.ratio_thresh = ratio_thresh
        self.detector = cv2.SIFT_create(
            nfeatures=n_features, contrastThreshold=contrast_threshold
        )
        # FLANN is far faster than brute force at SIFT's descriptor count.
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=64)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        self._validate(img1, img2)

        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return empty_result(self.name)

        knn_matches = self.flann.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)

        good_pts1, good_pts2, scores = [], [], []
        for pair in knn_matches:
            if len(pair) != 2:
                continue
            m, n = pair
            # Lowe's ratio test — keeps only distinctive matches.
            if m.distance < self.ratio_thresh * n.distance:
                good_pts1.append(kp1[m.queryIdx].pt)
                good_pts2.append(kp2[m.trainIdx].pt)
                # Convert distance to a 0-1 confidence-like score.
                scores.append(1.0 - (m.distance / (n.distance + 1e-6)))

        if not good_pts1:
            return empty_result(self.name)

        return MatchResult(
            pts1=np.asarray(good_pts1, dtype=np.float32),
            pts2=np.asarray(good_pts2, dtype=np.float32),
            scores=np.asarray(scores, dtype=np.float32),
            name=self.name,
        )
