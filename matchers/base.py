"""
Common interface every matcher backend implements, so the pipeline can
swap SIFT / LoFTR / RIFT in and out without touching downstream code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class MatchResult:
    """Uniform output shape for any matcher backend.

    pts1 / pts2 : (N, 2) float32 arrays of matched keypoint coordinates
                  in image1 / image2 pixel space, same length N, row i
                  in pts1 corresponds to row i in pts2.
    scores      : (N,) float32 confidence per match (1.0 if backend has
                  no notion of confidence, e.g. plain SIFT+BF).
    name        : matcher identifier, e.g. "sift", "loftr", "rift".
    """

    pts1: np.ndarray
    pts2: np.ndarray
    scores: np.ndarray
    name: str

    def __len__(self) -> int:
        return len(self.pts1)


class BaseMatcher(ABC):
    """All matchers take two grayscale uint8 images and return a MatchResult."""

    name: str = "base"

    @abstractmethod
    def match(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        raise NotImplementedError

    def _validate(self, img1: np.ndarray, img2: np.ndarray) -> None:
        if img1 is None or img2 is None:
            raise ValueError(f"[{self.name}] received a None image")
        if img1.ndim not in (2, 3) or img2.ndim not in (2, 3):
            raise ValueError(f"[{self.name}] images must be 2D or 3D arrays")
        if img1.size == 0 or img2.size == 0:
            raise ValueError(f"[{self.name}] received an empty image")


def empty_result(name: str) -> MatchResult:
    """Return a zero-match result — used when a matcher finds nothing or
    fails gracefully, so downstream code doesn't need special-case handling.
    """
    return MatchResult(
        pts1=np.zeros((0, 2), dtype=np.float32),
        pts2=np.zeros((0, 2), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
        name=name,
    )
