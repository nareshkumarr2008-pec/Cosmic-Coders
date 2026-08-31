"""
Generates a synthetic lunar-surface-like image pair for demo purposes.

Real Chandrayaan-2 OHRC/TMC pairs are the intended input, but weren't
bundled with this project. This script builds a plausible stand-in: a
crater field rendered twice under different simulated sun-incidence
angles (so shading direction differs, exactly the failure mode the
pipeline's preprocessing stage is built to survive), with the second
copy also put through a mild rotation + perspective warp so registration
actually has geometric work to do.

Not a substitute for validating against real ISSDC Pradan data before
a hackathon demo, but enough to exercise every stage of the pipeline
end-to-end.
"""
from __future__ import annotations

import os
import numpy as np
import cv2


def _crater_field(size=900, n_craters=220, seed=7):
    rng = np.random.default_rng(seed)
    heightmap = np.zeros((size, size), dtype=np.float64)

    # Base regolith roughness (low-frequency noise, upsampled).
    base = rng.normal(0, 1, (size // 20, size // 20))
    base = cv2.resize(base, (size, size), interpolation=cv2.INTER_CUBIC)
    heightmap += base * 3.0

    for _ in range(n_craters):
        cx, cy = rng.uniform(0, size, size=2)
        r = rng.uniform(6, 55) * (1.0 if rng.random() > 0.08 else rng.uniform(1.5, 3))
        depth = r * rng.uniform(0.25, 0.55)
        rim_h = depth * rng.uniform(0.15, 0.3)

        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        bowl = -depth * np.exp(-(dist ** 2) / (2 * (r * 0.55) ** 2))
        rim = rim_h * np.exp(-((dist - r) ** 2) / (2 * (r * 0.18) ** 2))
        heightmap += bowl + rim

    # Boulders: small bright/dark speckle.
    speckle = rng.normal(0, 1, (size, size))
    speckle = cv2.GaussianBlur(speckle, (0, 0), 0.8)
    heightmap += speckle * 0.6

    return heightmap


def _shade(heightmap, azimuth_deg, elevation_deg):
    """Simple Lambertian hillshade — mimics sun-angle-dependent shadowing."""
    gy, gx = np.gradient(heightmap)
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)

    light = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    normal = np.dstack([-gx, -gy, np.ones_like(heightmap)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True) + 1e-9

    shade = normal @ light
    shade = np.clip(shade, 0, None)

    albedo = 0.55  # lunar regolith is dark; keep dynamic range realistic
    img = (shade * 255 * albedo + 30).clip(0, 255).astype(np.uint8)
    return img


def generate(out_dir="data", size=900):
    os.makedirs(out_dir, exist_ok=True)
    heightmap = _crater_field(size=size)

    # Reference: high sun angle (fewer, shorter shadows).
    reference = _shade(heightmap, azimuth_deg=135, elevation_deg=55)

    # Target: differently-angled sun (different shadow lengths/direction),
    # then a real geometric offset applied on top — rotation + slight
    # perspective skew + scale, as if from a different orbit pass. Tuned
    # so classical SIFT still finds a working-but-imperfect registration
    # (illustrates the baseline succeeding partially, not just failing
    # outright — the bake-off / LoFTR pitch is about *improving on* this,
    # not rescuing a total failure).
    target_shaded = _shade(heightmap, azimuth_deg=175, elevation_deg=40)

    h, w = target_shaded.shape
    center = (w / 2, h / 2)
    rot = cv2.getRotationMatrix2D(center, angle=3.0, scale=1.015)
    rot_h = np.vstack([rot, [0, 0, 1]])

    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_pts = src_pts + np.float32(
        [[8, 5], [-5, 9], [4, -6], [-10, -3]]
    )
    persp_h = cv2.getPerspectiveTransform(src_pts, dst_pts)

    H = persp_h @ rot_h
    target = cv2.warpPerspective(target_shaded, H, (w, h), borderValue=20)

    # Mild sensor noise on both, independently (never shared statistics).
    rng = np.random.default_rng(3)
    reference = np.clip(
        reference.astype(np.int16) + rng.normal(0, 3, reference.shape), 0, 255
    ).astype(np.uint8)
    target = np.clip(
        target.astype(np.int16) + rng.normal(0, 3, target.shape), 0, 255
    ).astype(np.uint8)

    ref_path = os.path.join(out_dir, "reference.png")
    tgt_path = os.path.join(out_dir, "target.png")
    cv2.imwrite(ref_path, reference)
    cv2.imwrite(tgt_path, target)

    gt_path = os.path.join(out_dir, "ground_truth_H.npy")
    np.save(gt_path, H)

    print(f"Wrote {ref_path}, {tgt_path}, {gt_path}")
    return ref_path, tgt_path


if __name__ == "__main__":
    generate()
