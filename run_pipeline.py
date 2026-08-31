"""
End-to-end CLI: preprocessing -> matching -> outlier rejection + sub-pixel
refinement -> spatial uniformity -> registration -> evaluation.

Usage:
    python run_pipeline.py --img1 data/reference.png --img2 data/target.png \
        --matcher loftr --out outputs/

    python run_pipeline.py --img1 data/reference.png --img2 data/target.png \
        --matcher all --out outputs/
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

from matchers.base import MatchResult
from matchers.sift_matcher import SIFTMatcher
import matchers.loftr_matcher as loftr_mod
import matchers.rift_matcher as rift_mod
from utils.preprocess import preprocess_pair, ensure_bgr
from utils.refine import refine_matches
from utils.spatial import grid_cap_filter, distribution_report
from utils.register import fit_homography, warp_with_homography, make_checkerboard, make_diff_map
from utils.metrics import full_report, bakeoff_table


MATCHER_REGISTRY = {
    "sift": lambda: SIFTMatcher(),
    "loftr": lambda: loftr_mod.LoFTRMatcher() if loftr_mod.is_available() else None,
    "rift": lambda: rift_mod.RIFTMatcher() if rift_mod.is_available() else None,
}


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    # Normalize to 3-channel BGR: IMREAD_UNCHANGED preserves an alpha
    # channel for RGBA PNGs, and mixing a 4-channel image with a
    # 3-channel one downstream (e.g. make_checkerboard when --img1 and
    # --img2 have different channel counts) crashes with an OpenCV
    # arithm_op size-mismatch error. See utils/preprocess.ensure_bgr.
    return ensure_bgr(img)


def build_matcher(name: str):
    factory = MATCHER_REGISTRY.get(name)
    if factory is None:
        raise ValueError(f"Unknown matcher '{name}'. Choices: {list(MATCHER_REGISTRY)}")
    matcher = factory()
    if matcher is None:
        raise RuntimeError(
            f"Matcher '{name}' is not available in this environment "
            f"(missing optional dependency or backend not wired in)."
        )
    return matcher


def run_single_matcher(matcher_name: str, norm1, norm2, orig1, orig2, out_dir: str,
                        grid_size=(8, 8), max_per_tile=15, gsd=None) -> dict:
    matcher = build_matcher(matcher_name)
    raw_match: MatchResult = matcher.match(norm1, norm2)

    if len(raw_match) < 4:
        return {
            "matcher": matcher_name,
            "status": "failed",
            "reason": f"only {len(raw_match)} raw matches found (need >= 4)",
        }

    refined_match, H = refine_matches(norm1, norm2, raw_match)

    if len(refined_match) < 4 or H is None:
        return {
            "matcher": matcher_name,
            "status": "failed",
            "reason": "robust homography fit failed or too few inliers survived",
        }

    uniform_match = grid_cap_filter(
        refined_match, norm1.shape, grid_size=grid_size, max_per_tile=max_per_tile
    )

    H_final = fit_homography(uniform_match)
    if H_final is None:
        H_final = H

    report = full_report(raw_match, uniform_match, H_final, norm1.shape, grid_size, gsd)
    report["status"] = "ok"

    # Registration outputs (only written for the winning/selected matcher
    # by the caller, but harmless to compute per-matcher during a bake-off).
    warped = warp_with_homography(orig2, H_final, orig1.shape)
    checkerboard = make_checkerboard(orig1, warped)
    diff = make_diff_map(orig1, warped)

    prefix = os.path.join(out_dir, matcher_name)
    cv2.imwrite(f"{prefix}_registered.png", warped)
    cv2.imwrite(f"{prefix}_checkerboard.png", checkerboard)
    cv2.imwrite(f"{prefix}_diffmap.png", diff)

    report["outputs"] = {
        "registered": f"{prefix}_registered.png",
        "checkerboard": f"{prefix}_checkerboard.png",
        "diffmap": f"{prefix}_diffmap.png",
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Lunar image registration pipeline")
    parser.add_argument("--img1", required=True, help="Reference image path")
    parser.add_argument("--img2", required=True, help="Target image path (to be registered)")
    parser.add_argument(
        "--matcher", default="sift", choices=["sift", "loftr", "rift", "all"],
        help="Matcher backend, or 'all' to run the full bake-off",
    )
    parser.add_argument("--out", default="outputs/", help="Output directory")
    parser.add_argument("--grid-rows", type=int, default=8)
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--max-per-tile", type=int, default=15)
    parser.add_argument("--gsd", type=float, default=None, help="Ground sample distance, m/px")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    orig1 = load_image(args.img1)
    orig2 = load_image(args.img2)

    pre = preprocess_pair(orig1, orig2)
    norm1, norm2 = pre["norm1"], pre["norm2"]

    grid_size = (args.grid_rows, args.grid_cols)

    matcher_names = ["sift", "loftr", "rift"] if args.matcher == "all" else [args.matcher]

    reports = []
    for name in matcher_names:
        try:
            report = run_single_matcher(
                name, norm1, norm2, orig1, orig2, args.out,
                grid_size=grid_size, max_per_tile=args.max_per_tile, gsd=args.gsd,
            )
        except RuntimeError as e:
            report = {"matcher": name, "status": "unavailable", "reason": str(e)}
        reports.append(report)
        print(f"[{name}] status={report['status']}", file=sys.stderr)

    ok_reports = [r for r in reports if r.get("status") == "ok"]

    if args.matcher == "all":
        ranked = bakeoff_table(ok_reports)
        out_path = os.path.join(args.out, "bakeoff_report.json")
        with open(out_path, "w") as f:
            json.dump({"ranking": ranked, "all_results": reports}, f, indent=2)
        print(f"Bake-off report written to {out_path}")
        if ranked:
            print(f"Best matcher: {ranked[0]['matcher']} "
                  f"(RMSE={ranked[0]['rmse_px']}px, inlier_ratio={ranked[0]['inlier_ratio']})")
    else:
        out_path = os.path.join(args.out, "metrics.json")
        with open(out_path, "w") as f:
            json.dump(reports[0], f, indent=2)
        print(f"Metrics written to {out_path}")
        if reports[0].get("status") != "ok":
            print(f"WARNING: {reports[0].get('reason')}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
