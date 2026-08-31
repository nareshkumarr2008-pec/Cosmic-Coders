"""
Web API for the lunar image registration pipeline.

Wraps the existing CLI pipeline (run_pipeline.py / matchers / utils) in a
small Flask app so the same preprocessing -> matching -> refinement ->
spatial-uniformity -> registration -> evaluation stages can be driven
from a browser instead of the command line, with every intermediate
artifact (match visualization, checkerboard, diff map, coverage heatmap)
returned as an image the frontend can render directly.

No pipeline logic lives in this file — it only orchestrates calls into
matchers/ and utils/, exactly as run_pipeline.py does, then serializes
results to JSON.
"""
from __future__ import annotations

import base64
import io
import time
import traceback

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from matchers.base import MatchResult
from matchers.sift_matcher import SIFTMatcher
import matchers.loftr_matcher as loftr_mod
import matchers.rift_matcher as rift_mod
from utils.preprocess import preprocess_pair, to_grayscale, ensure_bgr
from utils.refine import refine_matches
from utils.spatial import grid_cap_filter, distribution_report
from utils.register import fit_homography, warp_with_homography, make_checkerboard, make_diff_map
from utils.metrics import full_report, bakeoff_table
from utils.visualize import draw_matches, heatmap_overlay, side_by_side

app = Flask(__name__, static_folder="webapp/static", static_url_path="/static")

MAX_DIM = 1600  # guard rail so a huge upload doesn't stall a demo

MATCHER_REGISTRY = {
    "sift": lambda: SIFTMatcher(),
    "loftr": lambda: loftr_mod.LoFTRMatcher() if loftr_mod.is_available() else None,
    "rift": lambda: rift_mod.RIFTMatcher() if rift_mod.is_available() else None,
}

MATCHER_LABELS = {
    "sift": "SIFT (baseline)",
    "loftr": "LoFTR (illumination-robust)",
    "rift": "RIFT (phase-congruency)",
}


def matcher_availability() -> dict:
    return {
        "sift": True,
        "loftr": loftr_mod.is_available(),
        "rift": rift_mod.is_available(),
    }


def build_matcher(name: str):
    factory = MATCHER_REGISTRY.get(name)
    if factory is None:
        raise ValueError(f"Unknown matcher '{name}'")
    matcher = factory()
    if matcher is None:
        raise RuntimeError(f"Matcher '{name}' is not available in this environment")
    return matcher


def _decode_upload(file_storage) -> np.ndarray:
    data = np.frombuffer(file_storage.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not decode image '{file_storage.filename}'")
    return _clamp_size(ensure_bgr(img))


def _clamp_size(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= MAX_DIM:
        return img
    scale = MAX_DIM / long_side
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _encode_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def run_one_matcher(name: str, norm1, norm2, orig1, orig2, grid_size, max_per_tile, gsd):
    t0 = time.time()
    matcher = build_matcher(name)
    raw_match: MatchResult = matcher.match(norm1, norm2)

    result = {
        "matcher": name,
        "label": MATCHER_LABELS.get(name, name),
        "raw_match_count": int(len(raw_match)),
    }

    if len(raw_match) < 4:
        result.update(status="failed", reason=f"only {len(raw_match)} raw matches found (need >= 4)")
        return result

    raw_preview = draw_matches(norm1, norm2, raw_match, max_lines=200)

    refined_match, H = refine_matches(norm1, norm2, raw_match)
    if len(refined_match) < 4 or H is None:
        result.update(
            status="failed",
            reason="robust homography fit failed or too few inliers survived",
            images={"raw_matches": _encode_png(raw_preview)},
        )
        return result

    uniform_match = grid_cap_filter(refined_match, norm1.shape, grid_size=grid_size, max_per_tile=max_per_tile)

    H_final = fit_homography(uniform_match)
    if H_final is None:
        H_final = H

    report = full_report(raw_match, uniform_match, H_final, norm1.shape, grid_size, gsd)
    report["matcher"] = name
    report["label"] = MATCHER_LABELS.get(name, name)
    report["raw_match_count"] = int(len(raw_match))
    report["status"] = "ok"
    report["elapsed_ms"] = round((time.time() - t0) * 1000, 1)

    warped = warp_with_homography(orig2, H_final, orig1.shape)
    checkerboard = make_checkerboard(orig1, warped)
    diff = make_diff_map(orig1, warped)
    inlier_preview = draw_matches(norm1, norm2, uniform_match, max_lines=250)
    dist = report["spatial_distribution"]
    heatmap = heatmap_overlay(orig1, dist["tile_counts"], tuple(dist["grid_size"]))

    report["images"] = {
        "raw_matches": _encode_png(raw_preview),
        "inlier_matches": _encode_png(inlier_preview),
        "registered": _encode_png(warped),
        "checkerboard": _encode_png(checkerboard),
        "diffmap": _encode_png(diff),
        "coverage_heatmap": _encode_png(heatmap),
    }
    return report


@app.route("/")
def index():
    return send_from_directory("webapp", "index.html")


@app.route("/api/status")
def status():
    return jsonify({"ok": True, "matchers": matcher_availability()})


@app.route("/api/sample-preview")
def sample_preview():
    ref = cv2.imread("data/reference.png", cv2.IMREAD_UNCHANGED)
    tgt = cv2.imread("data/target.png", cv2.IMREAD_UNCHANGED)
    if ref is None or tgt is None:
        return jsonify({"ok": False, "error": "sample data not found"}), 404
    return jsonify({"ok": True, "reference": _encode_png(ensure_bgr(ref)), "target": _encode_png(ensure_bgr(tgt))})


@app.route("/api/run", methods=["POST"])
def run():
    try:
        use_sample = request.form.get("use_sample") == "true"
        if use_sample:
            orig1 = cv2.imread("data/reference.png", cv2.IMREAD_UNCHANGED)
            orig2 = cv2.imread("data/target.png", cv2.IMREAD_UNCHANGED)
            if orig1 is None or orig2 is None:
                return jsonify({"ok": False, "error": "sample data not found"}), 404
            orig1, orig2 = ensure_bgr(orig1), ensure_bgr(orig2)
        else:
            if "img1" not in request.files or "img2" not in request.files:
                return jsonify({"ok": False, "error": "Both img1 and img2 are required"}), 400
            orig1 = _decode_upload(request.files["img1"])
            orig2 = _decode_upload(request.files["img2"])

        matcher_choice = request.form.get("matcher", "sift")
        grid_rows = int(request.form.get("grid_rows", 8))
        grid_cols = int(request.form.get("grid_cols", 8))
        max_per_tile = int(request.form.get("max_per_tile", 15))
        gsd_raw = request.form.get("gsd", "").strip()
        gsd = float(gsd_raw) if gsd_raw else None

        pre = preprocess_pair(orig1, orig2)
        norm1, norm2 = pre["norm1"], pre["norm2"]
        grid_size = (grid_rows, grid_cols)

        matcher_names = ["sift", "loftr", "rift"] if matcher_choice == "all" else [matcher_choice]

        reports = []
        for name in matcher_names:
            try:
                avail = matcher_availability()
                if not avail.get(name, False):
                    reports.append({
                        "matcher": name,
                        "label": MATCHER_LABELS.get(name, name),
                        "status": "unavailable",
                        "reason": f"'{name}' backend not installed in this environment",
                    })
                    continue
                reports.append(
                    run_one_matcher(name, norm1, norm2, orig1, orig2, grid_size, max_per_tile, gsd)
                )
            except Exception as e:  # noqa: BLE001
                reports.append({
                    "matcher": name,
                    "label": MATCHER_LABELS.get(name, name),
                    "status": "error",
                    "reason": str(e),
                })

        ok_reports = [r for r in reports if r.get("status") == "ok"]
        ranking = [r["matcher"] for r in bakeoff_table(ok_reports)] if ok_reports else []

        preprocessing_preview = {
            "clahe_reference": _encode_png(norm1),
            "clahe_target": _encode_png(norm2),
            "shadow_mask_reference": _encode_png(pre["mask1"]),
            "shadow_mask_target": _encode_png(pre["mask2"]),
        }

        return jsonify({
            "ok": True,
            "reports": reports,
            "ranking": ranking,
            "preprocessing": preprocessing_preview,
            "matcher_availability": matcher_availability(),
        })

    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    # debug=False by default: this is a hackathon prototype server, not a
    # production deployment. Set FLASK_DEBUG=1 in your shell if you want
    # the auto-reloader while editing the frontend.
    import os

    app.run(host="0.0.0.0", port=5050, debug=bool(os.environ.get("FLASK_DEBUG")))
