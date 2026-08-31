# Lunar Image Registration

Multi-modal / multi-illumination lunar image registration for Chandrayaan-2
(OHRC/TMC) image pairs — a hackathon deliverable set covering:

- Correspondence (matched keypoints) between two lunar images
- Sub-pixel accurate match locations
- Spatially uniform match distribution across the image
- A final registered/warped output image
- Quantitative evaluation metrics (RMSE, inlier ratio, spatial coverage)

Two ways to run it: the original **CLI pipeline**, and a **web app**
(prototype UI) built on top of the same pipeline code.

## Web app

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5050**.

- Upload a reference and target frame, or click "Use synthetic demo pair"
  to try it immediately with no data of your own — `data/reference.png` /
  `data/target.png` are a generated crater scene rendered under two
  different sun angles plus a rotation/perspective offset, regenerable
  with `python data/generate_samples.py`. Swap in real Chandrayaan-2
  OHRC/TMC frames from [ISSDC Pradan](https://pradan.issdc.gov.in/) for
  an actual demo.
- Pick a matcher (SIFT is always available; LoFTR needs `torch` +
  `kornia` installed — see below; RIFT is a stub, see
  `matchers/rift_matcher.py`), or "Run all" for a side-by-side bake-off.
- Tune grid rows/columns, max matches per tile, and optional ground
  sample distance (converts pixel RMSE to metres).
- Step through **Preprocess → Match → Register → Metrics** — each stage
  shows the actual intermediate artifact (CLAHE output, shadow mask,
  raw vs. refined correspondences, checkerboard/diff overlays, spatial
  coverage heatmap), not just a final number.

The web app (`app.py`, `webapp/`) is a thin Flask layer: it calls the
exact same `matchers/` and `utils/` functions as the CLI, then encodes
the intermediate images as base64 PNGs for the browser. No pipeline
logic was duplicated to build it.

## CLI pipeline

```bash
python run_pipeline.py --img1 data/reference.png --img2 data/target.png \
    --matcher loftr --out outputs/
```

Outputs land in `outputs/`:
- `registered.png` — warped target image in reference frame
- `checkerboard.png` — overlay visualization
- `metrics.json` — RMSE, inlier ratio, coverage score, etc.

### Bake-off (multiple matchers, one report)

```bash
python run_pipeline.py --img1 data/reference.png --img2 data/target.png \
    --matcher all --out outputs/
```

Runs SIFT, LoFTR, and RIFT (if available) on the same pair and writes a
comparison table to `outputs/bakeoff_report.json`.

## Pipeline stages

1. **Preprocessing** (`utils/preprocess.py`) — CLAHE illumination normalization
   + Otsu-based shadow masking on each image independently.
2. **Matching** (`matchers/`) — pluggable matcher backends behind a common
   interface: SIFT (baseline), LoFTR (primary, illumination-robust), and a
   RIFT-family stub (optional, phase-congruency based).
3. **Outlier rejection + sub-pixel refinement** (`utils/refine.py`) —
   MAGSAC++ (falls back to RANSAC) homography fit, followed by
   `cv2.cornerSubPix` refinement of inlier points.
4. **Spatial uniformity enforcement** (`utils/spatial.py`) — grid-capped
   filtering so matches don't cluster in one region.
5. **Registration** (`utils/register.py`) — final homography fit + image
   warp, plus a checkerboard overlay for visual QA.
6. **Evaluation** (`utils/metrics.py`) — RMSE, inlier count/ratio, and a
   spatial distribution / coverage score.

## Setup

```bash
pip install -r requirements.txt
```

LoFTR requires `kornia` (pulls in torch). If torch/kornia aren't available
in your environment, the pipeline automatically falls back to the SIFT
matcher — see `matchers/loftr_matcher.py`. The web app's matcher list
greys out anything not installed rather than letting you select it.

RIFT/RIFT2/sRIFD are MATLAB-origin methods. `matchers/rift_matcher.py` is a
stub with setup instructions; if no working Python port is wired in, it
raises `NotImplementedError` and the pipeline should not select it in the
bake-off.

## Project structure

```
lunar_registration/
├── README.md
├── requirements.txt
├── run_pipeline.py          # CLI entry point
├── app.py                   # Flask web API (new)
├── webapp/                  # frontend (new)
│   ├── index.html
│   └── static/
│       ├── styles.css
│       └── app.js
├── data/
│   ├── generate_samples.py  # synthetic demo pair generator (new)
│   ├── reference.png
│   └── target.png
├── matchers/
│   ├── base.py               # common matcher interface + validation
│   ├── sift_matcher.py       # baseline
│   ├── loftr_matcher.py      # primary candidate
│   └── rift_matcher.py       # stretch candidate / stub
└── utils/
    ├── preprocess.py          # CLAHE + shadow masking
    ├── refine.py              # MAGSAC++/RANSAC + sub-pixel refinement
    ├── spatial.py              # grid-capped uniform distribution filter
    ├── register.py             # transform fit + warp + checkerboard
    ├── metrics.py               # RMSE, inlier ratio, distribution score
    └── visualize.py             # match-line drawing + coverage heatmap (new)
```

## Notes for the SIH demo

- The bundled synthetic pair is tuned so classical SIFT partially
  succeeds (~55% inlier ratio) rather than failing outright — enough to
  show every stage of the pipeline working, while still visibly
  struggling with illumination change. That's the actual motivation for
  the SIFT → LoFTR → RIFT bake-off; install `torch`+`kornia` before the
  demo if you want LoFTR's improvement to show up next to it live.
- Swap the synthetic pair for real Chandrayaan-2 OHRC/TMC frames from
  ISSDC Pradan before presenting — synthetic data proves the pipeline
  runs, not that it works on the real sensor characteristics (real
  shot noise, real multi-orbit geometry, real crater statistics).
- If you have ground sample distance (GSD) metadata for your image
  pair, enter it in the web app's parameter panel — it converts pixel
  RMSE into a ground-distance error, which is the number a judge asking
  "how accurate is this in metres" will want.
