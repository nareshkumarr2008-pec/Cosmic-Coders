const state = {
  files: { ref: null, tgt: null },
  useSample: false,
  availability: { sift: true, loftr: false, rift: false },
  matcher: "sift",
  result: null,
  stage: "preprocess",
  activeMatcherTab: null,
};

const MATCHER_META = {
  sift: { name: "SIFT", desc: "Classical baseline — fast, fails hard under illumination change" },
  loftr: { name: "LoFTR", desc: "Learned, dense matching — robust to sun-angle change" },
  rift: { name: "RIFT", desc: "Phase-congruency — multi-modal, MATLAB-origin stub" },
  all: { name: "Run all (bake-off)", desc: "Every available matcher, ranked side by side" },
};

// ---------------------------------------------------------------- init

async function init() {
  wireDropzone("dz-ref", "file-ref", "preview-ref", "ref");
  wireDropzone("dz-tgt", "file-tgt", "preview-tgt", "tgt");
  document.getElementById("btn-sample").addEventListener("click", useSamplePair);
  document.getElementById("btn-run").addEventListener("click", runPipeline);

  wireSlider("grid-rows", "grid-rows-val");
  wireSlider("grid-cols", "grid-cols-val");
  wireSlider("max-per-tile", "max-per-tile-val");

  document.querySelectorAll(".step").forEach((btn) => {
    btn.addEventListener("click", () => setStage(btn.dataset.stage));
  });

  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    state.availability = data.matchers;
    setEnvStatus(true);
  } catch (e) {
    setEnvStatus(false);
  }
  renderMatcherList();
}

function setEnvStatus(ok) {
  const el = document.getElementById("env-status");
  if (ok) {
    el.textContent = "backend connected";
    el.className = "pill pill-ok";
  } else {
    el.textContent = "backend unreachable";
    el.className = "pill pill-warn";
  }
}

function wireSlider(id, outId) {
  const input = document.getElementById(id);
  const out = document.getElementById(outId);
  input.addEventListener("input", () => (out.textContent = input.value));
}

// ---------------------------------------------------------------- uploads

function wireDropzone(zoneId, inputId, previewId, slot) {
  const zone = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  const preview = document.getElementById(previewId);

  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files[0]) handleFile(input.files[0], zone, preview, slot);
  });

  ["dragover", "dragenter"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.add("drag-over");
    })
  );
  ["dragleave", "dragend", "drop"].forEach((evt) =>
    zone.addEventListener(evt, () => zone.classList.remove("drag-over"))
  );
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f, zone, preview, slot);
  });
}

function handleFile(file, zone, preview, slot) {
  state.files[slot] = file;
  state.useSample = false;
  const url = URL.createObjectURL(file);
  preview.src = url;
  zone.classList.add("filled");
  zone.querySelector(".dropzone-hint").textContent = file.name;
  setNote("");
}

async function useSamplePair() {
  state.useSample = true;
  state.files.ref = null;
  state.files.tgt = null;
  setNote("using synthetic demo pair…");
  try {
    const res = await fetch("/api/sample-preview");
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "failed to load sample");

    const refZone = document.getElementById("dz-ref");
    const tgtZone = document.getElementById("dz-tgt");
    document.getElementById("preview-ref").src = data.reference;
    document.getElementById("preview-tgt").src = data.target;
    refZone.classList.add("filled");
    tgtZone.classList.add("filled");
    refZone.querySelector(".dropzone-hint").textContent = "synthetic reference.png";
    tgtZone.querySelector(".dropzone-hint").textContent = "synthetic target.png";
    setNote("demo pair loaded — ready to run");
  } catch (e) {
    setNote(e.message, true);
  }
}

// ---------------------------------------------------------------- matcher list

function renderMatcherList() {
  const list = document.getElementById("matcher-list");
  list.innerHTML = "";
  const order = ["sift", "loftr", "rift", "all"];

  order.forEach((key) => {
    const meta = MATCHER_META[key];
    const isAvailable = key === "all" ? true : state.availability[key];
    const opt = document.createElement("div");
    opt.className = "matcher-opt" + (key === state.matcher ? " selected" : "") + (!isAvailable ? " disabled" : "");
    opt.innerHTML = `
      <div>
        <span class="matcher-opt-name">${meta.name}</span>
        <span class="matcher-opt-desc">${meta.desc}</span>
      </div>
      <span class="matcher-opt-badge ${isAvailable ? "ready" : ""}">${isAvailable ? "ready" : "not installed"}</span>
    `;
    if (isAvailable) {
      opt.addEventListener("click", () => {
        state.matcher = key;
        renderMatcherList();
      });
    }
    list.appendChild(opt);
  });
}

// ---------------------------------------------------------------- run

function setNote(text, isError = false) {
  const el = document.getElementById("run-note");
  el.textContent = text;
  el.className = "rail-note" + (isError ? " error" : "");
}

async function runPipeline() {
  const hasUpload = state.files.ref && state.files.tgt;
  if (!hasUpload && !state.useSample) {
    setNote("load a reference and target frame first", true);
    return;
  }

  const btn = document.getElementById("btn-run");
  btn.disabled = true;
  btn.querySelector(".run-btn-label").textContent = "Running…";
  setNote("preprocessing → matching → refining → registering → scoring");

  const form = new FormData();
  if (state.useSample) {
    form.append("use_sample", "true");
  } else {
    form.append("img1", state.files.ref);
    form.append("img2", state.files.tgt);
  }
  form.append("matcher", state.matcher);
  form.append("grid_rows", document.getElementById("grid-rows").value);
  form.append("grid_cols", document.getElementById("grid-cols").value);
  form.append("max_per_tile", document.getElementById("max-per-tile").value);
  form.append("gsd", document.getElementById("gsd").value);

  try {
    const res = await fetch("/api/run", { method: "POST", body: form });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "pipeline failed");

    state.result = data;
    const okReports = data.reports.filter((r) => r.status === "ok");
    state.activeMatcherTab = okReports.length ? okReports[0].matcher : data.reports[0].matcher;

    setNote(`done — ${okReports.length}/${data.reports.length} matcher(s) succeeded`);
    setStage("preprocess");
  } catch (e) {
    setNote(e.message, true);
  } finally {
    btn.disabled = false;
    btn.querySelector(".run-btn-label").textContent = "Run pipeline";
  }
}

// ---------------------------------------------------------------- stages

function setStage(stage) {
  state.stage = stage;
  document.querySelectorAll(".step").forEach((b) => b.classList.toggle("active", b.dataset.stage === stage));
  render();
}

function render() {
  const body = document.getElementById("viewport-body");
  const tabsEl = document.getElementById("matcher-tabs");

  if (!state.result) {
    tabsEl.hidden = true;
    return; // keep empty state markup
  }

  const reports = state.result.reports;
  const multi = reports.length > 1;
  tabsEl.hidden = !multi;
  if (multi) {
    tabsEl.innerHTML = "";
    reports.forEach((r) => {
      const tab = document.createElement("button");
      tab.className = "stage-tab" + (r.matcher === state.activeMatcherTab ? " active" : "");
      tab.textContent = `${r.label || r.matcher}${r.status !== "ok" ? " (" + r.status + ")" : ""}`;
      tab.addEventListener("click", () => {
        state.activeMatcherTab = r.matcher;
        render();
      });
      tabsEl.appendChild(tab);
    });
  }

  const report = reports.find((r) => r.matcher === state.activeMatcherTab) || reports[0];

  if (state.stage === "preprocess") body.innerHTML = "";
  body.innerHTML = "";

  if (state.stage === "preprocess") return renderPreprocess(body);
  if (report.status !== "ok") return renderUnavailable(body, report);
  if (state.stage === "match") return renderMatch(body, report);
  if (state.stage === "register") return renderRegister(body, report);
  if (state.stage === "metrics") return renderMetrics(body, report, reports);
}

function renderUnavailable(body, report) {
  const div = document.createElement("div");
  div.className = "status-banner" + (report.status === "unavailable" ? " unavailable" : "");
  div.textContent = `${report.label || report.matcher}: ${report.reason || report.status}`;
  body.appendChild(div);
}

function imageCard(src, caption, wide = false) {
  const fig = document.createElement("figure");
  fig.className = "image-card" + (wide ? " wide" : "");
  const img = document.createElement("img");
  img.src = src;
  img.loading = "lazy";
  const cap = document.createElement("figcaption");
  cap.textContent = caption;
  fig.appendChild(img);
  fig.appendChild(cap);
  return fig;
}

function sectionTitle(text, small = "") {
  const h = document.createElement("p");
  h.className = "panel-title";
  h.innerHTML = `<span>${text}</span>${small ? `<small>${small}</small>` : ""}`;
  return h;
}

function renderPreprocess(body) {
  const pre = state.result.preprocessing;
  const block = document.createElement("div");
  block.className = "section-block";
  block.appendChild(sectionTitle("Illumination normalization", "CLAHE, applied independently per frame"));
  const grid1 = document.createElement("div");
  grid1.className = "image-grid";
  grid1.appendChild(imageCard(pre.clahe_reference, "reference — CLAHE normalized"));
  grid1.appendChild(imageCard(pre.clahe_target, "target — CLAHE normalized"));
  block.appendChild(grid1);

  block.appendChild(sectionTitle("Shadow masking", "Otsu threshold + morphological cleanup, 255 = usable"));
  const grid2 = document.createElement("div");
  grid2.className = "image-grid";
  grid2.appendChild(imageCard(pre.shadow_mask_reference, "reference — shadow mask"));
  grid2.appendChild(imageCard(pre.shadow_mask_target, "target — shadow mask"));
  block.appendChild(grid2);

  body.appendChild(block);
}

function renderMatch(body, report) {
  const readouts = document.createElement("div");
  readouts.className = "readout-row";
  readouts.appendChild(readout("raw matches", report.raw_match_count));
  readouts.appendChild(readout("inliers kept", report.inlier_count, "accent"));
  readouts.appendChild(readout("inlier ratio", report.inlier_ratio));
  body.appendChild(readouts);

  const block = document.createElement("div");
  block.className = "section-block";
  block.appendChild(sectionTitle("Raw correspondences", `${report.label || report.matcher}, before outlier rejection`));
  block.appendChild(imageCard(report.images.raw_matches, "reference (left) ↔ target (right) — all candidate matches", true));
  body.appendChild(block);

  const block2 = document.createElement("div");
  block2.className = "section-block";
  block2.appendChild(sectionTitle("After refinement + spatial capping", "MAGSAC++/RANSAC inliers, sub-pixel refined, grid-capped"));
  block2.appendChild(imageCard(report.images.inlier_matches, "surviving correspondences used for the final homography", true));
  body.appendChild(block2);
}

function renderRegister(body, report) {
  const block = document.createElement("div");
  block.className = "section-block";
  block.appendChild(sectionTitle("Registered output", `${report.label || report.matcher} → warped into reference frame`));
  const grid = document.createElement("div");
  grid.className = "image-grid";
  grid.appendChild(imageCard(report.images.registered, "target warped into reference frame"));
  grid.appendChild(imageCard(report.images.checkerboard, "checkerboard overlay — QA for local misalignment"));
  block.appendChild(grid);
  body.appendChild(block);

  const block2 = document.createElement("div");
  block2.className = "section-block";
  block2.appendChild(sectionTitle("Difference map", "absolute pixel difference, reference vs. registered target"));
  block2.appendChild(imageCard(report.images.diffmap, "warmer = larger residual", true));
  body.appendChild(block2);
}

function renderMetrics(body, report, reports) {
  const readouts = document.createElement("div");
  readouts.className = "readout-row";
  readouts.appendChild(readout("RMSE (px)", report.rmse_px));
  readouts.appendChild(readout("mean error (px)", report.mean_error_px));
  readouts.appendChild(readout("max error (px)", report.max_error_px));
  readouts.appendChild(readout("inlier ratio", report.inlier_ratio, "accent"));
  readouts.appendChild(readout("coverage", report.spatial_distribution.coverage_pct + "%", "accent"));
  readouts.appendChild(readout("tile count std", report.spatial_distribution.count_std));
  if (report.rmse_ground_m !== undefined) {
    readouts.appendChild(readout("RMSE (ground, m)", report.rmse_ground_m, "warn"));
  }
  body.appendChild(readouts);

  const block = document.createElement("div");
  block.className = "section-block";
  block.appendChild(sectionTitle("Spatial coverage", `${report.spatial_distribution.grid_size.join(" × ")} grid, matches per tile`));
  block.appendChild(imageCard(report.images.coverage_heatmap, "brighter tile = more retained matches in that region", true));
  body.appendChild(block);

  const okReports = reports.filter((r) => r.status === "ok");
  if (okReports.length > 1) {
    const block2 = document.createElement("div");
    block2.className = "section-block";
    block2.appendChild(sectionTitle("Bake-off ranking", "sorted by RMSE, then inlier ratio"));
    block2.appendChild(bakeoffTable(state.result.ranking, reports));
    body.appendChild(block2);
  }
}

function bakeoffTable(ranking, reports) {
  const table = document.createElement("table");
  table.className = "bakeoff";
  table.innerHTML = `
    <thead><tr>
      <th>matcher</th><th>status</th><th>rmse (px)</th><th>inlier ratio</th><th>coverage</th><th>time (ms)</th>
    </tr></thead>
  `;
  const tbody = document.createElement("tbody");
  const order = ranking.length ? ranking : reports.map((r) => r.matcher);
  order.forEach((name) => {
    const r = reports.find((x) => x.matcher === name);
    if (!r) return;
    const tr = document.createElement("tr");
    if (r.status === "ok") {
      tr.innerHTML = `
        <td>${r.label || r.matcher}</td>
        <td>${r.status}</td>
        <td>${r.rmse_px ?? "—"}</td>
        <td>${r.inlier_ratio ?? "—"}</td>
        <td>${r.spatial_distribution?.coverage_pct ?? "—"}%</td>
        <td>${r.elapsed_ms ?? "—"}</td>
      `;
    } else {
      tr.innerHTML = `<td>${r.label || r.matcher}</td><td>${r.status}</td><td colspan="4">${r.reason || ""}</td>`;
    }
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  return table;
}

function readout(label, value, variant) {
  const div = document.createElement("div");
  div.className = "readout";
  div.innerHTML = `
    <div class="readout-label">${label}</div>
    <div class="readout-value ${variant || ""}">${value === null || value === undefined ? "—" : value}</div>
  `;
  return div;
}

init();
