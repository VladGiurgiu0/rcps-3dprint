/* RCPS GUI front-end: stage navigation, job polling, WebGL previews. */
"use strict";

const $ = (id) => document.getElementById(id);
const api = {
  get: async (p) => { const r = await fetch(p); const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText); return j; },
  post: async (p, body) => { const r = await fetch(p, { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText); return j; },
};

/* ---------- stage navigation ---------- */
const stages = ["setup", "pack", "mesh", "export"];
function show(stage) {
  stages.forEach((s) => {
    $(`stage-${s}`).classList.toggle("visible", s === stage);
    document.querySelector(`.step[data-stage=${s}]`)
      .classList.toggle("active", s === stage);
  });
}
document.querySelectorAll(".step").forEach((b) =>
  b.addEventListener("click", () => show(b.dataset.stage)));
function markDone(stage, done = true) {
  document.querySelector(`.step[data-stage=${stage}]`)
    .classList.toggle("done", done);
}

/* ---------- job polling ---------- */
function fmtElapsed(s) {
  const m = Math.floor(s / 60), ss = Math.floor(s % 60);
  return m ? `${m}:${String(ss).padStart(2, "0")}` : `${ss}s`;
}

async function pollJob(id, { logEl, onDone, onError, btn, cancelBtn }) {
  if (logEl) logEl.classList.remove("hidden");
  if (btn) { btn.disabled = true; btn.dataset.label = btn.textContent;
    btn.innerHTML = '<span class="spinner-inline"></span>' + btn.dataset.label; }
  if (cancelBtn) cancelBtn.classList.remove("hidden");
  const tick = async () => {
    let j;
    try { j = await api.get(`/api/job/${id}`); }
    catch (e) { if (onError) onError(e.message); return; }
    if (logEl) { logEl.textContent = j.log.join("\n");
      logEl.scrollTop = logEl.scrollHeight; }
    if (j.status === "running" || j.status === "pending") {
      if (btn) btn.innerHTML = '<span class="spinner-inline"></span>' +
        btn.dataset.label + " · " + fmtElapsed(j.elapsed_s);
      setTimeout(tick, 700); return;
    }
    if (btn) { btn.disabled = false; btn.textContent = btn.dataset.label; }
    if (cancelBtn) cancelBtn.classList.add("hidden");
    if (j.status === "done" && onDone) onDone(j);
    else if (onError && j.status !== "done") onError(j.error || j.status);
  };
  tick();
}
function alertErr(msg) { alert("Error: " + msg); }

/* ---------- three.js viewers (lazy; custom orbit, no addons) ---------- */
let threeReady = false;
window.addEventListener("three-ready", () => { threeReady = true; });

function makeViewer(el) {
  const W = el.clientWidth, H = el.clientHeight;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(W, H);
  el.innerHTML = ""; el.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, W / H, 0.1, 5000);
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 0.7);
  key.position.set(1, 1.2, 0.8); scene.add(key);
  const rim = new THREE.DirectionalLight(0xbfd6ff, 0.25);
  rim.position.set(-1, -0.5, -1); scene.add(rim);

  let theta = 0.9, phi = 1.1, dist = 160, target = new THREE.Vector3();
  const apply = () => {
    camera.position.set(
      target.x + dist * Math.sin(phi) * Math.cos(theta),
      target.y + dist * Math.cos(phi),
      target.z + dist * Math.sin(phi) * Math.sin(theta));
    camera.lookAt(target);
  };
  let drag = null;
  renderer.domElement.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY }; });
  window.addEventListener("pointermove", (e) => {
    if (!drag) return;
    theta += (e.clientX - drag.x) * 0.006;
    phi = Math.min(2.9, Math.max(0.2, phi - (e.clientY - drag.y) * 0.006));
    drag = { x: e.clientX, y: e.clientY }; apply();
  });
  window.addEventListener("pointerup", () => { drag = null; });
  renderer.domElement.addEventListener("wheel", (e) => {
    e.preventDefault();
    dist = Math.min(1200, Math.max(20, dist * (1 + e.deltaY * 0.001))); apply();
  }, { passive: false });

  const animate = () => { requestAnimationFrame(animate);
    renderer.render(scene, camera); };
  animate();
  return {
    scene,
    setTarget(c, d) { target.set(c[0], c[1], c[2]); dist = d; apply(); },
    clear() {
      for (let i = scene.children.length - 1; i >= 3; i--)
        scene.remove(scene.children[i]); },
  };
}

function addBoxOutline(viewer, box) {
  const g = new THREE.BoxGeometry(box[0], box[1], box[2]);
  const edges = new THREE.EdgesGeometry(g);
  const line = new THREE.LineSegments(edges,
    new THREE.LineBasicMaterial({ color: 0x86868b }));
  line.position.set(box[0] / 2, box[1] / 2, box[2] / 2);
  viewer.scene.add(line);
}

function showSpheres(el, data) {
  if (!threeReady) { el.innerHTML =
    "<p class='muted' style='padding:20px'>3D preview needs three.js " +
    "(offline and no vendored copy — see README).</p>"; return; }
  const viewer = makeViewer(el);
  viewer.clear();
  addBoxOutline(viewer, data.box_mm);
  const n = data.n;
  const geo = new THREE.SphereGeometry(0.5, 18, 14);
  const mat = new THREE.MeshStandardMaterial({
    color: 0x4d8fd1, roughness: 0.55, metalness: 0.05 });
  const inst = new THREE.InstancedMesh(geo, mat, n);
  const m = new THREE.Matrix4();
  for (let i = 0; i < n; i++) {
    const c = data.centers[i], d = data.diameters[i];
    m.makeScale(d, d, d);
    m.setPosition(c[0], c[1], c[2]);
    inst.setMatrixAt(i, m);
  }
  viewer.scene.add(inst);
  const b = data.box_mm;
  viewer.setTarget([b[0] / 2, b[1] / 2, b[2] / 2], Math.max(b[0], b[1], b[2]) * 2.4);
}

async function showMesh(el, jobId) {
  if (!threeReady) { el.innerHTML =
    "<p class='muted' style='padding:20px'>3D preview needs three.js.</p>"; return; }
  const buf = await (await fetch(`/api/mesh/data/${jobId}`)).arrayBuffer();
  const dv = new DataView(buf);
  const nV = dv.getUint32(0, true), nF = dv.getUint32(4, true);
  const V = new Float32Array(buf, 8, nV * 3);
  const F = new Uint32Array(buf, 8 + nV * 12, nF * 3);

  const viewer = makeViewer(el);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(V, 3));
  geo.setIndex(new THREE.BufferAttribute(F, 1));
  geo.computeVertexNormals();
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    color: 0xb8bcc4, roughness: 0.62, metalness: 0.12 }));
  viewer.scene.add(mesh);
  geo.computeBoundingBox();
  const bb = geo.boundingBox, c = new THREE.Vector3();
  bb.getCenter(c);
  addBoxOutline(viewer, [bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z]);
  viewer.setTarget([c.x, c.y, c.z],
    Math.max(bb.max.x - bb.min.x, bb.max.y - bb.min.y) * 2.4);
}

/* ---------- Setup stage ---------- */
$("btn-exe").onclick = async () => {
  try {
    const info = await api.post("/api/setup/exe", { path: $("exe-path").value });
    if (info.ok) { $("exe-ok").classList.remove("hidden"); markDone("setup"); }
    else alertErr("not an executable file: " + info.path);
  } catch (e) { alertErr(e.message); }
};
$("btn-clone").onclick = async () => {
  try {
    const j = await api.post("/api/setup/clone", {});
    pollJob(j.id, {
      logEl: $("log-setup"), btn: $("btn-clone"),
      onDone: (jj) => { $("exe-path").value = jj.result.exe_path;
        $("exe-ok").classList.remove("hidden"); markDone("setup"); },
      onError: alertErr,
    });
  } catch (e) { alertErr(e.message); }
};
$("btn-example").onclick = async () => {
  try {
    const r = await api.post("/api/use_example", { source: $("example-path").value });
    await refreshStatus();          // fresh run: pack done, mesh/export pending
    await refreshPackPreview();
    show("mesh");
  } catch (e) { alertErr(e.message); }
};

/* sidebar dots = what the CURRENT run folder already contains */
async function refreshStatus() {
  try {
    const s = await api.get("/api/state");
    markDone("setup", !!s.exe_path || s.packing_exists);
    const rs = s.run_status || {};
    markDone("pack", !!rs.pack);
    markDone("mesh", !!rs.mesh);
    markDone("export", !!rs.export);
    setPackBlocked(!!rs.pack);
    const sel = $("run-select");
    const current = s.current_run ? s.current_run.split("/").slice(-1)[0] : "";
    sel.innerHTML = "";
    if (!(s.runs || []).length)
      sel.innerHTML = '<option value="">none yet</option>';
    for (const r of s.runs || []) {
      const o = document.createElement("option");
      o.value = r; o.textContent = r.replace(/^run_/, "");
      if (r === current) o.selected = true;
      sel.appendChild(o);
    }
  } catch (e) { /* ignore */ }
}

function resetDerivedPanes() {
  $("pack-result").classList.add("hidden");
  $("pack-summary").textContent = "";
  $("mesh-result").classList.add("hidden");
  $("export-result").classList.add("hidden");
  $("slicer").classList.add("hidden");
}

$("btn-new-run").onclick = async () => {
  try {
    await api.post("/api/runs/new", {});
    resetDerivedPanes();
    await refreshStatus();
    show("pack");
  } catch (e) { alertErr(e.message); }
};

$("run-select").addEventListener("change", async () => {
  const name = $("run-select").value;
  if (!name) return;
  try {
    await api.post("/api/runs/select", { name });
    resetDerivedPanes();
    await refreshStatus();
    await refreshPackPreview();
  } catch (e) { alertErr(e.message); }
});

/* ---------- Pack stage ---------- */
function setPackBlocked(blocked) {
  $("pack-banner").classList.toggle("hidden", !blocked);
  $("btn-pack").disabled = blocked;
  $("pack-card").classList.toggle("dimmed", blocked);
}
$("pack-unlock").onclick = (e) => { e.preventDefault(); setPackBlocked(false); };

function packParams() {
  const L = parseFloat($("p-box").value);
  const d = parseFloat($("p-d").value);
  const phi = parseFloat($("p-phi").value);
  const n = Math.floor((1 - phi) * L * L * L / (Math.PI / 6 * d * d * d));
  return {
    n_particles: n,
    box_mm: [L, L, L],
    d_nominal_mm: d,
    seed: parseInt($("p-seed").value, 10),
    contraction_rate: parseFloat($("p-cr").value),
    steps_to_write: parseInt($("p-steps").value, 10) || 1000,
    stages: ["fba", "ls", "lsgd"],
  };
}
function updateN() {
  const p = packParams();
  $("p-n").value = p.n_particles;
}
["p-d", "p-box", "p-phi"].forEach((id) => $(id).addEventListener("input", updateN));

$("btn-pack").onclick = async () => {
  try {
    const j = await api.post("/api/pack/run", packParams());
    // a fresh run folder was just created: mesh/export start over
    markDone("mesh", false); markDone("export", false);
    pollJob(j.id, {
      logEl: $("log-pack"), btn: $("btn-pack"), cancelBtn: $("btn-pack-cancel"),
      onDone: async (jj) => {
        const m = jj.result;
        $("pack-summary").innerHTML =
          `φ requested <b>${m.phi_requested_at_nominal_d?.toFixed(4)}</b> → ` +
          `achieved <b>${m.phi_true_after_rescale?.toFixed(4)}</b> · ` +
          `true mean d = <b>${m.mean_true_diameter_mm?.toFixed(4)} mm</b> · ` +
          `rescale ×${m.scaling_factor?.toFixed(5)}`;
        $("pack-result").classList.remove("hidden");
        await refreshStatus();
        await refreshPackPreview();
      },
      onError: alertErr,
    });
    $("btn-pack-cancel").onclick = () => api.post(`/api/job/${j.id}/cancel`);
  } catch (e) { alertErr(e.message); }
};

async function refreshPackPreview() {
  try {
    const data = await api.get("/api/pack/preview");
    $("pack-result").classList.remove("hidden");
    if (!$("pack-summary").textContent)
      $("pack-summary").innerHTML = `${data.n} spheres · mean d = ` +
        `<b>${data.mean_d_mm} mm</b> · φ = <b>${data.phi.toFixed(4)}</b>`;
    $("dia-stored-info").textContent =
      `d = ${data.mean_d_mm} mm, φ = ${data.phi.toFixed(4)}`;
    const designLabel = $("dia-design-label");
    const designRadio = designLabel.querySelector("input");
    if (data.design) {
      designRadio.disabled = false;
      designLabel.classList.remove("dimmed");
      $("dia-design-info").textContent =
        `d = ${data.design.mean_d_mm} mm, φ = ${data.design.phi.toFixed(4)} ` +
        `(×${data.design.factor.toFixed(5)})`;
    } else {
      designRadio.disabled = true;
      designRadio.checked = false;
      designLabel.querySelector("input[name=dia]");
      document.querySelector("input[name=dia][value=stored]").checked = true;
      designLabel.classList.add("dimmed");
      $("dia-design-info").textContent =
        "unavailable — no .nfo / packing_meta.json next to the loaded packing";
    }
    renderPackMetrics(data.metrics);
    showSpheres($("view-pack"), data);
  } catch (e) { /* no packing yet */ }
}

/* RCP structure metrics (rcps/metrics.py; persisted to packing_metrics.json) */
function renderPackMetrics(m) {
  const line = $("pack-metrics"), det = $("pack-metrics-details");
  if (!m || !m.coordination) {
    line.classList.add("hidden"); det.classList.add("hidden"); return;
  }
  const co = m.coordination, kc = m.kozeny_carman, ck = m.rcp_checklist;
  const kBed = kc.k_m2_nominal_d ?? kc.k_m2_stored_d;
  const kExp = (x) => (x == null || !isFinite(x)) ? "–" : x.toExponential(2);
  line.innerHTML =
    `packing fraction <b>${m.packing_fraction.toFixed(4)}</b> ` +
    `(ε = ${m.porosity.toFixed(4)}) · ` +
    `z = <b>${co.z_no_rattlers.toFixed(2)}</b> (isostatic 6) · ` +
    `k<sub>KC</sub> ≈ <b>${kExp(kBed)} m²</b> · ` +
    (m.is_rcp_consistent
      ? `<span class="ok">✓ RCP-consistent</span>`
      : `<span class="err">deviates from RCP — see details</span>`);
  const pf = (b) => b ? `<span class="ok">✓</span>` : `<span class="err">✗</span>`;
  const rows = [
    [`packing fraction φ<sub>pack</sub>`, m.packing_fraction.toFixed(4),
     `0.642–0.649 sim. / 0.643–0.659 theory — Anzivino et al. 2023`,
     pf(ck.phi_in_rcp_window)],
    [`kissing number z (no rattlers)`, co.z_no_rattlers.toFixed(3),
     `6 = isostatic (Maxwell) — Anzivino et al. 2023, Eq. (1)`,
     pf(ck.isostatic)],
    [`rattler fraction (z<sub>i</sub> &lt; 4)`,
     `${(100 * co.rattler_fraction).toFixed(1)}% (${co.n_rattlers})`,
     `small (&lt; 5%); loose spheres are normal in jammed packings`,
     pf(ck.rattler_fraction_ok)],
    [`Berryman median NN / d`, m.berryman_median_nn_over_d.toFixed(5),
     `= 1 at RCP — Berryman 1983`, pf(ck.berryman)],
    [`k (Kozeny–Carman, printed d)`, `${kExp(kBed)} m²`,
     `Eq. (3.3) of De Paoli et al. 2024, k<sub>C</sub> = 5; creeping flow`, ``],
  ];
  $("pack-metrics-body").innerHTML =
    `<table class="mtable"><tr><th>quantity</th><th>value</th>` +
    `<th>RCP reference</th><th></th></tr>` +
    rows.map((r) => `<tr><td>${r[0]}</td><td><b>${r[1]}</b></td>` +
                    `<td>${r[2]}</td><td>${r[3]}</td></tr>`).join("") +
    `</table>`;
  line.classList.remove("hidden");
  det.classList.remove("hidden");
}

/* ---------- Mesh stage ---------- */
function diameterChoice() {
  return document.querySelector("input[name=dia]:checked").value;
}
function meshParams() {
  return {
    export_what: $("m-what").value,
    bridge_mode: $("m-bridge").value,
    radius_frac: parseFloat($("m-rf").value),
    diameter: diameterChoice(),
    backend: $("m-backend").value,
    iso2mesh: {
      angbound_deg: parseFloat($("a-ang").value),
      radbound: parseFloat($("a-rad").value),
      distbound: parseFloat($("a-dist").value),
      maxnode: parseFloat($("a-maxn").value),
    },
  };
}
$("btn-mesh").onclick = async () => {
  try {
    const j = await api.post("/api/mesh/preview", {
      vox_mm: parseFloat($("m-vox").value),
      ...meshParams(),
    });
    pollJob(j.id, {
      logEl: $("log-mesh"), btn: $("btn-mesh"),
      onDone: async (jj) => {
        await refreshStatus();
        const r = jj.result;
        $("mesh-summary").innerHTML =
          `${(r.n_faces / 1e6).toFixed(2)} M triangles · φ = <b>${r.porosity.toFixed(4)}</b>` +
          ` · printed d = <b>${r.diameter.mean_printed_diameter_mm} mm</b>` +
          (r.surface_area_mm2 ? ` · surface <b>${(r.surface_area_mm2 / 100).toFixed(1)} cm²</b>` +
            ` · specific surface <b>${r.specific_surface_per_mm.toFixed(3)} mm⁻¹</b>` : "") +
          ` · watertight: <b class="${r.watertight ? "ok" : "err"}">${r.watertight}</b>`;
        $("mesh-result").classList.remove("hidden");
        await showMesh($("view-mesh"), jj.id);
      },
      onError: alertErr,
    });
  } catch (e) { alertErr(e.message); }
};

/* ---------- Export stage: schematic + estimate ---------- */
function gridVals() {
  return [$("e-gx"), $("e-gy"), $("e-gz")].map((x) =>
    Math.max(1, parseInt(x.value, 10) || 1));
}
function uniqueTypes(g) {
  const cls = (N) => (N === 1 ? 1 : N === 2 ? 2 : 3);
  return cls(g[0]) * cls(g[1]) * cls(g[2]);
}
function schematicView(nA, nB, L, labelA, labelB, title) {
  // generic 2D grid view: nA cells along the horizontal axis, nB vertical
  const fa = nA * L, fb = nB * L;
  const pad = 34;
  const scale = Math.min((260 - pad) / fa, (150 - pad) / fb);
  const w = fa * scale, h = fb * scale;
  let cells = "";
  for (let i = 0; i < nA; i++)
    for (let j = 0; j < nB; j++)
      cells += `<rect x="${pad + i * L * scale}" y="${pad + (nB - 1 - j) * L * scale}"
        width="${L * scale}" height="${L * scale}" class="sch-cell"/>`;
  return `
  <svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg">
    ${cells}
    <line x1="${pad}" y1="${pad + h + 9}" x2="${pad + w}" y2="${pad + h + 9}" class="sch-dim"/>
    <text x="${pad + w / 2}" y="${pad + h + 22}" class="sch-txt" text-anchor="middle">${labelA}: ${fa} mm</text>
    <line x1="${pad - 9}" y1="${pad}" x2="${pad - 9}" y2="${pad + h}" class="sch-dim"/>
    <text x="${pad - 14}" y="${pad + h / 2}" class="sch-txt" text-anchor="middle"
      transform="rotate(-90 ${pad - 14} ${pad + h / 2})">${labelB}: ${fb} mm</text>
    <text x="${pad}" y="${pad - 12}" class="sch-txt">${title}</text>
  </svg>`;
}

function drawSchematic() {
  const g = gridVals();
  const L = parseFloat($("p-box").value) || 50;
  const fx = g[0] * L, fy = g[1] * L, fz = g[2] * L;
  $("e-schematic").innerHTML =
    schematicView(g[0], g[1], L, "x", "y", "top view (x–y)") +
    schematicView(g[0], g[2], L, "x", "z", "side view (x–z)");

  const n = g[0] * g[1] * g[2];
  const t = n === 1 ? 1 : uniqueTypes(g);
  $("e-estimate").textContent =
    `${n} tile(s) → ${t} unique mesh(es) to print · facility ` +
    `${fx} × ${fy} × ${fz} mm. Dry run is instant and writes the assembly plan.`;
}
["e-gx", "e-gy", "e-gz", "e-vox", "p-box"].forEach((id) =>
  $(id).addEventListener("input", drawSchematic));

/* ---------- SDF slicer ---------- */
const sdf = { job: null, meta: null, axis: 2, index: 0 };

function sdfColor(v, vmin, vmax) {
  // diverging map centered on F = 0: blue (inside) -> white -> warm (outside)
  if (v < 0) {
    const t = Math.min(1, v / (vmin || -1));        // 0 at surface, 1 deep inside
    return [Math.round(255 - 205 * t), Math.round(255 - 135 * t), 255];
  }
  const t = Math.min(1, v / (vmax || 1));
  return [255, Math.round(255 - 120 * t), Math.round(255 - 195 * t)];
}

function drawColorbar() {
  const [vmin, vmax] = sdf.meta.range;
  const cb = $("s-colorbar"), ctx = cb.getContext("2d");
  for (let y = 0; y < cb.height; y++) {
    const v = vmax - (y / (cb.height - 1)) * (vmax - vmin);
    const [r, g, b] = sdfColor(v, vmin, vmax);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(0, y, cb.width, 1);
  }
  // tick at F = 0
  const y0 = Math.round((vmax / (vmax - vmin)) * (cb.height - 1));
  ctx.fillStyle = "#1d1d1f";
  ctx.fillRect(0, y0, cb.width, 1.5);
  $("s-zero").style.top = `${(y0 / (cb.height - 1)) * 100}%`;
  $("s-max").textContent = `+${vmax.toFixed(2)} mm`;
  $("s-min").textContent = `${vmin.toFixed(2)} mm`;
}

async function drawSlice() {
  if (!sdf.job) return;
  const r = await fetch(`/api/sdf/slice/${sdf.job}?axis=${sdf.axis}&i=${sdf.index}`);
  if (!r.ok) return;
  const buf = await r.arrayBuffer();
  const dv = new DataView(buf);
  const w = dv.getUint32(0, true), h = dv.getUint32(4, true);
  const pos = dv.getFloat32(8, true);
  const data = new Float32Array(buf, 12, w * h);
  const [vmin, vmax] = sdf.meta.range;

  const cv = $("s-canvas");
  cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    const row = (h - 1 - y) * w;                    // physical axis points up
    for (let x = 0; x < w; x++) {
      const v = data[row + x];
      const [cr, cg, cb2] = sdfColor(v, vmin, vmax);
      const o = (y * w + x) * 4;
      img.data[o] = cr; img.data[o + 1] = cg; img.data[o + 2] = cb2;
      img.data[o + 3] = 255;
    }
  }
  // iso-line F = 0: mark sign changes against right/down neighbours
  for (let y = 0; y < h; y++) {
    const row = (h - 1 - y) * w;
    const rowD = (h - 1 - (y + 1)) * w;
    for (let x = 0; x < w; x++) {
      const v = data[row + x];
      const right = x + 1 < w ? data[row + x + 1] : v;
      const down = y + 1 < h ? data[rowD + x] : v;
      if ((v >= 0) !== (right >= 0) || (v >= 0) !== (down >= 0)) {
        const o = (y * w + x) * 4;
        img.data[o] = 29; img.data[o + 1] = 29; img.data[o + 2] = 31;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  const axName = ["x", "y", "z"][sdf.axis];
  $("s-pos").textContent = `${axName} = ${pos.toFixed(2)} mm`;
}

$("btn-sdf").onclick = async () => {
  try {
    const j = await api.post("/api/sdf/compute", {
      vox_mm: parseFloat($("s-vox").value), ...meshParams(),
    });
    pollJob(j.id, {
      logEl: $("log-sdf"), btn: $("btn-sdf"),
      onDone: async (jj) => {
        sdf.job = jj.id;
        sdf.meta = jj.result;
        sdf.index = Math.floor(jj.result.shape[sdf.axis] / 2);
        $("slicer").classList.remove("hidden");
        $("log-sdf").classList.add("hidden");
        const sl = $("s-slider");
        sl.max = jj.result.shape[sdf.axis] - 1;
        sl.value = sdf.index;
        drawColorbar();
        await drawSlice();
      },
      onError: alertErr,
    });
  } catch (e) { alertErr(e.message); }
};

$("s-slider").addEventListener("input", () => {
  sdf.index = parseInt($("s-slider").value, 10);
  drawSlice();
});
$("s-canvas").addEventListener("wheel", (e) => {
  e.preventDefault();
  if (!sdf.meta) return;
  const n = sdf.meta.shape[sdf.axis];
  sdf.index = Math.max(0, Math.min(n - 1, sdf.index + (e.deltaY > 0 ? 1 : -1)));
  $("s-slider").value = sdf.index;
  drawSlice();
}, { passive: false });
document.querySelectorAll("#s-axis button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll("#s-axis button").forEach((x) =>
      x.classList.remove("on"));
    b.classList.add("on");
    sdf.axis = parseInt(b.dataset.axis, 10);
    if (!sdf.meta) return;
    const n = sdf.meta.shape[sdf.axis];
    sdf.index = Math.floor(n / 2);
    $("s-slider").max = n - 1;
    $("s-slider").value = sdf.index;
    drawSlice();
  }));

async function runExport(dry) {
  try {
    const j = await api.post("/api/export/run", {
      vox_mm: parseFloat($("e-vox").value),
      grid: gridVals(),
      ...meshParams(),
      dry_run: dry,
    });
    pollJob(j.id, {
      logEl: $("log-export"), btn: dry ? $("btn-export-dry") : $("btn-export"),
      cancelBtn: $("btn-export-cancel"),
      onDone: async (jj) => {
        await refreshStatus();
        const r = jj.result;
        let txt = "";
        if (r.files) for (const [k, v] of Object.entries(r.files)) txt += `${k}: ${v}\n`;
        if (r.meshes) { txt += `assembly map: ${r.map_txt}\n`;
          for (const [k, v] of Object.entries(r.meshes)) txt += `${k}: ${v}\n`;
          if (!Object.keys(r.meshes).length) txt += "(dry run - no meshes)\n"; }
        $("export-files").textContent = txt || JSON.stringify(r, null, 2);
        $("export-result").classList.remove("hidden");
      },
      onError: alertErr,
    });
    $("btn-export-cancel").onclick = () => api.post(`/api/job/${j.id}/cancel`);
  } catch (e) { alertErr(e.message); }
}
$("btn-export").onclick = () => {
  const g = gridVals().join("·");
  if (confirm(`Start full-resolution export (${g} grid)? This is a long run.`))
    runExport(false);
};
$("btn-export-dry").onclick = () => runExport(true);

/* ---------- boot ---------- */
(async () => {
  try {
    const s = await api.get("/api/state");
    $("version").textContent = "v" + s.version;
    $("project-path").textContent = s.project_dir;
    if (s.exe_path) { $("exe-path").value = s.exe_path;
      $("exe-ok").classList.remove("hidden"); markDone("setup"); }
    const pp = s.pack_params || {};
    if (pp.box_mm) $("p-box").value = pp.box_mm[0];
    if (pp.d_nominal_mm) $("p-d").value = pp.d_nominal_mm;
    if (pp.seed != null) $("p-seed").value = pp.seed;
    if (pp.contraction_rate) $("p-cr").value = pp.contraction_rate;
    const mp = s.mesh_params || {};
    if (mp.backend) $("m-backend").value = mp.backend;
    updateN();
    drawSchematic();
    await refreshStatus();
    if (s.packing_exists) {
      window.addEventListener("three-ready", refreshPackPreview, { once: true });
      setTimeout(refreshPackPreview, 800);
    }
  } catch (e) { console.error(e); }
})();
