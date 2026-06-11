# Test fixtures

This directory holds large binary fixtures referenced by the validation
suite. The MATLAB-vs-Python comparison tests in
`tests/test_pipeline_e2e.py::TestPythonMatchesMatlabReference` are
gated on the presence of `reference.3mf`; when the file is absent they
auto-skip via the `matlab_reference_3mf` pytest fixture (see
`tests/conftest.py`).

## How to regenerate `reference.3mf`

Run `tests/fixtures/generate_reference.m` in MATLAB. It is fully
scripted: it executes an unmodified `matlab/legacy/RCPS_v4.m` with the canonical
configuration applied programmatically, writes `reference.3mf` and
`reference_info.txt` here, and appends an audit block (bridge settings,
ghost flag, MATLAB version) to the info file. **Do not run RCPS_v4.m by
hand for this** — the two pitfalls below are exactly why the scripted
path exists.

Dependencies on the MATLAB path: iso2mesh toolbox, `write3mf`
(<https://github.com/cvergari/write3mf>), Statistics & ML Toolbox
(`rangesearch`).

## Two audited RCPS_v4.m behaviours (2026-06-10)

1. **Ghost spheres require `'facility'` mode.** RCPS_v4.m only generates
   periodic ghost copies in `exportMode='facility'` (line ~162);
   `'tile'` mode meshes the raw spheres with no periodic images. A
   reference generated in tile mode loses the neighbour-sphere caps that
   protrude through the tile faces — for `data_example` that is
   5,592 mm³ of solid, φ = 0.408 instead of the periodic bulk 0.3633.
   The first (2026-06-05) reference was generated in tile mode and
   failed the comparison for precisely this reason; the Python pipeline
   matched the analytic periodic value to +0.013%. The canonical
   reference is therefore **facility 1×1×1 with `ghostTiles=1`**.

2. **Rigid −1-voxel frame offset.** The `v2s` branch of RCPS_v4.m maps
   mesh nodes to mm one voxel off: the flush-cut faces land at
   `[-0.1, 49.9]` instead of the physical `[0, 50]` (the Python port
   lands on `[0, 50]` to <1 µm; verified against the analytic packing).
   Decision: RCPS_v4.m stays frozen as the legacy reference; the pytest
   comparison detects the offset, asserts it is an integer number of
   voxels, removes it, and prints it. A cleaned MATLAB-only `RCPS_v5`
   correcting the frame at the source is on the roadmap.

## Canonical configuration for the reference

These are applied by `generate_reference.m` on top of the RCPS_v4.m
defaults (locked v1.0 values, the archived design audit §7) for
`data_example/packing.xyzd`:

| Parameter | Value |
|---|---|
| `field.mode` | `icsg` |
| `field.exportWhat` | `beads` |
| `field.exportMode` | **`facility`** (1×1×1 grid — *not* `tile`; see above) |
| `field.ghostTiles` | `1` |
| `field.padVox` | `1` |
| `field.bandVox` | `3` |
| `field.keepSides` | `{}` (all faces cut flush) |
| `grid.voxSize_mm` | `0.1` |
| `spheres.expansion_factor` | `1.00` |
| `spheres.contactTol_mm` | `0.20` |
| `bridge.mode` | `cylinders` |
| `bridge.radiusFrac` | `0.15` |
| `mesh.backend` / `mesh.method` | `matlab` / `iso2mesh` |
| `mesh.iso2mesh.angbound_deg` | `25` |
| `mesh.iso2mesh.radbound` | `1.0` |
| `mesh.iso2mesh.distbound` | `0.10` |
| `mesh.iso2mesh.maxnode` | `2e8` |
| `mesh.doRepair` | `false` |
| `mesh.doReducePatch` | `false` |
| `out.save3MF` | `true` |
| `out.saveSTL` | `false` |

## Expected comparison values (data_example, vox = 0.1)

Mesher-independent ground truth from `packing.xyzd` (718 spheres,
stored d = 5.9598 mm, periodic tile 50³): solid = 79,583 mm³,
φ = 0.36334. Python pipeline (cylinder bridges): 79,593 mm³,
φ = 0.363253. A correctly regenerated reference should land within the
test tolerances of these values (volume 0.5%, porosity 0.001, aligned
bbox 1 µm, surface area 1%).

Validated 2026-06-10 (regenerated reference): volume, porosity, bbox,
and absolute-frame tests all pass. Two metric-design notes from that
run: (1) vertex counts differ ~7% between the two CGAL builds despite
identical CGAL version and RNG seed (and even consecutive Python runs
differ by ~250 vertices) — counts fingerprint the binary/FP
environment, so the suite asserts **surface area (±1%)** instead and
prints counts as info. (2) The raw (`doRepair=false`) reference carries
~12 sliver defects out of 14.6M faces while remaining closed; the suite
requires the *Python* mesh to be strictly watertight, and the raw
reference to be closed (0 boundary edges) with a ≤32 defect budget for
non-manifold edges and degenerate faces. The slivers are asserted on
*raw* — they participate in the closure (filtering them exposes
boundary edges) and contribute zero volume to the divergence-theorem
integral.

Note on diameters: `packing.nfo` reports φ = 0.3504, which corresponds
to d = 6.000 mm (Baranau's convention stores pre-rescale diameters;
scale factor 1.00674). Both implementations consistently use the stored
diameters with `expansion_factor = 1.00`, so the comparison is
unaffected — but the physical tile at this setting realizes
φ = 0.363 / d = 5.96 mm, not φ = 0.350 / d = 6.0 mm. Decide the
convention explicitly before facility production runs.

## Practical notes

- The file is ~190 MB; keep it locally or commit via Git LFS (`*.3mf`
  is gitignored; the fixture is an explicit exception).
- The Python side of the comparison takes ~80 min at vox = 0.1. Set
  `RCPS_E2E_CACHE_DIR` (e.g. `~/.cache/rcps_e2e`) to cache it across
  pytest runs; the key hashes config + packing + `rcps/*.py` sources.
- Byte size and triangle count of the reference depend on the iso2mesh
  build; the load-bearing assertions are volume, porosity, aligned
  bbox, surface area, and closedness/watertightness.
