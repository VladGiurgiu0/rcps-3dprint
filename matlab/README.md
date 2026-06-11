# MATLAB implementations

| File | Status | Use it when |
|---|---|---|
| `RCPS_v5.m` | **Maintained** | You want to run the pipeline natively in MATLAB. Frame-correct (mesh spans exactly `[0, L]³`, self-asserted), periodic ghost spheres always on, no Python required. |
| `legacy/RCPS_v4.m` | Frozen | Only used by `tests/fixtures/generate_reference.m` to regenerate the cross-validation reference mesh. Known quirk: output frame is rigidly shifted by −1 voxel (documented; compensated in the test suite). Do not edit. |
| `legacy/mesh_from_raw.py` | Dead | The Python mesher that v4's `backend='python'` mode shelled out to. Superseded by the `rcps` package; kept for provenance (see the archived design audit). |

Python users: ignore this folder entirely — `pip install -e .` from the
repo root and use `rcps-build` (see the top-level README).

Dependencies for both .m scripts: [iso2mesh](http://iso2mesh.sf.net)
toolbox, [write3mf](https://github.com/cvergari/write3mf), and the
Statistics & Machine Learning Toolbox (`rangesearch`, bridges only).

Cross-validation status (2026-06-10, `data_example/` at vox = 0.1 mm):
the Python `rcps` package matches the v4-generated reference mesh to
Δ = 0.003% in solid volume, |Δφ| = 1.6×10⁻⁵ in porosity, Δ = 0.13% in
surface area, and 0.62 µm in bounding box; both agree with the analytic
packing volume to within 0.016%. `RCPS_v5.m` is additionally
frame-verified (bbox = [0, L]³ to <1 µm) and smoke-tested at coarse
resolution. See `tests/fixtures/README.md` for details.
