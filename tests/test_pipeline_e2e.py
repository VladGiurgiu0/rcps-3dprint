"""End-to-end validation suite for the RCPS pipeline (Task 8).

Two test groups:

1. *Self-consistency* (`TestSelfConsistencyAtCoarseVox`,
   `TestDiameterMode`, `TestKeepSidesGeometry`) — runs the Python pipeline
   at a coarse voxel size and asserts watertightness, porosity bounds,
   bbox correctness, and (for `diameter` mode) the printable contact-
   graph invariant. Requires `trimesh`, `pymeshfix`, and either
   `iso2mesh` or `scikit-image`. Slow tests are marked
   ``@pytest.mark.slow``.

2. *MATLAB reference* (`TestPythonMatchesMatlabReference`) — compares the
   Python pipeline output against ``tests/fixtures/reference.3mf``
   produced by ``RCPS_v4.m``. Auto-skips when the fixture is missing.
   See ``tests/fixtures/README.md`` for regeneration instructions.

The facility orchestrator tests (including the 2×2×1 e2e) live in
``tests/test_facility.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# -------- optional-dep gating --------

try:
    import trimesh  # noqa: F401
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
try:
    import iso2mesh.core  # noqa: F401
    HAS_ISO2MESH = True
except ImportError:
    HAS_ISO2MESH = False
try:
    import pymeshfix  # noqa: F401
    HAS_PYMESHFIX = True
except ImportError:
    HAS_PYMESHFIX = False
try:
    import skimage  # noqa: F401
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def _has_a_mesher() -> bool:
    return HAS_ISO2MESH or HAS_SKIMAGE


# Choose the meshing backend present in the test environment.
def _backend() -> str:
    if HAS_ISO2MESH:
        return "iso2mesh"
    return "skimage"


# -------- helper: build a config dict-style --------

def _config_dict(packing_path, out_dir, *, vox=2.0, bridge_mode="cylinders", keep_sides=(), backend=None, export_what="beads"):
    return {
        "paths": {
            "packing": str(packing_path),
            "root": str(packing_path.parent),
            "out_dir": str(out_dir),
        },
        "geom": {"tile_size_mm": [50, 50, 50]},
        "spheres": {"diameter_mm": 6.0, "expansion_factor": 1.0, "contact_tol_mm": 0.2},
        "grid": {"vox_size_mm": float(vox)},
        "field": {
            "export_what": export_what, "ghost_tiles": 1, "pad_vox": 1, "band_vox": 3,
            "keep_sides": list(keep_sides),
        },
        "bridge": {"mode": bridge_mode, "radius_frac": 0.15},
        "mesh": {"backend": backend or _backend()},
        "out": {"base_name": "e2e_test"},
    }


# =====================================================================
# Self-consistency suite (no MATLAB reference required)
# =====================================================================

@pytest.mark.skipif(
    not (HAS_TRIMESH and HAS_PYMESHFIX and HAS_SKIMAGE),
    reason="trimesh + pymeshfix + scikit-image required (coarse-vox tests pin skimage)",
)
class TestSelfConsistencyAtCoarseVox:
    """Coarse-vox self-consistency, pinned to the skimage backend.

    Rationale (audited 2026-06-10): at vox=2 there are only 3 voxels per
    sphere diameter and iso2mesh's `distbound` is in voxel units
    (0.1 vox = 0.2 mm allowed surface deviation), so CGAL legitimately
    inflates the solid (measured φ=0.261 vs marching-cubes φ≈0.36 ≈
    analytic 0.3633). The φ bounds below were calibrated on marching
    cubes, which tracks the trilinear level set tightly at any vox.
    iso2mesh quality is asserted at production vox=0.1 by
    `TestPythonMatchesMatlabReference` instead.
    """

    def test_pipeline_produces_watertight_mesh_at_vox_2(self, packing_xyzd_path, tmp_path):
        from rcps.config import RcpsConfig
        from rcps.mesh import diagnose
        from rcps.pipeline import run
        from tests._mesh_metrics import bbox_of, load_3mf, mesh_porosity_in_tile

        cfg = RcpsConfig.from_dict(_config_dict(packing_xyzd_path, tmp_path, backend="skimage"))
        out = run(cfg)
        V, F = load_3mf(out["3mf"])

        stats = diagnose(V, F)
        assert stats.watertight, stats.summary_line()
        assert stats.n_degenerate_faces == 0, stats.summary_line()
        assert stats.n_boundary_edges == 0
        assert stats.n_nonmanifold_edges == 0

        # bbox roughly matches tile.
        (xmin, ymin, zmin), (xmax, ymax, zmax) = bbox_of(V)
        assert -1.0 < xmin < 1.0 and -1.0 < ymin < 1.0 and -1.0 < zmin < 1.0
        assert 49.0 < xmax < 51.0 and 49.0 < ymax < 51.0 and 49.0 < zmax < 51.0

        # Porosity in the RCP range, with bridges contributing ~1% solid.
        phi = mesh_porosity_in_tile(V, F, (50, 50, 50))
        assert 0.30 < phi < 0.42, f"phi={phi:.4f}"

    def test_no_bridges_porosity_higher_than_with_bridges(self, packing_xyzd_path, tmp_path):
        """Cylinder bridges must add solid → reduce porosity."""
        from rcps.config import RcpsConfig
        from rcps.pipeline import run
        from tests._mesh_metrics import load_3mf, mesh_porosity_in_tile

        bare = run(RcpsConfig.from_dict(
            _config_dict(packing_xyzd_path, tmp_path / "bare", bridge_mode="none",
                         backend="skimage"),
        ))
        with_cyl = run(RcpsConfig.from_dict(
            _config_dict(packing_xyzd_path, tmp_path / "cyl", bridge_mode="cylinders",
                         backend="skimage"),
        ))
        V_bare, F_bare = load_3mf(bare["3mf"])
        V_cyl, F_cyl = load_3mf(with_cyl["3mf"])
        phi_bare = mesh_porosity_in_tile(V_bare, F_bare, (50, 50, 50))
        phi_cyl = mesh_porosity_in_tile(V_cyl, F_cyl, (50, 50, 50))
        assert phi_cyl < phi_bare, (phi_bare, phi_cyl)
        # The reduction is small but real (~1–3% depending on vox).
        assert (phi_bare - phi_cyl) > 0.002, (phi_bare, phi_cyl)


@pytest.mark.skipif(
    not (HAS_TRIMESH and HAS_PYMESHFIX and HAS_SKIMAGE),
    reason="trimesh + pymeshfix + scikit-image required",
)
class TestPoreMode:
    """End-to-end verification that Python pore export is correct.

    Background (2026-06-11): RCPS_v4.m's pore branch was a silent no-op
    (it returned the beads field; see `compose_field` docstring), and
    RCPS_v5.m now refuses 'pore' outright. The Python port fixed the
    composition (`pore = max(F_box, −F_beads)`); this test asserts the
    partition property end-to-end: beads + pore from the SAME grid must
    fill the tile box.

    Measurement details (audited 2026-06-13): volumes are summed PER
    CONNECTED COMPONENT in absolute value, because at coarse voxel sizes
    the pore space pinches into isolated pockets whose marching-cubes
    orientation can differ from the main body — a whole-mesh signed
    volume silently cancels them. The closure tolerance is 5% at
    vox=1 mm: measured closure converges 0.909 (vox=2) → 0.969 (vox=1)
    purely from discretization at sphere contacts; the failure mode this
    test exists to catch (pore ≡ beads, the v4 bug) produces closure
    ≈ 1.37, far outside the band.
    """

    def test_pore_and_beads_volumes_are_complementary(self, packing_xyzd_path, tmp_path):
        import trimesh

        from rcps.config import RcpsConfig
        from rcps.mesh import diagnose
        from rcps.pipeline import run
        from tests._mesh_metrics import load_3mf

        def comp_volume(V, F):
            mesh = trimesh.Trimesh(V, F, process=False)
            return float(sum(abs(c.volume)
                             for c in mesh.split(only_watertight=False)))

        beads = run(RcpsConfig.from_dict(_config_dict(
            packing_xyzd_path, tmp_path / "beads",
            vox=1.0, backend="skimage", export_what="beads",
        )))
        pore = run(RcpsConfig.from_dict(_config_dict(
            packing_xyzd_path, tmp_path / "pore",
            vox=1.0, backend="skimage", export_what="pore",
        )))
        Vb, Fb = load_3mf(beads["3mf"])
        Vp, Fp = load_3mf(pore["3mf"])

        sp = diagnose(Vp, Fp)
        assert sp.watertight, f"pore mesh: {sp.summary_line()}"

        vol_beads = comp_volume(Vb, Fb)
        vol_pore = comp_volume(Vp, Fp)
        v_box = 50.0 ** 3
        closure = (vol_beads + vol_pore) / v_box
        print(f"    [info] V_beads={vol_beads:.0f}, V_pore={vol_pore:.0f} mm³, "
              f"closure={closure:.4f} (1.0 = perfect partition)")
        assert abs(closure - 1.0) < 0.05, (
            f"beads + pore do not partition the box: closure={closure:.4f}"
        )
        # and the pore mesh is NOT secretly the beads mesh (the v4 bug):
        assert vol_pore < 0.7 * vol_beads, (vol_pore, vol_beads)


@pytest.mark.skipif(
    not (HAS_TRIMESH and HAS_PYMESHFIX and _has_a_mesher()),
    reason="trimesh + pymeshfix + (iso2mesh OR scikit-image) required",
)
class TestDiameterMode:
    """The new `diameter` bridge mode: validate the contact-graph invariant
    and overall mesh quality. No MATLAB reference exists for this mode."""

    def test_after_expand_to_touch_all_contacts_closed(self, packing_xyzd_path):
        """For every input contact pair, post-expansion gap ≤ 0."""
        from rcps.bridges import expand_to_touch, find_contact_pairs
        from rcps.field import cull_spheres, make_grid, replicate_with_ghost_tiles
        from rcps.io import load_packing_xyzd

        S = load_packing_xyzd(packing_xyzd_path)
        c, d = replicate_with_ghost_tiles(S[:, :3], S[:, 3], (50, 50, 50), 1)
        r = d / 2.0
        g = make_grid((50, 50, 50), 2.0, pad_vox=1, band_vox=3)
        bd = 3 * g.vox_size
        c, r, _ = cull_spheres(c, r, g, band_dist=bd)

        pairs = find_contact_pairs(c, r, contact_tol_mm=0.2)
        r_new = expand_to_touch(c, r, contact_tol_mm=0.2)

        # For every input contact (i, j): gap_new = ||c_i - c_j|| - (r_i_new + r_j_new) ≤ 0.
        diffs = c[pairs[:, 0]] - c[pairs[:, 1]]
        dist = np.sqrt((diffs * diffs).sum(axis=1))
        gaps_new = dist - r_new[pairs[:, 0]] - r_new[pairs[:, 1]]
        assert (gaps_new <= 1e-9).all(), (
            f"some contacts not closed: max gap after expansion = {gaps_new.max():.6g} mm"
        )

    def test_diameter_mode_mesh_is_watertight(self, packing_xyzd_path, tmp_path):
        from rcps.config import RcpsConfig
        from rcps.mesh import diagnose
        from rcps.pipeline import run
        from tests._mesh_metrics import load_3mf

        cfg = RcpsConfig.from_dict(_config_dict(packing_xyzd_path, tmp_path, bridge_mode="diameter"))
        out = run(cfg)
        V, F = load_3mf(out["3mf"])
        stats = diagnose(V, F)
        assert stats.watertight, stats.summary_line()
        assert stats.n_degenerate_faces == 0, stats.summary_line()

    def test_diameter_mode_porosity_within_bounds(self, packing_xyzd_path, tmp_path):
        """Diameter mode grows spheres slightly → porosity drops slightly vs no-bridges."""
        from rcps.config import RcpsConfig
        from rcps.pipeline import run
        from tests._mesh_metrics import load_3mf, mesh_porosity_in_tile

        bare = run(RcpsConfig.from_dict(_config_dict(packing_xyzd_path, tmp_path / "bare", bridge_mode="none")))
        diam = run(RcpsConfig.from_dict(_config_dict(packing_xyzd_path, tmp_path / "diam", bridge_mode="diameter")))
        V_b, F_b = load_3mf(bare["3mf"])
        V_d, F_d = load_3mf(diam["3mf"])
        phi_b = mesh_porosity_in_tile(V_b, F_b, (50, 50, 50))
        phi_d = mesh_porosity_in_tile(V_d, F_d, (50, 50, 50))
        # Diameter mode in data_example produces a max ∆r ≈ 0.25 mm at tol=0.2,
        # so the solid grows by a few %.
        assert phi_d <= phi_b + 1e-6, (phi_b, phi_d)


@pytest.mark.skipif(
    not (HAS_TRIMESH and HAS_PYMESHFIX and _has_a_mesher()),
    reason="trimesh + pymeshfix + (iso2mesh OR scikit-image) required",
)
class TestKeepSidesGeometry:
    """Partial `keep_sides` extends the mesh past the corresponding tile faces."""

    def test_S_kept_extends_past_z_W(self, packing_xyzd_path, tmp_path):
        from rcps.config import RcpsConfig
        from rcps.pipeline import run
        from tests._mesh_metrics import bbox_of, load_3mf

        cfg = RcpsConfig.from_dict(_config_dict(packing_xyzd_path, tmp_path, keep_sides=["S"]))
        out = run(cfg)
        V, F = load_3mf(out["3mf"])
        (_, _, zmin), (_, _, zmax) = bbox_of(V)
        # Bottom (-Z) is cut flush at z=0, top (+Z) protrudes past z=W=50.
        assert -1.0 < zmin < 1.0, zmin
        assert zmax > 50.0 + 0.5, f"S kept but zmax={zmax} did not protrude past 50"


# =====================================================================
# MATLAB reference comparison (auto-skip if fixture missing)
# =====================================================================

# Fixture path constant; the matlab_reference_3mf pytest fixture defined
# in `tests/conftest.py` auto-skips tests when this file is absent.

@pytest.mark.matlab_reference
@pytest.mark.slow
@pytest.mark.skipif(
    not (HAS_TRIMESH and HAS_PYMESHFIX and HAS_ISO2MESH),
    reason="MATLAB-vs-Python comparison requires iso2mesh + trimesh + pymeshfix",
)
class TestPythonMatchesMatlabReference:
    """Compare Python pipeline output to a MATLAB-produced reference.

    The reference .3mf is produced by running ``RCPS_v4.m`` with the
    locked recommended defaults on ``data_example/packing.xyzd`` and
    saving as ``tests/fixtures/reference.3mf``. See
    ``tests/fixtures/README.md``.

    Tolerances are tight because the SDF construction, bridges, and
    iso2mesh meshing are deterministic functions of the input — any
    drift indicates a real divergence between the implementations.
    """

    #: voxel size used by both the Python run and the MATLAB reference.
    VOX = 0.1

    @pytest.fixture(scope="class")
    def python_run(self, packing_xyzd_path, tmp_path_factory):
        """Run the full Python pipeline once per class at production vox=0.1.

        This run takes ~80 min. Set ``RCPS_E2E_CACHE_DIR`` to cache the
        produced ``.3mf`` across pytest invocations; the cache key hashes
        the config, the packing file, and every ``rcps/*.py`` source, so
        a stale cache cannot mask a code change.
        """
        import hashlib
        import json
        import shutil

        from rcps.config import RcpsConfig
        from rcps.pipeline import run
        from tests._mesh_metrics import load_3mf

        tmp = tmp_path_factory.mktemp("matlab_ref_compare")
        cfg_dict = _config_dict(
            packing_xyzd_path, tmp,
            vox=self.VOX,           # production resolution
            bridge_mode="cylinders",
            keep_sides=(),
            backend="iso2mesh",
        )

        cached = None
        cache_dir = os.environ.get("RCPS_E2E_CACHE_DIR")
        if cache_dir:
            import rcps

            h = hashlib.sha256()
            h.update(json.dumps(
                {k: v for k, v in cfg_dict.items() if k != "paths"},
                sort_keys=True,
            ).encode())
            h.update(packing_xyzd_path.read_bytes())
            for f in sorted(Path(rcps.__file__).parent.glob("*.py")):
                h.update(f.read_bytes())
            cached = Path(cache_dir) / f"python_ref_{h.hexdigest()[:16]}.3mf"
            if cached.exists():
                print(f"    [info] using cached python run: {cached}")
                return load_3mf(cached)

        out = run(RcpsConfig.from_dict(cfg_dict))
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out["3mf"], cached)
            print(f"    [info] cached python run -> {cached}")
        return load_3mf(out["3mf"])

    @pytest.fixture(scope="class")
    def matlab_run_aligned(self, matlab_reference_3mf):
        """Load the MATLAB mesh once and remove RCPS_v4.m's rigid frame offset.

        RCPS_v4.m's ``v2s`` branch maps NODE→mm one voxel off: with all
        tile faces cut flush, the mesh spans ``[-0.1, 49.9]`` instead of
        the physical ``[0, 50]`` (audited 2026-06-10 against the analytic
        packing; the Python port lands on ``[0, 50]`` to <1 µm). Decision:
        keep RCPS_v4.m as-is and compensate here, since a rigid translation
        has no effect on the printed geometry. Only an offset that is an
        integer multiple of the voxel size is removed — anything else
        would hide a real geometry difference and fails loudly.
        """
        from tests._mesh_metrics import bbox_of, load_3mf

        Vm, Fm = load_3mf(matlab_reference_3mf)
        bb_min, _ = bbox_of(Vm)
        offset = np.asarray(bb_min)          # flush cuts ⇒ true bbox_min = 0
        n_vox = np.round(offset / self.VOX)
        residual = np.abs(offset - n_vox * self.VOX)
        assert np.all(residual < 2e-3), (
            f"MATLAB frame offset {offset.tolist()} mm is not an integer "
            f"number of voxels (residual {residual.tolist()} mm); refusing "
            "to auto-align — investigate before relaxing this."
        )
        if np.any(n_vox != 0):
            print(f"    [info] removed rigid MATLAB frame offset: "
                  f"{(n_vox * self.VOX).tolist()} mm")
        return Vm - (n_vox * self.VOX).reshape(1, 3), Fm

    def test_python_mesh_is_in_absolute_tile_frame(self, python_run):
        """The Python mesh must span exactly [0, 50]³ — flush cuts at the
        tile faces. This is the absolute-frame claim the MATLAB reference
        cannot make (see `matlab_run_aligned`)."""
        from tests._mesh_metrics import bbox_of

        Vp, _ = python_run
        bb_min, bb_max = bbox_of(Vp)
        for i in range(3):
            assert abs(bb_min[i] - 0.0) < 5e-3, f"axis {i}: bbox_min={bb_min[i]}"
            assert abs(bb_max[i] - 50.0) < 5e-3, f"axis {i}: bbox_max={bb_max[i]}"

    def test_surface_area_within_1_percent(self, python_run, matlab_run_aligned):
        """Surface area is the tessellation-independent shape metric.

        Replaces the original vertex-count ±2% assertion (2026-06-10):
        with identical CGAL versions and RNG seeds, the two builds still
        differ by ~7% in vertex count, and even two consecutive Python
        runs differ by ~250 vertices — triangle counts fingerprint the
        CGAL binary/FP environment, not the geometry. Counts are still
        printed below for the record.
        """
        from tests._mesh_metrics import mesh_surface_area

        Vp, Fp = python_run
        Vm, Fm = matlab_run_aligned
        print(f"    [info] vertex counts (informational): py={Vp.shape[0]}, "
              f"matlab={Vm.shape[0]}, "
              f"delta={abs(Vp.shape[0]-Vm.shape[0])/max(1,Vm.shape[0])*100:.2f}%")
        Ap = mesh_surface_area(Vp, Fp)
        Am = mesh_surface_area(Vm, Fm)
        delta_pct = abs(Ap - Am) / Am * 100
        print(f"    [info] surface area: py={Ap:.2f}, matlab={Am:.2f} mm², "
              f"delta={delta_pct:.3f}%")
        assert delta_pct < 1.0, (
            f"surface-area divergence too large: py={Ap:.2f}, "
            f"matlab={Am:.2f} mm² ({delta_pct:.3f}% > 1%)"
        )

    def test_bbox_matches_within_1_micron(self, python_run, matlab_run_aligned):
        """After removing the documented rigid v4 frame offset, the two
        bounding boxes must agree to 1 µm."""
        from tests._mesh_metrics import bbox_distance, bbox_of

        Vp, _ = python_run
        Vm, _ = matlab_run_aligned
        d = bbox_distance(bbox_of(Vp), bbox_of(Vm))
        print(f"    [info] bbox L∞ distance (aligned): {d*1000:.3f} µm")
        assert d < 1e-3, f"bbox L∞ distance {d*1000:.3f} µm > 1 µm"

    def test_solid_volume_within_0_5_percent(self, python_run, matlab_run_aligned):
        from tests._mesh_metrics import mesh_volume_signed

        Vp, Fp = python_run
        Vm, Fm = matlab_run_aligned
        Vsol_p = abs(mesh_volume_signed(Vp, Fp))
        Vsol_m = abs(mesh_volume_signed(Vm, Fm))
        delta_pct = abs(Vsol_p - Vsol_m) / Vsol_m * 100
        print(f"    [info] solid volume: py={Vsol_p:.2f}, matlab={Vsol_m:.2f} mm³, "
              f"delta={delta_pct:.3f}%")
        assert delta_pct < 0.5, (
            f"solid volume divergence too large: {delta_pct:.3f}% > 0.5%"
        )

    def test_porosity_within_0_1_percent(self, python_run, matlab_run_aligned):
        from tests._mesh_metrics import mesh_porosity_in_tile

        Vp, Fp = python_run
        Vm, Fm = matlab_run_aligned
        phi_p = mesh_porosity_in_tile(Vp, Fp, (50, 50, 50))
        phi_m = mesh_porosity_in_tile(Vm, Fm, (50, 50, 50))
        delta = abs(phi_p - phi_m)
        print(f"    [info] porosity: py={phi_p:.6f}, matlab={phi_m:.6f}, "
              f"|Δ|={delta:.6f}")
        assert delta < 1e-3, f"|Δφ| = {delta:.6f} > 0.001 (0.1%)"

    def test_python_watertight_and_matlab_closed(self, python_run, matlab_run_aligned):
        """Python (the released product, pymeshfix-repaired) must be strictly
        watertight. The MATLAB reference is *raw* v2s output (canonical
        ``doRepair=false``): on the 2026-06-10 regeneration it carries 12
        degenerate faces + 12 non-manifold edges out of 14.6M faces
        (defect rate ~8e-7) while remaining CLOSED (0 boundary edges).

        Policy: assert on the RAW reference — closed + small defect
        budget. Do NOT filter the sliver faces first: they participate in
        the closure (removing them was measured to expose 7 boundary
        edges), and they are volumetrically irrelevant anyway — zero-area
        faces contribute nothing to the divergence-theorem integral, as
        proven by the 0.003% volume agreement with the Python mesh and
        the analytic packing.
        """
        from rcps.mesh import diagnose

        Vp, Fp = python_run
        Vm, Fm = matlab_run_aligned

        sp = diagnose(Vp, Fp)
        assert sp.watertight, f"python: {sp.summary_line()}"

        sm = diagnose(Vm, Fm)
        print(f"    [info] matlab raw reference: {sm.summary_line()}")
        assert sm.n_boundary_edges == 0, f"matlab not closed: {sm.summary_line()}"
        assert sm.n_nonmanifold_edges <= 32, (
            f"matlab defect budget exceeded (>32 non-manifold edges — "
            f"no longer a few CGAL slivers): {sm.summary_line()}"
        )
        assert sm.n_degenerate_faces <= 32, (
            f"matlab defect budget exceeded (>32 degenerate faces): "
            f"{sm.summary_line()}"
        )
