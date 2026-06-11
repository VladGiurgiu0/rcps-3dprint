"""Tests for the rcps-facility orchestrator.

Three tiers:

1. Pure-logic unit tests (keep-sides derivation, type dedup, diameter
   resolution) — run anywhere with numpy only.
2. Dry-run orchestration (assembly map without meshing) — numpy only.
3. A gated 2×2×1 end-to-end test at coarse vox with the skimage backend
   (slow; requires trimesh + pymeshfix + scikit-image).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rcps.facility import (
    derive_keep_sides,
    design_expansion_factor,
    parse_nfo_final_porosity,
    resolve_expansion_factor,
    run_facility,
    stored_porosity,
    type_tag,
    unique_tile_types,
)

# -------- optional-dep gating (e2e tier only) --------

try:
    import trimesh  # noqa: F401
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
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


# =====================================================================
# Tier 1 — keep-sides derivation
# =====================================================================

class TestDeriveKeepSides:

    def test_single_tile_all_flush(self):
        assert derive_keep_sides((0, 0, 0), (1, 1, 1)) == ()

    def test_two_by_one_by_one(self):
        # left tile keeps its +X face, right tile its -X face: they interlock.
        assert derive_keep_sides((0, 0, 0), (2, 1, 1)) == ("R",)
        assert derive_keep_sides((1, 0, 0), (2, 1, 1)) == ("L",)

    def test_two_by_two_by_one_corners(self):
        assert derive_keep_sides((0, 0, 0), (2, 2, 1)) == ("R", "A")
        assert derive_keep_sides((1, 0, 0), (2, 2, 1)) == ("L", "A")
        assert derive_keep_sides((0, 1, 0), (2, 2, 1)) == ("R", "P")
        assert derive_keep_sides((1, 1, 0), (2, 2, 1)) == ("L", "P")

    def test_interior_tile_keeps_all_six(self):
        assert derive_keep_sides((1, 1, 1), (3, 3, 3)) == (
            "L", "R", "P", "A", "I", "S",
        )

    def test_invalid_position_raises(self):
        with pytest.raises(ValueError):
            derive_keep_sides((2, 0, 0), (2, 1, 1))
        with pytest.raises(ValueError):
            derive_keep_sides((-1, 0, 0), (2, 1, 1))

    def test_invalid_grid_raises(self):
        with pytest.raises(ValueError):
            derive_keep_sides((0, 0, 0), (0, 1, 1))


class TestUniqueTileTypes:

    def test_1x1x1_one_type(self):
        types = unique_tile_types((1, 1, 1))
        assert list(types.keys()) == [()]
        assert types[()] == [(0, 0, 0)]

    def test_2x2x1_four_types(self):
        types = unique_tile_types((2, 2, 1))
        assert len(types) == 4
        assert sum(len(v) for v in types.values()) == 4

    def test_4x4x1_nine_types_sixteen_tiles(self):
        types = unique_tile_types((4, 4, 1))
        assert len(types) == 9
        copies = sorted(len(v) for v in types.values())
        assert copies == [1, 1, 1, 1, 2, 2, 2, 2, 4]
        assert sum(copies) == 16

    def test_3x3x3_27_types(self):
        types = unique_tile_types((3, 3, 3))
        assert len(types) == 27
        assert all(len(v) == 1 for v in types.values())

    def test_every_position_appears_exactly_once(self):
        types = unique_tile_types((4, 3, 2))
        seen = [p for v in types.values() for p in v]
        assert len(seen) == 24
        assert len(set(seen)) == 24

    def test_type_tag(self):
        assert type_tag(()) == "flush"
        assert type_tag(("L", "R")) == "keepLR"


# =====================================================================
# Tier 1 — half-open ownership (the assembly-collision guard)
# =====================================================================

class TestHalfOpenOwnership:

    def test_sphere_on_shared_plane_owned_exactly_once(self):
        """Center exactly at x=0 is kept; its periodic image at x=L is
        dropped — across identical adjacent tiles, the sphere prints
        exactly once. This is the rule that prevents assembly collisions."""
        from rcps.field import apply_keepsides_filter, make_grid

        g = make_grid(
            (50.0, 50.0, 50.0), 2.0, pad_vox=1, band_vox=3,
            keep_sides=("L", "R"), max_radius_mm=3.0,
        )
        centers = np.array([
            [0.0, 25.0, 25.0],    # exactly on the lower shared plane -> kept
            [50.0, 25.0, 25.0],   # periodic image on the upper plane -> dropped
            [25.0, 25.0, 25.0],   # interior -> kept
            [-3.0, 25.0, 25.0],   # beyond kept lower face -> dropped
            [53.0, 25.0, 25.0],   # beyond kept upper face -> dropped
        ])
        radii = np.full(len(centers), 3.0)
        _, _, keep = apply_keepsides_filter(centers, radii, g)
        assert keep.tolist() == [True, False, True, False, False]

    def test_no_keep_sides_is_noop(self):
        from rcps.field import apply_keepsides_filter, make_grid

        g = make_grid((50.0, 50.0, 50.0), 2.0, pad_vox=1, band_vox=3)
        centers = np.array([[0.0, 0.0, 0.0], [50.0, 50.0, 50.0]])
        _, _, keep = apply_keepsides_filter(centers, np.ones(2), g)
        assert keep.all()


# =====================================================================
# Tier 1 — diameter convention
# =====================================================================

class TestDiameterConvention:

    def test_stored_porosity_matches_analytic(self, packing_xyzd_path):
        from rcps.io import load_packing_xyzd

        S = load_packing_xyzd(packing_xyzd_path)
        phi = stored_porosity(S, (50.0, 50.0, 50.0))
        assert abs(phi - 0.363339) < 1e-4

    def test_nfo_parse_and_design_factor(self, packing_xyzd_path, packing_nfo_path):
        from rcps.io import load_packing_xyzd

        phi_nfo = parse_nfo_final_porosity(packing_nfo_path)
        assert abs(phi_nfo - 0.350369) < 1e-4

        S = load_packing_xyzd(packing_xyzd_path)
        f = design_expansion_factor(S, (50.0, 50.0, 50.0), phi_nfo)
        # rescales stored d = 5.9598 to the design d = 6.000
        assert abs(f - 1.00674) < 5e-4
        assert abs(float(np.mean(S[:, 3])) * f - 6.0) < 2e-3

    def test_resolve_modes(self, packing_xyzd_path):
        f, info = resolve_expansion_factor("stored", packing_xyzd_path, (50, 50, 50))
        assert f == 1.0 and info["mode"] == "stored"

        f, info = resolve_expansion_factor("design", packing_xyzd_path, (50, 50, 50))
        assert abs(f - 1.00674) < 5e-4
        assert abs(info["phi_realized"] - 0.350369) < 1e-4

        f, info = resolve_expansion_factor(1.02, packing_xyzd_path, (50, 50, 50))
        assert f == 1.02 and info["mode"] == "explicit"
        # phi must DROP when spheres grow
        assert info["phi_realized"] < info["phi_at_stored_diameters"]

    def test_bad_factor_raises(self, packing_xyzd_path):
        with pytest.raises(ValueError):
            resolve_expansion_factor(-1.0, packing_xyzd_path, (50, 50, 50))


# =====================================================================
# Tier 2 — dry-run orchestration (no meshing)
# =====================================================================

def _base_config(packing_path, out_dir):
    from rcps.config import RcpsConfig

    return RcpsConfig.from_dict({
        "paths": {
            "packing": str(packing_path),
            "root": str(packing_path.parent),
            "out_dir": str(out_dir),
        },
        "geom": {"tile_size_mm": [50, 50, 50]},
        "spheres": {"diameter_mm": 6.0, "expansion_factor": 1.0,
                    "contact_tol_mm": 0.2},
        "grid": {"vox_size_mm": 2.0},
        "field": {"export_what": "beads", "ghost_tiles": 1, "pad_vox": 1,
                  "band_vox": 3, "keep_sides": []},
        "bridge": {"mode": "cylinders", "radius_frac": 0.15},
        "mesh": {"backend": "skimage"},
        "out": {"base_name": "facility_test"},
    })


class TestDryRun:

    def test_dry_run_writes_assembly_map(self, packing_xyzd_path, tmp_path):
        import json

        cfg = _base_config(packing_xyzd_path, tmp_path)
        res = run_facility(cfg, (4, 4, 1), diameter="design",
                           out_dir=tmp_path / "fac", dry_run=True)
        assert res["n_types"] == 9 and res["n_tiles"] == 16
        assert res["meshes"] == {}
        assert res["map_json"].exists() and res["map_txt"].exists()

        amap = json.loads(res["map_json"].read_text())
        assert amap["grid"] == [4, 4, 1]
        assert amap["n_unique_types"] == 9
        assert sum(t["copies"] for t in amap["tile_types"]) == 16
        assert all(t["mesh"] is None for t in amap["tile_types"])
        assert abs(amap["diameter"]["expansion_factor"] - 1.00674) < 5e-4
        # every grid position appears exactly once in the map
        seen = {tuple(p) for t in amap["tile_types"] for p in t["positions"]}
        assert len(seen) == 16

    def test_dry_run_1x1x1_is_single_flush_tile(self, packing_xyzd_path, tmp_path):
        cfg = _base_config(packing_xyzd_path, tmp_path)
        res = run_facility(cfg, (1, 1, 1), diameter="stored",
                           out_dir=tmp_path / "fac1", dry_run=True)
        assert res["n_types"] == 1 and res["n_tiles"] == 1
        assert "flush" in res["map_txt"].read_text()


# =====================================================================
# Tier 3 — gated 2×2×1 end-to-end at coarse vox (slow)
# =====================================================================

@pytest.mark.slow
@pytest.mark.skipif(
    not (HAS_TRIMESH and HAS_PYMESHFIX and HAS_SKIMAGE),
    reason="trimesh + pymeshfix + scikit-image required",
)
class TestFacility2x2x1EndToEnd:

    def test_four_meshes_interlock_geometry(self, packing_xyzd_path, tmp_path):
        """2×2×1 facility at vox=2: 4 unique corner tiles. Each mesh must
        be watertight, protrude past its kept faces, and stay flush at
        its cut faces."""
        from rcps.mesh import diagnose
        from tests._mesh_metrics import bbox_of, load_3mf

        cfg = _base_config(packing_xyzd_path, tmp_path)
        res = run_facility(cfg, (2, 2, 1), diameter="stored",
                           out_dir=tmp_path / "fac22", dry_run=False)
        assert res["n_types"] == 4
        assert len(res["meshes"]) == 4

        L = 50.0
        for tag, path in res["meshes"].items():
            V, F = load_3mf(Path(path))
            stats = diagnose(V, F)
            assert stats.watertight, f"{tag}: {stats.summary_line()}"

            (xmin, ymin, zmin), (xmax, ymax, zmax) = bbox_of(V)
            keeps = tag.replace("keep", "")
            # kept faces protrude clearly; flush faces stay near the plane
            if "L" in keeps:
                assert xmin < -0.5, (tag, xmin)
            else:
                assert -1.0 < xmin < 1.0, (tag, xmin)
            if "R" in keeps:
                assert xmax > L + 0.5, (tag, xmax)
            else:
                assert L - 1.0 < xmax < L + 1.0, (tag, xmax)
            if "P" in keeps:
                assert ymin < -0.5, (tag, ymin)
            else:
                assert -1.0 < ymin < 1.0, (tag, ymin)
            if "A" in keeps:
                assert ymax > L + 0.5, (tag, ymax)
            else:
                assert L - 1.0 < ymax < L + 1.0, (tag, ymax)
            # z faces are all exterior in a 2x2x1 grid -> flush
            assert -1.0 < zmin < 1.0 and L - 1.0 < zmax < L + 1.0, (tag, zmin, zmax)
