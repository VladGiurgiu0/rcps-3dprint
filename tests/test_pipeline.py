"""Tests for rcps.pipeline (Task 7).

The full e2e test requires iso2mesh + trimesh + pymeshfix and is gated.
A skimage-only e2e test runs when scikit-image is available.
The dry-run integration tests (steps 1-6 of the pipeline, no meshing)
work in any environment with numpy + pyyaml.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rcps.config import RcpsConfig

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


def _example_config(packing_xyzd_path: Path, tmp_path: Path, **overrides) -> RcpsConfig:
    """Build a programmatic config pointing at data_example/packing.xyzd.

    Uses a coarse voxel size by default so tests run in seconds, not minutes.
    """
    base = {
        "paths": {
            "packing": str(packing_xyzd_path),
            "root": str(packing_xyzd_path.parent),
            "out_dir": str(tmp_path),
        },
        "geom": {"tile_size_mm": [50, 50, 50]},
        "spheres": {"diameter_mm": 6.0, "expansion_factor": 1.0, "contact_tol_mm": 0.2},
        "grid": {"vox_size_mm": 2.0},
        "field": {"export_what": "beads", "ghost_tiles": 1, "pad_vox": 1, "band_vox": 3},
        "bridge": {"mode": "cylinders", "radius_frac": 0.15},
        "out": {"base_name": "test_tile"},
    }
    # Deep-merge overrides (the base intentionally omits optional sections
    # like "mesh", so absent keys must be created, not KeyError'd).
    for k, v in overrides.items():
        if isinstance(v, dict):
            base.setdefault(k, {}).update(v)
        else:
            base[k] = v
    return RcpsConfig.from_dict(base)


# =====================================================================
# Dry-run integration: pipeline pieces wired correctly (no meshing)
# =====================================================================

class TestPipelineDryRun:
    """Drive the pre-meshing portion of the pipeline directly and check
    the data shapes and porosity. Validates that config → field → bridges
    composition is correct without needing iso2mesh installed.
    """

    def test_beads_pipeline_shapes_and_porosity(self, packing_xyzd_path, tmp_path):
        from rcps.bridges import add_cylinders
        from rcps.field import (
            apply_keepsides_filter,
            build_beads_sdf_icsg,
            build_box_sdf,
            compose_field,
            cull_spheres,
            make_grid,
            replicate_with_ghost_tiles,
        )
        from rcps.io import load_packing_xyzd

        c = _example_config(packing_xyzd_path, tmp_path)

        S = load_packing_xyzd(c.paths.packing)
        centers, diameters = replicate_with_ghost_tiles(
            S[:, :3], S[:, 3], c.geom.tile_size_mm, c.field.ghost_tiles,
        )
        radii = (diameters / 2.0) * c.spheres.expansion_factor
        g = make_grid(c.geom.tile_size_mm, c.grid.vox_size_mm,
                      pad_vox=c.field.pad_vox, band_vox=c.field.band_vox,
                      keep_sides=c.field.keep_sides)
        band_dist = max(2, c.field.band_vox) * g.vox_size
        centers, radii, _ = cull_spheres(centers, radii, g, band_dist=band_dist)
        centers, radii, _ = apply_keepsides_filter(centers, radii, g)
        F_beads = build_beads_sdf_icsg(centers, radii, g, band_vox=c.field.band_vox)
        F_beads = add_cylinders(
            F_beads, centers, radii, g,
            contact_tol_mm=c.spheres.contact_tol_mm,
            radius_frac=c.bridge.radius_frac, band_vox=c.field.band_vox,
        )
        F_box = build_box_sdf(g, max_radius_mm=float(radii.max()))
        F = compose_field(F_beads, F_box, c.field.export_what)

        # Sanity: bounds, porosity in RCP range.
        n_box = int((F_box < 0).sum())
        n_solid = int((F < 0).sum())
        phi = 1.0 - n_solid / max(1, n_box)
        # Bridges add a bit of solid, so phi should be slightly below the
        # bare-beads value of 0.367 at vox=2.
        assert 0.30 < phi < 0.42, f"phi={phi:.4f}"


# =====================================================================
# Full e2e via rcps.pipeline.run — skipped if deps missing
# =====================================================================

@pytest.mark.skipif(
    not (HAS_SKIMAGE and HAS_TRIMESH),
    reason="scikit-image + trimesh required for skimage backend",
)
class TestPipelineRunSkimage:

    def test_run_with_skimage_backend(self, packing_xyzd_path, tmp_path):
        from rcps.pipeline import run

        c = _example_config(
            packing_xyzd_path, tmp_path,
            mesh={"backend": "skimage"},
            # Skip pymeshfix if not present by sending a config that lets
            # repair_and_finalize raise — but the default path uses meshfix.
            # We rely on test gating above to ensure trimesh is present;
            # skimage path still requires meshfix in repair_and_finalize.
        )
        if not HAS_PYMESHFIX:
            pytest.skip("pymeshfix not installed (repair step requires it)")
        written = run(c)
        assert "3mf" in written and Path(written["3mf"]).is_file()
        assert "info" in written and Path(written["info"]).is_file()
        assert "config_json" in written and Path(written["config_json"]).is_file()


@pytest.mark.skipif(
    not (HAS_ISO2MESH and HAS_TRIMESH and HAS_PYMESHFIX),
    reason="iso2mesh + trimesh + pymeshfix required",
)
class TestPipelineRunIso2Mesh:

    def test_run_with_iso2mesh_backend_writes_all_artefacts(self, packing_xyzd_path, tmp_path):
        from rcps.pipeline import run

        c = _example_config(packing_xyzd_path, tmp_path)
        written = run(c)
        for kind in ("3mf", "info", "config_json"):
            assert kind in written
            assert Path(written[kind]).is_file()

    def test_run_with_diameter_bridge_mode(self, packing_xyzd_path, tmp_path):
        from rcps.pipeline import run

        c = _example_config(
            packing_xyzd_path, tmp_path,
            bridge={"mode": "diameter", "radius_frac": 0.15},
        )
        written = run(c)
        assert Path(written["3mf"]).is_file()
