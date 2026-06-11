"""Tests for rcps.config (Task 7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rcps.config import (
    BridgeConfig,
    FieldConfig,
    GeomConfig,
    Iso2MeshConfig,
    MeshConfig,
    OutConfig,
    PathsConfig,
    RcpsConfig,
    SpheresConfig,
)

# =====================================================================
# Section dataclasses — validation rules
# =====================================================================

class TestPathsConfig:

    def test_coerces_strings_to_path(self):
        p = PathsConfig(packing="data/p.xyzd", root=".", out_dir="./out")
        assert isinstance(p.packing, Path) and str(p.packing) == "data/p.xyzd"
        assert isinstance(p.root, Path)
        assert isinstance(p.out_dir, Path)


class TestGeomConfig:

    def test_accepts_list_for_tile_size(self):
        g = GeomConfig(tile_size_mm=[50.0, 50.0, 50.0])
        assert g.tile_size_mm == (50.0, 50.0, 50.0)

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="3 entries"):
            GeomConfig(tile_size_mm=[50.0, 50.0])

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError, match="positive"):
            GeomConfig(tile_size_mm=[50.0, 0.0, 50.0])

    def test_rejects_bad_porosity(self):
        with pytest.raises(ValueError, match="target_porosity"):
            GeomConfig(tile_size_mm=[10, 10, 10], target_porosity=1.0)
        with pytest.raises(ValueError, match="target_porosity"):
            GeomConfig(tile_size_mm=[10, 10, 10], target_porosity=-0.1)


class TestSpheresConfig:

    def test_defaults(self):
        s = SpheresConfig(diameter_mm=6.0)
        assert s.expansion_factor == 1.0
        assert s.contact_tol_mm == 0.20

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            SpheresConfig(diameter_mm=0)
        with pytest.raises(ValueError):
            SpheresConfig(diameter_mm=6.0, expansion_factor=0)
        with pytest.raises(ValueError):
            SpheresConfig(diameter_mm=6.0, contact_tol_mm=-0.1)


class TestFieldConfig:

    def test_defaults(self):
        f = FieldConfig()
        assert f.export_what == "beads"
        assert f.ghost_tiles == 1
        assert f.keep_sides == []

    def test_export_what_enum(self):
        with pytest.raises(ValueError, match="export_what"):
            FieldConfig(export_what="solid")

    def test_keep_sides_uppercase_unique(self):
        f = FieldConfig(keep_sides=["s", "S", "r"])
        assert f.keep_sides == ["S", "R"]

    def test_keep_sides_rejects_invalid_label(self):
        with pytest.raises(ValueError, match="keep_sides"):
            FieldConfig(keep_sides=["X"])

    def test_rejects_negative_ints(self):
        with pytest.raises(ValueError):
            FieldConfig(ghost_tiles=-1)
        with pytest.raises(ValueError):
            FieldConfig(band_vox=0)


class TestBridgeConfig:

    def test_defaults(self):
        b = BridgeConfig()
        assert b.mode == "cylinders"
        assert b.radius_frac == 0.15

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="bridge.mode"):
            BridgeConfig(mode="caps")

    def test_rejects_bad_radius_frac(self):
        with pytest.raises(ValueError):
            BridgeConfig(radius_frac=0.0)
        with pytest.raises(ValueError):
            BridgeConfig(radius_frac=1.5)


class TestMeshConfig:

    def test_defaults(self):
        m = MeshConfig()
        assert m.backend == "iso2mesh"
        assert m.iso2mesh.angbound_deg == 25.0

    def test_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="mesh.backend"):
            MeshConfig(backend="cgal")


class TestIso2MeshConfig:

    def test_rejects_bad_angbound(self):
        with pytest.raises(ValueError):
            Iso2MeshConfig(angbound_deg=0.0)
        with pytest.raises(ValueError):
            Iso2MeshConfig(angbound_deg=90.0)


class TestOutConfig:

    def test_must_emit_some_mesh(self):
        with pytest.raises(ValueError, match="at least one"):
            OutConfig(save_3mf=False, save_stl=False)


# =====================================================================
# RcpsConfig — top-level
# =====================================================================

class TestRcpsConfigFromDict:

    def test_minimal_config(self):
        d = {
            "paths": {"packing": "p.xyzd"},
            "geom": {"tile_size_mm": [50, 50, 50]},
            "spheres": {"diameter_mm": 6.0},
        }
        c = RcpsConfig.from_dict(d)
        # Defaults flowed through.
        assert c.field.export_what == "beads"
        assert c.bridge.mode == "cylinders"
        assert c.grid.vox_size_mm == 0.1

    def test_missing_required_section_raises(self):
        with pytest.raises(ValueError, match="paths"):
            RcpsConfig.from_dict({"geom": {"tile_size_mm": [50, 50, 50]}, "spheres": {"diameter_mm": 6}})

    def test_full_config_passes_through(self):
        d = {
            "paths": {"packing": "p.xyzd", "root": ".", "out_dir": "./out"},
            "geom": {"tile_size_mm": [50, 50, 50], "target_porosity": 0.35},
            "spheres": {"diameter_mm": 6.0, "expansion_factor": 1.0, "contact_tol_mm": 0.2},
            "field": {
                "export_what": "beads", "ghost_tiles": 1, "pad_vox": 1,
                "band_vox": 3, "keep_sides": ["S", "R"],
            },
            "grid": {"vox_size_mm": 0.1},
            "bridge": {"mode": "cylinders", "radius_frac": 0.15},
            "mesh": {
                "backend": "iso2mesh",
                "iso2mesh": {
                    "angbound_deg": 25.0, "radbound": 1.0,
                    "distbound": 0.1, "maxnode": 200_000_000,
                },
            },
            "out": {"base_name": "x", "save_3mf": True, "save_stl": False,
                    "write_info_txt": True, "write_config_json": True},
        }
        c = RcpsConfig.from_dict(d)
        assert c.field.keep_sides == ["S", "R"]
        assert c.mesh.iso2mesh.angbound_deg == 25.0
        assert c.out.base_name == "x"


class TestRcpsConfigFromYaml:

    yaml = pytest.importorskip("yaml")  # PyYAML required

    def test_loads_example_config(self):
        repo = Path(__file__).resolve().parent.parent
        example = repo / "examples" / "config_50mm_d6_phi035.yaml"
        c = RcpsConfig.from_yaml(example)

        # Relative `paths.packing: ../data_example/packing.xyzd` was resolved
        # against the YAML's directory.
        assert c.paths.packing.is_absolute()
        assert c.paths.packing.parts[-2:] == ("data_example", "packing.xyzd")
        assert c.paths.packing.is_file(), f"resolved path missing: {c.paths.packing}"

        # Defaults from the example.
        assert c.geom.tile_size_mm == (50.0, 50.0, 50.0)
        assert c.spheres.diameter_mm == 6.0
        assert c.bridge.mode == "cylinders"
        assert c._source_yaml == example.resolve()

    def test_missing_yaml_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RcpsConfig.from_yaml(tmp_path / "nope.yaml")

    def test_non_mapping_root_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("- 1\n- 2\n")
        with pytest.raises(ValueError, match="mapping"):
            RcpsConfig.from_yaml(bad)

    def test_relative_paths_resolved_against_config_dir(self, tmp_path):
        # Put a packing file under tmp/data_example/, and a config under tmp/cfg/
        (tmp_path / "data_example").mkdir()
        packing = tmp_path / "data_example" / "p.xyzd"
        packing.write_bytes(b"\x00" * 32)  # any non-empty file
        cfgdir = tmp_path / "cfg"
        cfgdir.mkdir()
        cfgfile = cfgdir / "c.yaml"
        cfgfile.write_text(
            "paths:\n"
            "  packing: ../data_example/p.xyzd\n"
            "geom:\n"
            "  tile_size_mm: [50, 50, 50]\n"
            "spheres:\n"
            "  diameter_mm: 6.0\n"
        )
        c = RcpsConfig.from_yaml(cfgfile)
        assert c.paths.packing == packing.resolve()


class TestRcpsConfigToDict:

    def test_paths_serialised_as_strings(self):
        d = {
            "paths": {"packing": "/abs/p.xyzd", "root": "/abs", "out_dir": "/abs/out"},
            "geom": {"tile_size_mm": [50, 50, 50]},
            "spheres": {"diameter_mm": 6.0},
        }
        c = RcpsConfig.from_dict(d)
        out = c.to_dict()
        assert isinstance(out["paths"]["packing"], str)
        # JSON-serialisable round-trip.
        s = json.dumps(out)
        round_tripped = json.loads(s)
        assert round_tripped["geom"]["tile_size_mm"] == [50, 50, 50]

    def test_source_yaml_not_leaked(self):
        d = {
            "paths": {"packing": "p.xyzd"},
            "geom": {"tile_size_mm": [50, 50, 50]},
            "spheres": {"diameter_mm": 6.0},
        }
        c = RcpsConfig.from_dict(d)
        c._source_yaml = Path("/tmp/cfg.yaml")
        out = c.to_dict()
        assert "_source_yaml" not in out
