"""Tests for rcps.cli (Task 7)."""

from __future__ import annotations

import pytest

from rcps.cli import (
    EXIT_BAD_USAGE,
    EXIT_MISSING_FILE,
    EXIT_OK,
    _build_parser,
    build_main,
    facility_main,
)


class TestBuildParser:

    def test_required_config_argument(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _build_parser().parse_args([])
        assert exc.value.code == 2

    def test_help_works(self, capsys):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--help"])
        out = capsys.readouterr().out
        assert "rcps-build" in out and "YAML config" in out

    def test_version_works(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _build_parser().parse_args(["--version"])
        # argparse `--version` exits cleanly.
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "rcps-3dprint" in out

    def test_out_dir_override_parses(self, tmp_path):
        ns = _build_parser().parse_args([str(tmp_path / "c.yaml"), "--out-dir", str(tmp_path / "out")])
        assert ns.out_dir == tmp_path / "out"

    def test_log_level_choices(self):
        ns = _build_parser().parse_args(["c.yaml", "--log-level", "DEBUG"])
        assert ns.log_level == "DEBUG"
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["c.yaml", "--log-level", "TRACE"])


class TestBuildMainExitCodes:

    def test_missing_config_yields_missing_file(self, tmp_path, capsys):
        rc = build_main([str(tmp_path / "does-not-exist.yaml")])
        assert rc == EXIT_MISSING_FILE
        # Error message should mention the missing file
        err = capsys.readouterr().err
        assert "does-not-exist" in err

    def test_bad_config_yields_bad_usage(self, tmp_path, capsys):
        """A YAML file with invalid content should exit with bad-usage code."""
        bad = tmp_path / "bad.yaml"
        # Missing required `spheres` section.
        bad.write_text(
            "paths:\n"
            "  packing: p.xyzd\n"
            "geom:\n"
            "  tile_size_mm: [50, 50, 50]\n"
        )
        rc = build_main([str(bad)])
        assert rc == EXIT_BAD_USAGE
        err = capsys.readouterr().err
        assert "spheres" in err.lower() or "config error" in err.lower()


class TestFacilityMain:
    """rcps-facility is implemented (2026-06-11); these replace the old
    placeholder tests."""

    def test_missing_required_args_exits_2(self):
        # argparse exits with SystemExit(2) when config/--grid/--diameter
        # are missing — that is the standard CLI contract.
        with pytest.raises(SystemExit) as exc:
            facility_main([])
        assert exc.value.code == 2

    def test_help_works(self):
        with pytest.raises(SystemExit) as exc:
            facility_main(["--help"])
        assert exc.value.code == 0

    def test_missing_config_file_exit_code(self, tmp_path):
        rc = facility_main([str(tmp_path / "nope.yaml"),
                            "--grid", "2", "2", "1", "--diameter", "stored"])
        assert rc == EXIT_MISSING_FILE

    def test_dry_run_end_to_end(self, packing_xyzd_path, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "paths:\n"
            f"  packing: {packing_xyzd_path}\n"
            f"  root: {packing_xyzd_path.parent}\n"
            f"  out_dir: {tmp_path / 'out'}\n"
            "geom:\n  tile_size_mm: [50, 50, 50]\n"
            "spheres:\n  diameter_mm: 6.0\n"
            "grid:\n  vox_size_mm: 2.0\n"
            "field:\n  export_what: beads\n"
            "bridge:\n  mode: cylinders\n"
            "mesh:\n  backend: skimage\n"
            "out:\n  base_name: cli_test\n"
        )
        rc = facility_main([str(cfg), "--grid", "4", "4", "1",
                            "--diameter", "design", "--dry-run",
                            "--out-dir", str(tmp_path / "fac")])
        assert rc == EXIT_OK
        assert (tmp_path / "fac" / "facility_map.txt").exists()
