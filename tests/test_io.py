"""Tests for rcps.io (Task 3)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rcps.io import (
    estimate_n_spheres,
    load_packing_xyzd,
    sha256_of_file,
    write_3mf,
    write_config_json,
    write_info_txt,
)

# ---------------------------------------------------------------------
# load_packing_xyzd
# ---------------------------------------------------------------------

class TestLoadPackingXyzd:
    """Validate the binary loader against the canonical example."""

    def test_shape_and_dtype(self, packing_xyzd_path):
        """data_example/packing.xyzd yields (718, 4) float64, per packing.nfo."""
        S = load_packing_xyzd(packing_xyzd_path)
        assert S.shape == (718, 4), f"expected (718, 4), got {S.shape}"
        assert S.dtype == np.float64

    def test_coordinates_in_tile_box(self, packing_xyzd_path):
        """All centers lie within the 50x50x50 mm tile."""
        S = load_packing_xyzd(packing_xyzd_path)
        x, y, z = S[:, 0], S[:, 1], S[:, 2]
        # Centers (not surfaces) lie inside [r, 50-r] for spheres of d=6 → r=3.
        # Allow [0, 50] envelope to avoid coupling this test to keepSides logic.
        assert x.min() >= 0.0 and x.max() <= 50.0
        assert y.min() >= 0.0 and y.max() <= 50.0
        assert z.min() >= 0.0 and z.max() <= 50.0

    def test_monodisperse_uniform_diameter(self, packing_xyzd_path):
        """All spheres share the same diameter (within float roundoff).

        Note: the actual diameter stored in the file (~5.96 mm) is slightly
        smaller than the nominal 6 mm given to the packing generator.
        Baranau's contraction algorithm scales the spheres during packing;
        the as-stored diameter is the convergence diameter. See
        `test_porosity_from_loaded_data_is_in_rcp_range` for the consequence.
        """
        S = load_packing_xyzd(packing_xyzd_path)
        d = S[:, 3]
        assert d.min() == d.max(), (
            f"non-monodisperse: range [{d.min()}, {d.max()}]"
        )
        # Soft sanity bound: actual diameter within 10% of the nominal 6 mm.
        assert 5.4 < d[0] < 6.6, f"diameter {d[0]} outside reasonable bounds"

    def test_porosity_from_loaded_data_is_in_rcp_range(self, packing_xyzd_path):
        """Porosity computed from N and the *actual* stored diameter lies near 0.36.

        IMPORTANT — the `packing.nfo` "Final Porosity" of 0.35037 is computed
        with the *nominal* d=6 mm given as input to packing-generation, not the
        as-stored diameter ~5.96 mm. With d≈5.96 and N=718 the achievable
        porosity is ≈0.363, very close to the random-close-packing limit for
        monodisperse hard spheres (φ_RCP ≈ 0.36 ± 0.02; Baranau & Tallarek 2014).
        This is why the MATLAB pipeline uses the *loaded* diameter d0 = S(:,4)
        for meshing, treating `p.spheres.diameter` as nominal/initial-guess only.
        """
        S = load_packing_xyzd(packing_xyzd_path)
        N, d = S.shape[0], float(S[0, 3])
        V_sph = (4.0 / 3.0) * np.pi * (d / 2.0) ** 3
        V_box = 50.0 ** 3
        phi = 1.0 - (N * V_sph) / V_box
        assert 0.30 < phi < 0.42, f"unexpected porosity {phi:.4f}"

    def test_nominal_porosity_matches_nfo(self, packing_xyzd_path, packing_nfo_path):
        """If we compute porosity with the *nominal* d=6 (not the stored ~5.96),
        we recover the .nfo's "Final Porosity" of 0.35037.

        This documents the convention: .nfo "Final Porosity" is computed with
        nominal d, while packing.xyzd stores the (smaller) convergence d.
        """
        S = load_packing_xyzd(packing_xyzd_path)
        N = S.shape[0]
        d_nominal = 6.0
        V_sph = (4.0 / 3.0) * np.pi * (d_nominal / 2.0) ** 3
        V_box = 50.0 ** 3
        phi_nominal = 1.0 - (N * V_sph) / V_box
        assert phi_nominal == pytest.approx(0.35037, abs=5e-5), (
            f"got phi_nominal={phi_nominal}"
        )

        # Cross-check against packing.nfo's "Final Porosity" line.
        for line in packing_nfo_path.read_text().splitlines():
            if "Final Porosity" in line:
                nfo_phi = float(line.split(":")[1].split()[0])
                assert nfo_phi == pytest.approx(phi_nominal, abs=1e-6), (
                    f"nfo={nfo_phi}, computed_nominal={phi_nominal}"
                )
                break
        else:
            pytest.fail("packing.nfo did not contain a 'Final Porosity' line")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_packing_xyzd(tmp_path / "does_not_exist.xyzd")

    def test_bad_length_raises(self, tmp_path):
        bad = tmp_path / "bad.xyzd"
        # Write 5 doubles (not a multiple of 4 quadruples)
        np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="<f8").tofile(bad)
        with pytest.raises(ValueError, match="not divisible by 4"):
            load_packing_xyzd(bad)

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.xyzd"
        empty.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            load_packing_xyzd(empty)

    def test_roundtrip_known_payload(self, tmp_path):
        """Write a known 2-sphere payload and read it back exactly."""
        ref = np.array(
            [[1.0, 2.0, 3.0, 6.0],
             [10.5, 20.25, 30.125, 6.0]],
            dtype=np.float64,
        )
        # Write in interleaved little-endian, the convention packing-generation produces.
        path = tmp_path / "tiny.xyzd"
        ref.astype("<f8").tofile(path)
        loaded = load_packing_xyzd(path)
        np.testing.assert_array_equal(loaded, ref)


# ---------------------------------------------------------------------
# estimate_n_spheres
# ---------------------------------------------------------------------

class TestEstimateNSpheres:

    def test_matches_data_example(self):
        """50³ mm tile, d=6 mm, phi=0.35 → 718 spheres (per data_example/packing.nfo)."""
        assert estimate_n_spheres((50, 50, 50), 6.0, 0.35) == 718

    def test_scaling_with_tile_volume(self):
        """Doubling all tile dimensions multiplies N by ~8.

        The exact value differs by up to 7 due to per-scale floor()
        rounding: ``floor(8x) − 8·floor(x) ∈ [0, 7]``.
        """
        n1 = estimate_n_spheres((50, 50, 50), 6.0, 0.35)
        n2 = estimate_n_spheres((100, 100, 100), 6.0, 0.35)
        assert abs(8 * n1 - n2) <= 7, f"8·{n1}={8 * n1} vs {n2}"

    def test_scaling_with_diameter(self):
        """Doubling diameter shrinks N by ~8 (V_sphere ∝ d³)."""
        n1 = estimate_n_spheres((50, 50, 50), 6.0, 0.35)
        n2 = estimate_n_spheres((50, 50, 50), 12.0, 0.35)
        assert abs(n1 // 8 - n2) <= 7, f"{n1}//8={n1 // 8} vs {n2}"

    def test_phi_zero_max_packing(self):
        """phi=0 means box is completely solid spheres."""
        n = estimate_n_spheres((50, 50, 50), 6.0, 0.0)
        # 50³ / V_sphere(d=6) = 125000 / 113.097... ≈ 1105
        assert n == 1105

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            estimate_n_spheres((-50, 50, 50), 6.0, 0.35)
        with pytest.raises(ValueError):
            estimate_n_spheres((50, 50, 50), 0.0, 0.35)
        with pytest.raises(ValueError):
            estimate_n_spheres((50, 50, 50), 6.0, 1.0)
        with pytest.raises(ValueError):
            estimate_n_spheres((50, 50, 50), 6.0, -0.1)


# ---------------------------------------------------------------------
# sha256_of_file
# ---------------------------------------------------------------------

class TestSha256OfFile:

    def test_known_hash_empty_file(self, tmp_path):
        """SHA-256 of an empty file is the well-known constant."""
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        assert sha256_of_file(p) == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_known_hash_abc(self, tmp_path):
        """SHA-256('abc') is the well-known constant."""
        p = tmp_path / "abc.txt"
        p.write_bytes(b"abc")
        assert sha256_of_file(p) == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )

    def test_chunked_matches_oneshot(self, tmp_path):
        """Chunked reading produces the same hash as a single-shot read."""
        # ~5 MB random payload to exercise multiple chunks at the default 1 MiB chunk size.
        rng = np.random.default_rng(0)
        payload = rng.bytes(5_000_000)
        p = tmp_path / "blob.bin"
        p.write_bytes(payload)

        import hashlib
        one_shot = hashlib.sha256(payload).hexdigest()
        chunked = sha256_of_file(p, chunk_size=1 << 12)  # 4 KiB chunks, many iters
        assert chunked == one_shot


# ---------------------------------------------------------------------
# write_info_txt / write_config_json
# ---------------------------------------------------------------------

class TestSidecars:

    @pytest.fixture
    def example_config(self):
        return {
            "paths": {"root": ".", "packing": "packing.xyzd", "out_dir": "out"},
            "geom": {"tile_size_mm": [50.0, 50.0, 50.0], "target_porosity": 0.35},
            "spheres": {"diameter_mm": 6.0, "expansion_factor": 1.0, "contact_tol_mm": 0.2},
            "field": {
                "export_what": "beads",
                "ghost_tiles": 1,
                "pad_vox": 1,
                "band_vox": 3,
                "keep_sides": ["S", "R"],
            },
            "grid": {"vox_size_mm": 0.1},
            "bridge": {"mode": "cylinders", "radius_frac": 0.15},
            "mesh": {
                "iso2mesh": {
                    "angbound_deg": 25.0,
                    "radbound": 1.0,
                    "distbound": 0.1,
                    "maxnode": 200_000_000,
                }
            },
            "out": {"base_name": "test"},
        }

    def test_info_txt_writes_flat_dotted_keys(self, tmp_path, example_config):
        p = tmp_path / "info.txt"
        write_info_txt(p, example_config)
        text = p.read_text()
        # Spot-check some flattened keys.
        assert "geom.target_porosity: 0.35" in text
        assert "mesh.iso2mesh.angbound_deg: 25" in text
        assert "field.keep_sides: S R" in text
        # Header line includes the rcps version.
        from rcps import __version__
        assert __version__ in text

    def test_info_txt_includes_runtime_section(self, tmp_path, example_config):
        p = tmp_path / "info.txt"
        write_info_txt(p, example_config, runtime={"snapped_vox_size_mm": 0.1, "grid_dims": [502, 502, 502]})
        text = p.read_text()
        assert "runtime.snapped_vox_size_mm: 0.1" in text
        assert "runtime.grid_dims: 502 502 502" in text

    def test_config_json_roundtrip(self, tmp_path, example_config, packing_xyzd_path):
        p = tmp_path / "config.json"
        write_config_json(
            p,
            example_config,
            packing_path=packing_xyzd_path,
            runtime={"elapsed_seconds": 26.5, "grid_dims": [502, 502, 502]},
        )
        payload = json.loads(p.read_text())
        assert payload["config"] == example_config
        assert payload["input"]["packing_sha256"] == sha256_of_file(packing_xyzd_path)
        assert payload["runtime"]["elapsed_seconds"] == 26.5
        assert "rcps_version" in payload and "written_at" in payload

    def test_config_json_handles_numpy_types(self, tmp_path):
        """numpy scalars and arrays are coerced to native JSON types."""
        cfg = {
            "n": np.int64(42),
            "x": np.float32(3.14),
            "arr": np.array([1.0, 2.0, 3.0]),
        }
        p = tmp_path / "config.json"
        write_config_json(p, cfg)
        payload = json.loads(p.read_text())
        assert payload["config"]["n"] == 42
        assert payload["config"]["x"] == pytest.approx(3.14, rel=1e-6)
        assert payload["config"]["arr"] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------
# write_3mf  (trimesh-dependent; auto-skip if trimesh missing)
# ---------------------------------------------------------------------

trimesh = pytest.importorskip("trimesh", reason="trimesh not installed")


class TestWrite3mf:

    @pytest.fixture
    def tetrahedron(self):
        V = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        F = np.array([
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ])
        return V, F

    def test_writes_file_with_vertex_count(self, tmp_path, tetrahedron):
        V, F = tetrahedron
        out = write_3mf(V, F, tmp_path / "tet.3mf")
        assert out.is_file()
        assert out.stat().st_size > 0
        loaded = trimesh.load(out)
        # trimesh may return a Scene; normalize to Trimesh.
        if isinstance(loaded, trimesh.Scene):
            loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        assert loaded.vertices.shape == (4, 3)
        assert loaded.faces.shape == (4, 3)
        # bounding box matches input
        np.testing.assert_allclose(
            loaded.bounds, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], atol=1e-9,
        )

    def test_units_millimeter_in_3mf(self, tmp_path, tetrahedron):
        """The 3MF unit metadata must be 'millimeter' for PreForm compatibility."""
        V, F = tetrahedron
        out = write_3mf(V, F, tmp_path / "tet.3mf")
        # 3MF files are zip archives; inspect 3D/3dmodel.model.
        import zipfile
        with zipfile.ZipFile(out) as z:
            with z.open("3D/3dmodel.model") as f:
                model_xml = f.read().decode("utf-8")
        assert 'unit="millimeter"' in model_xml, model_xml[:500]

    def test_invalid_shape_raises(self, tmp_path):
        with pytest.raises(ValueError, match="vertices must be"):
            write_3mf(np.zeros((4, 2)), np.zeros((1, 3), dtype=int), tmp_path / "x.3mf")
        with pytest.raises(ValueError, match="faces must be"):
            write_3mf(np.zeros((4, 3)), np.zeros((1, 4), dtype=int), tmp_path / "x.3mf")

    def test_out_of_bounds_face_raises(self, tmp_path):
        with pytest.raises(ValueError, match="out of bounds"):
            write_3mf(np.zeros((3, 3)), np.array([[0, 1, 5]]), tmp_path / "x.3mf")
