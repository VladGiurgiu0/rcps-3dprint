"""Tests for rcps.metrics — RCP diagnostics and Kozeny-Carman permeability.

Reference values for the example packing (data_example/, N = 718, 50 mm
periodic box, stored d = 5.9598 mm) were cross-validated with an
independent MATLAB implementation (GRS_GRC rcp_analysis.m) and a pure-
numpy reference: phi = 0.636661, Nc = 2115 (tol 1e-4), z_backbone =
6.0371, 18 rattlers, Berryman ratio = 1.000000, k_KC(stored) = 2.335e-8 m^2.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rcps.metrics import (
    ISOSTATIC_Z_3D,
    berryman_ratio,
    coordination_number,
    kozeny_carman,
    packing_fraction,
    rcp_metrics,
    rdf,
)


def _sc_lattice(n_side: int = 4, a: float = 1.0):
    """Simple-cubic lattice of touching unit spheres in a periodic box.

    Analytic ground truth: phi = pi/6, every sphere has exactly z = 6
    touching neighbors, median NN distance = d.
    """
    g = np.arange(n_side) * a
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    centers = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(float)
    diameters = np.full(centers.shape[0], a)
    box = [n_side * a] * 3
    return centers, diameters, box


class TestSimpleCubicLattice:
    """Exact analytic checks on a periodic simple-cubic crystal."""

    def test_packing_fraction_is_pi_over_6(self):
        c, d, box = _sc_lattice()
        assert packing_fraction(d, box) == pytest.approx(np.pi / 6.0, rel=1e-12)

    def test_coordination_number_is_exactly_6(self):
        c, d, box = _sc_lattice()
        co = coordination_number(c, d, box)
        assert co["z_mean"] == pytest.approx(6.0)
        assert co["z_no_rattlers"] == pytest.approx(6.0)
        assert co["n_rattlers"] == 0
        assert co["n_contacts"] == 3 * c.shape[0]  # 3 bonds per site (periodic)

    def test_berryman_ratio_is_one(self):
        c, d, box = _sc_lattice()
        assert berryman_ratio(c, d, box) == pytest.approx(1.0, abs=1e-12)

    def test_rdf_has_no_pairs_below_contact(self):
        c, d, box = _sc_lattice()
        out = rdf(c, d, box)
        r = np.array(out["r_over_d"])
        g = np.array(out["g"])
        assert g[r < 0.99].sum() == 0.0
        # nearest-neighbor peak at r = d must be present
        assert g[np.abs(r - 1.0) < 0.02].max() > 1.0

    def test_sc_is_not_rcp(self):
        """phi = 0.524 is far below the RCP window -> verdict must be False."""
        c, d, box = _sc_lattice()
        m = rcp_metrics(c, d, box, include_rdf=False)
        assert m["rcp_checklist"]["isostatic"] is True  # z = 6, but...
        assert m["rcp_checklist"]["phi_in_rcp_window"] is False
        assert m["is_rcp_consistent"] is False


class TestKozenyCarman:
    """k = d^2/(36 kC) * eps^3/(1-eps)^2 — De Paoli et al. (2024), Eq. (3.3)."""

    def test_reference_value(self):
        # eps = 0.36, d = 6 mm, kC = 5: k = 36e-6/180 * 0.36^3/0.64^2 m^2
        k = kozeny_carman(0.36, 6.0)
        assert k == pytest.approx(2.2781e-8, rel=1e-3)

    def test_carman_constant_scaling(self):
        assert kozeny_carman(0.36, 6.0, k_c=10.0) == pytest.approx(
            0.5 * kozeny_carman(0.36, 6.0), rel=1e-12
        )

    def test_diameter_squared_scaling(self):
        assert kozeny_carman(0.36, 2.0) == pytest.approx(
            kozeny_carman(0.36, 1.0) * 4.0, rel=1e-12
        )

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_porosity_raises(self, bad):
        with pytest.raises(ValueError):
            kozeny_carman(bad, 6.0)


class TestExamplePacking:
    """The shipped example packing must be RCP-consistent (validated refs)."""

    @pytest.fixture(scope="class")
    def metrics(self, packing_xyzd_path):
        S = np.fromfile(packing_xyzd_path, dtype="<f8").reshape(-1, 4)
        return rcp_metrics(S[:, :3], S[:, 3], [50.0, 50.0, 50.0], d_nominal_mm=6.0)

    def test_packing_fraction(self, metrics):
        assert metrics["packing_fraction"] == pytest.approx(0.636661, abs=1e-5)

    def test_isostatic_backbone(self, metrics):
        co = metrics["coordination"]
        assert co["n_contacts"] == 2115
        assert co["z_no_rattlers"] == pytest.approx(6.0371, abs=1e-3)
        assert co["n_rattlers"] == 18
        assert abs(co["z_no_rattlers"] - ISOSTATIC_Z_3D) < 0.25

    def test_berryman(self, metrics):
        assert metrics["berryman_median_nn_over_d"] == pytest.approx(1.0, abs=1e-6)

    def test_kozeny_carman_both_diameters(self, metrics):
        kc = metrics["kozeny_carman"]
        assert kc["k_m2_stored_d"] == pytest.approx(2.335e-8, rel=1e-3)
        assert kc["k_m2_nominal_d"] == pytest.approx(2.038e-8, rel=1e-3)
        assert kc["porosity_nominal"] == pytest.approx(0.3504, abs=1e-4)

    def test_verdict_and_serializable(self, metrics):
        assert metrics["is_rcp_consistent"] is True
        json.dumps(metrics)  # must round-trip to packing_metrics.json


class TestPreviewIntegration:
    """packing_preview must expose metrics and persist packing_metrics.json."""

    def test_preview_writes_metrics_json(self, tmp_path, packing_xyzd_path,
                                         packing_nfo_path):
        import shutil

        from rcps_gui import packgen

        shutil.copy(packing_xyzd_path, tmp_path / "packing.xyzd")
        shutil.copy(packing_nfo_path, tmp_path / "packing.nfo")
        data = packgen.packing_preview(tmp_path)
        m = data["metrics"]
        assert m is not None
        assert m["is_rcp_consistent"] is True
        mpath = tmp_path / "packing_metrics.json"
        assert mpath.exists()
        on_disk = json.loads(mpath.read_text())
        assert on_disk["packing_fraction"] == pytest.approx(0.636661, abs=1e-5)
        assert "references" in on_disk  # citations travel with the data
        # second call must re-read, not recompute/overwrite
        data2 = packgen.packing_preview(tmp_path)
        assert data2["metrics"]["packing_fraction"] == m["packing_fraction"]
