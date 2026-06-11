"""Tests for rcps.field (Task 4)."""

from __future__ import annotations

import numpy as np
import pytest

from rcps.field import (
    apply_keepsides_filter,
    build_beads_sdf_icsg,
    build_box_sdf,
    compose_field,
    cull_spheres,
    make_grid,
    replicate_with_ghost_tiles,
    snap_grid,
)

# =====================================================================
# replicate_with_ghost_tiles
# =====================================================================

class TestReplicateWithGhostTiles:

    def test_zero_ghosts_is_noop(self):
        c = np.array([[1.0, 2.0, 3.0]])
        d = np.array([6.0])
        c2, d2 = replicate_with_ghost_tiles(c, d, (50, 50, 50), ghost_tiles=0)
        assert c2 is c and d2 is d  # no copy, no work

    def test_g1_yields_27_copies(self):
        rng = np.random.default_rng(0)
        c = rng.uniform(0, 50, size=(10, 3))
        d = np.full(10, 6.0)
        c2, d2 = replicate_with_ghost_tiles(c, d, (50, 50, 50), ghost_tiles=1)
        assert c2.shape == (270, 3)
        assert d2.shape == (270,)

    def test_diameters_preserved(self):
        c = np.zeros((3, 3))
        d = np.array([5.0, 6.0, 7.0])
        c2, d2 = replicate_with_ghost_tiles(c, d, (10, 10, 10), ghost_tiles=1)
        # Each original diameter appears 27 times.
        for di in d:
            assert int(np.count_nonzero(d2 == di)) == 27

    def test_shifts_match_tile_size(self):
        c = np.array([[1.0, 2.0, 3.0]])
        d = np.array([6.0])
        L, H, W = 50.0, 60.0, 70.0
        c2, _ = replicate_with_ghost_tiles(c, d, (L, H, W), ghost_tiles=1)
        # The (-1,-1,-1) copy is at (1-L, 2-H, 3-W); the (+1,+1,+1) at (1+L, 2+H, 3+W).
        assert np.any(
            np.isclose(c2, [1 - L, 2 - H, 3 - W], atol=1e-12).all(axis=1)
        )
        assert np.any(
            np.isclose(c2, [1 + L, 2 + H, 3 + W], atol=1e-12).all(axis=1)
        )
        # Original is present once.
        assert int(np.count_nonzero(np.isclose(c2, [1, 2, 3], atol=1e-12).all(axis=1))) == 1

    def test_negative_ghosts_raises(self):
        with pytest.raises(ValueError):
            replicate_with_ghost_tiles(np.zeros((1, 3)), np.array([6.0]), (50, 50, 50), ghost_tiles=-1)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            replicate_with_ghost_tiles(np.zeros((3, 3)), np.array([6.0]), (50, 50, 50))


# =====================================================================
# snap_grid
# =====================================================================

class TestSnapGrid:

    def test_exact_divisor_no_change(self):
        nx, ny, nz, vox = snap_grid((50, 50, 50), 0.1)
        assert (nx, ny, nz) == (500, 500, 500)
        assert vox == pytest.approx(0.1, abs=1e-12)

    def test_snaps_when_not_divisible(self):
        # 50 / 0.07 ≈ 714.28 → round to 714 → vox = 50/714 = 0.0700280…
        nx, _, _, vox = snap_grid((50, 50, 50), 0.07)
        assert nx == round(50 / 0.07)
        assert 50 / vox == pytest.approx(nx, abs=1e-9)

    def test_minimum_nx_is_8(self):
        # Very coarse target — minimum nx clamp kicks in.
        nx, _, _, vox = snap_grid((1.0, 1.0, 1.0), 0.5)
        assert nx >= 8
        assert vox <= 0.5

    def test_non_divisible_HW_raises(self):
        # L=50 divides exactly at vox=0.1 (nx=500), but H=33 / 0.1 = 330 (clean)
        # so cook a value that's *almost* but not exactly divisible.
        with pytest.raises(ValueError, match="not divisible"):
            snap_grid((50, 33.333333333, 50), 0.1)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            snap_grid((-1, 50, 50), 0.1)
        with pytest.raises(ValueError):
            snap_grid((50, 50, 50), 0.0)


# =====================================================================
# make_grid
# =====================================================================

class TestMakeGrid:

    def test_no_keepsides_basic_pad(self):
        g = make_grid((10, 10, 10), 1.0, pad_vox=1, band_vox=3, keep_sides=())
        assert g.nx == 12 and g.ny == 12 and g.nz == 12
        assert g.vox_size == pytest.approx(1.0, abs=1e-12)
        assert g.origin == (-1.0, -1.0, -1.0)
        assert g.tile_size == (10.0, 10.0, 10.0)
        assert g.pad_vox == 1
        assert g.keep_sides == ()

    def test_keep_sides_extra_padding(self):
        # With max_radius_mm=3, vox=1, band_vox=3 → band_dist=3, extra pad
        # = ceil((3+3)/1)+1 = 7. Total pad = 1 + 7 = 8 → nx = 10 + 16 = 26.
        g = make_grid(
            (10, 10, 10), 1.0,
            pad_vox=1, band_vox=3,
            keep_sides=["S", "R"],
            max_radius_mm=3.0,
        )
        assert g.pad_vox == 8
        assert g.nx == 26 and g.ny == 26 and g.nz == 26
        assert g.keep_sides == ("S", "R")

    def test_keep_sides_requires_max_radius(self):
        with pytest.raises(ValueError, match="max_radius_mm"):
            make_grid((10, 10, 10), 1.0, keep_sides=["S"], max_radius_mm=0)

    def test_invalid_keep_sides_label_raises(self):
        with pytest.raises(ValueError, match="invalid keep_sides"):
            make_grid((10, 10, 10), 1.0, keep_sides=["X"])

    def test_keep_sides_normalised_uppercase_unique(self):
        g = make_grid(
            (10, 10, 10), 1.0,
            pad_vox=1, keep_sides=["s", "S", "r"], max_radius_mm=3.0,
        )
        assert g.keep_sides == ("S", "R")

    def test_oversize_grid_raises(self):
        # Force a huge grid: 1000 mm at vox=0.01 → 100k per side, 1e15 voxels.
        with pytest.raises(ValueError, match="too large"):
            make_grid((1000, 1000, 1000), 0.01)

    def test_grid_helpers(self):
        g = make_grid((10, 10, 10), 1.0, pad_vox=2)
        assert g.shape == (14, 14, 14)
        assert g.n_voxels == 14 ** 3
        assert g.memory_float32_bytes == 4 * 14 ** 3
        x = g.x_vec()
        assert x.shape == (14,)
        assert x[0] == pytest.approx(-2.0)
        assert x[-1] == pytest.approx(11.0)


# =====================================================================
# build_box_sdf
# =====================================================================

class TestBuildBoxSdf:

    def test_strictly_negative_inside_box(self):
        g = make_grid((10, 10, 10), 1.0, pad_vox=2)
        F_box = build_box_sdf(g, max_radius_mm=0.0)
        # Center of the tile (x=y=z=5) is at voxel index (5+2, 5+2, 5+2) = (7,7,7).
        # Distance to nearest face = 5 mm; F_box(centre) = -5.
        assert F_box[7, 7, 7] == pytest.approx(-5.0, abs=1e-6)

    def test_strictly_positive_outside_box(self):
        g = make_grid((10, 10, 10), 1.0, pad_vox=2)
        F_box = build_box_sdf(g, max_radius_mm=0.0)
        # Voxel (0,0,0) is at (-2,-2,-2), corner-distance = sqrt(12).
        assert F_box[0, 0, 0] == pytest.approx(np.sqrt(12), abs=1e-5)

    def test_keep_side_S_extends_upward(self):
        # When 'S' is kept, the +Z face is extended outward by ext_mm.
        # Voxels above z=W=10 should no longer be "outside" the (extended) box.
        g = make_grid(
            (10, 10, 10), 1.0,
            pad_vox=1, band_vox=3,
            keep_sides=["S"], max_radius_mm=3.0,
        )
        F_box = build_box_sdf(g, max_radius_mm=3.0)
        # Voxel near z=W in original tile coords: pick the column at x=y=5 mm.
        # The +Z face is at z = W + ext_mm. Any voxel with z < W + ext_mm and inside [0,L]^2 in xy
        # should have F_box < 0.
        L, H, W = g.tile_size
        vox = g.vox_size
        band_dist = 3 * vox
        ext_mm = 2 * (3.0 + band_dist) + 2 * vox
        # Voxel at (5, 5, W + ext_mm/2) is inside the extended box.
        ix = int(round((5.0 - g.origin[0]) / vox))
        iy = int(round((5.0 - g.origin[1]) / vox))
        z_target = W + ext_mm / 2
        iz = int(round((z_target - g.origin[2]) / vox))
        assert F_box[ix, iy, iz] < 0

    def test_F_box_dtype_is_float32(self):
        g = make_grid((10, 10, 10), 1.0)
        F_box = build_box_sdf(g, max_radius_mm=0.0)
        assert F_box.dtype == np.float32


# =====================================================================
# cull_spheres
# =====================================================================

class TestCullSpheres:

    def test_keeps_central_sphere_drops_distant(self):
        g = make_grid((10, 10, 10), 1.0, pad_vox=1)
        # Centre sphere inside; far sphere clearly outside.
        c = np.array([[5.0, 5.0, 5.0], [100.0, 100.0, 100.0]])
        r = np.array([3.0, 3.0])
        cK, rK, keep = cull_spheres(c, r, g, band_dist=0.5)
        assert cK.shape == (1, 3) and rK.shape == (1,)
        assert keep.tolist() == [True, False]

    def test_keep_boundary_sphere(self):
        g = make_grid((10, 10, 10), 1.0, pad_vox=1)
        # Sphere whose bounding box just touches the grid corner.
        c = np.array([[-2.0, -2.0, -2.0]])
        r = np.array([3.0])
        _, _, keep = cull_spheres(c, r, g, band_dist=0.0)
        # bbox extends from -5 to 1; grid x_vec spans [-1, 10]. Overlaps.
        assert keep[0]


# =====================================================================
# apply_keepsides_filter
# =====================================================================

class TestKeepsidesFilter:

    def test_noop_when_empty(self):
        c = np.array([[5.0, 5.0, 5.0], [60.0, 5.0, 5.0]])
        r = np.array([3.0, 3.0])
        g = make_grid((50, 50, 50), 1.0)
        cK, rK, keep = apply_keepsides_filter(c, r, g)
        assert keep.tolist() == [True, True]

    def test_R_kept_drops_right_ghosts(self):
        # R kept → drop spheres with x > L.
        L = 50.0
        g = make_grid((L, L, L), 1.0, pad_vox=1, keep_sides=["R"], max_radius_mm=3.0)
        c = np.array([[5.0, 5.0, 5.0], [L + 5.0, 5.0, 5.0], [L - 1.0, 5.0, 5.0]])
        r = np.array([3.0, 3.0, 3.0])
        cK, _, keep = apply_keepsides_filter(c, r, g)
        assert keep.tolist() == [True, False, True]

    def test_L_kept_drops_left_ghosts(self):
        L = 50.0
        g = make_grid((L, L, L), 1.0, pad_vox=1, keep_sides=["L"], max_radius_mm=3.0)
        c = np.array([[5.0, 5.0, 5.0], [-5.0, 5.0, 5.0]])
        r = np.array([3.0, 3.0])
        _, _, keep = apply_keepsides_filter(c, r, g)
        assert keep.tolist() == [True, False]

    def test_L_and_R_kept_keeps_only_in_tile(self):
        L = 50.0
        g = make_grid(
            (L, L, L), 1.0,
            pad_vox=1, keep_sides=["L", "R"], max_radius_mm=3.0,
        )
        c = np.array([
            [-5.0, 5.0, 5.0],   # left ghost  → drop
            [5.0, 5.0, 5.0],    # in tile     → keep
            [L + 5, 5.0, 5.0],  # right ghost → drop
        ])
        r = np.array([3.0, 3.0, 3.0])
        _, _, keep = apply_keepsides_filter(c, r, g)
        assert keep.tolist() == [False, True, False]


# =====================================================================
# build_beads_sdf_icsg
# =====================================================================

class TestBuildBeadsSdfIcsg:

    def test_single_sphere_centred_in_grid(self):
        """At a sphere centre the SDF equals -R; at distance R it's ~0."""
        g = make_grid((10, 10, 10), 0.5, pad_vox=1)
        c = np.array([[5.0, 5.0, 5.0]])
        r = np.array([2.0])
        F = build_beads_sdf_icsg(c, r, g, band_vox=4)
        # Index of centre.
        i = int(round((5.0 - g.origin[0]) / g.vox_size))
        j = int(round((5.0 - g.origin[1]) / g.vox_size))
        k = int(round((5.0 - g.origin[2]) / g.vox_size))
        assert F[i, j, k] == pytest.approx(-2.0, abs=g.vox_size)
        # A voxel at distance ~R along +x should give SDF ≈ 0.
        i2 = int(round((5.0 + 2.0 - g.origin[0]) / g.vox_size))
        assert abs(float(F[i2, j, k])) <= g.vox_size

    def test_far_voxel_stays_at_band_dist(self):
        """Voxels untouched by any sphere keep the initial +band_dist value."""
        g = make_grid((10, 10, 10), 0.5, pad_vox=1)
        c = np.array([[1.0, 1.0, 1.0]])
        r = np.array([0.5])  # tiny sphere in a corner
        F = build_beads_sdf_icsg(c, r, g, band_vox=2)
        band_dist = 2 * g.vox_size
        # Far corner — should remain at +band_dist.
        far = F[-1, -1, -1]
        assert far == pytest.approx(band_dist, abs=1e-6)

    def test_porosity_from_F_matches_analytical_for_single_sphere(self):
        """Counting F<0 voxels approximates the sphere volume."""
        g = make_grid((10, 10, 10), 0.25, pad_vox=1)
        c = np.array([[5.0, 5.0, 5.0]])
        R = 2.0
        r = np.array([R])
        F = build_beads_sdf_icsg(c, r, g, band_vox=4)
        F_box = build_box_sdf(g, max_radius_mm=R)
        F_final = compose_field(F, F_box, "beads")
        n_solid = int((F_final < 0).sum())
        V_solid = n_solid * g.vox_size ** 3
        V_analytical = (4.0 / 3.0) * np.pi * R ** 3
        assert V_solid == pytest.approx(V_analytical, rel=0.05)  # 5% at vox=0.25

    def test_empty_inputs_return_band_dist_field(self):
        g = make_grid((10, 10, 10), 1.0)
        c = np.zeros((0, 3))
        r = np.zeros((0,))
        F = build_beads_sdf_icsg(c, r, g, band_vox=3)
        band_dist = 3 * g.vox_size
        assert np.all(F == np.float32(band_dist))

    def test_F_dtype_is_float32(self):
        g = make_grid((10, 10, 10), 1.0)
        c = np.array([[5.0, 5.0, 5.0]])
        r = np.array([2.0])
        F = build_beads_sdf_icsg(c, r, g, band_vox=3)
        assert F.dtype == np.float32


# =====================================================================
# compose_field
# =====================================================================

class TestComposeField:

    def test_beads_inside_box(self):
        F_beads = np.array([[-1.0]], dtype=np.float32)  # inside bead
        F_box = np.array([[-2.0]], dtype=np.float32)    # inside box
        F = compose_field(F_beads.reshape(1, 1, 1), F_box.reshape(1, 1, 1), "beads")
        assert float(F.ravel()[0]) == pytest.approx(-1.0)

    def test_beads_outside_box(self):
        F_beads = np.array([[-1.0]], dtype=np.float32)   # inside bead
        F_box = np.array([[+3.0]], dtype=np.float32)     # outside box
        F = compose_field(F_beads.reshape(1, 1, 1), F_box.reshape(1, 1, 1), "beads")
        assert float(F.ravel()[0]) == pytest.approx(+3.0)  # box wins

    def test_pore_inside_box_outside_bead(self):
        F_beads = np.array([[+2.0]], dtype=np.float32)   # outside bead
        F_box = np.array([[-3.0]], dtype=np.float32)     # inside box
        F = compose_field(F_beads.reshape(1, 1, 1), F_box.reshape(1, 1, 1), "pore")
        # Pore = box ∩ ¬beads → max(F_box, -F_beads) = max(-3, -2) = -2.
        assert float(F.ravel()[0]) == pytest.approx(-2.0)

    def test_pore_inside_box_inside_bead_is_positive(self):
        F_beads = np.array([[-1.5]], dtype=np.float32)   # inside bead
        F_box = np.array([[-3.0]], dtype=np.float32)     # inside box
        F = compose_field(F_beads.reshape(1, 1, 1), F_box.reshape(1, 1, 1), "pore")
        # Pore = max(-3, -(-1.5)) = max(-3, 1.5) = 1.5 → outside pore (inside a bead).
        assert float(F.ravel()[0]) == pytest.approx(1.5)

    def test_invalid_export_what(self):
        F = np.zeros((1, 1, 1), dtype=np.float32)
        with pytest.raises(ValueError):
            compose_field(F, F, "solid")  # type: ignore[arg-type]

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            compose_field(
                np.zeros((1, 1, 1), dtype=np.float32),
                np.zeros((2, 1, 1), dtype=np.float32),
                "beads",
            )


# =====================================================================
# End-to-end: tile-only smoke against data_example
# =====================================================================

class TestEndToEndOnDataExample:
    """Coarse tile-only smoke test against the canonical packing.

    Uses ``vox=2 mm`` to keep memory/time tiny (~5 ms). The validation suite
    in Task 8 re-runs at ``vox=0.1 mm`` against the MATLAB reference.
    """

    def test_porosity_from_full_pipeline_in_rcp_range(self, packing_xyzd_path):
        from rcps.io import load_packing_xyzd

        S = load_packing_xyzd(packing_xyzd_path)
        centers = S[:, :3]
        diameters = S[:, 3]

        # Ghost-tile replication (g=1) then we cull.
        c, d = replicate_with_ghost_tiles(centers, diameters, (50, 50, 50), 1)
        r = d / 2.0

        g = make_grid((50, 50, 50), 2.0, pad_vox=1, band_vox=3, keep_sides=())
        band_dist = 3 * g.vox_size

        c, r, _ = cull_spheres(c, r, g, band_dist=band_dist)
        F_beads = build_beads_sdf_icsg(c, r, g, band_vox=3)
        F_box = build_box_sdf(g, max_radius_mm=float(r.max()))
        F = compose_field(F_beads, F_box, "beads")

        # Voxel-count porosity inside the (un-extended) tile box.
        # F_box < 0 → inside tile box.  F < 0 → inside beads.
        in_box = F_box < 0
        in_beads_and_box = F < 0
        n_box = int(in_box.sum())
        n_solid = int(in_beads_and_box.sum())
        phi_grid = 1.0 - n_solid / max(1, n_box)
        # vox=2 is coarse; tolerate ±0.04 vs the actual ~0.363.
        assert 0.30 < phi_grid < 0.42, f"phi_grid={phi_grid:.4f}"

    def test_pore_vs_beads_have_inverted_solid_volume(self, packing_xyzd_path):
        """Pore and beads modes should partition the box, modulo surface voxels."""
        from rcps.io import load_packing_xyzd

        S = load_packing_xyzd(packing_xyzd_path)
        centers, diameters = S[:, :3], S[:, 3]
        c, d = replicate_with_ghost_tiles(centers, diameters, (50, 50, 50), 1)
        r = d / 2.0

        g = make_grid((50, 50, 50), 2.0, pad_vox=1, keep_sides=())
        band_dist = 3 * g.vox_size
        c, r, _ = cull_spheres(c, r, g, band_dist=band_dist)
        F_beads_raw = build_beads_sdf_icsg(c, r, g, band_vox=3)
        F_box = build_box_sdf(g, max_radius_mm=float(r.max()))

        F_solid = compose_field(F_beads_raw, F_box, "beads")
        F_pore = compose_field(F_beads_raw, F_box, "pore")

        n_solid = int((F_solid < 0).sum())
        n_pore = int((F_pore < 0).sum())
        n_box = int((F_box < 0).sum())
        # Solid + pore ≈ box (small surface-voxel overlap allowed).
        assert abs(n_solid + n_pore - n_box) < 0.02 * n_box, (n_solid, n_pore, n_box)
