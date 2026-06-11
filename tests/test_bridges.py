"""Tests for rcps.bridges (Task 5)."""

from __future__ import annotations

import numpy as np
import pytest

from rcps.bridges import (
    _brute_force_query_pairs,
    add_cylinders,
    expand_to_touch,
    find_contact_pairs,
)
from rcps.field import (
    build_beads_sdf_icsg,
    build_box_sdf,
    compose_field,
    make_grid,
)

# =====================================================================
# _brute_force_query_pairs (the NumPy fallback used when scipy missing)
# =====================================================================

class TestBruteForceQueryPairs:

    def test_three_collinear_close_to_each_other(self):
        c = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        pairs = _brute_force_query_pairs(c, max_d=1.5)
        s = {(int(a), int(b)) for a, b in pairs.tolist()}
        # (0,1) and (1,2) within 1.5; (0,2) at distance 2 → excluded.
        assert s == {(0, 1), (1, 2)}

    def test_no_pairs_when_max_d_zero(self):
        c = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        pairs = _brute_force_query_pairs(c, max_d=0.0)
        assert pairs.shape == (0, 2)

    def test_singleton_returns_empty(self):
        pairs = _brute_force_query_pairs(np.array([[1.0, 2.0, 3.0]]), max_d=10.0)
        assert pairs.shape == (0, 2)

    def test_only_upper_triangle(self):
        rng = np.random.default_rng(0)
        c = rng.uniform(-1, 1, size=(20, 3))
        pairs = _brute_force_query_pairs(c, max_d=1.0)
        # i < j strictly.
        assert (pairs[:, 0] < pairs[:, 1]).all()
        # No duplicates.
        assert len({tuple(p) for p in pairs.tolist()}) == pairs.shape[0]


# =====================================================================
# find_contact_pairs
# =====================================================================

class TestFindContactPairs:

    def test_two_close_spheres_are_a_contact(self):
        c = np.array([[0.0, 0.0, 0.0], [5.5, 0.0, 0.0]])  # gap = 5.5 - 3 - 3 = -0.5
        r = np.array([3.0, 3.0])
        pairs = find_contact_pairs(c, r, contact_tol_mm=0.2)
        assert pairs.shape == (1, 2)
        assert pairs[0].tolist() == [0, 1]

    def test_two_distant_spheres_are_not_a_contact(self):
        c = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])  # gap = 4
        r = np.array([3.0, 3.0])
        pairs = find_contact_pairs(c, r, contact_tol_mm=0.2)
        assert pairs.shape == (0, 2)

    def test_gap_exactly_at_tol_is_excluded(self):
        # contact criterion uses strict <, so gap == tol should be excluded.
        c = np.array([[0.0, 0.0, 0.0], [6.2, 0.0, 0.0]])  # gap = 0.2
        r = np.array([3.0, 3.0])
        pairs = find_contact_pairs(c, r, contact_tol_mm=0.2)
        assert pairs.shape == (0, 2)

    def test_gap_just_below_tol_is_a_contact(self):
        c = np.array([[0.0, 0.0, 0.0], [6.19, 0.0, 0.0]])  # gap = 0.19
        r = np.array([3.0, 3.0])
        pairs = find_contact_pairs(c, r, contact_tol_mm=0.2)
        assert pairs.shape == (1, 2)

    def test_singleton_returns_empty(self):
        pairs = find_contact_pairs(
            np.array([[0.0, 0.0, 0.0]]), np.array([3.0]), contact_tol_mm=1.0
        )
        assert pairs.shape == (0, 2)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            find_contact_pairs(np.zeros((3, 3)), np.array([1.0]), contact_tol_mm=1.0)

    def test_three_sphere_chain(self):
        # Three collinear spheres, each pair just-touching (gap = 0.0 → contact since 0 < 0.5).
        c = np.array([[0, 0, 0], [6, 0, 0], [12, 0, 0]], dtype=float)
        r = np.array([3.0, 3.0, 3.0])
        pairs = find_contact_pairs(c, r, contact_tol_mm=0.5)
        s = {tuple(p) for p in pairs.tolist()}
        # (0,1) and (1,2) are contacts; (0,2) has gap = 12-6 = 6 → not a contact.
        assert s == {(0, 1), (1, 2)}


# =====================================================================
# expand_to_touch  (the new `diameter` bridge mode)
# =====================================================================

class TestExpandToTouch:

    def test_no_contacts_returns_unchanged(self):
        c = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        r = np.array([3.0, 3.0])
        new_r = expand_to_touch(c, r, contact_tol_mm=0.5)
        np.testing.assert_allclose(new_r, r)
        # Must be a copy (caller shouldn't worry about aliasing).
        assert new_r is not r

    def test_two_spheres_with_positive_gap_both_grow_by_half_gap(self):
        c = np.array([[0.0, 0.0, 0.0], [6.4, 0.0, 0.0]])  # gap = 0.4
        r = np.array([3.0, 3.0])
        new_r = expand_to_touch(c, r, contact_tol_mm=0.5)
        np.testing.assert_allclose(new_r, [3.2, 3.2], atol=1e-12)
        # After expansion, the pair *touches* (new sum of radii equals distance).
        assert np.isclose(new_r[0] + new_r[1], 6.4)

    def test_already_touching_pair_does_not_shrink(self):
        c = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])  # gap = 0.0
        r = np.array([3.0, 3.0])
        new_r = expand_to_touch(c, r, contact_tol_mm=0.5)
        np.testing.assert_allclose(new_r, r)

    def test_overlapping_pair_does_not_shrink(self):
        c = np.array([[0.0, 0.0, 0.0], [5.5, 0.0, 0.0]])  # gap = -0.5
        r = np.array([3.0, 3.0])
        new_r = expand_to_touch(c, r, contact_tol_mm=0.5)
        np.testing.assert_allclose(new_r, r)

    def test_three_sphere_chain_uniform_growth(self):
        # Three collinear spheres with gap 0.4 between each pair.
        c = np.array([[0.0, 0, 0], [6.4, 0, 0], [12.8, 0, 0]])
        r = np.array([3.0, 3.0, 3.0])
        new_r = expand_to_touch(c, r, contact_tol_mm=0.5)
        # All three see exactly one neighbour with gap=0.4 → each grows by 0.2.
        # (Middle sphere sees two neighbours, both with gap=0.4 → max = 0.4 → grow by 0.2.)
        np.testing.assert_allclose(new_r, [3.2, 3.2, 3.2], atol=1e-12)

    def test_pair_symmetric_when_only_one_pair(self):
        """For a single isolated pair, both spheres grow by exactly the same amount."""
        c = np.array([[0.0, 0, 0], [7.5, 0, 0]])
        r = np.array([3.0, 4.0])
        # gap = 7.5 - 7 = 0.5; we need contact_tol > 0.5 to count it as a contact.
        new_r = expand_to_touch(c, r, contact_tol_mm=0.6)
        delta = new_r - r
        # Pair-symmetric closure: both grow by gap/2 = 0.25.
        np.testing.assert_allclose(delta, [0.25, 0.25], atol=1e-12)

    def test_sphere_with_two_unequal_neighbours_grows_by_max(self):
        """Middle sphere has neighbours at gap 0.1 and 0.5 → it grows by 0.25 (max(g/2))."""
        c = np.array([
            [0.0, 0.0, 0.0],    # left  neighbour
            [6.5, 0.0, 0.0],    # middle: gap with left = 0.5
            [12.6, 0.0, 0.0],   # right neighbour: gap with middle = 0.1
        ])
        r = np.array([3.0, 3.0, 3.0])
        new_r = expand_to_touch(c, r, contact_tol_mm=0.6)
        # Middle's worst gap is 0.5 → +0.25.
        # Left's only contact (with middle) has gap 0.5 → +0.25.
        # Right's only contact (with middle) has gap 0.1 → +0.05.
        np.testing.assert_allclose(new_r, [3.25, 3.25, 3.05], atol=1e-12)

    def test_no_spheres_returns_empty(self):
        new_r = expand_to_touch(np.zeros((0, 3)), np.zeros((0,)), contact_tol_mm=1.0)
        assert new_r.shape == (0,)

    def test_radii_never_decrease(self):
        """No matter the configuration, output radii must be ≥ input radii pointwise."""
        rng = np.random.default_rng(42)
        c = rng.uniform(0, 10, size=(20, 3))
        r = rng.uniform(0.5, 1.5, size=20)
        new_r = expand_to_touch(c, r, contact_tol_mm=0.3)
        assert (new_r >= r - 1e-12).all()


# =====================================================================
# add_cylinders
# =====================================================================

class TestAddCylinders:

    def _two_sphere_scene(self, gap_mm: float = 0.2):
        """Two spheres of r=2 separated by `gap_mm`. Centred in a 10 mm tile."""
        d = 2 * 2.0 + gap_mm  # centre-to-centre
        c1 = np.array([5.0 - d / 2, 5.0, 5.0])
        c2 = np.array([5.0 + d / 2, 5.0, 5.0])
        return np.stack([c1, c2]), np.array([2.0, 2.0])

    def test_no_contacts_leaves_F_beads_unchanged(self):
        c, r = self._two_sphere_scene(gap_mm=2.0)  # gap > tol → no contact
        g = make_grid((10, 10, 10), 0.5, pad_vox=1)
        F0 = build_beads_sdf_icsg(c, r, g, band_vox=3)
        F_after = add_cylinders(
            F0.copy(), c, r, g,
            contact_tol_mm=0.3, radius_frac=0.5, band_vox=3,
        )
        np.testing.assert_array_equal(F_after, F0)

    def test_cylinder_axis_voxel_becomes_negative(self):
        c, r = self._two_sphere_scene(gap_mm=0.2)
        g = make_grid((10, 10, 10), 0.5, pad_vox=1)
        F = build_beads_sdf_icsg(c, r, g, band_vox=3)
        # Voxel on the inter-centre axis midway between the two spheres,
        # at the cylinder radius midpoint (i.e., right at the centre).
        mid = (c[0] + c[1]) / 2
        ix = int(round((mid[0] - g.origin[0]) / g.vox_size))
        iy = int(round((mid[1] - g.origin[1]) / g.vox_size))
        iz = int(round((mid[2] - g.origin[2]) / g.vox_size))
        # Before bridges: midpoint is in the gap, F > 0.
        assert F[ix, iy, iz] > 0
        # After cylinders with radius_frac=0.5 → r_cyl = 1.0:
        F = add_cylinders(F, c, r, g, contact_tol_mm=0.3, radius_frac=0.5, band_vox=3)
        # Midpoint of segment, on the axis → distance = 0; SDF = 0 - 1.0 - 0.25*vox < 0.
        assert F[ix, iy, iz] < 0, F[ix, iy, iz]

    def test_added_solid_volume_matches_cylinder(self):
        """Voxel-count change ≈ cylinder volume between the two sphere surfaces."""
        c, r = self._two_sphere_scene(gap_mm=0.2)
        g = make_grid((10, 10, 10), 0.25, pad_vox=1)
        F = build_beads_sdf_icsg(c, r, g, band_vox=4)
        F_box = build_box_sdf(g, max_radius_mm=float(r.max()))
        F0 = compose_field(F, F_box, "beads")
        n0 = int((F0 < 0).sum())

        # Mutate F in place with cylinders.
        radius_frac = 0.5
        Fc = add_cylinders(F.copy(), c, r, g, contact_tol_mm=0.3, radius_frac=radius_frac, band_vox=4)
        F_after = compose_field(Fc, F_box, "beads")
        n_after = int((F_after < 0).sum())

        # Expected added volume: a small region between two spheres of r=2,
        # gap=0.2, cylinder radius 1.0. We don't compute the exact 3D
        # intersection analytically — just assert the added solid is positive
        # and within an order-of-magnitude of the bare cylinder volume of
        # length=gap, radius=r_cyl.  V_bare = π · r_cyl² · gap ≈ 0.63 mm³ →
        # ~40 voxels at vox=0.25. Allow a generous range to absorb the
        # `-0.25·vox` offset and the sphere intersection.
        added_voxels = n_after - n0
        added_volume = added_voxels * g.vox_size ** 3
        assert added_voxels > 20, f"added voxels too small: {added_voxels}"
        assert added_volume < 5.0, f"added volume too large: {added_volume}"

    def test_F_beads_dtype_check(self):
        c, r = self._two_sphere_scene(gap_mm=0.2)
        g = make_grid((10, 10, 10), 0.5, pad_vox=1)
        F64 = np.zeros(g.shape, dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            add_cylinders(F64, c, r, g, contact_tol_mm=0.3, radius_frac=0.5)

    def test_F_beads_shape_check(self):
        c, r = self._two_sphere_scene(gap_mm=0.2)
        g = make_grid((10, 10, 10), 0.5, pad_vox=1)
        F_wrong = np.zeros((1, 1, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match grid"):
            add_cylinders(F_wrong, c, r, g, contact_tol_mm=0.3, radius_frac=0.5)

    def test_invalid_radius_frac_raises(self):
        c, r = self._two_sphere_scene(gap_mm=0.2)
        g = make_grid((10, 10, 10), 0.5, pad_vox=1)
        F = np.zeros(g.shape, dtype=np.float32)
        with pytest.raises(ValueError, match="radius_frac"):
            add_cylinders(F, c, r, g, contact_tol_mm=0.3, radius_frac=0.0)
        with pytest.raises(ValueError, match="radius_frac"):
            add_cylinders(F, c, r, g, contact_tol_mm=0.3, radius_frac=1.5)


# =====================================================================
# End-to-end smoke against data_example
# =====================================================================

class TestEndToEndOnDataExample:
    """At vox=2 mm, with bridges turned on, verify behaviour is consistent.

    The actual data has spheres with very small gaps (most pairs already
    touching), so the diameter mode produces tiny changes. The cylinders
    mode adds visible bridge volume.
    """

    def _setup(self, packing_xyzd_path, vox=2.0, keep_sides=()):
        from rcps.field import cull_spheres, replicate_with_ghost_tiles
        from rcps.io import load_packing_xyzd

        S = load_packing_xyzd(packing_xyzd_path)
        c0, d0 = S[:, :3], S[:, 3]
        c, d = replicate_with_ghost_tiles(c0, d0, (50, 50, 50), 1)
        r = d / 2.0
        max_r = float(r.max())
        g = make_grid((50, 50, 50), vox, pad_vox=1, band_vox=3,
                      keep_sides=keep_sides, max_radius_mm=max_r if keep_sides else 0.0)
        band_dist = 3 * g.vox_size
        c, r, _ = cull_spheres(c, r, g, band_dist=band_dist)
        return c, r, g

    def test_data_example_has_many_contact_pairs(self, packing_xyzd_path):
        c, r, _ = self._setup(packing_xyzd_path)
        pairs = find_contact_pairs(c, r, contact_tol_mm=0.2)
        assert pairs.shape[0] > 500, f"only {pairs.shape[0]} pairs"
        print(f"    [info] data_example contact pairs (tol=0.2): {pairs.shape[0]}")

    def test_cylinders_add_solid_volume(self, packing_xyzd_path):
        c, r, g = self._setup(packing_xyzd_path)
        F_beads = build_beads_sdf_icsg(c, r, g, band_vox=3)
        F_box = build_box_sdf(g, max_radius_mm=float(r.max()))
        F_no_bridges = compose_field(F_beads.copy(), F_box, "beads")
        n0 = int((F_no_bridges < 0).sum())

        F_with = add_cylinders(F_beads, c, r, g, contact_tol_mm=0.2, radius_frac=0.15, band_vox=3)
        F_final = compose_field(F_with, F_box, "beads")
        n1 = int((F_final < 0).sum())
        # Bridges must add solid (cylinders fill gaps between spheres).
        # The data_example actually has mostly-touching spheres, so the
        # bridges contribution is modest; assert positive monotonicity only.
        assert n1 >= n0, f"cylinders removed solid: before={n0}, after={n1}"
        print(f"    [info] solid voxels: bare={n0}, with cylinders={n1} (Δ={n1-n0})")

    def test_diameter_mode_grows_only_open_contacts(self, packing_xyzd_path):
        c, r, _ = self._setup(packing_xyzd_path)
        # Use the ORIGINAL stored radii (no post-cull adjustment).
        r_new = expand_to_touch(c, r, contact_tol_mm=0.5)
        # All radii must be non-decreasing.
        assert (r_new >= r - 1e-12).all()
        # Expansion is small because most pairs already touch (most gaps ≤ 0).
        max_delta = float((r_new - r).max())
        print(f"    [info] diameter mode: max ∆r = {max_delta:.6f} mm")
        assert max_delta < 0.5, f"unexpectedly large expansion: {max_delta} mm"
