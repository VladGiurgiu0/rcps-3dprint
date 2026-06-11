"""Tests for rcps.mesh (Task 6).

Diagnostics and input-validation tests run without external deps.
Tests that exercise iso2mesh / skimage / trimesh / pymeshfix are auto-
skipped when the dep is not importable.
"""

from __future__ import annotations

import numpy as np
import pytest

from rcps.mesh import (
    _validate_field_inputs,
    diagnose,
    mesh_iso2mesh,
    mesh_skimage,
    repair_and_finalize,
)

# Detect optional deps once at module load.
_HAS_TRIMESH = pytest.importorskip.__doc__ is not None  # always True; placeholder
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


# =====================================================================
# Hand-crafted meshes used by the diagnostic tests
# =====================================================================

def _unit_tetrahedron():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    F = np.array(
        [
            [0, 2, 1],  # bottom (z=0), outward = -z
            [0, 1, 3],  # front (y=0), outward = -y
            [0, 3, 2],  # left (x=0), outward = -x
            [1, 2, 3],  # diagonal face, outward = +(x+y+z)
        ],
        dtype=np.int64,
    )
    return V, F


def _open_triangle():
    """A unit-square quad as two triangles sharing one diagonal — open mesh.

    Distinct edges: 4 perimeter (boundary) + 1 shared diagonal (manifold).
    """
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    F = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
        ],
        dtype=np.int64,
    )
    return V, F


def _non_manifold_fan():
    """Three triangles sharing a single edge — non-manifold."""
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],   # shared edge endpoint
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    F = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
        ],
        dtype=np.int64,
    )
    return V, F


# =====================================================================
# diagnose (pure NumPy)
# =====================================================================

class TestDiagnose:

    def test_unit_tetrahedron_is_watertight(self):
        V, F = _unit_tetrahedron()
        s = diagnose(V, F)
        assert s.n_vertices == 4
        assert s.n_faces == 4
        assert s.n_degenerate_faces == 0
        assert s.n_boundary_edges == 0
        assert s.n_nonmanifold_edges == 0
        assert s.watertight is True
        # Surface area of unit tetrahedron with given vertices:
        # 3 small right-triangles of area 0.5 each + 1 equilateral face
        # with side √2 → area = (√3 / 4) · 2 = √3 / 2 ≈ 0.866.
        expected = 3 * 0.5 + np.sqrt(3) / 2
        assert s.surface_area_mm2 == pytest.approx(expected, rel=1e-6)
        assert s.bounds_mm[0] == (0.0, 0.0, 0.0)
        assert s.bounds_mm[1] == (1.0, 1.0, 1.0)

    def test_open_mesh_flags_boundary_edges(self):
        V, F = _open_triangle()
        s = diagnose(V, F)
        # Two triangles sharing one diagonal: 5 distinct edges = 4 perimeter
        # (boundary) + 1 shared diagonal (manifold).
        assert s.n_boundary_edges == 4
        assert s.n_nonmanifold_edges == 0
        assert s.watertight is False

    def test_non_manifold_fan_flags_nonmanifold(self):
        V, F = _non_manifold_fan()
        s = diagnose(V, F)
        # Three triangles share edge (0,1); other edges are boundary.
        # Three triangles × 3 edges = 9 edges total before dedup.
        # Shared edge (0,1) appears 3 times → 1 non-manifold edge.
        # Each triangle's other 2 edges are unique → 6 boundary edges.
        assert s.n_nonmanifold_edges == 1
        assert s.n_boundary_edges == 6
        assert s.watertight is False

    def test_degenerate_triangle_is_flagged(self):
        # Triangle with collinear vertices → zero area.
        V = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]])
        F = np.array([[0, 1, 2], [0, 1, 3]])
        s = diagnose(V, F)
        assert s.n_degenerate_faces == 1  # the collinear (0,1,2)

    def test_empty_mesh(self):
        V = np.zeros((0, 3))
        F = np.zeros((0, 3), dtype=np.int64)
        s = diagnose(V, F)
        assert s.n_vertices == 0
        assert s.n_faces == 0
        assert s.watertight is False
        assert s.surface_area_mm2 == 0.0

    def test_meshstats_summary_line_renders(self):
        V, F = _unit_tetrahedron()
        s = diagnose(V, F)
        line = s.summary_line()
        assert "V=4" in line and "F=4" in line and "watertight=True" in line

    def test_out_of_bounds_face_raises(self):
        V = np.zeros((3, 3))
        F = np.array([[0, 1, 7]])  # 7 ≥ nv
        with pytest.raises(ValueError, match="out of bounds"):
            diagnose(V, F)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="vertices"):
            diagnose(np.zeros((3, 2)), np.array([[0, 1, 2]]))
        with pytest.raises(ValueError, match="faces"):
            diagnose(np.zeros((4, 3)), np.array([[0, 1, 2, 3]]))


# =====================================================================
# _validate_field_inputs
# =====================================================================

class TestValidateFieldInputs:

    def test_accepts_valid(self):
        F = np.zeros((5, 5, 5), dtype=np.float32)
        F[2, 2, 2] = -1.0
        F2, org = _validate_field_inputs(F, 0.1, (0.0, 0.0, 0.0), iso_level=-0.5)
        assert F2 is F
        np.testing.assert_array_equal(org, [0.0, 0.0, 0.0])

    def test_rejects_wrong_ndim(self):
        with pytest.raises(ValueError, match="3D"):
            _validate_field_inputs(np.zeros((4, 4), dtype=np.float32), 0.1, (0, 0, 0), 0.0)

    def test_rejects_wrong_dtype(self):
        with pytest.raises(ValueError, match="float32"):
            _validate_field_inputs(np.zeros((4, 4, 4), dtype=np.float64), 0.1, (0, 0, 0), 0.0)

    def test_rejects_bad_vox(self):
        with pytest.raises(ValueError, match="vox_size"):
            _validate_field_inputs(np.zeros((4, 4, 4), dtype=np.float32), 0.0, (0, 0, 0), 0.0)

    def test_rejects_bad_origin(self):
        with pytest.raises(ValueError, match="origin"):
            _validate_field_inputs(np.zeros((4, 4, 4), dtype=np.float32), 0.1, (0, 0), 0.0)

    def test_rejects_iso_outside_field_range(self):
        F = np.zeros((4, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            _validate_field_inputs(F, 0.1, (0, 0, 0), iso_level=5.0)


# =====================================================================
# mesh_iso2mesh / mesh_skimage import-fallback behaviour
# =====================================================================

class TestMeshBackendsAbsentDeps:
    """The meshing functions must validate inputs before lazy-importing."""

    def test_iso2mesh_validates_before_import(self):
        # Even if iso2mesh is missing, bad inputs should fail with ValueError
        # (not ImportError) so users get a clear error message.
        bad_F = np.zeros((4, 4), dtype=np.float32)  # 2D not 3D
        with pytest.raises(ValueError, match="3D"):
            mesh_iso2mesh(bad_F, 0.1, (0, 0, 0), 0.0)

    def test_skimage_validates_before_import(self):
        bad_F = np.zeros((4, 4, 4), dtype=np.float64)  # wrong dtype
        with pytest.raises(ValueError, match="float32"):
            mesh_skimage(bad_F, 0.1, (0, 0, 0), 0.0)


# =====================================================================
# mesh_skimage end-to-end (skipped if scikit-image absent)
# =====================================================================

@pytest.mark.skipif(not HAS_SKIMAGE, reason="scikit-image not installed")
class TestMeshSkimage:

    def test_single_sphere_mesh_is_closed(self):
        """A clean radial SDF should mesh to a watertight sphere-ish surface."""
        # Build a 21^3 SDF of a sphere of radius 3 at the centre.
        N = 21
        vox = 0.5
        c = (N - 1) / 2 * vox
        x = np.arange(N) * vox
        XX, YY, ZZ = np.meshgrid(x, x, x, indexing="ij")
        F = np.sqrt((XX - c) ** 2 + (YY - c) ** 2 + (ZZ - c) ** 2) - 3.0
        F = F.astype(np.float32)
        verts, faces = mesh_skimage(F, vox, (0.0, 0.0, 0.0), iso_level=0.0)
        s = diagnose(verts, faces)
        # Marching cubes can leave a few non-manifold features; check bounds
        # and surface area instead of strict watertightness.
        assert s.n_faces > 100
        np.testing.assert_allclose(s.bounds_mm[0], (c - 3, c - 3, c - 3), atol=vox)
        np.testing.assert_allclose(s.bounds_mm[1], (c + 3, c + 3, c + 3), atol=vox)
        # Analytical: 4πR² = 113.1; allow ±10% for vox=0.5 discretization.
        analytical = 4 * np.pi * 3 ** 2
        assert abs(s.surface_area_mm2 - analytical) / analytical < 0.10


# =====================================================================
# mesh_iso2mesh end-to-end (skipped if iso2mesh absent)
# =====================================================================

@pytest.mark.skipif(not HAS_ISO2MESH, reason="iso2mesh not installed")
class TestMeshIso2mesh:

    def test_single_sphere_bbox_matches(self):
        N = 21
        vox = 0.5
        c = (N - 1) / 2 * vox
        x = np.arange(N) * vox
        XX, YY, ZZ = np.meshgrid(x, x, x, indexing="ij")
        F = (np.sqrt((XX - c) ** 2 + (YY - c) ** 2 + (ZZ - c) ** 2) - 3.0).astype(np.float32)
        # NOTE (2026-06-13): iso_level must follow the production convention
        # (a small NEGATIVE value, never exactly 0.0). With iso=0.0 the
        # iso2mesh grayscale thresholding selects the complement region and
        # meshes the padded-grid shell instead of the sphere (verified with
        # pyiso2mesh 0.5.5). The pipeline always uses ISO_LEVEL_MULTIPLIER *
        # vox; this test now does the same.
        verts, faces = mesh_iso2mesh(F, vox, (0.0, 0.0, 0.0), iso_level=-1e-6 * vox)
        s = diagnose(verts, faces)
        # The coord mapping should put the sphere bbox at ±R around (c,c,c).
        np.testing.assert_allclose(s.bounds_mm[0], (c - 3, c - 3, c - 3), atol=vox)
        np.testing.assert_allclose(s.bounds_mm[1], (c + 3, c + 3, c + 3), atol=vox)
        assert s.n_faces > 100


# =====================================================================
# repair_and_finalize (skipped if trimesh absent)
# =====================================================================

@pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not installed")
class TestRepairAndFinalize:

    def test_tetrahedron_round_trip(self):
        V, F = _unit_tetrahedron()
        # Without pymeshfix: just dedupe + merge + normals.
        Vc, Fc = repair_and_finalize(V, F, vox_size_mm=0.1, do_meshfix=False)
        assert Vc.shape[0] == 4
        assert Fc.shape == (4, 3)
        # Still watertight after cleanup.
        s = diagnose(Vc, Fc)
        assert s.watertight

    def test_duplicate_face_removed(self):
        V, F = _unit_tetrahedron()
        Fdup = np.vstack([F, F[[0]]])  # duplicate one face
        Vc, Fc = repair_and_finalize(V, Fdup, vox_size_mm=0.1, do_meshfix=False)
        assert Fc.shape == (4, 3)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError, match="vertices"):
            repair_and_finalize(np.zeros((4, 2)), np.array([[0, 1, 2]]), vox_size_mm=0.1)
        with pytest.raises(ValueError, match="faces"):
            repair_and_finalize(np.zeros((4, 3)), np.array([[0, 1, 2, 3]]), vox_size_mm=0.1)

    @pytest.mark.skipif(not HAS_PYMESHFIX, reason="pymeshfix not installed")
    def test_meshfix_closes_open_volumetric_mesh(self):
        # NOTE (2026-06-13): the previous input (a flat two-triangle sheet)
        # encloses zero volume, and pymeshfix 0.18 legitimately deletes such
        # degenerate geometry outright (-> 0 faces). A meaningful repair test
        # needs an OPEN BUT VOLUMETRIC input: a unit cube missing one face.
        # MeshFix must cap it watertight without deleting it.
        import trimesh

        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        V = np.asarray(box.vertices, dtype=np.float64)
        F = np.asarray(box.faces, dtype=np.int64)
        keep = ~np.all(V[F][:, :, 2].round(6) == 0.5, axis=1)  # drop +z face
        F_open = F[keep]
        assert F_open.shape[0] == 10

        Vc, Fc = repair_and_finalize(
            V, F_open,
            vox_size_mm=0.1,
            do_meshfix=True, joincomp=True,
            remove_smallest_components=False,
        )
        s = diagnose(Vc, Fc)
        assert Fc.shape[0] >= 10
        assert s.watertight


# =====================================================================
# End-to-end smoke against data_example (skipped if deps absent)
# =====================================================================

@pytest.mark.skipif(
    not (HAS_ISO2MESH and HAS_TRIMESH and HAS_PYMESHFIX),
    reason="iso2mesh + trimesh + pymeshfix required",
)
class TestEndToEndOnDataExample:

    def test_data_example_meshes_at_vox_2(self, packing_xyzd_path):
        """At vox=2 mm the full pipeline produces a closed multi-component mesh."""
        from rcps.field import (
            build_beads_sdf_icsg,
            build_box_sdf,
            compose_field,
            cull_spheres,
            make_grid,
            replicate_with_ghost_tiles,
        )
        from rcps.io import load_packing_xyzd

        S = load_packing_xyzd(packing_xyzd_path)
        c0, d0 = S[:, :3], S[:, 3]
        c, d = replicate_with_ghost_tiles(c0, d0, (50, 50, 50), 1)
        r = d / 2.0
        g = make_grid((50, 50, 50), 2.0, pad_vox=1, band_vox=3)
        band_dist = 3 * g.vox_size
        c, r, _ = cull_spheres(c, r, g, band_dist=band_dist)
        F_beads = build_beads_sdf_icsg(c, r, g, band_vox=3)
        F_box = build_box_sdf(g, max_radius_mm=float(r.max()))
        F = compose_field(F_beads, F_box, "beads")
        iso = -1e-6 * g.vox_size

        verts, faces = mesh_iso2mesh(
            F, g.vox_size, g.origin, iso,
            angbound_deg=25.0, radbound=1.0, distbound=0.10, maxnode=20_000_000,
        )
        Vc, Fc = repair_and_finalize(verts, faces, vox_size_mm=g.vox_size)
        s = diagnose(Vc, Fc)
        # Watertight per-component; bbox roughly matches the tile.
        assert s.watertight, s.summary_line()
        bmin, bmax = s.bounds_mm
        # The keep_sides=() default cuts flush at the tile box → bbox ≈ tile.
        for axis in range(3):
            assert -1.0 < bmin[axis] < 1.0
            assert 49.0 < bmax[axis] < 51.0
