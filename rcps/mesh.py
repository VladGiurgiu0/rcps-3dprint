"""Iso-surface meshing and repair for the RCPS pipeline.

Ports the surviving logic from ``matlab/legacy/mesh_from_raw.py`` (see
the archived design audit §4 for the per-line disposition). Three responsibilities:

1. **Iso-surface extraction** — `mesh_iso2mesh` (primary backend, CGAL via
   `pyiso2mesh.vol2restrictedtri`) and `mesh_skimage` (fallback marching
   cubes from `scikit-image`).
2. **Repair and cleanup** — `repair_and_finalize` deduplicates faces,
   merges near-duplicate vertices, drops degenerate triangles, runs
   `pymeshfix` (multi-component-safe), and orients normals outward.
3. **Diagnostics** — `diagnose` (pure NumPy) returns a `MeshStats`
   dataclass with vertex/face counts, degenerate count, boundary and
   non-manifold edge counts, watertightness, surface area, and bounds.

Sign convention reminder
------------------------
The SDF passed in here uses ``F < 0`` inside the exported phase
(beads or pore). `mesh_iso2mesh` internally negates ``F`` and the
``iso_level`` to match `iso2mesh`'s "positive inside" convention.

Coordinate mapping (load-bearing detail)
----------------------------------------
``mesh_from_raw.py:160`` uses ``origin_corner_p = origin − 0.5·vox − 1.0·vox``
because the field is ``np.pad``-ed with width 1 before being passed to
`vol2restrictedtri`. The ``−0.5·vox`` accounts for voxel-center vs
voxel-corner convention; the ``−1.0·vox`` undoes the padding offset.
**Do not change** without verifying the bounding-box equality assertion
in the validation suite (Task 8). See the archived design audit §5 risk 3.

Heavy dependencies are imported lazily so the module loads (and
diagnostics work) without them installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


# =====================================================================
# Iso-surface extraction
# =====================================================================

def _validate_field_inputs(
    field: NDArray[np.float32],
    vox_size: float,
    origin: tuple[float, float, float] | NDArray[np.float64],
    iso_level: float,
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Common input validation for the meshing backends."""
    if field.ndim != 3:
        raise ValueError(f"field must be 3D, got shape {field.shape}")
    if field.dtype != np.float32:
        raise ValueError(f"field must be float32, got {field.dtype}")
    if vox_size <= 0:
        raise ValueError(f"vox_size must be positive, got {vox_size}")
    org = np.asarray(origin, dtype=np.float64).reshape(-1)
    if org.shape != (3,):
        raise ValueError(f"origin must be length-3, got shape {org.shape}")
    fmin, fmax = float(field.min()), float(field.max())
    if not (fmin <= iso_level <= fmax):
        raise ValueError(
            f"iso_level {iso_level} is outside the field range [{fmin}, {fmax}] "
            f"— the iso-surface would be empty."
        )
    return field, org


def mesh_iso2mesh(
    field: NDArray[np.float32],
    vox_size: float,
    origin: tuple[float, float, float] | NDArray[np.float64],
    iso_level: float,
    *,
    angbound_deg: float = 25.0,
    radbound: float = 1.0,
    distbound: float = 0.10,
    maxnode: int = 200_000_000,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Mesh the ``F = iso_level`` iso-surface using CGAL via pyiso2mesh.

    Implementation faithfully ports ``matlab/legacy/mesh_from_raw.py:mesh_pyiso2mesh``,
    including:

    - The sign flip ``F → −F``, ``iso → −iso`` so that iso2mesh's
      "positive inside" convention matches our "negative inside" SDF.
    - The ``np.pad(F, pad_width=1, constant_values=−1.0)`` step so the
      iso-surface is fully closed at the grid boundary.
    - The ``cent = idx.mean + 0.5 + 1.0`` seed-point computation
      (``+0.5`` for voxel-center, ``+1.0`` for the padding shift).
    - ``brad = 2·Σ shape²`` bounding sphere radius.
    - The coordinate mapping
      ``verts = (origin − 0.5·vox − 1.0·vox) + nodes · vox``.

    Parameters
    ----------
    field
        3D ``float32`` signed-distance field.
    vox_size
        Voxel edge length in mm.
    origin
        Physical coordinate of voxel ``(0, 0, 0)`` in mm.
    iso_level
        Iso-value (typically a small negative number,
        e.g. ``-1e-6 * vox_size``).
    angbound_deg, radbound, distbound, maxnode
        iso2mesh quality knobs. Defaults match the recommended values
        from ``RCPS_v4.m``.

    Returns
    -------
    verts, faces
        ``(Nv, 3)`` float64 mm coordinates and ``(Nf, 3)`` int64 0-based
        triangle indices.

    Raises
    ------
    ImportError
        If ``iso2mesh`` is not installed.
    ValueError
        On invalid inputs or empty iso-surface interior.
    """
    field, origin = _validate_field_inputs(field, vox_size, origin, iso_level)

    # Sign flip (iso2mesh expects "positive inside").
    F = -field
    iso = -iso_level

    mask = F >= iso
    idx = np.argwhere(mask)
    if idx.size == 0:
        raise ValueError(
            "mesh_iso2mesh: empty interior (no voxels with F >= iso_level "
            "after sign flip). Check the iso_level."
        )

    # Pad with -1.0 so the iso-surface is fully closed at the boundary.
    imgp = np.pad(F.astype(np.float32, copy=False), pad_width=1,
                  mode="constant", constant_values=-1.0)

    # Seed point for CGAL: interior centroid in padded-image coords.
    # +0.5 → voxel-center convention; +1.0 → padding shift.
    cent = idx.mean(axis=0).astype(float) + 0.5 + 1.0

    # Bounding sphere radius for CGAL; "2·sum(shape²)" is the formula
    # used by mesh_from_raw.py (generously oversized; CGAL clamps as needed).
    brad = 2.0 * float(np.sum(np.asarray(imgp.shape, dtype=float) ** 2))

    try:
        from iso2mesh.core import vol2restrictedtri  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "mesh_iso2mesh requires the `iso2mesh` package (CGAL-based). "
            "Install with `pip install iso2mesh`. Falls back available: "
            "`mesh_skimage` (lower quality, no CGAL surface reconstruction)."
        ) from e

    log.info(
        "vol2restrictedtri: angbound=%.3g° radbound=%.3g distbound=%.3g maxnode=%d",
        angbound_deg, radbound, distbound, maxnode,
    )
    node, elem = vol2restrictedtri(
        imgp,
        float(iso),
        cent.tolist(),
        brad,
        float(angbound_deg),
        float(radbound),
        float(distbound),
        int(maxnode),
    )

    node = np.asarray(node, dtype=np.float64)
    elem = np.asarray(elem, dtype=np.int64)
    if elem.shape[1] > 3:
        # iso2mesh sometimes returns a region-id column; drop it.
        elem = elem[:, :3]
    faces = elem - 1  # 1-based → 0-based

    # Physical-coord mapping. See module docstring for the derivation of
    # the "-0.5·vox - 1.0·vox" offset.
    origin_corner_p = origin - 0.5 * float(vox_size) - 1.0 * float(vox_size)
    verts = origin_corner_p.reshape(1, 3) + node * float(vox_size)

    log.info(
        "iso2mesh produced %d verts, %d faces; bbox %s..%s",
        verts.shape[0], faces.shape[0],
        verts.min(axis=0).tolist(), verts.max(axis=0).tolist(),
    )
    return verts, faces.astype(np.int64, copy=False)


def mesh_skimage(
    field: NDArray[np.float32],
    vox_size: float,
    origin: tuple[float, float, float] | NDArray[np.float64],
    iso_level: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Mesh the iso-surface using scikit-image's marching cubes.

    Fallback backend used when `iso2mesh` (CGAL) is unavailable. Produces
    a regular-grid marching-cubes mesh that is denser and less geometrically
    adaptive than iso2mesh, but suitable for users without CGAL.

    No sign flip is needed: ``skimage.measure.marching_cubes`` extracts
    the iso-surface at ``level``, irrespective of sign convention.

    Raises
    ------
    ImportError
        If ``scikit-image`` is not installed.
    """
    field, origin = _validate_field_inputs(field, vox_size, origin, iso_level)

    try:
        from skimage import measure  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "mesh_skimage requires `scikit-image`. "
            "Install with `pip install scikit-image`."
        ) from e

    verts_ijk, faces, _normals, _values = measure.marching_cubes(
        volume=field,
        level=float(iso_level),
        spacing=(float(vox_size), float(vox_size), float(vox_size)),
    )
    verts = verts_ijk + origin.reshape(1, 3)
    faces = np.asarray(faces, dtype=np.int64)
    log.info(
        "skimage marching_cubes produced %d verts, %d faces; bbox %s..%s",
        verts.shape[0], faces.shape[0],
        verts.min(axis=0).tolist(), verts.max(axis=0).tolist(),
    )
    return verts.astype(np.float64, copy=False), faces


# =====================================================================
# Repair and cleanup
# =====================================================================

def repair_and_finalize(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    *,
    vox_size_mm: float,
    do_meshfix: bool = True,
    joincomp: bool = True,
    remove_smallest_components: bool = False,
    merge_tol_mm: float | None = None,
    deg_height_mm: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Post-mesh cleanup: dedupe, merge, repair, orient.

    Ports ``matlab/legacy/mesh_from_raw.py:export_trimesh_stl`` (the name is
    misleading there — the function did cleanup, not just STL writing).
    Steps:

    1. Remove duplicate faces (order-independent).
    2. Merge near-duplicate vertices via decimal rounding; the precision
       is set by ``merge_tol_mm`` (defaults to ``1e-4 · vox_size``).
    3. Drop degenerate triangles whose height is below ``deg_height_mm``
       (defaults to ``1e-3 · vox_size``).
    4. (Optional) Run ``pymeshfix.MeshFix(...)`` with
       ``joincomp=True, remove_smallest_components=False`` — multi-
       component-safe; required when ``keep_sides`` is partial.
    5. Fix normals and ensure outward orientation via centroid test.

    Parameters
    ----------
    vertices, faces
        Mesh from `mesh_iso2mesh` or `mesh_skimage`.
    vox_size_mm
        Voxel size used to derive default tolerances.
    do_meshfix
        Run pymeshfix? Set False for speed when upstream mesh is known clean.
    joincomp
        Forwarded to ``pymeshfix.MeshFix.repair``. ``True`` lets meshfix
        merge components that should be connected.
    remove_smallest_components
        Forwarded to pymeshfix. **Keep this ``False``** when meshing with
        partial ``keep_sides`` — small valid components must not be dropped.
    merge_tol_mm
        Vertex merge tolerance. None → default ``max(1e-6, 1e-4 · vox)``.
    deg_height_mm
        Min triangle height. None → default ``max(1e-6, 1e-3 · vox)``.

    Returns
    -------
    verts_clean, faces_clean
        Clean ``(Nv, 3)`` float64 / ``(Nf, 3)`` int64.

    Raises
    ------
    ImportError
        If ``trimesh`` is not installed, or if ``do_meshfix=True`` and
        ``pymeshfix`` is not installed.
    """
    try:
        import trimesh  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "repair_and_finalize requires `trimesh` "
            "(`pip install trimesh`)."
        ) from e

    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"vertices must be (N, 3), got shape {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"faces must be (M, 3), got shape {F.shape}")
    if F.size and (F.max() >= V.shape[0] or F.min() < 0):
        raise ValueError(
            f"face indices out of bounds: min={F.min()}, max={F.max()}, "
            f"#vertices={V.shape[0]}"
        )

    vox = float(vox_size_mm)
    merge_tol = (
        merge_tol_mm
        if merge_tol_mm is not None
        else max(1e-6, 1e-4 * vox)
    )
    deg_height = (
        deg_height_mm
        if deg_height_mm is not None
        else max(1e-6, 1e-3 * vox)
    )

    # 1) Remove duplicate faces (order-independent).
    Fkey = np.sort(F, axis=1)
    _, keep = np.unique(Fkey, axis=0, return_index=True)
    F = F[np.sort(keep)]

    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)

    # 2) Merge near-duplicate vertices via digit rounding.
    digits = int(np.ceil(-np.log10(merge_tol))) if merge_tol > 0 else None
    if digits is not None:
        mesh.merge_vertices(digits_vertex=digits)

    # 3) Drop degenerate triangles by minimum height.
    ok = trimesh.triangles.nondegenerate(mesh.triangles, height=float(deg_height))
    if ok is not None and ok.size == mesh.faces.shape[0]:
        mesh.update_faces(ok)
        mesh.remove_unreferenced_vertices()

    # 4) Optional pymeshfix repair.
    if do_meshfix:
        try:
            from pymeshfix import MeshFix  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "do_meshfix=True requires `pymeshfix` "
                "(`pip install pymeshfix`). "
                "Pass `do_meshfix=False` to skip."
            ) from e
        mf = MeshFix(mesh.vertices, mesh.faces)
        # IMPORTANT: keep all components — porous bead packs may be
        # multi-component when keep_sides is partial.
        mf.repair(
            joincomp=bool(joincomp),
            remove_smallest_components=bool(remove_smallest_components),
        )
        mesh = trimesh.Trimesh(
            vertices=np.asarray(mf.points),
            faces=np.asarray(mf.faces, dtype=np.int64),
            process=False,
        )
        log.info(
            "pymeshfix repair: V %d, F %d (joincomp=%s, drop_small=%s)",
            mesh.vertices.shape[0], mesh.faces.shape[0],
            joincomp, remove_smallest_components,
        )

    # 5) Normals + global outward orientation.
    trimesh.repair.fix_normals(mesh, multibody=True)
    fc = mesh.triangles_center
    fn = mesh.face_normals
    c = mesh.centroid
    s = float(np.mean(np.einsum("ij,ij->i", fn, fc - c)))
    if np.isfinite(s) and s < 0:
        mesh.invert()
        log.info("inverted mesh winding for outward orientation")

    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
    )


# =====================================================================
# Diagnostics (pure NumPy)
# =====================================================================

@dataclass(frozen=True)
class MeshStats:
    """Immutable summary of mesh quality and geometry."""

    n_vertices: int
    n_faces: int
    n_degenerate_faces: int
    n_boundary_edges: int
    n_nonmanifold_edges: int
    watertight: bool
    surface_area_mm2: float
    bounds_mm: tuple[tuple[float, float, float], tuple[float, float, float]]

    def summary_line(self) -> str:
        """One-line text summary suitable for log output."""
        return (
            f"V={self.n_vertices}, F={self.n_faces}, "
            f"deg={self.n_degenerate_faces}, "
            f"bdry={self.n_boundary_edges}, nonmanifold={self.n_nonmanifold_edges}, "
            f"watertight={self.watertight}, area={self.surface_area_mm2:.2f} mm²"
        )


def diagnose(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    *,
    area_tol_mm2: float = 1e-12,
) -> MeshStats:
    """Compute mesh-quality diagnostics using only NumPy.

    Equivalent to the diagnostic block from ``RCPS_v4.m`` lines 686–707
    and ``mesh_from_raw.py`` lines 261–276, but reimplemented in pure
    NumPy so the function loads/runs without `trimesh`.

    Watertightness is defined as zero boundary edges AND zero non-manifold
    edges. This is per-mesh: when the mesh has multiple disconnected
    components (e.g., partial ``keep_sides``), the property holds iff
    every component is individually watertight. See the archived design audit §5
    risk 2.

    Parameters
    ----------
    vertices, faces
        Mesh arrays. ``vertices``: ``(Nv, 3)`` float; ``faces``: ``(Nf, 3)`` int.
    area_tol_mm2
        Triangles with area below this are flagged degenerate.

    Returns
    -------
    MeshStats
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"vertices must be (N, 3), got shape {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"faces must be (M, 3), got shape {F.shape}")

    nv = int(V.shape[0])
    nf = int(F.shape[0])

    if nf == 0:
        return MeshStats(
            n_vertices=nv, n_faces=0,
            n_degenerate_faces=0,
            n_boundary_edges=0,
            n_nonmanifold_edges=0,
            watertight=False,
            surface_area_mm2=0.0,
            bounds_mm=(
                tuple(V.min(axis=0).tolist()) if nv else (0.0, 0.0, 0.0),
                tuple(V.max(axis=0).tolist()) if nv else (0.0, 0.0, 0.0),
            ),
        )

    if F.size and (F.max() >= nv or F.min() < 0):
        raise ValueError(
            f"face indices out of bounds: min={F.min()}, max={F.max()}, "
            f"#vertices={nv}"
        )

    # Triangle areas via |e1 × e2| / 2.
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    cross = np.cross(e1, e2)
    tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
    n_degenerate = int(np.count_nonzero(tri_areas < area_tol_mm2))
    surface_area = float(tri_areas.sum())

    # Edge multiplicity. Each face contributes 3 edges; sort endpoints
    # so the edge (a, b) and (b, a) collapse.
    edges = np.empty((3 * nf, 2), dtype=np.int64)
    edges[0::3] = F[:, [0, 1]]
    edges[1::3] = F[:, [1, 2]]
    edges[2::3] = F[:, [2, 0]]
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    n_boundary = int(np.count_nonzero(counts == 1))
    n_nonmanifold = int(np.count_nonzero(counts > 2))
    watertight = (n_boundary == 0) and (n_nonmanifold == 0)

    bmin = tuple(V.min(axis=0).tolist())
    bmax = tuple(V.max(axis=0).tolist())

    stats = MeshStats(
        n_vertices=nv,
        n_faces=nf,
        n_degenerate_faces=n_degenerate,
        n_boundary_edges=n_boundary,
        n_nonmanifold_edges=n_nonmanifold,
        watertight=watertight,
        surface_area_mm2=surface_area,
        bounds_mm=(bmin, bmax),
    )
    log.info("diagnose: %s", stats.summary_line())
    return stats


__all__ = [
    "MeshStats",
    "diagnose",
    "mesh_iso2mesh",
    "mesh_skimage",
    "repair_and_finalize",
]
