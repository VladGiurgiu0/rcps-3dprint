"""Pure-NumPy mesh metrics for the validation suite.

These functions intentionally avoid trimesh so the metrics can be computed
in any environment, including the MATLAB reference comparison on a CI
runner. trimesh is used only to *load* a `.3mf` file (lazy import).

Implementation notes
--------------------
- `mesh_volume_signed` uses the divergence theorem: enclosed volume of a
  closed triangle mesh equals ``(1/6) Σ v0 · (v1 × v2)`` over all faces.
  For watertight meshes with consistent outward normals, the sum is the
  positive enclosed volume. The function returns the signed value;
  callers should ``abs()`` it when normals orientation is uncertain.
- `mesh_porosity` divides the absolute solid volume by the tile box
  volume — it's a *bulk* porosity inside the tile, regardless of how
  many disconnected components the mesh has.
- `bbox_distance` returns the L∞ distance between two axis-aligned
  bounding boxes, in mm. The validation suite asserts ``< 1e-3`` (1 µm).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def mesh_volume_signed(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
) -> float:
    """Divergence-theorem signed volume of a closed triangle mesh.

    For a watertight mesh with consistent outward normals, the result
    is the positive enclosed volume. If normals are reversed, the result
    is the negative of the volume.

    Returns 0.0 for an empty face array.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if F.size == 0:
        return 0.0
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)


def mesh_surface_area(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
) -> float:
    """Total surface area in mm² — tessellation-independent geometry.

    Unlike vertex/face counts (which fingerprint the CGAL build and FP
    environment), the surface area is a physical property of the part
    and is the right cross-implementation comparison metric.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if F.size == 0:
        return 0.0
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    return float(0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum())


def drop_degenerate_and_duplicate_faces(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    *,
    area_tol_mm2: float = 1e-12,
) -> tuple[NDArray[np.int64], int, int]:
    """Remove exact-duplicate faces (same vertex set) and ~zero-area faces.

    Minimal, deterministic cleanup for *raw* iso2mesh output (the MATLAB
    reference is generated with ``doRepair=false``). Returns
    ``(faces_clean, n_duplicates_removed, n_degenerate_removed)``.
    Does not move, merge, or add any vertex — pure face filtering.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if F.size == 0:
        return F, 0, 0

    key = np.sort(F, axis=1)
    _, first_idx = np.unique(key, axis=0, return_index=True)
    dup_mask = np.ones(len(F), dtype=bool)
    dup_mask[:] = False
    dup_mask[first_idx] = True          # keep first occurrence of each set
    n_dup = int((~dup_mask).sum())
    F1 = F[dup_mask]

    e1 = V[F1[:, 1]] - V[F1[:, 0]]
    e2 = V[F1[:, 2]] - V[F1[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    deg_mask = area > area_tol_mm2
    n_deg = int((~deg_mask).sum())

    return F1[deg_mask], n_dup, n_deg


def mesh_porosity_in_tile(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    tile_size_mm: tuple[float, float, float],
) -> float:
    """Bulk porosity inside the tile: ``1 − |V_solid| / V_tile``.

    Assumes the mesh encloses solid (beads + bridges) inside the tile box.
    Uses ``abs(mesh_volume_signed)`` so orientation issues don't flip the sign.
    """
    L, H, W = (float(x) for x in tile_size_mm)
    V_box = L * H * W
    V_solid = abs(mesh_volume_signed(vertices, faces))
    return 1.0 - V_solid / V_box


def bbox_of(vertices: NDArray[np.float64]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return ``(min, max)`` axis-aligned bounding box as nested tuples."""
    V = np.asarray(vertices, dtype=np.float64)
    if V.size == 0:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    return tuple(V.min(axis=0).tolist()), tuple(V.max(axis=0).tolist())


def bbox_distance(
    a: tuple[tuple[float, ...], tuple[float, ...]],
    b: tuple[tuple[float, ...], tuple[float, ...]],
) -> float:
    """L∞ distance between two AABBs across both corners."""
    a_min, a_max = a
    b_min, b_max = b
    diffs = []
    for i in range(3):
        diffs.append(abs(a_min[i] - b_min[i]))
        diffs.append(abs(a_max[i] - b_max[i]))
    return max(diffs)


def load_3mf(path: Path) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Load a .3mf file via trimesh and return ``(vertices, faces)``.

    A trimesh ``Scene`` is flattened by concatenating its geometry; the
    return arrays are float64 / int64 contiguous.
    """
    import trimesh  # lazy

    loaded = trimesh.load(str(path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    V = np.asarray(loaded.vertices, dtype=np.float64)
    F = np.asarray(loaded.faces, dtype=np.int64)
    return V, F


__all__ = [
    "bbox_distance",
    "bbox_of",
    "load_3mf",
    "mesh_porosity_in_tile",
    "mesh_volume_signed",
]
