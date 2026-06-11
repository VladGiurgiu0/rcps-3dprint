"""Signed-distance field construction for the RCPS pipeline.

Ports STEPs 2–4c of ``RCPS_v4.m`` to NumPy. See the archived design audit §1 for the
disposition of each MATLAB section.

Conventions
-----------
- Coordinates are in millimetres.
- RAS axis labels for ``keep_sides``: ``L``/``R`` = -X/+X, ``P``/``A`` = -Y/+Y,
  ``I``/``S`` = -Z/+Z. Listing a face in ``keep_sides`` means "do **not** cut
  beads at that face" — used when the tile dovetails with a neighbouring tile.
- All 3D scalar fields are stored as ``float32`` (matches MATLAB ``single``).
  1D coordinate vectors and 1D sphere arrays stay ``float64``.

Sign conventions for the SDF::

    F < 0   →  inside  (beads, or pore, depending on `export_what`)
    F = 0   →  surface (iso-surface extracted by the mesher)
    F > 0   →  outside

Memory
------
The dense ``float32`` field array dominates memory: ``nx·ny·nz · 4 bytes``.
At ``vox=0.1 mm`` on a 50×50×50 mm tile, this is ~500 MiB. Use a coarser
voxel size for unit tests; the MATLAB feasibility guard at ~5·10⁸ voxels is
preserved (``MAX_VOXELS``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


# =====================================================================
# Constants & types
# =====================================================================

# RAS face label set. L/R toggle -X/+X, P/A toggle -Y/+Y, I/S toggle -Z/+Z.
VALID_FACES: frozenset[str] = frozenset({"L", "R", "P", "A", "I", "S"})

# Hard upper bound on grid voxel count (mirrors the MATLAB guard at lines 240–245).
# 5·10⁸ float32 voxels = ~1.86 GiB for one field. Above this we refuse to
# proceed and recommend an adaptive/sparse representation.
MAX_VOXELS: int = 500_000_000

ExportWhat = Literal["beads", "pore"]


@dataclass(frozen=True)
class Grid:
    """Voxel grid describing the SDF evaluation domain.

    Attributes
    ----------
    nx, ny, nz
        Voxel counts along x, y, z.
    vox_size
        Voxel edge length in mm (already snapped to divide ``tile_size``).
    origin
        Physical coordinate of voxel ``(0, 0, 0)`` in mm. With non-zero
        padding the origin is negative (the grid extends into ``x < 0``).
    tile_size
        ``(L, H, W)`` of the *original* tile (without padding) in mm.
    pad_vox
        Padding (in voxels) on each side of the tile, accounting for both
        ``pad_vox`` and any extra padding required by ``keep_sides``.
    keep_sides
        Tuple of RAS face labels (subset of L/R/P/A/I/S) whose beads are
        not cut by the box SDF.
    """

    nx: int
    ny: int
    nz: int
    vox_size: float
    origin: tuple[float, float, float]
    tile_size: tuple[float, float, float]
    pad_vox: int
    keep_sides: tuple[str, ...] = _dc_field(default_factory=tuple)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def n_voxels(self) -> int:
        return int(self.nx) * int(self.ny) * int(self.nz)

    @property
    def memory_float32_bytes(self) -> int:
        return 4 * self.n_voxels

    def x_vec(self) -> NDArray[np.float64]:
        return self.origin[0] + np.arange(self.nx, dtype=np.float64) * self.vox_size

    def y_vec(self) -> NDArray[np.float64]:
        return self.origin[1] + np.arange(self.ny, dtype=np.float64) * self.vox_size

    def z_vec(self) -> NDArray[np.float64]:
        return self.origin[2] + np.arange(self.nz, dtype=np.float64) * self.vox_size


# =====================================================================
# 1. Replicate spheres with ghost tiles
# =====================================================================

def replicate_with_ghost_tiles(
    centers: NDArray[np.float64],
    diameters: NDArray[np.float64],
    tile_size_mm: tuple[float, float, float] | Sequence[float],
    ghost_tiles: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Periodic-tile replication of sphere positions.

    The packing is periodic by construction (Baranau's domain has periodic
    boundary conditions). For boundary-correct SDF construction near tile
    faces — and for the ``keep_sides`` dovetailing workflow — we replicate
    each sphere into ``(2g+1)³`` positions where ``g = ghost_tiles``.
    Spheres outside the relevant grid extent are dropped later by
    ``cull_spheres``.

    Ports ``RCPS_v4.m`` lines 161–183 (facility branch). In v1.0 we run this
    unconditionally in tile-only mode; ``ghost_tiles=0`` is a no-op.

    Parameters
    ----------
    centers
        ``(N, 3)`` original sphere centers.
    diameters
        ``(N,)`` per-sphere diameters.
    tile_size_mm
        ``(L, H, W)`` tile dimensions.
    ghost_tiles
        Number of periodic shells on each side.

    Returns
    -------
    centers_out, diameters_out
        Replicated arrays of length ``N · (2·ghost_tiles + 1)³``.

    Notes
    -----
    For ``g=1`` and N=718 the output is 19 386 spheres. Most get culled
    before the SDF loop, so this is cheap.
    """
    centers = np.asarray(centers, dtype=np.float64)
    diameters = np.asarray(diameters, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"centers must be (N, 3), got {centers.shape}")
    if diameters.shape != (centers.shape[0],):
        raise ValueError(
            f"diameters shape {diameters.shape} does not match N={centers.shape[0]}"
        )
    if ghost_tiles < 0:
        raise ValueError(f"ghost_tiles must be non-negative, got {ghost_tiles}")

    if ghost_tiles == 0:
        return centers, diameters

    L, H, W = float(tile_size_mm[0]), float(tile_size_mm[1]), float(tile_size_mm[2])

    g = ghost_tiles
    txs, tys, tzs = (
        np.arange(-g, g + 1, dtype=np.int64),
        np.arange(-g, g + 1, dtype=np.int64),
        np.arange(-g, g + 1, dtype=np.int64),
    )
    # Cartesian product of shifts.
    TX, TY, TZ = np.meshgrid(txs, tys, tzs, indexing="ij")
    shifts = np.stack(
        [TX.ravel() * L, TY.ravel() * H, TZ.ravel() * W], axis=1
    )  # ((2g+1)^3, 3)

    # Tile centers: outer-sum (N, 1, 3) + (1, M, 3) → (N, M, 3) → (N·M, 3).
    rep = centers[:, None, :] + shifts[None, :, :]
    n_copies = shifts.shape[0]
    centers_out = rep.reshape(-1, 3)
    diameters_out = np.tile(diameters, n_copies)

    log.info(
        "ghost-tile replication: N=%d × (2·%d+1)^3 = %d spheres",
        centers.shape[0], g, centers_out.shape[0],
    )
    return centers_out, diameters_out


# =====================================================================
# 2. Snap voxel grid to a divisor of the tile size
# =====================================================================

def snap_grid(
    tile_size_mm: tuple[float, float, float] | Sequence[float],
    vox_target_mm: float,
) -> tuple[int, int, int, float]:
    """Snap ``vox_size`` so the tile divides exactly along all axes.

    Ports ``RCPS_v4.m`` STEP 3 (lines 198–217). The number of voxels along
    X is ``max(8, round(L / vox_target))``; ``vox_snapped = L / nx``. Then
    ``ny = H / vox_snapped`` and ``nz = W / vox_snapped`` must be integer
    to within ``1e-10``, otherwise the grid is not divisible and the
    function raises.

    Returns
    -------
    nx, ny, nz
        Voxel counts (excluding padding).
    vox_snapped
        Adjusted voxel size in mm.
    """
    L, H, W = (float(s) for s in tile_size_mm)
    if min(L, H, W) <= 0:
        raise ValueError(f"tile dims must be positive, got {tile_size_mm}")
    if vox_target_mm <= 0:
        raise ValueError(f"vox_target_mm must be positive, got {vox_target_mm}")

    nx = int(max(8, round(L / vox_target_mm)))
    vox_snapped = L / nx
    ny_f = H / vox_snapped
    nz_f = W / vox_snapped
    ny = int(round(ny_f))
    nz = int(round(nz_f))
    if abs(ny_f - ny) > 1e-10:
        raise ValueError(
            f"H={H} not divisible by snapped vox_size={vox_snapped}; H/vox={ny_f}"
        )
    if abs(nz_f - nz) > 1e-10:
        raise ValueError(
            f"W={W} not divisible by snapped vox_size={vox_snapped}; W/vox={nz_f}"
        )
    log.info(
        "snap_grid: nx=%d ny=%d nz=%d, vox=%.6g mm (target %.6g)",
        nx, ny, nz, vox_snapped, vox_target_mm,
    )
    return nx, ny, nz, vox_snapped


# =====================================================================
# 3. Build the full grid (padding + keepSides accommodation)
# =====================================================================

def _validate_keep_sides(keep_sides: Sequence[str] | None) -> tuple[str, ...]:
    """Normalise keep_sides to an upper-case tuple of valid RAS labels."""
    if keep_sides is None:
        return ()
    out: list[str] = []
    for s in keep_sides:
        u = str(s).upper()
        if u not in VALID_FACES:
            raise ValueError(
                f"invalid keep_sides label {s!r}; valid: {sorted(VALID_FACES)}"
            )
        if u not in out:
            out.append(u)
    return tuple(out)


def make_grid(
    tile_size_mm: tuple[float, float, float] | Sequence[float],
    vox_target_mm: float,
    *,
    pad_vox: int = 1,
    band_vox: int = 3,
    keep_sides: Sequence[str] | None = None,
    max_radius_mm: float = 0.0,
) -> Grid:
    """Construct the padded SDF evaluation grid.

    Ports ``RCPS_v4.m`` lines 222–251. The grid extends ``pad_vox`` voxels
    on each side of the tile (so the box walls can be extracted cleanly);
    if any face is in ``keep_sides``, an extra ``ceil((rMax + bandDist)/vox)
    + 1`` voxels of padding are added so beads protruding past the tile
    are fully covered.

    Parameters
    ----------
    tile_size_mm
        ``(L, H, W)`` original tile dims.
    vox_target_mm
        Target voxel size in mm (snapped before grid construction).
    pad_vox
        Base padding in voxels (≥1 recommended so the box SDF resolves).
    band_vox
        Narrow-band width in voxels; affects how much extra padding is
        required for kept faces. ``max(2, band_vox)`` is used (matches MATLAB).
    keep_sides
        RAS labels (subset of L/R/P/A/I/S) whose beads are not cut.
    max_radius_mm
        Maximum sphere radius after expansion factor. Required when any
        face is kept (to size the extra padding); ignored otherwise.

    Returns
    -------
    Grid
        Immutable description of the padded voxel grid.
    """
    ks = _validate_keep_sides(keep_sides)
    nx0, ny0, nz0, vox = snap_grid(tile_size_mm, vox_target_mm)

    band_vox_eff = max(2, int(band_vox))
    band_dist = band_vox_eff * vox

    pad = int(pad_vox)
    if ks:
        if max_radius_mm <= 0:
            raise ValueError(
                "max_radius_mm must be positive when keep_sides is non-empty "
                "(used to size the extra grid padding)"
            )
        pad_extra = int(np.ceil((max_radius_mm + band_dist) / vox)) + 1
        pad += pad_extra
        log.info(
            "keep_sides %s → extra padding %d voxels (total pad=%d)",
            ks, pad_extra, pad,
        )

    nx = nx0 + 2 * pad
    ny = ny0 + 2 * pad
    nz = nz0 + 2 * pad

    n_vox = nx * ny * nz
    if n_vox > MAX_VOXELS:
        raise ValueError(
            f"grid too large: {nx}×{ny}×{nz} = {n_vox:.3g} voxels "
            f"({4 * n_vox / 1024**3:.2f} GiB for one float32 field). "
            f"Use a coarser vox_size or split the tile."
        )

    origin = (-pad * vox, -pad * vox, -pad * vox)
    L, H, W = (float(s) for s in tile_size_mm)
    g = Grid(
        nx=nx, ny=ny, nz=nz,
        vox_size=vox,
        origin=origin,
        tile_size=(L, H, W),
        pad_vox=pad,
        keep_sides=ks,
    )
    log.info(
        "grid: %d×%d×%d, vox=%.6g mm, origin=%s, %.1f MiB",
        nx, ny, nz, vox, origin, g.memory_float32_bytes / 1024**2,
    )
    return g


# =====================================================================
# 4. Box SDF (with RAS face extension for keep_sides)
# =====================================================================

def build_box_sdf(grid: Grid, *, max_radius_mm: float, band_vox: int = 3) -> NDArray[np.float32]:
    """Construct the signed-distance field of the tile bounding box.

    Ports ``RCPS_v4.m`` lines 253–294. The box has corners at
    ``(0,0,0)`` and ``tile_size`` by default. For each face listed in
    ``grid.keep_sides``, the box is extended outward by
    ``ext_mm = 2·(rMax + bandDist) + 2·vox``, so that the far face of the
    box lies *outside* the padded grid and the box SDF stops clipping
    beads at that face.

    Sign convention: ``F_box < 0`` strictly inside the (possibly extended)
    box, ``F_box > 0`` outside.

    Returns
    -------
    ndarray, shape ``(nx, ny, nz)``, dtype float32
        Signed distance to the box surface.
    """
    L, H, W = grid.tile_size
    vox = grid.vox_size
    band_vox_eff = max(2, int(band_vox))
    band_dist = band_vox_eff * vox
    ext_mm = 2.0 * (max_radius_mm + band_dist) + 2.0 * vox

    xmin, xmax = 0.0, L
    ymin, ymax = 0.0, H
    zmin, zmax = 0.0, W
    ks = grid.keep_sides
    if "L" in ks:
        xmin -= ext_mm
    if "R" in ks:
        xmax += ext_mm
    if "P" in ks:
        ymin -= ext_mm
    if "A" in ks:
        ymax += ext_mm
    if "I" in ks:
        zmin -= ext_mm
    if "S" in ks:
        zmax += ext_mm

    cx, bx = 0.5 * (xmin + xmax), 0.5 * (xmax - xmin)
    cy, by = 0.5 * (ymin + ymax), 0.5 * (ymax - ymin)
    cz, bz = 0.5 * (zmin + zmax), 0.5 * (zmax - zmin)

    qx = np.abs(grid.x_vec() - cx) - bx
    qy = np.abs(grid.y_vec() - cy) - by
    qz = np.abs(grid.z_vec() - cz) - bz

    # Broadcasting trick avoids allocating a full ndgrid.
    QX = qx.astype(np.float32).reshape(-1, 1, 1)
    QY = qy.astype(np.float32).reshape(1, -1, 1)
    QZ = qz.astype(np.float32).reshape(1, 1, -1)

    # outside = sqrt(max(QX,0)^2 + max(QY,0)^2 + max(QZ,0)^2)
    QXp = np.maximum(QX, np.float32(0.0))
    QYp = np.maximum(QY, np.float32(0.0))
    QZp = np.maximum(QZ, np.float32(0.0))
    outside = np.sqrt(QXp * QXp + QYp * QYp + QZp * QZp)

    # inside = min(max(QX, QY, QZ), 0)   (broadcasted max-of-three)
    inside = np.minimum(np.maximum(np.maximum(QX, QY), QZ), np.float32(0.0))
    F_box = outside + inside
    return F_box.astype(np.float32, copy=False)


# =====================================================================
# 5. Sphere culling (drop spheres that can't influence the grid)
# =====================================================================

def cull_spheres(
    centers: NDArray[np.float64],
    radii: NDArray[np.float64],
    grid: Grid,
    band_dist: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Drop spheres whose ICSG bounding box doesn't intersect the grid.

    Ports ``RCPS_v4.m`` lines 296–318. A sphere of centre ``(xc, yc, zc)``
    and effective radius ``r + band_dist`` can only affect voxels inside
    its bounding box; if that box is fully outside the grid extent, the
    sphere contributes nothing and is removed. Major speedup when ghost
    tiles inflate the input list 27× — most ghosts get culled here.

    Returns
    -------
    centers_kept, radii_kept, keep_mask
        Filtered centres/radii plus the boolean mask used (for diagnostics
        and joint filtering of any per-sphere arrays maintained by the
        caller).
    """
    xMin, xMax = float(grid.x_vec()[0]), float(grid.x_vec()[-1])
    yMin, yMax = float(grid.y_vec()[0]), float(grid.y_vec()[-1])
    zMin, zMax = float(grid.z_vec()[0]), float(grid.z_vec()[-1])

    rad_cull = radii + float(band_dist)

    keep = (
        (centers[:, 0] + rad_cull >= xMin) & (centers[:, 0] - rad_cull <= xMax)
        & (centers[:, 1] + rad_cull >= yMin) & (centers[:, 1] - rad_cull <= yMax)
        & (centers[:, 2] + rad_cull >= zMin) & (centers[:, 2] - rad_cull <= zMax)
    )
    n_before = centers.shape[0]
    n_after = int(keep.sum())
    log.info(
        "cull_spheres: %d → %d (%.1f%% removed)",
        n_before, n_after, 100.0 * (n_before - n_after) / max(1, n_before),
    )
    return centers[keep], radii[keep], keep


def apply_keepsides_filter(
    centers: NDArray[np.float64],
    radii: NDArray[np.float64],
    grid: Grid,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Drop ghost spheres beyond any kept face (half-open ownership).

    Ports ``RCPS_v4.m`` lines 320–338, with one deliberate change. A
    "kept" face means *the current tile's beads protrude past that face
    uncut* — it does **not** mean "include ghost spheres from the
    neighbouring tile". So when ``L`` is kept, we drop spheres with
    ``x < 0``; when ``R`` is kept, we drop spheres with ``x ≥ L``.

    **Half-open ownership rule (2026-06-11, deviates from RCPS_v4).**
    Ownership intervals are half-open: ``[0, L)`` per kept axis — closed
    at the lower face, *open* at the upper face. RCPS_v4 used closed
    intervals on both sides, so a sphere centered exactly on a shared
    facility plane (center at ``x = 0``, periodic image at ``x = L``)
    was kept *twice* within the tile, and therefore printed twice when
    adjacent interlocking tiles were assembled — a physical collision.
    With the half-open rule the ``x = 0`` instance is kept and its
    ``x = L`` image is dropped: every sphere is owned by exactly one
    tile across the assembly. The equality tolerance is 1 nm
    (``1e-6`` mm), far below any printable feature.

    No-op if ``grid.keep_sides`` is empty.
    """
    if not grid.keep_sides:
        keep = np.ones(centers.shape[0], dtype=bool)
        return centers, radii, keep

    L, H, W = grid.tile_size
    ks = grid.keep_sides
    tol = 1e-6  # mm; equality tolerance for "centered exactly on a face"

    keep = np.ones(centers.shape[0], dtype=bool)
    if "L" in ks:
        keep &= centers[:, 0] >= -tol          # closed at lower face
    if "R" in ks:
        keep &= centers[:, 0] < L - tol        # open at upper face
    if "P" in ks:
        keep &= centers[:, 1] >= -tol
    if "A" in ks:
        keep &= centers[:, 1] < H - tol
    if "I" in ks:
        keep &= centers[:, 2] >= -tol
    if "S" in ks:
        keep &= centers[:, 2] < W - tol

    n_before, n_after = centers.shape[0], int(keep.sum())
    log.info(
        "keep_sides %s filter: %d → %d kept",
        ks, n_before, n_after,
    )
    return centers[keep], radii[keep], keep


# =====================================================================
# 6. ICSG narrow-band beads SDF
# =====================================================================

def build_beads_sdf_icsg(
    centers: NDArray[np.float64],
    radii: NDArray[np.float64],
    grid: Grid,
    band_vox: int = 3,
) -> NDArray[np.float32]:
    """ICSG narrow-band union-of-spheres signed-distance field.

    Ports ``RCPS_v4.m`` lines 343–378. For each sphere, evaluates the
    local distance field on its bounding-box subgrid (radius ``r +
    band_dist`` from the sphere centre) and ``min``-reduces it into the
    global field ``F``. Voxels never touched by any sphere remain at
    ``+band_dist`` — i.e., positive, "just outside the narrow band".

    Sign convention: ``F < 0`` inside any sphere, ``F = 0`` on the union
    surface, ``F > 0`` outside.

    Parameters
    ----------
    centers
        ``(N, 3)`` sphere centres in mm.
    radii
        ``(N,)`` sphere radii in mm (already post-expansion-factor and
        post-``diameter`` bridge mode if applicable).
    grid
        Padded SDF grid from :func:`make_grid`.
    band_vox
        Narrow-band half-width in voxels; ``max(2, band_vox)`` is used.

    Returns
    -------
    ndarray of shape ``(grid.nx, grid.ny, grid.nz)`` and dtype ``float32``.
    """
    if centers.shape[0] != radii.shape[0]:
        raise ValueError(
            f"centers/radii length mismatch: {centers.shape[0]} vs {radii.shape[0]}"
        )

    nx, ny, nz = grid.shape
    vox = grid.vox_size
    band_vox_eff = max(2, int(band_vox))
    band_dist = float(band_vox_eff * vox)

    F = np.full((nx, ny, nz), np.float32(band_dist), dtype=np.float32)
    if centers.size == 0:
        return F

    xVec = grid.x_vec()
    yVec = grid.y_vec()
    zVec = grid.z_vec()
    ox, oy, oz = grid.origin

    # The MATLAB inner loop. We could vectorise across spheres using
    # scatter-reduce, but the per-sphere subgrid is small and the loop is
    # fast enough in NumPy. Profile before optimising further.
    for i in range(centers.shape[0]):
        xc, yc, zc = float(centers[i, 0]), float(centers[i, 1]), float(centers[i, 2])
        R = float(radii[i])
        rad = R + band_dist

        ix_lo = max(0, int(np.floor((xc - rad - ox) / vox)))
        ix_hi = min(nx, int(np.ceil((xc + rad - ox) / vox)) + 1)
        iy_lo = max(0, int(np.floor((yc - rad - oy) / vox)))
        iy_hi = min(ny, int(np.ceil((yc + rad - oy) / vox)) + 1)
        iz_lo = max(0, int(np.floor((zc - rad - oz) / vox)))
        iz_hi = min(nz, int(np.ceil((zc + rad - oz) / vox)) + 1)

        if ix_lo >= ix_hi or iy_lo >= iy_hi or iz_lo >= iz_hi:
            continue

        X = (xVec[ix_lo:ix_hi] - xc).astype(np.float32)
        Y = (yVec[iy_lo:iy_hi] - yc).astype(np.float32)
        Z = (zVec[iz_lo:iz_hi] - zc).astype(np.float32)
        # Broadcasted Euclidean distance minus R; no full meshgrid allocation.
        d = np.sqrt(
            X[:, None, None] ** 2
            + Y[None, :, None] ** 2
            + Z[None, None, :] ** 2
        ) - np.float32(R)

        blk = F[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi]
        np.minimum(blk, d, out=blk)

    log.info(
        "build_beads_sdf_icsg: %d spheres → F range [%.4g, %.4g] mm",
        centers.shape[0], float(F.min()), float(F.max()),
    )
    return F


# =====================================================================
# 7. Compose the final field (beads or pore)
# =====================================================================

def compose_field(
    F_beads_raw: NDArray[np.float32],
    F_box: NDArray[np.float32],
    export_what: ExportWhat,
) -> NDArray[np.float32]:
    """Combine beads SDF and box SDF into the final iso-surface field.

    Sign convention is preserved (``F < 0`` inside the exported phase).

    Modes:

    - ``"beads"`` returns ``max(F_beads_raw, F_box)`` — i.e., beads ∩ box.
      Voxels inside any sphere *and* inside the box are negative; outside
      either, positive.

    - ``"pore"`` returns ``max(F_box, -F_beads_raw)`` — i.e., box ∩ ¬beads.
      Voxels inside the box *and* outside every sphere are negative.

    **Note on the MATLAB pore branch.** ``RCPS_v4.m`` lines 487–511
    confine ``F_beads`` in-place via ``F_beads = max(F_beads, F_box)`` and
    *then* in the ``'pore'`` branch returns ``max(F_box, F_beads)``. Since
    the pre-confined ``F_beads`` already satisfies ``F_beads ≥ F_box``
    pointwise, that expression is identical to ``F_beads`` and the pore
    mesh is silently the same as beads. This Python port fixes the bug by
    keeping ``F_beads_raw`` un-confined and composing only at the end —
    see the archived design audit §1 line 483–511 disposition.
    """
    if F_beads_raw.shape != F_box.shape:
        raise ValueError(
            f"shape mismatch: F_beads_raw {F_beads_raw.shape} vs F_box {F_box.shape}"
        )
    if export_what == "beads":
        return np.maximum(F_beads_raw, F_box)
    if export_what == "pore":
        return np.maximum(F_box, -F_beads_raw)
    raise ValueError(f"export_what must be 'beads' or 'pore', got {export_what!r}")
