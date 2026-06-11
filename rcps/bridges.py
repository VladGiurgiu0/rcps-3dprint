"""Sphere-to-sphere bridge geometries for the RCPS pipeline.

Three modes; see the archived design audit §1 and §7 for the locked v1.0 contract.

- ``none`` — no bridges. Spheres remain disconnected. Useful as a
  reference geometry; not printable as a single piece.

- ``cylinders`` — for each contacting pair ``(i, j)`` add a capped
  cylinder of radius ``radius_frac · min(r_i, r_j)`` along the inter-
  center axis, ``min``-combined into the beads SDF. Ports
  ``RCPS_v4.m`` STEP 4b (lines 410–481), preserving the
  ``−0.25 · voxSize`` ICSG band offset.

- ``diameter`` (new in v1.0; not in MATLAB code) — for each sphere ``i``
  set ``r_i_new = r_i + max_{j ∈ contacts(i)}(gap_ij / 2)`` so each
  sphere expands enough to close its *worst* contact halfway.
  Smaller-gap contacts overlap harmlessly (the ICSG SDF merges
  overlapping spheres into a smooth bridge — equivalent to a single
  smooth bicone where the spheres meet).

Contact criterion
-----------------
A pair ``(i, j)`` is "in contact" iff::

    gap_ij = ||c_i - c_j|| - (r_i + r_j) < contact_tol_mm

Neighbour search
----------------
Uses ``scipy.spatial.cKDTree.query_pairs`` when scipy is installed
(O(N log N + K)). Falls back to a NumPy pairwise scan when scipy is
unavailable (O(N²) but adequate for the post-cull sphere counts
encountered in practice: a few thousand).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rcps.field import Grid

log = logging.getLogger(__name__)


BridgeMode = Literal["none", "cylinders", "diameter"]


# =====================================================================
# Contact-pair finder
# =====================================================================

def _query_pairs_within(centers: NDArray[np.float64], max_d: float) -> NDArray[np.int64]:
    """Return ``(K, 2)`` ``int64`` array of ``i < j`` pairs with
    ``||c_i - c_j|| < max_d``.

    Uses ``scipy.spatial.cKDTree.query_pairs`` when scipy is available;
    falls back to a vectorised NumPy pairwise scan otherwise. Block-wise
    scan above 5 000 spheres to keep peak memory in check.
    """
    N = int(centers.shape[0])
    if N < 2 or max_d <= 0:
        return np.zeros((0, 2), dtype=np.int64)

    try:
        from scipy.spatial import cKDTree  # noqa: PLC0415
        tree = cKDTree(centers)
        pairs = tree.query_pairs(float(max_d), output_type="ndarray")
        return pairs.astype(np.int64, copy=False) if pairs.size else np.zeros((0, 2), dtype=np.int64)
    except ImportError:
        return _brute_force_query_pairs(centers, float(max_d))


def _brute_force_query_pairs(centers: NDArray[np.float64], max_d: float) -> NDArray[np.int64]:
    """Pure-NumPy fallback for ``query_pairs``. Returns ``(K, 2)`` int64."""
    N = int(centers.shape[0])
    max_d2 = float(max_d) ** 2

    # Block over rows to bound peak memory at ~block · N · 8 bytes.
    block = 2000 if N > 5000 else N
    out: list[tuple[int, int]] = []
    for i_start in range(0, N, block):
        i_end = min(N, i_start + block)
        ci = centers[i_start:i_end]                  # (b, 3)
        cj = centers[i_start:]                       # (N - i_start, 3)
        d2 = ((ci[:, None, :] - cj[None, :, :]) ** 2).sum(axis=-1)  # (b, N-i_start)
        for k in range(i_end - i_start):
            i = i_start + k
            mask = d2[k] < max_d2
            mask[: (i - i_start) + 1] = False  # enforce j > i (skip i itself and lower j)
            j_local = np.flatnonzero(mask)
            for j_off in j_local:
                out.append((i, int(i_start + j_off)))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(out, dtype=np.int64)


def find_contact_pairs(
    centers: NDArray[np.float64],
    radii: NDArray[np.float64],
    contact_tol_mm: float,
) -> NDArray[np.int64]:
    """Return ``(K, 2)`` int64 array of contact pairs ``(i, j)``, ``i < j``.

    A pair is a "contact" iff ``gap_ij = ||c_i - c_j|| - (r_i + r_j)
    < contact_tol_mm``. Used by both bridge modes.
    """
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if centers.shape[0] != radii.shape[0]:
        raise ValueError(
            f"centers/radii length mismatch: {centers.shape[0]} vs {radii.shape[0]}"
        )
    if centers.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.int64)

    # Search radius: pairs with d_ij ≤ r_i + r_j + tol; bounded above by 2·r_max + tol.
    search_r = 2.0 * float(radii.max()) + float(contact_tol_mm)
    pairs_all = _query_pairs_within(centers, search_r)
    if pairs_all.shape[0] == 0:
        return pairs_all

    diffs = centers[pairs_all[:, 0]] - centers[pairs_all[:, 1]]
    d = np.sqrt((diffs * diffs).sum(axis=1))
    gaps = d - radii[pairs_all[:, 0]] - radii[pairs_all[:, 1]]
    is_contact = gaps < float(contact_tol_mm)
    pairs = pairs_all[is_contact]
    log.info(
        "find_contact_pairs: %d candidates → %d contacts (tol=%.3g mm)",
        pairs_all.shape[0], pairs.shape[0], contact_tol_mm,
    )
    return pairs.astype(np.int64, copy=False)


# =====================================================================
# `diameter` mode — expand sphere radii to close worst contact halfway
# =====================================================================

def expand_to_touch(
    centers: NDArray[np.float64],
    radii: NDArray[np.float64],
    contact_tol_mm: float,
) -> NDArray[np.float64]:
    """Per-sphere radius expansion that closes each sphere's worst contact.

    For each sphere ``i``::

        r_i_new = r_i + max(0, max_{j ∈ contacts(i)} gap_ij / 2)

    where ``contacts(i) = {j : gap_ij < contact_tol}``. The outer
    ``max(0, …)`` ensures that spheres whose contacts are already
    closed (gap ≤ 0 to every neighbour) do not shrink.

    The resulting bridge geometry is generated by the regular
    :func:`rcps.field.build_beads_sdf_icsg` call applied **after**
    ``expand_to_touch`` — overlapping spheres merge cleanly in the SDF.

    Parameters
    ----------
    centers
        ``(N, 3)`` sphere centres in mm.
    radii
        ``(N,)`` original sphere radii in mm.
    contact_tol_mm
        Pair is a "contact" iff gap_ij < contact_tol_mm.

    Returns
    -------
    ndarray of shape ``(N,)``
        New radii, ``≥`` the input radii pointwise.
    """
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)

    pairs = find_contact_pairs(centers, radii, contact_tol_mm)
    if pairs.shape[0] == 0:
        log.info("expand_to_touch: no contacts found, returning radii unchanged")
        return radii.copy()

    diffs = centers[pairs[:, 0]] - centers[pairs[:, 1]]
    d = np.sqrt((diffs * diffs).sum(axis=1))
    gaps = d - radii[pairs[:, 0]] - radii[pairs[:, 1]]
    half_gaps = np.maximum(gaps / 2.0, 0.0)  # only positive gaps grow a sphere

    max_half_gap = np.zeros_like(radii)
    # Each pair credits both endpoints with the same half-gap; the np.maximum.at
    # ufunc accumulates the per-sphere max.
    np.maximum.at(max_half_gap, pairs[:, 0], half_gaps)
    np.maximum.at(max_half_gap, pairs[:, 1], half_gaps)

    new_radii = radii + max_half_gap
    delta = new_radii - radii
    log.info(
        "expand_to_touch: %d contacts; max ∆r=%.4g mm, mean ∆r=%.4g mm",
        pairs.shape[0], float(delta.max()), float(delta.mean()),
    )
    return new_radii


# =====================================================================
# `cylinders` mode — add capped-cylinder bridges to F_beads
# =====================================================================

def add_cylinders(
    F_beads: NDArray[np.float32],
    centers: NDArray[np.float64],
    radii: NDArray[np.float64],
    grid: Grid,
    *,
    contact_tol_mm: float,
    radius_frac: float,
    band_vox: int = 3,
) -> NDArray[np.float32]:
    """Add capped-cylinder bridges between contacting spheres (in place).

    Ports ``RCPS_v4.m`` lines 410–481. The cylinder SDF on a local subgrid
    is ``min``-combined into ``F_beads``, with the same ``−0.25 · voxSize``
    ICSG band offset used by MATLAB to keep the bridge robustly merged
    with the beads at the contact points.

    Parameters
    ----------
    F_beads
        ``(nx, ny, nz)`` ``float32`` beads SDF (in-place modified).
    centers, radii
        Sphere geometry (post-cull, post-keepSides filter).
    grid
        Padded SDF grid.
    contact_tol_mm
        Contact criterion (same as ``find_contact_pairs``).
    radius_frac
        Cylinder radius = ``radius_frac · min(r_i, r_j)``.
    band_vox
        Narrow-band half-width in voxels; ``max(2, band_vox)`` is used.

    Returns
    -------
    The same ``F_beads`` array (modified in place).
    """
    if F_beads.dtype != np.float32:
        raise ValueError(f"F_beads must be float32, got {F_beads.dtype}")
    if F_beads.shape != grid.shape:
        raise ValueError(
            f"F_beads shape {F_beads.shape} does not match grid shape {grid.shape}"
        )
    if not (0.0 < radius_frac <= 1.0):
        raise ValueError(
            f"radius_frac must lie in (0, 1], got {radius_frac}"
        )

    pairs = find_contact_pairs(centers, radii, contact_tol_mm)
    if pairs.shape[0] == 0:
        log.info("add_cylinders: no contact pairs; F_beads unchanged")
        return F_beads

    nx, ny, nz = grid.shape
    vox = float(grid.vox_size)
    ox, oy, oz = grid.origin
    xVec = grid.x_vec()
    yVec = grid.y_vec()
    zVec = grid.z_vec()

    band_vox_eff = max(2, int(band_vox))
    band_dist = band_vox_eff * vox
    rcyl_offset = np.float32(0.25 * vox)

    n_added = 0
    for idx in range(pairs.shape[0]):
        i, j = int(pairs[idx, 0]), int(pairs[idx, 1])
        c1 = centers[i]
        c2 = centers[j]
        r1 = float(radii[i])
        r2 = float(radii[j])

        v = c2 - c1
        Lc = float(np.sqrt((v * v).sum()))
        if Lc <= 0.0:
            continue

        rcyl = float(radius_frac) * min(r1, r2)
        pad = rcyl + band_dist

        xmin = min(c1[0], c2[0]) - pad
        xmax = max(c1[0], c2[0]) + pad
        ymin = min(c1[1], c2[1]) - pad
        ymax = max(c1[1], c2[1]) + pad
        zmin = min(c1[2], c2[2]) - pad
        zmax = max(c1[2], c2[2]) + pad

        ix_lo = max(0, int(np.floor((xmin - ox) / vox)))
        ix_hi = min(nx, int(np.ceil((xmax - ox) / vox)) + 1)
        iy_lo = max(0, int(np.floor((ymin - oy) / vox)))
        iy_hi = min(ny, int(np.ceil((ymax - oy) / vox)) + 1)
        iz_lo = max(0, int(np.floor((zmin - oz) / vox)))
        iz_hi = min(nz, int(np.ceil((zmax - oz) / vox)) + 1)

        if ix_lo >= ix_hi or iy_lo >= iy_hi or iz_lo >= iz_hi:
            continue

        X = xVec[ix_lo:ix_hi].astype(np.float32)
        Y = yVec[iy_lo:iy_hi].astype(np.float32)
        Z = zVec[iz_lo:iz_hi].astype(np.float32)

        u = v / Lc  # unit axis vector, float64
        ux32, uy32, uz32 = np.float32(u[0]), np.float32(u[1]), np.float32(u[2])
        c1x, c1y, c1z = np.float32(c1[0]), np.float32(c1[1]), np.float32(c1[2])

        dx = X[:, None, None] - c1x
        dy = Y[None, :, None] - c1y
        dz = Z[None, None, :] - c1z

        # Projection of (p - c1) onto u, clamped to [0, Lc].
        t = dx * ux32 + dy * uy32 + dz * uz32
        np.clip(t, np.float32(0.0), np.float32(Lc), out=t)

        # Closest point on the segment.
        cxi = c1x + t * ux32
        cyi = c1y + t * uy32
        czi = c1z + t * uz32

        # Euclidean distance from each voxel centre to the closest segment point.
        dist = np.sqrt(
            (X[:, None, None] - cxi) ** 2
            + (Y[None, :, None] - cyi) ** 2
            + (Z[None, None, :] - czi) ** 2
        )
        local = dist - np.float32(rcyl) - rcyl_offset

        blk = F_beads[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi]
        np.minimum(blk, local, out=blk)
        n_added += 1

    log.info("add_cylinders: %d/%d cylinder bridges added", n_added, pairs.shape[0])
    return F_beads
