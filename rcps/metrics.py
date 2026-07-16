"""Structural diagnostics for sphere packings: is this packing RCP?

Computes the coordinate-based quantities used in the literature to decide
whether an equal-sphere packing is random-close-packed (RCP), plus a
Kozeny-Carman permeability estimate for the printed bed.

Definitions and exact sources
-----------------------------
packing fraction
    ``phi = (4/3) pi (sigma/2)^3 rho`` (monodisperse; inline definition in
    Sec. II A of [1]).  Porosity is ``eps = 1 - phi``.
kissing / coordination number
    ``z = 4 pi rho int_0^{sigma+} g(r) r^2 dr`` — Eq. (1) of [1].
    Operationally ``z = 2 Nc / N``: each of the ``Nc`` contact pairs is
    shared by two spheres.  Mechanical stability (jamming) requires the
    Maxwell isostatic value ``z_c = 2 d_dim = 6`` in 3D (Sec. II A of [1]).
RCP definition used here
    "the densest isostatic jammed packing, i.e., the right-most point on
    the MRJ line" in the (phi, z) plane — Fig. 2 of [1].  Classic
    simulation values: phi_RCP ~ 0.642-0.649 (endnote [103] of [1]);
    Percus-Yevick-based predictions 0.6433-0.6590 (Table 1 of [1]).
radial distribution function
    ``g(r) = g_c(r) + g_BC(r)`` — Eq. (2) of [1]; the contact term is a
    Dirac delta, ``g_c(r) = g0 g(sigma; phi) delta(r - sigma)`` — Eq. (3)
    of [1].  Near contact the continuous part diverges as
    ``g_BC(r) ~ (r - sigma)^(-1/2)`` [2].
Berryman criterion
    RCP occurs at the minimum phi where the median nearest-neighbor
    distance equals the sphere diameter [3].
Kozeny-Carman permeability
    ``k = d^2/(36 kC) * eps^3/(1 - eps)^2`` with the Carman constant
    ``kC = 5`` (so ``36 kC = 180``) — Eq. (3.3) of [4].  For random sphere
    packs the best-fit constant is C = 4.83 +/- 0.06 (vs. Carman's 5) in
    the porosity window 0.27 < eps < 0.38 — Eqs. (2) and (10) of [5].

References
----------
.. [1] Anzivino, Casiulis, Zhang, Moussa, Martiniani, Zaccone,
       J. Chem. Phys. 158, 044901 (2023). doi:10.1063/5.0137111
.. [2] Donev, Torquato, Stillinger, Phys. Rev. E 71, 011105 (2005).
       doi:10.1103/PhysRevE.71.011105
.. [3] Berryman, Phys. Rev. A 27, 1053 (1983).
       doi:10.1103/PhysRevA.27.1053
.. [4] De Paoli, Howland, Verzicco, Lohse, J. Fluid Mech. 987, A1 (2024).
       doi:10.1017/jfm.2024.328
.. [5] Vasseur, Wadsworth, Coumans, Dingwell, Phys. Rev. E 103, 062613
       (2021). doi:10.1103/PhysRevE.103.062613
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

try:  # same lazy-scipy pattern as rcps.bridges
    from scipy.spatial import cKDTree

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - scipy is a hard dep, but be safe
    _HAVE_SCIPY = False

#: Maxwell isostatic coordination number in 3D, z_c = 2*d_dim ([1], Sec. II A).
ISOSTATIC_Z_3D = 6.0

#: phi_RCP from classic simulations (endnote [103] of [1]).
PHI_RCP_SIMULATIONS = (0.642, 0.649)

#: phi_RCP from Percus-Yevick-based theory (Table 1 of [1]).
PHI_RCP_PY_THEORY = (0.6433, 0.6590)

#: Carman constant kC in k = d^2/(36 kC) * eps^3/(1-eps)^2 ([4], Eq. (3.3)).
CARMAN_CONSTANT = 5.0

#: A sphere with fewer contacts than d_dim + 1 = 4 cannot be locally
#: mechanically stable in 3D; such "rattlers" are conventionally excluded
#: when quoting z of the jammed backbone (cf. [1], Sec. II A).
MIN_STABLE_CONTACTS_3D = 4

REFERENCES: dict[str, str] = {
    "packing_fraction": (
        "phi = (4/3) pi (sigma/2)^3 rho; Anzivino et al., J. Chem. Phys. 158, "
        "044901 (2023), Sec. II A. doi:10.1063/5.0137111"
    ),
    "coordination_number": (
        "z = 4 pi rho int_0^{sigma+} g(r) r^2 dr, Eq. (1) of Anzivino et al. "
        "(2023) (= 2 Nc/N); isostatic z_c = 2d = 6 in 3D (Maxwell), Sec. II A."
    ),
    "rcp_definition": (
        "RCP = densest isostatic jammed packing (end of the z = 6 plateau in "
        "the (phi, z) plane); Anzivino et al. (2023), Fig. 2. Classic "
        "simulations: phi_RCP = 0.642-0.649 (endnote [103])."
    ),
    "rdf": (
        "g(r) = g_c(r) + g_BC(r), Eqs. (2)-(3) of Anzivino et al. (2023); "
        "near-contact divergence g_BC ~ (r-sigma)^(-1/2): Donev, Torquato & "
        "Stillinger, Phys. Rev. E 71, 011105 (2005)."
    ),
    "berryman": (
        "median nearest-neighbor distance = sphere diameter at RCP; Berryman, "
        "Phys. Rev. A 27, 1053 (1983)."
    ),
    "kozeny_carman": (
        "k = d^2/(36 kC) * eps^3/(1-eps)^2, kC = 5; De Paoli et al., J. Fluid "
        "Mech. 987, A1 (2024), Eq. (3.3). Validity for random sphere packs "
        "(0.27 < eps < 0.38): Vasseur et al., Phys. Rev. E 103, 062613 (2021), "
        "Eqs. (2), (10)."
    ),
}


# =====================================================================
# pair geometry (periodic minimum image)
# =====================================================================

def _pair_distances(
    centers: NDArray[np.float64],
    box_mm: NDArray[np.float64],
    r_max: float,
    periodic: bool = True,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """All pairs (i < j) with center distance < ``r_max``.

    Uses a periodic cKDTree when scipy is available, otherwise an O(N^2)
    minimum-image fallback.  Returns ``(i, j, r)``.
    """
    c = np.asarray(centers, dtype=np.float64)
    box = np.asarray(box_mm, dtype=np.float64)
    n = c.shape[0]

    if _HAVE_SCIPY:
        if periodic:
            tree = cKDTree(np.mod(c, box), boxsize=box)
        else:
            tree = cKDTree(c)
        pairs = tree.query_pairs(r_max, output_type="ndarray")
        if pairs.size == 0:
            empty = np.empty(0)
            return empty.astype(np.int64), empty.astype(np.int64), empty
        i, j = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
        dv = c[i] - c[j]
        if periodic:
            dv -= box * np.round(dv / box)
        return i, j, np.sqrt(np.sum(dv * dv, axis=1))

    # numpy fallback
    iu, ju = np.triu_indices(n, k=1)
    dv = c[iu] - c[ju]
    if periodic:
        dv -= box * np.round(dv / box)
    r = np.sqrt(np.sum(dv * dv, axis=1))
    keep = r < r_max
    return iu[keep].astype(np.int64), ju[keep].astype(np.int64), r[keep]


def _nearest_neighbor_distances(
    centers: NDArray[np.float64],
    box_mm: NDArray[np.float64],
    periodic: bool = True,
) -> NDArray[np.float64]:
    """Distance from every sphere to its nearest neighbor (min-image)."""
    c = np.asarray(centers, dtype=np.float64)
    box = np.asarray(box_mm, dtype=np.float64)
    if _HAVE_SCIPY:
        tree = cKDTree(np.mod(c, box), boxsize=box) if periodic else cKDTree(c)
        dist, _ = tree.query(np.mod(c, box) if periodic else c, k=2)
        return np.asarray(dist[:, 1], dtype=np.float64)
    dv = c[:, None, :] - c[None, :, :]
    if periodic:
        dv -= box * np.round(dv / box)
    r = np.sqrt(np.sum(dv * dv, axis=2))
    np.fill_diagonal(r, np.inf)
    return r.min(axis=1)


# =====================================================================
# individual metrics
# =====================================================================

def packing_fraction(
    diameters_mm: NDArray[np.float64], box_mm: NDArray[np.float64] | list[float]
) -> float:
    """Global packing fraction phi = sum(pi/6 d^3) / V_box.

    Monodisperse definition phi = (4/3) pi (sigma/2)^3 rho, Sec. II A of
    Anzivino et al. (2023) [1].  Porosity (GUI convention) is 1 - phi.
    Assumes the box is a periodic tile so every sphere volume counts once.
    """
    d = np.asarray(diameters_mm, dtype=np.float64)
    v_box = float(np.prod(np.asarray(box_mm, dtype=np.float64)))
    return float(np.sum(np.pi / 6.0 * d**3) / v_box)


def coordination_number(
    centers_mm: NDArray[np.float64],
    diameters_mm: NDArray[np.float64],
    box_mm: NDArray[np.float64] | list[float],
    *,
    tol_rel: float = 1e-4,
    periodic: bool = True,
) -> dict[str, Any]:
    """Mean kissing (coordination) number z and rattler statistics.

    A pair is in contact when ``r_ij < (R_i + R_j)(1 + tol_rel)``.  Then
    ``z = 2 Nc / N`` (equivalent to Eq. (1) of Anzivino et al. (2023) [1]).
    Rattlers (< 4 contacts, not locally stable in 3D) are excluded from
    ``z_no_rattlers``, the value to compare with the isostatic z_c = 6.
    """
    c = np.asarray(centers_mm, dtype=np.float64)
    d = np.asarray(diameters_mm, dtype=np.float64)
    box = np.asarray(box_mm, dtype=np.float64)
    n = c.shape[0]

    r_max = float(d.max()) * (1.0 + tol_rel) * 1.0001
    i, j, r = _pair_distances(c, box, r_max, periodic)
    touch = r < 0.5 * (d[i] + d[j]) * (1.0 + tol_rel)
    i, j = i[touch], j[touch]

    z = np.bincount(i, minlength=n) + np.bincount(j, minlength=n)
    rattler = z < MIN_STABLE_CONTACTS_3D
    backbone = z[~rattler]
    return {
        "n_contacts": int(i.size),
        "z_mean": float(z.mean()),
        "z_no_rattlers": float(backbone.mean()) if backbone.size else float("nan"),
        "n_rattlers": int(rattler.sum()),
        "rattler_fraction": float(rattler.mean()),
        "tol_rel": float(tol_rel),
        "isostatic_z": ISOSTATIC_Z_3D,
    }


def rdf(
    centers_mm: NDArray[np.float64],
    diameters_mm: NDArray[np.float64],
    box_mm: NDArray[np.float64] | list[float],
    *,
    r_max_over_d: float = 2.5,
    bin_width_over_d: float = 0.02,
    periodic: bool = True,
) -> dict[str, Any]:
    """Radial distribution function g(r) on bins of r / mean(d).

    ``g(r) = counts / (N/2 * rho * 4 pi r^2 dr)`` with pairs counted once.
    In a jammed packing g(r) splits into a contact Dirac delta plus a
    continuous part, Eqs. (2)-(3) of Anzivino et al. (2023) [1]; RCP
    signatures are the strong contact peak and the split second peak at
    r = sqrt(3) d and 2 d.
    """
    c = np.asarray(centers_mm, dtype=np.float64)
    dmean = float(np.mean(np.asarray(diameters_mm, dtype=np.float64)))
    box = np.asarray(box_mm, dtype=np.float64)
    n = c.shape[0]
    v_box = float(np.prod(box))
    rho = n / v_box

    edges = np.arange(0.8, r_max_over_d + bin_width_over_d, bin_width_over_d) * dmean
    _, _, r = _pair_distances(c, box, float(edges[-1]), periodic)
    counts, _ = np.histogram(r, bins=edges)
    rc = 0.5 * (edges[1:] + edges[:-1])
    shell = 4.0 * np.pi * rc**2 * np.diff(edges)
    g = counts / (0.5 * n * rho * shell)
    return {
        "r_over_d": np.round(rc / dmean, 4).tolist(),
        "g": np.round(g, 4).tolist(),
        "bin_width_over_d": float(bin_width_over_d),
    }


def berryman_ratio(
    centers_mm: NDArray[np.float64],
    diameters_mm: NDArray[np.float64],
    box_mm: NDArray[np.float64] | list[float],
    *,
    periodic: bool = True,
) -> float:
    """Median nearest-neighbor distance over mean diameter (= 1 at RCP [3])."""
    nn = _nearest_neighbor_distances(
        np.asarray(centers_mm, dtype=np.float64),
        np.asarray(box_mm, dtype=np.float64),
        periodic,
    )
    return float(np.median(nn) / np.mean(np.asarray(diameters_mm, dtype=np.float64)))


def kozeny_carman(porosity: float, diameter_mm: float, k_c: float = CARMAN_CONSTANT) -> float:
    """Kozeny-Carman permeability k [m^2] of a monodisperse sphere pack.

    ``k = d^2 / (36 kC) * eps^3 / (1 - eps)^2`` — Eq. (3.3) of De Paoli et
    al., J. Fluid Mech. 987, A1 (2024) [4], with the Carman constant
    kC = 5 (36 kC = 180).  Creeping (Darcy) flow only.  For random sphere
    packs the form is accurate for 0.27 < eps < 0.38 (best fit C = 4.83
    +/- 0.06): Vasseur et al., Phys. Rev. E 103, 062613 (2021) [5].

    Parameters
    ----------
    porosity : void fraction eps = 1 - phi (NOT the packing fraction).
    diameter_mm : sphere (grain) diameter in mm.
    k_c : Carman constant (default 5).
    """
    if not 0.0 < porosity < 1.0:
        raise ValueError(f"porosity must be in (0, 1), got {porosity}")
    d_m = float(diameter_mm) * 1e-3
    eps = float(porosity)
    return d_m**2 / (36.0 * k_c) * eps**3 / (1.0 - eps) ** 2


# =====================================================================
# assembled report
# =====================================================================

def rcp_metrics(
    centers_mm: NDArray[np.float64],
    diameters_mm: NDArray[np.float64],
    box_mm: NDArray[np.float64] | list[float],
    *,
    d_nominal_mm: float | None = None,
    tol_rel: float = 1e-4,
    periodic: bool = True,
    include_rdf: bool = True,
) -> dict[str, Any]:
    """Full RCP diagnostic report for one packing (JSON-serializable).

    All structural quantities use the STORED (true jammed hard-sphere)
    diameters.  If ``d_nominal_mm`` is given (the printed/nominal sphere
    size, e.g. 6.0 mm), the Kozeny-Carman estimate is also evaluated for
    the printed bed at its nominal porosity.

    See module docstring for the equations and their exact sources.
    """
    c = np.asarray(centers_mm, dtype=np.float64)
    d = np.asarray(diameters_mm, dtype=np.float64)
    box = np.asarray(box_mm, dtype=np.float64)

    phi = packing_fraction(d, box)
    eps = 1.0 - phi
    coord = coordination_number(c, d, box, tol_rel=tol_rel, periodic=periodic)
    berry = berryman_ratio(c, d, box, periodic=periodic)
    d_mean = float(np.mean(d))

    kc: dict[str, Any] = {
        "carman_constant": CARMAN_CONSTANT,
        "k_m2_stored_d": kozeny_carman(eps, d_mean),
        "d_stored_mm": d_mean,
        "porosity_stored": eps,
    }
    if d_nominal_mm is not None:
        v_box = float(np.prod(box))
        phi_nom = len(d) * np.pi / 6.0 * float(d_nominal_mm) ** 3 / v_box
        kc.update(
            {
                "k_m2_nominal_d": kozeny_carman(1.0 - phi_nom, float(d_nominal_mm)),
                "d_nominal_mm": float(d_nominal_mm),
                "porosity_nominal": 1.0 - phi_nom,
            }
        )

    z_bb = coord["z_no_rattlers"]
    checklist = {
        "phi_in_rcp_window": bool(0.630 <= phi <= 0.660),
        "isostatic": bool(abs(z_bb - ISOSTATIC_Z_3D) < 0.25),
        "rattler_fraction_ok": bool(coord["rattler_fraction"] < 0.05),
        "berryman": bool(abs(berry - 1.0) < 5e-3),
    }

    out: dict[str, Any] = {
        "n_spheres": int(c.shape[0]),
        "box_mm": [float(b) for b in box],
        "mean_diameter_mm": d_mean,
        "packing_fraction": phi,
        "porosity": eps,
        "coordination": coord,
        "berryman_median_nn_over_d": berry,
        "kozeny_carman": kc,
        "phi_rcp_simulations": list(PHI_RCP_SIMULATIONS),
        "phi_rcp_py_theory": list(PHI_RCP_PY_THEORY),
        "rcp_checklist": checklist,
        "is_rcp_consistent": bool(all(checklist.values())),
        "references": REFERENCES,
    }
    if include_rdf:
        out["rdf"] = rdf(c, d, box, periodic=periodic)
    return out
