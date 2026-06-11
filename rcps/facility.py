"""Multi-tile facility orchestrator (``rcps-facility``).

Generates the set of interlocking tile meshes for an ``Nx × Ny × Nz``
facility built from a single periodic packing tile.

How interlocking works
----------------------
Every *shared* (interior) face of the facility is "kept" on both of the
tiles adjacent to it: neither tile cuts its beads at that plane, and each
tile contains exactly the spheres whose centers it owns (half-open
ownership, see :func:`rcps.field.apply_keepsides_filter`). Because the
packing is periodic with the tile period, the protrusions of one tile fit
the cavities of its neighbour exactly. *Exterior* faces are cut flush
(with periodic ghost caps included), which preserves the bulk porosity at
the facility walls.

Tile types
----------
Tiles are distinguished only by *which* faces are kept, which depends on
grid position. A ``4 × 4 × 1`` facility therefore needs only 9 distinct
meshes (4 corner types, 4 edge types, 1 interior type) for its 16 tiles.
The orchestrator runs the pipeline once per unique type and writes an
assembly map saying which mesh goes at which grid position.

Diameter convention
-------------------
(Audited 2026-06-11 against ``notebooks/packing_generation.ipynb``.) For
``data_example`` the stored diameters (5.9598 mm) are the TRUE jammed
diameters: the notebook already applied Baranau's issue-30 rescale and
then set the ``.nfo``'s Final Porosity to the *requested* value. The
generator was asked for φ = 0.3504 (d = 6.0) but jammed at φ = 0.3633 —
a Δφ that shifts Kozeny–Carman permeability by ≈16 %, so the printed
convention is an explicit, per-job choice:

- ``"stored"`` — print the true jammed packing exactly (tangent
  contacts, φ = 0.3633 for data_example; factor 1.0);
- ``"design"`` — inflate diameters so the realized porosity equals the
  ``.nfo``'s value (d = 6.000, φ = 0.3504 for data_example; creates
  ≈40 µm overlaps at contacts — stronger printed bonds, but no longer
  the as-jammed geometry);
- a float — use that expansion factor directly.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rcps._version import __version__
from rcps.config import RcpsConfig
from rcps.io import load_packing_xyzd

log = logging.getLogger(__name__)

# RAS face labels per axis: lower face (at coordinate 0) / upper face (at L).
_LOWER = {0: "L", 1: "P", 2: "I"}
_UPPER = {0: "R", 1: "A", 2: "S"}


# =====================================================================
# Keep-sides derivation
# =====================================================================

def _validate_grid(grid: Sequence[int]) -> tuple[int, int, int]:
    g = tuple(int(x) for x in grid)
    if len(g) != 3 or any(x < 1 for x in g):
        raise ValueError(f"grid must be three integers >= 1, got {grid!r}")
    return g


def derive_keep_sides(
    pos: Sequence[int], grid: Sequence[int]
) -> tuple[str, ...]:
    """Keep-sides for the tile at grid position ``pos`` (0-based).

    A face is kept iff a neighbouring tile exists across it (interior,
    shared face); exterior faces are cut flush. Order of the returned
    labels is deterministic: per axis x, y, z — lower then upper.
    """
    g = _validate_grid(grid)
    p = tuple(int(x) for x in pos)
    if len(p) != 3 or any(not (0 <= p[ax] < g[ax]) for ax in range(3)):
        raise ValueError(f"pos {pos!r} outside grid {grid!r}")

    ks: list[str] = []
    for ax in range(3):
        if p[ax] > 0:
            ks.append(_LOWER[ax])
        if p[ax] < g[ax] - 1:
            ks.append(_UPPER[ax])
    return tuple(ks)


def unique_tile_types(
    grid: Sequence[int],
) -> dict[tuple[str, ...], list[tuple[int, int, int]]]:
    """Map each unique keep-sides set to its list of grid positions.

    Deterministic iteration order (z-major, then y, then x). For an
    ``N×M×1`` facility with N, M ≥ 3 this yields 9 types; ``2×2×1``
    yields 4; ``1×1×1`` yields 1 (all faces flush).
    """
    g = _validate_grid(grid)
    types: dict[tuple[str, ...], list[tuple[int, int, int]]] = {}
    for k in range(g[2]):
        for j in range(g[1]):
            for i in range(g[0]):
                ks = derive_keep_sides((i, j, k), g)
                types.setdefault(ks, []).append((i, j, k))
    return types


def type_tag(keep_sides: tuple[str, ...]) -> str:
    """Short filesystem-safe tag for a keep-sides set."""
    return "flush" if not keep_sides else "keep" + "".join(keep_sides)


# =====================================================================
# Diameter convention
# =====================================================================

def parse_nfo_porosity(nfo_path: Path, which: str = "theoretical") -> float:
    """Extract a porosity from a PackingGeneration ``.nfo`` file.

    ``which='theoretical'`` is the porosity at the *nominal* (requested)
    diameters — the right target for ``design`` mode. ``which='final'``
    is the porosity the generator reports as achieved (beware: legacy
    issue-30 postprocessing scripts overwrite this line).
    """
    label = {"theoretical": "Theoretical", "final": "Final"}[which]
    text = Path(nfo_path).read_text(errors="ignore")
    m = re.search(label + r"\s+Porosity:\s*([0-9.eE+-]+)", text)
    if not m:
        raise ValueError(f"no '{label} Porosity' line found in {nfo_path}")
    phi = float(m.group(1))
    if not (0.0 < phi < 1.0):
        raise ValueError(f"parsed porosity {phi} from {nfo_path} is not in (0, 1)")
    return phi


def parse_nfo_final_porosity(nfo_path: Path) -> float:
    """Back-compat wrapper; see :func:`parse_nfo_porosity`."""
    return parse_nfo_porosity(nfo_path, "final")


def stored_porosity(
    packing: np.ndarray, tile_size_mm: Sequence[float]
) -> float:
    """Porosity of the periodic tile at the diameters stored in the file."""
    v_solid = float(np.sum(np.pi / 6.0 * packing[:, 3] ** 3))
    v_box = float(np.prod([float(x) for x in tile_size_mm]))
    return 1.0 - v_solid / v_box

def design_expansion_factor(
    packing: np.ndarray,
    tile_size_mm: Sequence[float],
    phi_target: float,
) -> float:
    """Uniform diameter scale so the periodic tile realizes ``phi_target``.

    ``factor = ((1 − φ_target) · V_box / Σ (π/6) d_i³)^(1/3)`` — exact for
    a periodic tile as long as overlap volumes stay negligible (tangent
    contacts), which holds for the small corrections this is meant for.
    """
    v_solid = float(np.sum(np.pi / 6.0 * packing[:, 3] ** 3))
    v_box = float(np.prod([float(x) for x in tile_size_mm]))
    if not (0.0 < phi_target < 1.0):
        raise ValueError(f"phi_target must be in (0, 1), got {phi_target}")
    return float(((1.0 - phi_target) * v_box / v_solid) ** (1.0 / 3.0))


def resolve_expansion_factor(
    diameter: str | float,
    packing_path: Path,
    tile_size_mm: Sequence[float],
) -> tuple[float, dict[str, Any]]:
    """Turn the per-job diameter choice into a concrete expansion factor.

    Returns ``(factor, info)`` where ``info`` documents the choice for the
    assembly map (mode, φ at stored diameters, φ realized, factor).
    """
    packing = load_packing_xyzd(packing_path)
    phi_stored = stored_porosity(packing, tile_size_mm)

    if isinstance(diameter, str) and diameter == "stored":
        factor = 1.0
        phi_real = phi_stored
    elif isinstance(diameter, str) and diameter == "design":
        # target = porosity at the NOMINAL (requested) diameters. Prefer
        # the GUI's packing_meta.json (honest provenance); fall back to
        # the .nfo's Theoretical Porosity line. For data_example both
        # give 0.3504 (its .nfo was equalized by the legacy notebook).
        phi_target = None
        meta = Path(packing_path).parent / "packing_meta.json"
        if meta.exists():
            try:
                phi_target = json.loads(meta.read_text()).get(
                    "phi_requested_at_nominal_d")
            except (json.JSONDecodeError, OSError):
                phi_target = None
        if phi_target is None:
            nfo = Path(packing_path).with_suffix(".nfo")
            if not nfo.exists():
                raise FileNotFoundError(
                    f"--diameter design needs packing_meta.json or the "
                    f"packing .nfo next to the xyzd (looked for {nfo}); "
                    f"use --diameter stored or give a numeric factor."
                )
            phi_target = parse_nfo_porosity(nfo, "theoretical")
        phi_real = float(phi_target)
        factor = design_expansion_factor(packing, tile_size_mm, phi_real)
    else:
        factor = float(diameter)
        if factor <= 0:
            raise ValueError(f"expansion factor must be positive, got {factor}")
        # realized porosity for a uniform rescale of all diameters:
        phi_real = 1.0 - (1.0 - phi_stored) * factor**3

    info = {
        "mode": diameter if isinstance(diameter, str) else "explicit",
        "expansion_factor": factor,
        "phi_at_stored_diameters": round(phi_stored, 6),
        "phi_realized": round(phi_real, 6),
        "mean_stored_diameter_mm": round(float(np.mean(packing[:, 3])), 6),
        "mean_printed_diameter_mm": round(float(np.mean(packing[:, 3])) * factor, 6),
    }
    return factor, info


# =====================================================================
# Orchestrator
# =====================================================================

def run_facility(
    config: RcpsConfig,
    grid: Sequence[int],
    *,
    diameter: str | float = "stored",
    out_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate all unique tile meshes + assembly map for an N×M×K facility.

    Parameters
    ----------
    config
        Base single-tile configuration. ``field.keep_sides``,
        ``spheres.expansion_factor``, ``out.base_name`` and
        ``paths.out_dir`` are overridden per tile type; everything else
        (voxel size, bridges, meshing backend, …) applies to every tile.
    grid
        ``(Nx, Ny, Nz)`` tiles per axis.
    diameter
        ``"stored"`` | ``"design"`` | explicit float factor. See module
        docstring.
    out_dir
        Output directory (default: ``<config out_dir>/facility_NxMxK``).
    dry_run
        If True, write the assembly map and print the plan without
        running the meshing pipeline (useful: each unique type at
        production resolution is an ~80 min run).

    Returns
    -------
    dict with keys ``map_json``, ``map_txt`` (paths), ``meshes``
    (tag → path, empty for dry runs), ``expansion_factor``, ``n_types``,
    ``n_tiles``.
    """
    g = _validate_grid(grid)
    base = config.to_dict()

    factor, dia_info = resolve_expansion_factor(
        diameter, Path(config.paths.packing), config.geom.tile_size_mm,
    )
    if config.spheres.expansion_factor != 1.0 and factor != config.spheres.expansion_factor:
        log.warning(
            "config expansion_factor=%.6g is overridden by --diameter -> %.6g",
            config.spheres.expansion_factor, factor,
        )

    types = unique_tile_types(g)
    n_tiles = g[0] * g[1] * g[2]

    if out_dir is None:
        out_dir = Path(config.paths.out_dir) / f"facility_{g[0]}x{g[1]}x{g[2]}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "facility %dx%dx%d: %d tiles, %d unique types, expansion %.6g (phi %.4f)",
        *g, n_tiles, len(types), factor, dia_info["phi_realized"],
    )

    # ---- run the pipeline once per unique type --------------------------
    meshes: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for ks, positions in types.items():
        tag = type_tag(ks)
        entry: dict[str, Any] = {
            "tag": tag,
            "keep_sides": list(ks),
            "copies": len(positions),
            "positions": [list(p) for p in positions],
            "mesh": None,
        }
        if not dry_run:
            from rcps.pipeline import run  # heavy deps load lazily

            d = json.loads(json.dumps(base))  # deep copy
            d["field"]["keep_sides"] = list(ks)
            d["spheres"]["expansion_factor"] = factor
            d["out"]["base_name"] = f"{base['out']['base_name']}_{tag}"
            d["paths"]["out_dir"] = str(out_dir)
            cfg_t = RcpsConfig.from_dict(d)
            log.info("tile type %s (%d copies): running pipeline ...", tag, len(positions))
            written = run(cfg_t)
            entry["mesh"] = str(written["3mf"].name)
            meshes[tag] = str(written["3mf"])
        entries.append(entry)

    # ---- assembly map ----------------------------------------------------
    tag_by_pos: dict[tuple[int, int, int], str] = {}
    for e in entries:
        for p in e["positions"]:
            tag_by_pos[tuple(p)] = e["tag"]

    amap = {
        "generator": f"rcps-facility {__version__}",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grid": list(g),
        "tile_size_mm": list(config.geom.tile_size_mm),
        "facility_size_mm": [g[ax] * config.geom.tile_size_mm[ax] for ax in range(3)],
        "diameter": dia_info,
        "dry_run": dry_run,
        "n_tiles": n_tiles,
        "n_unique_types": len(types),
        "tile_types": entries,
    }
    map_json = out_dir / "facility_map.json"
    map_json.write_text(json.dumps(amap, indent=2))

    # human-readable layout, one block per z-layer, y rows top-down
    lines = [
        f"rcps-facility assembly map ({g[0]}x{g[1]}x{g[2]} tiles, "
        f"tile {config.geom.tile_size_mm} mm)",
        f"diameter mode: {dia_info['mode']} -> factor {factor:.6g}, "
        f"printed d = {dia_info['mean_printed_diameter_mm']:.4f} mm, "
        f"phi = {dia_info['phi_realized']:.4f}",
        "",
        "Tile types (print `copies` of each mesh):",
    ]
    for e in entries:
        mesh = e["mesh"] or "(dry run - not generated)"
        lines.append(
            f"  {e['tag']:>14s}  x{e['copies']:<3d} keep={','.join(e['keep_sides']) or '-':<14s} {mesh}"
        )
    for k in range(g[2]):
        lines += ["", f"Layer z={k} (rows: y from top={g[1]-1} to 0; cols: x from 0 to {g[0]-1}):"]
        for j in reversed(range(g[1])):
            lines.append("  " + " | ".join(
                f"{tag_by_pos[(i, j, k)]:>14s}" for i in range(g[0])
            ))
    lines += [
        "",
        "Assembly rule: place each tile at position (i,j,k)*tile_size; kept",
        "faces interlock with the neighbour; exterior faces are flush.",
    ]
    map_txt = out_dir / "facility_map.txt"
    map_txt.write_text("\n".join(lines) + "\n")

    log.info("assembly map written: %s", map_json)
    return {
        "map_json": map_json,
        "map_txt": map_txt,
        "meshes": meshes,
        "expansion_factor": factor,
        "n_types": len(types),
        "n_tiles": n_tiles,
    }
