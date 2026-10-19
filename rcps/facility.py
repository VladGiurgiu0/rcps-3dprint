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

Columns (round pipes)
---------------------
With ``geom.shape == "cylinder"`` the facility is a ``1 × 1 × K`` column
of segments stacked along the cylinder axis (the packing's
``geom.flow_axis``, relabeled to z). Only the end caps interlock, so a
column needs at most three meshes — bottom (flat inlet face), middle,
top (flat outlet face) — whatever K is. The tile of the grid is the
printed *segment* (``geom.segment_size_mm``: packing period ×
``geom.segment_periods`` along z); :func:`column_segments` converts a
column length into K.

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
from rcps.io import load_packing_xyzd, sha256_of_file

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


def column_segments(column_length_mm: float, segment_length_mm: float) -> int:
    """Number of segments K in a column; the length must be a whole multiple."""
    if column_length_mm <= 0 or segment_length_mm <= 0:
        raise ValueError(
            f"lengths must be positive, got column={column_length_mm}, "
            f"segment={segment_length_mm}"
        )
    k = round(column_length_mm / segment_length_mm)
    if k < 1 or abs(k * segment_length_mm - column_length_mm) > 1e-6:
        raise ValueError(
            f"column length {column_length_mm} mm is not a whole multiple of the "
            f"segment length {segment_length_mm} mm"
        )
    return int(k)


_COLUMN_ROLE = {("S",): "bottom", ("I", "S"): "middle", ("I",): "top", (): "single"}

_DEFAULT_GEOMETRY = {"shape": "box", "cylinder_diameter_mm": None,
                     "flow_axis": "z", "segment_periods": 1}


def _plan_signature(amap: dict[str, Any]) -> dict[str, Any]:
    """What makes two assembly maps the same plan (maps written before the
    cylinder feature have no ``geometry`` key: they are box / z / 1 period)."""
    return {
        "grid": [int(x) for x in amap.get("grid", [])],
        "tile_size_mm": [round(float(x), 6) for x in amap.get("tile_size_mm", [])],
        "geometry": amap.get("geometry", _DEFAULT_GEOMETRY),
        "expansion_factor": round(float(amap.get("diameter", {}).get("expansion_factor", 1.0)), 6),
    }


def check_no_other_plan(out_dir: Path, amap: dict[str, Any]) -> None:
    """Refuse to overwrite an assembly map (and its meshes) of a *different* plan.

    Re-running the same plan (e.g. a dry run followed by the real export)
    is allowed; anything else must go to another folder.
    """
    f = Path(out_dir) / "facility_map.json"
    if not f.exists():
        return
    try:
        old = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return
    a, b = _plan_signature(old), _plan_signature(amap)
    if a != b:
        g = a["geometry"]
        what = (f"{g['shape']}, grid {a['grid']}, tile {a['tile_size_mm']} mm, "
                f"flow axis {g['flow_axis']}, expansion x{a['expansion_factor']}")
        raise FileExistsError(
            f"{Path(out_dir).name}/ already holds a different plan ({what}). "
            f"Move or rename that folder first so its map and meshes are not "
            f"overwritten, then export again."
        )


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
# Printed-medium record (what must match between exports of one packing)
# =====================================================================

# Settings that define the printed medium itself (must match for "the same
# medium"), and settings that only set the mesh accuracy (should match).
# Container settings (shape, D, flow axis, segment, grid) are meant to differ
# between e.g. a box facility and a cylinder column and are not listed.
MEDIUM_KEYS: tuple[str, ...] = (
    "packing_sha256", "expansion_factor", "export_what",
    "bridge_mode", "radius_frac", "contact_tol_mm",
)
ACCURACY_KEYS: tuple[str, ...] = (
    "vox_size_mm", "backend", "angbound_deg", "radbound", "distbound",
)


def medium_from_config(
    cfg: dict[str, Any], packing_sha256: str | None = None
) -> dict[str, Any]:
    """Medium + accuracy settings from an ``RcpsConfig.to_dict()`` dict."""
    sph, br, mesh = cfg["spheres"], cfg["bridge"], cfg.get("mesh", {})
    i2m = mesh.get("iso2mesh", {})
    return {
        "packing_sha256": packing_sha256,
        "expansion_factor": round(float(sph["expansion_factor"]), 6),
        "export_what": cfg["field"]["export_what"],
        "bridge_mode": br["mode"],
        "radius_frac": float(br["radius_frac"]),
        "contact_tol_mm": float(sph["contact_tol_mm"]),
        "vox_size_mm": float(cfg["grid"]["vox_size_mm"]),
        "backend": mesh.get("backend"),
        "angbound_deg": i2m.get("angbound_deg"),
        "radbound": i2m.get("radbound"),
        "distbound": i2m.get("distbound"),
    }


def recorded_medium(folder: Path) -> dict[str, Any] | None:
    """Medium settings recorded in an export folder, or ``None``.

    Meshes are the ground truth: their ``*.config.json`` sidecars are read
    first (this also covers folders exported before the map carried a
    ``medium`` block). Without meshes, the ``medium`` block of
    ``facility_map.json`` (a dry-run plan) is used. Returns
    ``{"source": "meshes"|"plan", "medium": {...}, "mixed": bool}``.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return None
    found: list[dict[str, Any]] = []
    for f in sorted(folder.glob("*.config.json")):
        try:
            d = json.loads(f.read_text())
            found.append(medium_from_config(
                d["config"], (d.get("input") or {}).get("packing_sha256")))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    if found:
        mixed = any(m != found[0] for m in found[1:])
        return {"source": "meshes", "medium": found[0], "mixed": mixed}
    fm = folder / "facility_map.json"
    if fm.exists():
        try:
            med = json.loads(fm.read_text()).get("medium")
        except (OSError, json.JSONDecodeError):
            med = None
        if med:
            return {"source": "plan", "medium": med, "mixed": False}
        return {"source": "plan", "medium": None, "mixed": False}
    return None


def compare_medium(here: dict[str, Any], there: dict[str, Any]) -> list[dict[str, Any]]:
    """Differences between two medium records (keys missing on either side,
    or not applicable to the bridge mode, are skipped)."""
    out = []
    for key in MEDIUM_KEYS + ACCURACY_KEYS:
        a, b = here.get(key), there.get(key)
        if a is None or b is None:
            continue
        if key == "radius_frac" and "cylinders" not in (here.get("bridge_mode"),
                                                         there.get("bridge_mode")):
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            same = abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
        else:
            same = a == b
        if not same:
            out.append({"key": key, "here": a, "there": b,
                        "kind": "medium" if key in MEDIUM_KEYS else "accuracy"})
    return out


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
    if config.geom.shape == "cylinder" and (g[0] != 1 or g[1] != 1):
        raise ValueError(
            f"a cylinder facility is a 1x1xK column, got grid {list(g)}"
        )
    base = config.to_dict()
    seg = config.geom.segment_size_mm   # the grid's tile = one printed segment

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
    check_no_other_plan(out_dir, {
        "grid": list(g),
        "tile_size_mm": list(seg),
        "geometry": {
            "shape": config.geom.shape,
            "cylinder_diameter_mm": (config.geom.effective_cylinder_diameter_mm
                                     if config.geom.shape == "cylinder" else None),
            "flow_axis": config.geom.flow_axis,
            "segment_periods": config.geom.segment_periods,
        },
        "diameter": {"expansion_factor": factor},
    })
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
            # .3mf is the default output, but out.save_3mf/save_stl are a
            # per-job choice: record whichever mesh file was actually written.
            mesh_path = written.get("3mf") or written.get("stl")
            if mesh_path is None:  # pragma: no cover — OutConfig forbids this
                raise RuntimeError(
                    f"tile type {tag}: pipeline wrote no mesh file "
                    f"(out.save_3mf and out.save_stl both off?)"
                )
            entry["mesh"] = str(mesh_path.name)
            meshes[tag] = str(mesh_path)
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
        "tile_size_mm": list(seg),
        "facility_size_mm": [g[ax] * seg[ax] for ax in range(3)],
        "packing_period_mm": list(config.geom.tile_size_mm),
        "geometry": {
            "shape": config.geom.shape,
            "cylinder_diameter_mm": (config.geom.effective_cylinder_diameter_mm
                                     if config.geom.shape == "cylinder" else None),
            "flow_axis": config.geom.flow_axis,
            "segment_periods": config.geom.segment_periods,
        },
        "diameter": dia_info,
        "medium": medium_from_config(
            {**base, "spheres": {**base["spheres"], "expansion_factor": factor}},
            packing_sha256=sha256_of_file(config.paths.packing),
        ),
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
        f"tile {seg} mm)",
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
    if config.geom.shape == "cylinder":
        D = config.geom.effective_cylinder_diameter_mm
        lines += [
            "",
            f"Column: {g[2]} segment(s) of {seg[2]:g} mm, D = {D:g} mm, "
            f"total {g[2] * seg[2]:g} mm. Segment axis = packing "
            f"{config.geom.flow_axis.upper()} axis (mesh z).",
            "Order from z = 0 (inlet end) upwards:",
        ]
        for k in range(g[2]):
            tag = tag_by_pos[(0, 0, k)]
            ks = next(tuple(e["keep_sides"]) for e in entries if e["tag"] == tag)
            lines.append(f"  {k + 1:>3d}  z = {k * seg[2]:>7g} mm  "
                         f"{_COLUMN_ROLE.get(ks, '?'):>6s}  {tag}")
    else:
        for k in range(g[2]):
            lines += ["", f"Layer z={k} (rows: y from top={g[1]-1} to 0; "
                          f"cols: x from 0 to {g[0]-1}):"]
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
