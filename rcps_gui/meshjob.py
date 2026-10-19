"""Meshing + export jobs for the GUI (thin layer over the rcps API)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from rcps_gui.jobs import Job

# Export sub-folder of a run, per printed shape. Runs exported before this
# split keep their legacy ``export/`` folder (read for status only).
EXPORT_DIRS = {"box": "export_box", "cylinder": "export_cylindrical"}


def _have(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def _backend(prefer: str = "skimage") -> str:
    """Resolve the 'auto' backend choice.

    Previews prefer skimage (fast, robust marching cubes); production
    exports prefer iso2mesh (the validated CGAL backend). An explicit
    user choice bypasses this entirely.
    """
    order = ["skimage", "iso2mesh"] if prefer == "skimage" else ["iso2mesh", "skimage"]
    for name in order:
        if _have("skimage" if name == "skimage" else "iso2mesh.core"):
            return name
    raise ImportError(
        "neither scikit-image nor iso2mesh is installed - "
        "pip install scikit-image (preview) and iso2mesh (production)."
    )


def _out_dict(p: dict[str, Any]) -> dict[str, Any]:
    """Resolve the Export-stage file-format choice into ``out.save_*`` flags.

    ``export_format`` is ``"3mf"`` (default, unchanged behaviour), ``"stl"``
    or ``"both"``. Anything unrecognised falls back to .3mf so a stale saved
    state can never produce a run with no mesh output.
    """
    fmt = str(p.get("export_format", "3mf")).lower()
    save_3mf = fmt in ("3mf", "both")
    save_stl = fmt in ("stl", "both")
    if not (save_3mf or save_stl):
        save_3mf = True
    return {
        "base_name": p.get("base_name", "rcps_gui"),
        "save_3mf": save_3mf,
        "save_stl": save_stl,
    }


def _config_dict(packing: Path, out_dir: Path, p: dict[str, Any],
                 *, prefer_backend: str = "skimage") -> dict[str, Any]:
    box = [float(x) for x in p.get("box_mm", [50.0, 50.0, 50.0])]
    backend = p.get("backend") or "auto"
    if backend == "auto":
        backend = _backend(prefer_backend)
    mesh: dict[str, Any] = {"backend": backend}
    if p.get("iso2mesh"):
        i2m = p["iso2mesh"]
        mesh["iso2mesh"] = {
            "angbound_deg": float(i2m.get("angbound_deg", 25.0)),
            "radbound": float(i2m.get("radbound", 1.0)),
            "distbound": float(i2m.get("distbound", 0.10)),
            "maxnode": int(float(i2m.get("maxnode", 2e8))),
        }
    return {
        "paths": {
            "packing": str(packing),
            "root": str(packing.parent),
            "out_dir": str(out_dir),
        },
        "geom": {
            "tile_size_mm": box,
            "shape": p.get("shape") or "box",
            "cylinder_diameter_mm": p.get("cylinder_diameter_mm") or None,
            "flow_axis": p.get("flow_axis") or "z",
            "segment_periods": int(p.get("segment_periods") or 1),
        },
        "spheres": {
            "diameter_mm": float(p.get("d_nominal_mm", 6.0)),
            "expansion_factor": float(p.get("expansion_factor", 1.0)),
            "contact_tol_mm": float(p.get("contact_tol_mm", 0.2)),
        },
        "grid": {"vox_size_mm": float(p.get("vox_mm", 1.0))},
        "field": {
            "export_what": p.get("export_what", "beads"),
            "ghost_tiles": 1, "pad_vox": 1, "band_vox": 3,
            "keep_sides": list(p.get("keep_sides", [])),
        },
        "bridge": {
            "mode": p.get("bridge_mode", "cylinders"),
            "radius_frac": float(p.get("radius_frac", 0.15)),
        },
        "mesh": mesh,
        "out": _out_dict(p),
    }


def resolve_expansion(diameter: str | float, packing: Path,
                      box: list[float]) -> tuple[float, dict[str, Any]]:
    from rcps.facility import resolve_expansion_factor
    return resolve_expansion_factor(diameter, packing, box)


def _contact_stats(packing: Path, p: dict[str, Any], factor: float,
                   job: Job) -> dict[str, Any] | None:
    """Contacts at the chosen tolerance, on the periodic tile.

    Same criterion as the bridge builder (gap < contact_tol_mm) with the
    printed radii, so it reports what will actually be bridged and how many
    spheres stay loose. Purely diagnostic — never allowed to break meshing.
    """
    try:
        from rcps.io import load_packing_xyzd
        from rcps.metrics import contact_stats

        S = load_packing_xyzd(packing)
        cs = contact_stats(
            S[:, :3], S[:, 3],
            [float(x) for x in p.get("box_mm", [50.0, 50.0, 50.0])],
            contact_tol_mm=float(p.get("contact_tol_mm", 0.2)),
            expansion_factor=float(factor),
        )
        job.log(f"contacts at tol={cs['contact_tol_mm']} mm: {cs['n_contacts']} "
                f"pairs, {cs['contacts_per_sphere']:.2f} per sphere; "
                f"{cs['n_isolated']} loose, {cs['n_under_coordinated']} with "
                f"< {cs['min_stable_contacts']} contacts (of {cs['n_spheres']})")
        return cs
    except Exception as e:  # noqa: BLE001 — diagnostics must not break meshing
        job.log(f"contact statistics unavailable: {e}")
        return None


def preview_job(job: Job, packing: str | Path, out_dir: str | Path,
                params: dict[str, Any]) -> None:
    """Coarse-vox meshing run; stores a binary triangle buffer for WebGL.

    Buffer layout (little-endian): ``u32 nV | u32 nF | f32 V[3*nV] | u32 F[3*nF]``.
    """
    from rcps.config import RcpsConfig
    from rcps.mesh import diagnose
    from rcps.pipeline import run

    packing = Path(packing)
    out = Path(out_dir) / "preview"
    p = dict(params)
    p.setdefault("vox_mm", 1.0)
    p["base_name"] = "preview"
    p["export_format"] = "3mf"     # the preview reads back written["3mf"]

    factor, dia_info = resolve_expansion(p.get("diameter", "stored"), packing,
                                         [float(x) for x in p.get("box_mm", [50, 50, 50])])
    p["expansion_factor"] = factor
    job.log(f"diameter mode {dia_info['mode']}: factor {factor:.6g}, "
            f"phi_realized {dia_info['phi_realized']}")
    contacts = _contact_stats(packing, p, factor, job)

    cfg = RcpsConfig.from_dict(_config_dict(packing, out, p, prefer_backend="skimage"))
    job.log(f"meshing preview at vox={p['vox_mm']} mm ({cfg.mesh.backend}) ...")
    written = run(cfg)

    import trimesh
    mesh = trimesh.load(str(written["3mf"]), force="mesh")
    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.faces, dtype=np.uint32)
    stats = diagnose(np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int64))

    job.result_bytes = (
        struct.pack("<II", V.shape[0], F.shape[0])
        + V.tobytes() + F.tobytes()
    )
    v_bulk = _bulk_volume(cfg)
    job.result = {
        "n_vertices": int(V.shape[0]),
        "n_faces": int(F.shape[0]),
        "watertight": bool(stats.watertight),
        "porosity": round(1.0 - abs(float(mesh.volume)) / v_bulk, 6),
        # specific surface a_s = A_solid / V_bulk — the Kozeny–Carman input
        "surface_area_mm2": round(float(stats.surface_area_mm2), 1),
        "specific_surface_per_mm": round(float(stats.surface_area_mm2) / v_bulk, 4),
        "diameter": dia_info,
        "contacts": contacts,
        "vox_mm": p["vox_mm"],
        "mesh_path": str(written["3mf"]),
        "shape": cfg.geom.shape,
        "segment_size_mm": list(cfg.geom.segment_size_mm),
        "cylinder_diameter_mm": (cfg.geom.effective_cylinder_diameter_mm
                                 if cfg.geom.shape == "cylinder" else None),
    }


def _bulk_volume(cfg: Any) -> float:
    """Bulk (container) volume of one printed segment, mm³ — porosity basis."""
    seg = cfg.geom.segment_size_mm
    if cfg.geom.shape == "cylinder":
        D = cfg.geom.effective_cylinder_diameter_mm
        return float(np.pi / 4.0 * D * D * seg[2])
    return float(np.prod(seg))


def sdf_job(job: Job, packing: str | Path, params: dict[str, Any]) -> None:
    """Compute the signed-distance field WITHOUT meshing (for the slicer).

    Mirrors pipeline steps 1–6 (load → ghosts → grid → cull → beads SDF →
    bridges → box → compose) and keeps the float32 volume on the job for
    the slice endpoint. Respects the Mesh-stage settings: printed-diameter
    choice, bridges, beads/pore.
    """
    from rcps.bridges import add_cylinders, expand_to_touch
    from rcps.config import GeomConfig
    from rcps.field import (
        apply_keepsides_filter,
        build_beads_sdf_icsg,
        build_box_sdf,
        build_cylinder_sdf,
        compose_field,
        cull_spheres,
        make_grid,
        orient_and_stack,
        replicate_with_ghost_tiles,
    )
    from rcps.io import load_packing_xyzd

    packing = Path(packing)
    p = dict(params)
    box = [float(x) for x in p.get("box_mm", [50.0, 50.0, 50.0])]
    vox = float(p.get("vox_mm", 0.5))
    bridge_mode = p.get("bridge_mode", "cylinders")
    export_what = p.get("export_what", "beads")
    band_vox = 3

    factor, dia_info = resolve_expansion(p.get("diameter", "stored"), packing, box)
    job.log(f"field at vox={vox} mm, diameter mode {dia_info['mode']} "
            f"(x{factor:.6g}), bridges={bridge_mode}, export={export_what}, "
            f"shape={p.get('shape') or 'box'}")

    geom = GeomConfig(
        tile_size_mm=tuple(box), shape=p.get("shape") or "box",
        cylinder_diameter_mm=p.get("cylinder_diameter_mm") or None,
        flow_axis=p.get("flow_axis") or "z",
        segment_periods=int(p.get("segment_periods") or 1),
    )
    S = load_packing_xyzd(packing)
    c0, d0, seg = orient_and_stack(S[:, :3], S[:, 3], tuple(box),
                                   flow_axis=geom.flow_axis,
                                   segment_periods=geom.segment_periods)
    centers, diameters = replicate_with_ghost_tiles(c0, d0, seg, 1)
    radii = (diameters / 2.0) * factor
    if bridge_mode == "diameter":
        radii = expand_to_touch(centers, radii,
                                contact_tol_mm=float(p.get("contact_tol_mm", 0.2)))
    max_radius = float(radii.max())

    grid = make_grid(seg, vox, pad_vox=1, band_vox=band_vox)
    band_dist = max(2, band_vox) * grid.vox_size
    centers, radii, _ = cull_spheres(centers, radii, grid, band_dist=band_dist)
    centers, radii, _ = apply_keepsides_filter(centers, radii, grid)
    job.log(f"grid {grid.nx}x{grid.ny}x{grid.nz}, {centers.shape[0]} spheres kept")

    F_beads = build_beads_sdf_icsg(centers, radii, grid, band_vox=band_vox)
    if bridge_mode == "cylinders":
        F_beads = add_cylinders(
            F_beads, centers, radii, grid,
            contact_tol_mm=float(p.get("contact_tol_mm", 0.2)),
            radius_frac=float(p.get("radius_frac", 0.15)),
            band_vox=band_vox,
        )
    if geom.shape == "cylinder":
        F_box = build_cylinder_sdf(grid, geom.effective_cylinder_diameter_mm,
                                   max_radius_mm=max_radius, band_vox=band_vox)
    else:
        F_box = build_box_sdf(grid, max_radius_mm=max_radius, band_vox=band_vox)
    F = compose_field(F_beads, F_box, export_what)

    job.volume = np.ascontiguousarray(F, dtype=np.float32)
    job.result = {
        "shape": [int(grid.nx), int(grid.ny), int(grid.nz)],
        "vox_mm": float(grid.vox_size),
        "origin_mm": [float(x) for x in grid.origin],
        "box_mm": list(seg),
        "range": [round(float(F.min()), 4), round(float(F.max()), 4)],
        "diameter": dia_info,
        "export_what": export_what,
    }
    job.log(f"field ready: range [{job.result['range'][0]}, "
            f"{job.result['range'][1]}] mm")


def sdf_slice(job: Job, axis: int, index: int) -> bytes:
    """Binary slice: u32 w | u32 h | f32 pos_mm | f32[w*h] row-major (h rows)."""
    import struct

    F = job.volume
    if F is None:
        raise ValueError("no field computed for this job")
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    n = F.shape[axis]
    index = max(0, min(int(index), n - 1))
    plane = np.take(F, index, axis=axis)
    # orient: rows = second remaining axis, columns = first remaining axis
    plane = np.ascontiguousarray(plane.T, dtype=np.float32)  # (h, w)
    h, w = plane.shape
    origin = job.result["origin_mm"][axis]
    pos = origin + index * job.result["vox_mm"]
    return struct.pack("<IIf", w, h, pos) + plane.tobytes()


def export_job(job: Job, packing: str | Path, out_dir: str | Path,
               params: dict[str, Any]) -> None:
    """Full-resolution export: single tile, or facility grid via rcps-facility."""
    from rcps.config import RcpsConfig

    packing = Path(packing)
    p = dict(params)
    # one export folder per shape, so a cylinder column never overwrites
    # the box facility of the same run (and vice versa)
    out = Path(out_dir) / EXPORT_DIRS.get(p.get("shape") or "box", "export_box")
    p.setdefault("vox_mm", 0.1)
    if p.get("dry_run") and p.get("backend") in (None, "auto"):
        # a dry run never meshes — don't require a meshing backend, but
        # record the one a real export would use (it goes into the map)
        try:
            p["backend"] = _backend("iso2mesh")
        except ImportError:
            p["backend"] = "skimage"
    grid = [int(x) for x in p.get("grid", [1, 1, 1])]
    box = [float(x) for x in p.get("box_mm", [50, 50, 50])]
    diameter = p.get("diameter", "stored")

    factor, dia_info = resolve_expansion(diameter, packing, box)
    p["expansion_factor"] = factor
    job.log(f"diameter mode {dia_info['mode']}: factor {factor:.6g} "
            f"(printed mean d = {dia_info['mean_printed_diameter_mm']} mm, "
            f"phi = {dia_info['phi_realized']})")

    if grid == [1, 1, 1] and not p.get("dry_run"):
        from rcps.pipeline import run

        p["base_name"] = p.get("base_name", "rcps_tile")
        cfg = RcpsConfig.from_dict(_config_dict(packing, out, p,
                                                prefer_backend="iso2mesh"))
        job.log(f"single tile at vox={p['vox_mm']} mm ({cfg.mesh.backend}) ...")
        written = run(cfg)
        job.result = {
            "files": {k: str(v) for k, v in written.items()},
            "diameter": dia_info,
            "out_dir": str(out),
        }
    else:
        from rcps.facility import run_facility

        cfg = RcpsConfig.from_dict(_config_dict(packing, out, p,
                                                prefer_backend="iso2mesh"))
        what = ("column" if cfg.geom.shape == "cylinder" else "facility")
        job.log(f"{what} {grid[0]}x{grid[1]}x{grid[2]} at vox={p['vox_mm']} mm "
                f"- one meshing run per unique tile type ...")
        res = run_facility(cfg, grid, diameter=diameter, out_dir=out,
                           dry_run=bool(p.get("dry_run", False)))
        job.result = {
            "map_txt": str(res["map_txt"]),
            "map_json": str(res["map_json"]),
            "meshes": res["meshes"],
            "n_types": res["n_types"],
            "n_tiles": res["n_tiles"],
            "diameter": dia_info,
            "out_dir": str(out),
        }


# ---------------------------------------------------------------------
# Same medium as the other exports of this run?
# ---------------------------------------------------------------------

_LABELS = {
    "packing_sha256": "packing file", "expansion_factor": "printed-diameter factor",
    "export_what": "beads/pore", "bridge_mode": "bridges",
    "radius_frac": "bridge radius", "contact_tol_mm": "contact tolerance [mm]",
    "vox_size_mm": "production voxel [mm]", "backend": "mesher",
    "angbound_deg": "angbound", "radbound": "radbound", "distbound": "distbound",
}


def _gui_settings(med: dict[str, Any], packing: Path, box: list[float]) -> dict[str, Any]:
    """Recorded medium -> GUI field values ("Use those settings")."""
    f = float(med.get("expansion_factor", 1.0))
    diameter = None
    if abs(f - 1.0) < 1e-6:
        diameter = "stored"
    else:
        try:
            fd, _ = resolve_expansion("design", packing, box)
            if abs(fd - f) < 1e-5:
                diameter = "design"
        except (FileNotFoundError, ValueError):
            pass
    backend = med.get("backend")
    try:
        if backend == _backend("iso2mesh"):
            backend = "auto"            # 'auto' already exports with it
    except ImportError:
        pass
    return {
        "export_what": med.get("export_what"), "bridge_mode": med.get("bridge_mode"),
        "radius_frac": med.get("radius_frac"), "contact_tol_mm": med.get("contact_tol_mm"),
        "diameter": diameter, "vox_mm": med.get("vox_size_mm"), "backend": backend,
        "iso2mesh": {k: med.get(k) for k in ("angbound_deg", "radbound", "distbound")},
    }


def compare_exports(run_dir: str | Path, packing: str | Path,
                    params: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare the current Mesh/Export settings with every export folder of
    the run (legacy ``export/`` included). Read-only; never blocks."""
    from rcps.config import RcpsConfig
    from rcps.facility import compare_medium, medium_from_config, recorded_medium
    from rcps.io import sha256_of_file

    run_dir, packing = Path(run_dir), Path(packing)
    p = dict(params)
    box = [float(x) for x in p.get("box_mm", [50, 50, 50])]
    if p.get("backend") in (None, "auto"):
        try:
            p["backend"] = _backend("iso2mesh")
        except ImportError:
            p["backend"] = "skimage"
    p["expansion_factor"], _ = resolve_expansion(p.get("diameter", "stored"), packing, box)
    cfg = RcpsConfig.from_dict(_config_dict(packing, run_dir, p))
    here = medium_from_config(cfg.to_dict(), sha256_of_file(packing))

    out = []
    for name in ("export", *EXPORT_DIRS.values()):
        rec = recorded_medium(run_dir / name)
        if rec is None:
            continue
        item: dict[str, Any] = {"folder": name, "source": rec["source"],
                                "mixed": rec["mixed"], "diffs": [], "settings": None}
        if rec["medium"] is None:
            item["status"] = "unknown"
        else:
            diffs = compare_medium(here, rec["medium"])
            for d in diffs:
                d["label"] = _LABELS.get(d["key"], d["key"])
            item["diffs"] = diffs
            item["status"] = ("same" if not diffs else
                              "differs" if any(d["kind"] == "medium" for d in diffs)
                              else "accuracy")
            item["settings"] = _gui_settings(rec["medium"], packing, box)
            item["summary"] = (
                f"bridges {rec['medium']['bridge_mode']} {rec['medium']['radius_frac']:g}, "
                f"tol {rec['medium']['contact_tol_mm']:g} mm, "
                f"d x{rec['medium']['expansion_factor']:g}, "
                f"vox {rec['medium']['vox_size_mm']:g} mm, {rec['medium']['backend']}")
        out.append(item)
    return out
