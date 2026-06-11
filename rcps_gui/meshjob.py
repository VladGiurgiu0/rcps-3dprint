"""Meshing + export jobs for the GUI (thin layer over the rcps API)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from rcps_gui.jobs import Job


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
        "geom": {"tile_size_mm": box},
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
        "out": {"base_name": p.get("base_name", "rcps_gui")},
    }


def resolve_expansion(diameter: str | float, packing: Path,
                      box: list[float]) -> tuple[float, dict[str, Any]]:
    from rcps.facility import resolve_expansion_factor
    return resolve_expansion_factor(diameter, packing, box)


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

    factor, dia_info = resolve_expansion(p.get("diameter", "stored"), packing,
                                         [float(x) for x in p.get("box_mm", [50, 50, 50])])
    p["expansion_factor"] = factor
    job.log(f"diameter mode {dia_info['mode']}: factor {factor:.6g}, "
            f"phi_realized {dia_info['phi_realized']}")

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
    v_bulk = float(np.prod(p.get("box_mm", [50, 50, 50])))
    job.result = {
        "n_vertices": int(V.shape[0]),
        "n_faces": int(F.shape[0]),
        "watertight": bool(stats.watertight),
        "porosity": round(1.0 - abs(float(mesh.volume)) / v_bulk, 6),
        # specific surface a_s = A_solid / V_bulk — the Kozeny–Carman input
        "surface_area_mm2": round(float(stats.surface_area_mm2), 1),
        "specific_surface_per_mm": round(float(stats.surface_area_mm2) / v_bulk, 4),
        "diameter": dia_info,
        "vox_mm": p["vox_mm"],
        "mesh_path": str(written["3mf"]),
    }


def sdf_job(job: Job, packing: str | Path, params: dict[str, Any]) -> None:
    """Compute the signed-distance field WITHOUT meshing (for the slicer).

    Mirrors pipeline steps 1–6 (load → ghosts → grid → cull → beads SDF →
    bridges → box → compose) and keeps the float32 volume on the job for
    the slice endpoint. Respects the Mesh-stage settings: printed-diameter
    choice, bridges, beads/pore.
    """
    from rcps.bridges import add_cylinders, expand_to_touch
    from rcps.field import (
        apply_keepsides_filter,
        build_beads_sdf_icsg,
        build_box_sdf,
        compose_field,
        cull_spheres,
        make_grid,
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
            f"(x{factor:.6g}), bridges={bridge_mode}, export={export_what}")

    S = load_packing_xyzd(packing)
    centers, diameters = replicate_with_ghost_tiles(
        S[:, :3], S[:, 3], tuple(box), 1)
    radii = (diameters / 2.0) * factor
    if bridge_mode == "diameter":
        radii = expand_to_touch(centers, radii,
                                contact_tol_mm=float(p.get("contact_tol_mm", 0.2)))
    max_radius = float(radii.max())

    grid = make_grid(tuple(box), vox, pad_vox=1, band_vox=band_vox)
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
    F_box = build_box_sdf(grid, max_radius_mm=max_radius, band_vox=band_vox)
    F = compose_field(F_beads, F_box, export_what)

    job.volume = np.ascontiguousarray(F, dtype=np.float32)
    job.result = {
        "shape": [int(grid.nx), int(grid.ny), int(grid.nz)],
        "vox_mm": float(grid.vox_size),
        "origin_mm": [float(x) for x in grid.origin],
        "box_mm": box,
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
    out = Path(out_dir) / "export"
    p = dict(params)
    p.setdefault("vox_mm", 0.1)
    if p.get("dry_run") and p.get("backend") in (None, "auto"):
        # a dry run never meshes — don't require a meshing backend
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
        }
    else:
        from rcps.facility import run_facility

        cfg = RcpsConfig.from_dict(_config_dict(packing, out, p,
                                                prefer_backend="iso2mesh"))
        job.log(f"facility {grid[0]}x{grid[1]}x{grid[2]} at vox={p['vox_mm']} mm "
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
        }
