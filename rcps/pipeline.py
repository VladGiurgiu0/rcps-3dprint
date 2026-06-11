"""End-to-end pipeline orchestration.

``run(config)`` chains the IO, field, bridges, and mesh modules into a
single per-tile pipeline and writes the .3mf plus the two reproducibility
sidecars. Order of operations (locked in the archived design audit §6):

    load packing
        → (optional) expand_to_touch          [bridge.mode == 'diameter']
        → ghost-tile replication
        → apply expansion_factor
        → make_grid (using post-expansion max radius)
        → cull_spheres
        → apply_keepsides_filter
        → build_beads_sdf_icsg
        → (optional) add_cylinders             [bridge.mode == 'cylinders']
        → build_box_sdf
        → compose_field
        → mesh_iso2mesh / mesh_skimage
        → repair_and_finalize
        → diagnose
        → write_3mf (+ write_stl if requested)
        → write_info_txt
        → write_config_json

The pipeline returns a dict of the written output paths.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from rcps._version import __version__
from rcps.bridges import add_cylinders, expand_to_touch
from rcps.config import RcpsConfig
from rcps.field import (
    apply_keepsides_filter,
    build_beads_sdf_icsg,
    build_box_sdf,
    compose_field,
    cull_spheres,
    make_grid,
    replicate_with_ghost_tiles,
)
from rcps.io import (
    load_packing_xyzd,
    sha256_of_file,
    write_3mf,
    write_config_json,
    write_info_txt,
    write_stl,
)
from rcps.mesh import (
    MeshStats,
    diagnose,
    mesh_iso2mesh,
    mesh_skimage,
    repair_and_finalize,
)

log = logging.getLogger(__name__)


# Iso level used by the mesher. A tiny negative number nudges the
# extracted surface just inside the beads/pore region to avoid the
# float32 grey zone at F == 0. Matches RCPS_v4.m line 485.
ISO_LEVEL_MULTIPLIER: float = -1e-6


def run(config: RcpsConfig) -> dict[str, Path]:
    """Execute the per-tile pipeline.

    Parameters
    ----------
    config
        A validated :class:`RcpsConfig`. Relative paths in
        ``config.paths`` must already be absolute (use
        :meth:`RcpsConfig.from_yaml` to resolve them automatically).

    Returns
    -------
    dict[str, Path]
        Mapping from output kind (``"3mf"`` / ``"stl"`` / ``"info"`` /
        ``"config_json"``) to the path written.

    Raises
    ------
    FileNotFoundError, ValueError, ImportError
        Propagated from the lower-level modules.
    """
    t_start = time.perf_counter()
    log.info("rcps pipeline v%s starting", __version__)

    out_dir = Path(config.paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Load packing.xyzd
    # -------------------------------------------------------------
    S = load_packing_xyzd(config.paths.packing)
    centers0 = S[:, :3].copy()
    diameters0 = S[:, 3].copy()
    log.info(
        "step 1/9 load: %d spheres, diameter=%g mm",
        S.shape[0], float(diameters0[0]),
    )

    # -------------------------------------------------------------
    # 2. Ghost-tile replication
    # -------------------------------------------------------------
    centers, diameters = replicate_with_ghost_tiles(
        centers0, diameters0,
        tile_size_mm=config.geom.tile_size_mm,
        ghost_tiles=config.field.ghost_tiles,
    )
    radii = (diameters / 2.0) * config.spheres.expansion_factor
    log.info(
        "step 2/9 ghost-tile replication: %d spheres after ×(2g+1)³ = %d",
        S.shape[0], centers.shape[0],
    )

    # -------------------------------------------------------------
    # 3. Apply `diameter` bridge mode BEFORE grid sizing (it may grow radii)
    # -------------------------------------------------------------
    if config.bridge.mode == "diameter":
        radii_before = radii.copy()
        radii = expand_to_touch(
            centers, radii,
            contact_tol_mm=config.spheres.contact_tol_mm,
        )
        log.info(
            "step 3/9 diameter mode: max ∆r = %.4g mm (mean %.4g mm)",
            float((radii - radii_before).max()),
            float((radii - radii_before).mean()),
        )
    else:
        log.info("step 3/9 diameter mode: skipped (bridge.mode=%r)", config.bridge.mode)

    max_radius = float(radii.max())

    # -------------------------------------------------------------
    # 4. Build padded SDF grid
    # -------------------------------------------------------------
    grid = make_grid(
        config.geom.tile_size_mm,
        config.grid.vox_size_mm,
        pad_vox=config.field.pad_vox,
        band_vox=config.field.band_vox,
        keep_sides=config.field.keep_sides,
        max_radius_mm=max_radius if config.field.keep_sides else 0.0,
    )
    log.info(
        "step 4/9 grid: %d × %d × %d voxels (vox=%.6g mm, %.1f MiB)",
        grid.nx, grid.ny, grid.nz, grid.vox_size,
        grid.memory_float32_bytes / 1024**2,
    )
    band_dist = max(2, config.field.band_vox) * grid.vox_size

    # -------------------------------------------------------------
    # 5. Cull spheres + keepSides ghost filter
    # -------------------------------------------------------------
    centers, radii, _ = cull_spheres(centers, radii, grid, band_dist=band_dist)
    centers, radii, _ = apply_keepsides_filter(centers, radii, grid)
    n_spheres_kept = int(centers.shape[0])
    log.info("step 5/9 culled+filtered: %d spheres kept", n_spheres_kept)

    # -------------------------------------------------------------
    # 6. ICSG beads SDF + (optional) cylinder bridges + box SDF + compose
    # -------------------------------------------------------------
    t_field = time.perf_counter()
    F_beads = build_beads_sdf_icsg(centers, radii, grid, band_vox=config.field.band_vox)
    if config.bridge.mode == "cylinders":
        F_beads = add_cylinders(
            F_beads, centers, radii, grid,
            contact_tol_mm=config.spheres.contact_tol_mm,
            radius_frac=config.bridge.radius_frac,
            band_vox=config.field.band_vox,
        )
    F_box = build_box_sdf(grid, max_radius_mm=max_radius, band_vox=config.field.band_vox)
    F = compose_field(F_beads, F_box, config.field.export_what)
    log.info(
        "step 6/9 SDF built in %.2f s (bridge.mode=%r, export_what=%r)",
        time.perf_counter() - t_field,
        config.bridge.mode, config.field.export_what,
    )

    # -------------------------------------------------------------
    # 7. Mesh the iso-surface
    # -------------------------------------------------------------
    iso_level = ISO_LEVEL_MULTIPLIER * grid.vox_size
    t_mesh = time.perf_counter()
    if config.mesh.backend == "iso2mesh":
        verts, faces = mesh_iso2mesh(
            F, grid.vox_size, grid.origin, iso_level,
            angbound_deg=config.mesh.iso2mesh.angbound_deg,
            radbound=config.mesh.iso2mesh.radbound,
            distbound=config.mesh.iso2mesh.distbound,
            maxnode=config.mesh.iso2mesh.maxnode,
        )
    elif config.mesh.backend == "skimage":
        verts, faces = mesh_skimage(F, grid.vox_size, grid.origin, iso_level)
    else:  # pragma: no cover — config schema rejects this
        raise ValueError(f"unknown mesh backend {config.mesh.backend!r}")
    log.info(
        "step 7/9 mesh (%s): %d verts, %d faces in %.2f s",
        config.mesh.backend, verts.shape[0], faces.shape[0],
        time.perf_counter() - t_mesh,
    )

    # -------------------------------------------------------------
    # 8. Repair + diagnose
    # -------------------------------------------------------------
    verts, faces = repair_and_finalize(verts, faces, vox_size_mm=grid.vox_size)
    stats = diagnose(verts, faces)
    log.info("step 8/9 repair+diagnose: %s", stats.summary_line())

    # -------------------------------------------------------------
    # 9. Write outputs + sidecars
    # -------------------------------------------------------------
    base = config.out.base_name
    written: dict[str, Path] = {}

    if config.out.save_3mf:
        written["3mf"] = write_3mf(verts, faces, out_dir / f"{base}.3mf")
    if config.out.save_stl:
        written["stl"] = write_stl(verts, faces, out_dir / f"{base}.stl")

    elapsed = time.perf_counter() - t_start
    runtime: dict[str, Any] = {
        "snapped_vox_size_mm": float(grid.vox_size),
        "grid_dims": list(grid.shape),
        "grid_origin_mm": list(grid.origin),
        "iso_level": float(iso_level),
        "n_spheres_loaded": int(S.shape[0]),
        "n_spheres_kept": n_spheres_kept,
        "max_radius_mm": max_radius,
        "elapsed_seconds": elapsed,
        "mesh_stats": _meshstats_to_dict(stats),
    }
    if config.out.write_info_txt:
        written["info"] = write_info_txt(
            out_dir / f"{base}_info.txt",
            config.to_dict(),
            runtime=runtime,
        )
    if config.out.write_config_json:
        written["config_json"] = write_config_json(
            out_dir / f"{base}.config.json",
            config.to_dict(),
            packing_path=config.paths.packing,
            packing_sha256=sha256_of_file(config.paths.packing),
            runtime=runtime,
        )

    log.info(
        "rcps pipeline complete in %.1f s; outputs: %s",
        elapsed, list(written.keys()),
    )
    return written


def _meshstats_to_dict(stats: MeshStats) -> dict[str, Any]:
    """Convert :class:`MeshStats` into JSON-safe nested dict for sidecars."""
    return {
        "n_vertices": stats.n_vertices,
        "n_faces": stats.n_faces,
        "n_degenerate_faces": stats.n_degenerate_faces,
        "n_boundary_edges": stats.n_boundary_edges,
        "n_nonmanifold_edges": stats.n_nonmanifold_edges,
        "watertight": bool(stats.watertight),
        "surface_area_mm2": float(stats.surface_area_mm2),
        "bounds_mm": {
            "min": [float(v) for v in stats.bounds_mm[0]],
            "max": [float(v) for v in stats.bounds_mm[1]],
        },
    }


__all__ = ["run", "ISO_LEVEL_MULTIPLIER"]
