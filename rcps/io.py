"""I/O for the RCPS pipeline.

Implements packing.xyzd loading, .3mf / .stl writers, and reproducibility
sidecars. See the archived design audit §1 (lines 131–142 of RCPS_v4.m) and §3
(write3mf replacement plan) for behaviour contract.

The trimesh-based writers import lazily, so the module loads even if
``trimesh`` is not installed — useful for environments that only need the
binary packing loader (e.g., a stripped-down Colab cell).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rcps._version import __version__

log = logging.getLogger(__name__)


# =====================================================================
# Packing.xyzd loader
# =====================================================================

def load_packing_xyzd(path: str | os.PathLike[str]) -> NDArray[np.float64]:
    """Load a Baranau ``packing.xyzd`` binary file.

    The file is a little-endian float64 stream of interleaved
    ``(x, y, z, d)`` quadruples — one per sphere. This mirrors the MATLAB
    reader in ``RCPS_v4.m`` lines 131–142::

        fid = fopen(path, 'r', 'ieee-le');
        raw = fread(fid, Inf, 'double=>double');
        S = reshape(raw, 4, []).';   % [x y z d]

    Parameters
    ----------
    path
        Path to the ``packing.xyzd`` file.

    Returns
    -------
    ndarray of shape ``(N, 4)``, dtype ``float64``
        Columns are ``(x, y, z, d)`` in mm. ``d`` is the **diameter**, not
        the radius (Baranau's convention).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file's byte length is not a multiple of ``4 * 8 = 32``
        (i.e., the stream is not a clean sequence of float64 quadruples),
        or if no spheres are found.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"packing.xyzd not found: {p}")

    raw = np.fromfile(p, dtype="<f8")
    if raw.size == 0:
        raise ValueError(f"packing.xyzd is empty: {p}")
    if raw.size % 4 != 0:
        raise ValueError(
            f"Invalid packing.xyzd stream: {raw.size} float64 values "
            f"not divisible by 4 (file {p}, size={p.stat().st_size} bytes)."
        )

    arr = raw.reshape(-1, 4)
    log.info("loaded packing: %d spheres from %s", arr.shape[0], p)
    return arr


def estimate_n_spheres(
    tile_size_mm: tuple[float, float, float] | Sequence[float],
    sphere_diameter_mm: float,
    target_porosity: float,
) -> int:
    """Initial guess of how many spheres fit a tile at a target porosity.

    Mirrors MATLAB ``RCPS_v4.m`` lines 107–129 (STEP 0). Used by the Colab
    notebook to set the ``-N`` argument for ``packing-generation``.

    Parameters
    ----------
    tile_size_mm
        ``(L, H, W)`` tile box dimensions in mm.
    sphere_diameter_mm
        Sphere diameter in mm.
    target_porosity
        Desired porosity ``phi = V_empty / V_total``, in ``[0, 1)``.

    Returns
    -------
    int
        ``floor(V_box * (1 - phi) / V_sphere)``.

    Raises
    ------
    ValueError
        If inputs are non-positive or porosity is out of range.

    Examples
    --------
    >>> estimate_n_spheres((50, 50, 50), 6.0, 0.35)
    718
    """
    L, H, W = tile_size_mm
    if min(L, H, W) <= 0:
        raise ValueError(f"tile dimensions must be positive, got {tile_size_mm}")
    if sphere_diameter_mm <= 0:
        raise ValueError(f"sphere diameter must be positive, got {sphere_diameter_mm}")
    if not (0.0 <= target_porosity < 1.0):
        raise ValueError(f"target_porosity must lie in [0, 1), got {target_porosity}")

    box_volume = float(L) * float(H) * float(W)
    sphere_volume = (4.0 / 3.0) * np.pi * (sphere_diameter_mm / 2.0) ** 3
    effective_volume = box_volume * (1.0 - target_porosity)
    return int(np.floor(effective_volume / sphere_volume))


# =====================================================================
# Mesh writers (lazy trimesh import)
# =====================================================================

def _build_trimesh(vertices: ArrayLike, faces: ArrayLike, units: str = "mm"):
    """Construct a trimesh.Trimesh without auto-processing, units set.

    Lazy import so the module loads without ``trimesh`` available.
    """
    import trimesh  # noqa: PLC0415  (lazy)

    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)

    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"vertices must be (N, 3), got shape {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"faces must be (M, 3) triangular, got shape {F.shape}")
    if F.size and (F.max() >= V.shape[0] or F.min() < 0):
        raise ValueError(
            f"face indices out of bounds: min={F.min()}, max={F.max()}, "
            f"#vertices={V.shape[0]}"
        )

    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    # Set units explicitly so the 3MF metadata says `unit="millimeter"`,
    # the convention expected by PreForm and other slicers.
    mesh.units = units
    return mesh


def write_3mf(
    vertices: ArrayLike,
    faces: ArrayLike,
    path: str | os.PathLike[str],
) -> Path:
    """Write a triangular mesh to a standard 3MF file.

    Replaces the MATLAB ``write3mf`` writer (cvergari/write3mf, MIT) called
    from ``RCPS_v4.m:719``. Uses ``trimesh.exchange.threemf.export_3MF``,
    which writes full float-precision vertex coordinates. The MATLAB
    writer used ``%.2f`` (10 µm quantisation), unsuitable for
    ``voxSize_mm = 0.1`` output.

    Parameters
    ----------
    vertices
        ``(N, 3)`` array of vertex coordinates in mm.
    faces
        ``(M, 3)`` array of triangle vertex indices (0-based).
    path
        Output ``.3mf`` file path. Parent directories are created.

    Returns
    -------
    Path
        The path that was written (for chaining).
    """
    from trimesh.exchange.threemf import export_3MF  # noqa: PLC0415

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    mesh = _build_trimesh(vertices, faces, units="mm")
    data = export_3MF(mesh)
    out.write_bytes(data)
    log.info(
        "wrote 3MF: %s (V=%d, F=%d, %.1f KiB)",
        out, mesh.vertices.shape[0], mesh.faces.shape[0], len(data) / 1024.0,
    )
    return out


def write_stl(
    vertices: ArrayLike,
    faces: ArrayLike,
    path: str | os.PathLike[str],
    *,
    binary: bool = True,
) -> Path:
    """Write a triangular mesh to STL.

    Optional fallback writer for users who explicitly need STL. Not used
    in the default v1.0 pipeline (.3mf only).
    """
    from trimesh.exchange.stl import export_stl, export_stl_ascii  # noqa: PLC0415

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    mesh = _build_trimesh(vertices, faces, units="mm")
    data = export_stl(mesh) if binary else export_stl_ascii(mesh)
    out.write_bytes(data) if isinstance(data, (bytes, bytearray)) else out.write_text(data)
    log.info("wrote STL: %s (V=%d, F=%d)", out, mesh.vertices.shape[0], mesh.faces.shape[0])
    return out


# =====================================================================
# Reproducibility sidecars
# =====================================================================

def sha256_of_file(path: str | os.PathLike[str], *, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks.

    Used by ``write_config_json`` to fingerprint the input ``packing.xyzd``
    so a downstream consumer can verify they have the exact packing that
    produced a given .3mf.
    """
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_info_txt(
    path: str | os.PathLike[str],
    config_dict: Mapping[str, Any],
    runtime: Mapping[str, Any] | None = None,
) -> Path:
    """Write a human-readable parameter dump (key: value).

    Compatible in spirit with the MATLAB ``_info.txt`` from ``RCPS_v4.m``
    lines 749–768: flat ``dotted.key: value`` lines, one per parameter.
    Nested config sections are flattened with dot-joined keys.

    Parameters
    ----------
    path
        Output ``.txt`` path.
    config_dict
        Plain dict of the full RcpsConfig (e.g. ``config.model_dump()``).
    runtime
        Optional dict of derived/runtime values (snapped voxel size, grid
        dims, mesh stats, elapsed seconds).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    flat = _flatten_dict(config_dict)
    if runtime:
        for k, v in _flatten_dict(runtime, prefix="runtime").items():
            flat[k] = v

    lines = [
        f"# rcps-3dprint v{__version__} — written {_iso_now()}",
        "",
    ]
    for k in sorted(flat.keys()):
        lines.append(f"{k}: {_format_scalar(flat[k])}")
    out.write_text("\n".join(lines) + "\n")
    log.info("wrote info.txt: %s (%d keys)", out, len(flat))
    return out


def write_config_json(
    path: str | os.PathLike[str],
    config_dict: Mapping[str, Any],
    *,
    packing_path: str | os.PathLike[str] | None = None,
    packing_sha256: str | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> Path:
    """Write a machine-readable reproducibility sidecar.

    Schema::

        {
          "rcps_version": "0.1.0",
          "written_at":   "2026-05-28T14:00:00+00:00",
          "config":       { ...full pydantic dump... },
          "input": {
            "packing_path":   "data_example/packing.xyzd",
            "packing_sha256": "ab12…"
          },
          "runtime": {
            "snapped_vox_size_mm": 0.1,
            "grid_dims": [502, 502, 502],
            "n_spheres_loaded": 718,
            "n_spheres_kept":   718,
            "mesh_stats": {...},
            "elapsed_seconds": 26.5
          }
        }

    Either pass ``packing_sha256`` directly, or pass ``packing_path`` and
    the hash will be computed on the fly. Passing neither omits the input
    block.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if packing_sha256 is None and packing_path is not None:
        packing_sha256 = sha256_of_file(packing_path)

    payload: dict[str, Any] = {
        "rcps_version": __version__,
        "written_at": _iso_now(),
        "config": _to_json_safe(config_dict),
    }
    if packing_path is not None or packing_sha256 is not None:
        payload["input"] = {
            "packing_path": str(packing_path) if packing_path is not None else None,
            "packing_sha256": packing_sha256,
        }
    if runtime:
        payload["runtime"] = _to_json_safe(runtime)

    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    log.info("wrote config.json: %s", out)
    return out


# =====================================================================
# Internal helpers
# =====================================================================

def _iso_now() -> str:
    """ISO-8601 timestamp in UTC with timezone suffix."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _flatten_dict(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings into dotted keys.

    Lists/tuples of scalars are preserved as-is (formatted in the value);
    nested dicts are recursed.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, Mapping):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


def _format_scalar(v: Any) -> str:
    """Format a scalar/list value for the human-readable info.txt."""
    if isinstance(v, float):
        # Match MATLAB's %.15g for fractional values; ints render as ints.
        return f"{v:.15g}"
    if isinstance(v, (list, tuple)):
        return " ".join(_format_scalar(x) for x in v)
    if isinstance(v, np.ndarray):
        return " ".join(_format_scalar(x) for x in v.tolist())
    return str(v)


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays into JSON-native types."""
    if isinstance(obj, Mapping):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj
