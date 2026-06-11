"""Configuration schema for the RCPS pipeline.

YAML in, validated `RcpsConfig` out. See the archived design audit §2 for the
field-by-field mapping from the MATLAB ``p`` struct.

Implementation note
-------------------
The plan committed to "pydantic v2", but the schema is small, fixed,
and only needs (a) type coercion from YAML and (b) range/enum
validation. Stdlib `dataclasses` cover both with one fewer runtime
dependency and a simpler test story. The public API
(`RcpsConfig.from_yaml`, `.to_dict`) is the same shape pydantic would
have offered.

Relative path resolution
------------------------
`RcpsConfig.from_yaml(path)` resolves relative path entries in
`config.paths` against the directory containing the YAML file — not
against the CWD. This makes example configs portable.

Informational vs. driving fields
--------------------------------
``spheres.diameter_mm`` and ``geom.target_porosity`` are informational
only — they feed :func:`rcps.io.estimate_n_spheres` to help the user
seed `packing-generation` in Colab. The pipeline's actual geometry comes
from the loaded ``packing.xyzd``. See the archived design audit §5 items 0a–0b.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass
from dataclasses import field as dc_field  # avoid shadowing by `RcpsConfig.field`
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# =====================================================================
# Constants and enum-like sets
# =====================================================================

VALID_EXPORT_WHAT = frozenset({"beads", "pore"})
VALID_BRIDGE_MODES = frozenset({"none", "cylinders", "diameter"})
VALID_MESH_BACKENDS = frozenset({"iso2mesh", "skimage"})
VALID_KEEP_SIDES = frozenset({"L", "R", "P", "A", "I", "S"})


# =====================================================================
# Section dataclasses
# =====================================================================

@dataclass
class PathsConfig:
    """Input/output paths. Relative entries resolve to the YAML file's dir."""

    packing: Path
    root: Path = dc_field(default_factory=lambda: Path("."))
    out_dir: Path = dc_field(default_factory=lambda: Path("./out"))

    def __post_init__(self) -> None:
        # Coerce strings to Path even when constructed programmatically.
        self.packing = Path(self.packing)
        self.root = Path(self.root)
        self.out_dir = Path(self.out_dir)


@dataclass
class GeomConfig:
    """Tile geometry."""

    tile_size_mm: tuple[float, float, float]
    target_porosity: float = 0.35  # informational

    def __post_init__(self) -> None:
        # YAML loads tuples as lists; coerce.
        t = tuple(float(x) for x in self.tile_size_mm)
        if len(t) != 3:
            raise ValueError(
                f"geom.tile_size_mm must have exactly 3 entries, got {len(t)}"
            )
        if min(t) <= 0:
            raise ValueError(f"geom.tile_size_mm must be positive, got {t}")
        self.tile_size_mm = t
        if not (0.0 <= self.target_porosity < 1.0):
            raise ValueError(
                f"geom.target_porosity must lie in [0, 1), got {self.target_porosity}"
            )


@dataclass
class SpheresConfig:
    """Sphere geometry and print options."""

    diameter_mm: float  # informational; actual d comes from packing.xyzd
    expansion_factor: float = 1.0
    contact_tol_mm: float = 0.20

    def __post_init__(self) -> None:
        if self.diameter_mm <= 0:
            raise ValueError(
                f"spheres.diameter_mm must be positive, got {self.diameter_mm}"
            )
        if self.expansion_factor <= 0:
            raise ValueError(
                f"spheres.expansion_factor must be positive, "
                f"got {self.expansion_factor}"
            )
        if self.contact_tol_mm < 0:
            raise ValueError(
                f"spheres.contact_tol_mm must be ≥ 0, got {self.contact_tol_mm}"
            )


@dataclass
class FieldConfig:
    """ICSG SDF construction parameters."""

    export_what: str = "beads"
    ghost_tiles: int = 1
    pad_vox: int = 1
    band_vox: int = 3
    keep_sides: list[str] = dc_field(default_factory=list)

    def __post_init__(self) -> None:
        if self.export_what not in VALID_EXPORT_WHAT:
            raise ValueError(
                f"field.export_what must be one of {sorted(VALID_EXPORT_WHAT)}, "
                f"got {self.export_what!r}"
            )
        if self.ghost_tiles < 0:
            raise ValueError(
                f"field.ghost_tiles must be ≥ 0, got {self.ghost_tiles}"
            )
        if self.pad_vox < 0:
            raise ValueError(f"field.pad_vox must be ≥ 0, got {self.pad_vox}")
        if self.band_vox < 1:
            raise ValueError(f"field.band_vox must be ≥ 1, got {self.band_vox}")

        # Normalise keep_sides to upper-case, validate, dedupe in input order.
        seen: list[str] = []
        for s in self.keep_sides:
            u = str(s).upper()
            if u not in VALID_KEEP_SIDES:
                raise ValueError(
                    f"field.keep_sides label {s!r} invalid; "
                    f"valid: {sorted(VALID_KEEP_SIDES)}"
                )
            if u not in seen:
                seen.append(u)
        self.keep_sides = seen


@dataclass
class GridConfig:
    """Voxel grid."""

    vox_size_mm: float = 0.1

    def __post_init__(self) -> None:
        if self.vox_size_mm <= 0:
            raise ValueError(
                f"grid.vox_size_mm must be positive, got {self.vox_size_mm}"
            )


@dataclass
class BridgeConfig:
    """Sphere-to-sphere bridge mode and parameters."""

    mode: str = "cylinders"
    radius_frac: float = 0.15

    def __post_init__(self) -> None:
        if self.mode not in VALID_BRIDGE_MODES:
            raise ValueError(
                f"bridge.mode must be one of {sorted(VALID_BRIDGE_MODES)}, "
                f"got {self.mode!r}"
            )
        if not (0.0 < self.radius_frac <= 1.0):
            raise ValueError(
                f"bridge.radius_frac must lie in (0, 1], got {self.radius_frac}"
            )


@dataclass
class Iso2MeshConfig:
    """iso2mesh quality knobs (see ``RCPS_v4.m`` lines 67–73)."""

    angbound_deg: float = 25.0
    radbound: float = 1.0
    distbound: float = 0.10
    maxnode: int = 200_000_000

    def __post_init__(self) -> None:
        if not (0.0 < self.angbound_deg < 90.0):
            raise ValueError(
                f"mesh.iso2mesh.angbound_deg must lie in (0, 90), "
                f"got {self.angbound_deg}"
            )
        if self.radbound <= 0:
            raise ValueError(
                f"mesh.iso2mesh.radbound must be > 0, got {self.radbound}"
            )
        if self.distbound <= 0:
            raise ValueError(
                f"mesh.iso2mesh.distbound must be > 0, got {self.distbound}"
            )
        if self.maxnode < 1000:
            raise ValueError(
                f"mesh.iso2mesh.maxnode must be ≥ 1000, got {self.maxnode}"
            )


@dataclass
class MeshConfig:
    """Meshing backend and its parameters."""

    backend: str = "iso2mesh"
    iso2mesh: Iso2MeshConfig = dc_field(default_factory=Iso2MeshConfig)

    def __post_init__(self) -> None:
        if self.backend not in VALID_MESH_BACKENDS:
            raise ValueError(
                f"mesh.backend must be one of {sorted(VALID_MESH_BACKENDS)}, "
                f"got {self.backend!r}"
            )


@dataclass
class OutConfig:
    """Output file selection."""

    base_name: str = "rcps_tile"
    save_3mf: bool = True
    save_stl: bool = False
    write_info_txt: bool = True
    write_config_json: bool = True

    def __post_init__(self) -> None:
        if not self.base_name:
            raise ValueError("out.base_name must be a non-empty string")
        if not (self.save_3mf or self.save_stl):
            raise ValueError(
                "out: at least one of save_3mf / save_stl must be True "
                "(otherwise the pipeline produces no mesh output)"
            )


# =====================================================================
# Top-level config
# =====================================================================

@dataclass
class RcpsConfig:
    """Full RCPS pipeline configuration.

    Build by hand for programmatic use, or load from YAML::

        config = RcpsConfig.from_yaml("examples/config_50mm_d6_phi035.yaml")
    """

    paths: PathsConfig
    geom: GeomConfig
    spheres: SpheresConfig
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    grid: GridConfig = dc_field(default_factory=GridConfig)
    bridge: BridgeConfig = dc_field(default_factory=BridgeConfig)
    mesh: MeshConfig = dc_field(default_factory=MeshConfig)
    out: OutConfig = dc_field(default_factory=OutConfig)

    # Path to the YAML file from which this config was loaded (None if
    # built programmatically). Used for diagnostics in the sidecars.
    _source_yaml: Path | None = dc_field(default=None, repr=False, compare=False)

    # ---------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> RcpsConfig:
        """Load a YAML config file, validate it, and resolve relative paths.

        Relative paths in ``config.paths`` are resolved against the
        directory containing the YAML file, so example configs work
        regardless of the user's current directory.
        """
        try:
            import yaml  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "RcpsConfig.from_yaml requires `PyYAML` "
                "(`pip install pyyaml`)."
            ) from e

        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"config YAML not found: {p}")
        data = yaml.safe_load(p.read_text())
        if not isinstance(data, dict):
            raise ValueError(
                f"YAML root must be a mapping, got {type(data).__name__}: {p}"
            )

        config = cls.from_dict(data)
        config._source_yaml = p
        config._resolve_paths_relative_to(p.parent)
        log.info("loaded config: %s", p)
        return config

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RcpsConfig:
        """Build from a plain dict (e.g., the parsed YAML or a Python dict)."""
        # Required top-level sections.
        for required in ("paths", "geom", "spheres"):
            if required not in d:
                raise ValueError(f"config: missing required section '{required}'")

        paths = PathsConfig(**d["paths"])
        geom = GeomConfig(**d["geom"])
        spheres = SpheresConfig(**d["spheres"])

        # Optional sections — use defaults if absent.
        field_cfg = FieldConfig(**d.get("field", {}))
        grid_cfg = GridConfig(**d.get("grid", {}))
        bridge_cfg = BridgeConfig(**d.get("bridge", {}))

        mesh_d = d.get("mesh", {})
        iso2mesh_cfg = Iso2MeshConfig(**(mesh_d.get("iso2mesh", {})))
        mesh_cfg = MeshConfig(
            backend=mesh_d.get("backend", "iso2mesh"),
            iso2mesh=iso2mesh_cfg,
        )
        out_cfg = OutConfig(**d.get("out", {}))

        return cls(
            paths=paths, geom=geom, spheres=spheres,
            field=field_cfg, grid=grid_cfg, bridge=bridge_cfg,
            mesh=mesh_cfg, out=out_cfg,
        )

    # ---------------------------------------------------------------
    # Path handling
    # ---------------------------------------------------------------

    def _resolve_paths_relative_to(self, base: Path) -> None:
        """Resolve `paths.{root,packing,out_dir}` relative to ``base``."""
        for attr in ("root", "packing", "out_dir"):
            p = getattr(self.paths, attr)
            if not isinstance(p, Path):
                p = Path(p)
            if not p.is_absolute():
                p = (base / p).resolve()
            setattr(self.paths, attr, p)

    # ---------------------------------------------------------------
    # Serialisation
    # ---------------------------------------------------------------

    def to_dict(self, *, paths_as_str: bool = True) -> dict[str, Any]:
        """Plain-dict form, suitable for JSON or YAML serialisation.

        ``Path`` objects are converted to ``str`` when ``paths_as_str``
        is True (default) so the result is JSON-serialisable.
        """
        d = dataclasses.asdict(self)
        d.pop("_source_yaml", None)
        if paths_as_str:
            for k in ("root", "packing", "out_dir"):
                if k in d.get("paths", {}):
                    d["paths"][k] = str(d["paths"][k])
        return d


__all__ = [
    "BridgeConfig",
    "FieldConfig",
    "GeomConfig",
    "GridConfig",
    "Iso2MeshConfig",
    "MeshConfig",
    "OutConfig",
    "PathsConfig",
    "RcpsConfig",
    "SpheresConfig",
    "VALID_BRIDGE_MODES",
    "VALID_EXPORT_WHAT",
    "VALID_KEEP_SIDES",
    "VALID_MESH_BACKENDS",
]
