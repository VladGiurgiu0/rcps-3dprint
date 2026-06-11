"""Command-line entry points for the RCPS pipeline.

Console scripts declared in ``pyproject.toml [project.scripts]``::

    rcps-build      → rcps.cli:build_main
    rcps-facility   → rcps.cli:facility_main   (Task 15 — placeholder)

``rcps-build``
~~~~~~~~~~~~~~

Single-tile pipeline. Usage::

    rcps-build path/to/config.yaml [--log-level INFO] [--out-dir PATH]

The config file is a YAML document conforming to
:class:`rcps.config.RcpsConfig`. ``--out-dir`` overrides the config's
``paths.out_dir`` when given. Exit codes::

    0  success
    2  bad CLI usage / config validation error
    3  missing required file (packing.xyzd, config.yaml, …)
    4  meshing or repair failed (ImportError or runtime error)
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from rcps._version import __version__

log = logging.getLogger(__name__)


# Standardised exit codes (used by `rcps-build` and `rcps-facility`).
EXIT_OK = 0
EXIT_BAD_USAGE = 2
EXIT_MISSING_FILE = 3
EXIT_RUNTIME = 4


# =====================================================================
# rcps-build — single-tile entry point
# =====================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rcps-build",
        description=(
            "Build a single-tile printable .3mf from a packing.xyzd plus a "
            "YAML config. Produces .3mf + _info.txt + .config.json sidecars."
        ),
    )
    p.add_argument(
        "config",
        type=Path,
        help="Path to the YAML config file (see examples/config_50mm_d6_phi035.yaml).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override paths.out_dir from the config.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Default: INFO.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"rcps-3dprint {__version__}",
    )
    return p


def build_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``rcps-build`` console script."""
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_level)

    # Lazy imports so `--help` works without numpy/pyyaml/etc. installed.
    try:
        from rcps.config import RcpsConfig
        from rcps.pipeline import run
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_RUNTIME

    try:
        config = RcpsConfig.from_yaml(args.config)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_MISSING_FILE
    except ValueError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_BAD_USAGE

    if args.out_dir is not None:
        config.paths.out_dir = args.out_dir.resolve()

    try:
        written = run(config)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_MISSING_FILE
    except (ImportError, ValueError, RuntimeError) as e:
        print(f"runtime error: {e}", file=sys.stderr)
        return EXIT_RUNTIME

    # Final summary — one line per artefact written.
    print()
    print("rcps-build complete:")
    for kind, path in written.items():
        print(f"  {kind:>11s}  {path}")
    return EXIT_OK


# =====================================================================
# rcps-facility — multi-tile orchestrator
# =====================================================================

def _diameter_arg(value: str):
    """``stored`` | ``design`` | explicit float expansion factor."""
    if value in ("stored", "design"):
        return value
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--diameter must be 'stored', 'design', or a float, got {value!r}"
        ) from None


def _facility_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rcps-facility",
        description=(
            "Generate all unique interlocking tile meshes + assembly map "
            "for an Nx x Ny x Nz facility from a single-tile YAML config. "
            "Interior faces are kept (tiles interlock); exterior faces are "
            "cut flush. Runs the pipeline once per unique tile type "
            "(e.g. 4x4x1 -> 9 meshes for 16 tiles)."
        ),
    )
    p.add_argument("config", type=Path, help="Single-tile YAML config.")
    p.add_argument(
        "--grid", nargs=3, type=int, required=True, metavar=("NX", "NY", "NZ"),
        help="Tiles per axis, e.g. --grid 4 4 1.",
    )
    p.add_argument(
        "--diameter", type=_diameter_arg, required=True,
        help=(
            "Diameter convention (explicit per print job): 'stored' = print "
            "the true jammed packing exactly as in packing.xyzd (tangent "
            "contacts); 'design' = inflate so the realized porosity matches "
            "the packing's .nfo (small contact overlaps, stronger printed "
            "bonds); or an explicit float expansion factor."
        ),
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: <config out_dir>/facility_NxMxK).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Write the assembly map and print the plan without meshing "
            "(each unique type at production resolution is a long run)."
        ),
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Default: INFO.",
    )
    p.add_argument(
        "--version", action="version", version=f"rcps-3dprint {__version__}",
    )
    return p


def facility_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``rcps-facility`` console script."""
    args = _facility_parser().parse_args(argv)
    _configure_logging(args.log_level)

    try:
        from rcps.config import RcpsConfig
        from rcps.facility import run_facility
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_RUNTIME

    try:
        config = RcpsConfig.from_yaml(args.config)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_MISSING_FILE
    except ValueError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_BAD_USAGE

    try:
        result = run_facility(
            config,
            args.grid,
            diameter=args.diameter,
            out_dir=args.out_dir,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_MISSING_FILE
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_BAD_USAGE
    except (ImportError, RuntimeError) as e:
        print(f"runtime error: {e}", file=sys.stderr)
        return EXIT_RUNTIME

    print()
    mode = "dry run (no meshes generated)" if args.dry_run else "complete"
    print(f"rcps-facility {mode}:")
    print(f"  {result['n_tiles']} tiles, {result['n_types']} unique meshes, "
          f"expansion factor {result['expansion_factor']:.6g}")
    for tag, path in result["meshes"].items():
        print(f"  {tag:>14s}  {path}")
    print(f"  assembly map: {result['map_txt']}")
    return EXIT_OK


# =====================================================================
# Helpers
# =====================================================================

def _configure_logging(level: str) -> None:
    """Set up root logging with a compact one-line format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(build_main())
