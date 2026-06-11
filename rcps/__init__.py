"""rcps-3dprint — reproducible random close-packed sphere lattices for 3D printing.

Public API (populated during Tasks 3–7):

    from rcps.config import RcpsConfig
    from rcps.pipeline import run

    config = RcpsConfig.from_yaml("examples/config_50mm_d6_phi035.yaml")
    run(config)

The package consumes a `packing.xyzd` file produced by Baranau's
`packing-generation` (run separately in Google Colab; see
`notebooks/packing_generation_colab.ipynb`) and produces a printable `.3mf`
tile plus reproducibility sidecars.

License: MIT. See LICENSE and NOTICE for third-party attribution.
"""

from rcps._version import __version__

__all__ = ["__version__"]
