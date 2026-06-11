"""rcps_gui — local browser GUI guiding the full RCPS workflow.

`rcps-gui` starts a local web server (stdlib only, fully on-device) and
opens the browser. Four guided stages: Setup (packing-generator
executable) → Pack (run the packing simulation) → Mesh (coarse preview)
→ Export (full-resolution printable .3mf, single tile or facility).
"""

from rcps._version import __version__  # noqa: F401
