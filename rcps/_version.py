"""Single source of truth for the package version.

Read by `pyproject.toml` via `[tool.hatch.version]` and re-exported by
`rcps/__init__.py`. Bump on every release; the `.config.json` sidecar
records this string for reproducibility.
"""

__version__ = "0.1.0.dev0"
