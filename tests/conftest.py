"""Shared pytest fixtures.

Fixture sources:
- `data_example/packing.xyzd`: the canonical reference packing (50 mm cube,
  6 mm spheres, 718 spheres, phi≈0.350).
- `tests/fixtures/reference.3mf`: a MATLAB-produced .3mf for the e2e
  validation suite (Task 8 produces this). Tests marked
  `@pytest.mark.matlab_reference` are skipped if this file is missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_EXAMPLE = REPO_ROOT / "data_example"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def packing_xyzd_path() -> Path:
    """Path to the canonical example packing.xyzd."""
    p = DATA_EXAMPLE / "packing.xyzd"
    assert p.exists(), f"Reference packing missing: {p}"
    return p


@pytest.fixture(scope="session")
def packing_nfo_path() -> Path:
    """Path to the packing.nfo metadata file (N=718, dims=50x50x50, phi≈0.350)."""
    p = DATA_EXAMPLE / "packing.nfo"
    assert p.exists(), f"Reference packing.nfo missing: {p}"
    return p


@pytest.fixture(scope="session")
def matlab_reference_3mf() -> Path:
    """Path to the MATLAB-produced reference .3mf for e2e validation.

    Tests using this fixture should be marked `@pytest.mark.matlab_reference`
    so they are skipped if the file is absent (e.g., on a fresh checkout
    before Task 8 has run MATLAB).
    """
    p = FIXTURES / "reference.3mf"
    if not p.exists():
        pytest.skip(f"MATLAB reference .3mf not available at {p}")
    return p
