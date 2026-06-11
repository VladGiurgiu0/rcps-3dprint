"""PackingGeneration (Baranau) integration: build, configure, run, rescale.

Reproduces the proven workflow from ``notebooks/packing_generation.ipynb``:

1. ``git clone https://github.com/VasiliBaranov/packing-generation``
   then ``make`` in ``_Release`` (auto-build path), or the user points
   the GUI at an existing ``PackingGeneration.exe`` (manual fallback).
2. Write ``diameters.txt`` (uniform nominal d) + ``generation.conf``.
3. Three-stage run: ``-fba`` (Generation start: 1), then ``-ls`` and
   ``-lsgd`` (Generation start: 0), removing ``packing.nfo`` between
   stages.
4. Apply the Baranau diameter rescale (issue #30):
   ``d_true = d_stored x ((1 - phi_final) / (1 - phi_theoretical))^(1/3)``
   writing the TRUE diameters into packing.xyzd. Unlike the notebook,
   the .nfo is left untouched; the full provenance (requested vs
   achieved porosity, scaling factor) goes into ``packing_meta.json``.

Terminology that the UI surfaces (audited 2026-06-11): the generator is
*asked* for a porosity (e.g. 0.350 at d = 6.0) but typically jams
earlier (phi ~ 0.363 for monodisperse); after the rescale, packing.xyzd
holds the true tangent-contact diameters. Printing at expansion 1.0
reproduces that true packing; rescaling up to the nominal d creates
small contact overlaps (stronger printed bonds, lower phi).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from rcps_gui.jobs import Job

PACKGEN_REPO = "https://github.com/VasiliBaranov/packing-generation.git"
EXE_NAME = "PackingGeneration.exe"

CONF_TEMPLATE = """Particles count: {n}
Packing size: {bx:.6g} {by:.6g} {bz:.6g}
Generation start: {start}
Seed: {seed}
Steps to write: {steps_to_write}
Boundaries mode: 1
Contraction rate: {contraction_rate:.6g}
"""

DEFAULTS: dict[str, Any] = {
    # the proven data_example workflow (notebook, seed 809)
    "n_particles": 718,
    "box_mm": [50.0, 50.0, 50.0],
    "d_nominal_mm": 6.0,
    "seed": 809,
    "contraction_rate": 1e-3,
    "steps_to_write": 1000,
    "stages": ["fba", "ls", "lsgd"],
}


# =====================================================================
# Executable discovery / build
# =====================================================================

def validate_exe(path: str | Path) -> dict[str, Any]:
    """Check that the given path looks like a runnable PackingGeneration."""
    p = Path(path).expanduser()
    ok = p.is_file()
    executable = ok and bool(p.stat().st_mode & 0o111)
    return {
        "path": str(p),
        "exists": ok,
        "executable": executable,
        "ok": ok and executable,
    }


def _cxx_flags(job: Job, workdir: Path) -> list[str]:
    """Preflight the C++ toolchain; return extra flags if libc++ is broken.

    Known macOS failure mode (seen 2026-06-11): after an OS/CLT update,
    ``g++`` cannot find ``<string>`` because the CommandLineTools-local
    ``c++/v1`` headers are stale while the SDK's copy is intact. The fix
    is ``-nostdinc++ -isystem <SDK>/usr/include/c++/v1``.
    """
    import subprocess as sp

    test = workdir / "_cxx_test.cpp"
    test.write_text("#include <string>\nint main(){return 0;}\n")
    out = workdir / "_cxx_test.out"

    def compiles(extra: list[str]) -> bool:
        r = sp.run(["g++", *extra, str(test), "-o", str(out)],
                   capture_output=True, text=True)
        return r.returncode == 0

    try:
        if compiles([]):
            return []
        job.log("toolchain preflight: g++ cannot find <string> - trying the "
                "macOS SDK libc++ fallback ...")
        r = sp.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True)
        sdk = r.stdout.strip()
        if sdk:
            flags = ["-nostdinc++", "-isystem", f"{sdk}/usr/include/c++/v1"]
            if compiles(flags):
                job.log(f"preflight OK with SDK libc++: {sdk}")
                return flags
        raise RuntimeError(
            "the C++ toolchain cannot compile a trivial program "
            "(<string> not found). On macOS, reinstall the Command Line "
            "Tools: sudo rm -rf /Library/Developer/CommandLineTools && "
            "xcode-select --install - or use 'select existing executable'."
        )
    finally:
        test.unlink(missing_ok=True)
        out.unlink(missing_ok=True)


def clone_and_build(job: Job, dest_dir: str | Path) -> str:
    """Clone PackingGeneration and compile it directly. Returns exe path.

    We compile the 89 ``PackingGeneration/**/*.cpp`` sources with one
    compiler invocation instead of running the repo's Eclipse-generated
    makefile: the makefile also builds the ``Tests/`` tree, which fails
    under modern clang (template strictness in Assert.h) and is not
    needed for the executable.
    """
    dest = Path(dest_dir).expanduser() / "packing-generation"
    if not shutil.which("git"):
        raise RuntimeError(
            "git not found on PATH - install git, or use 'select existing "
            "executable' instead."
        )
    if not (shutil.which("g++") or shutil.which("c++")):
        raise RuntimeError(
            "no C++ compiler found (on macOS: xcode-select --install), "
            "or use 'select existing executable' instead."
        )
    if dest.exists():
        job.log(f"repo already present at {dest}; skipping clone")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        job.run_streamed(["git", "clone", "--depth", "1", PACKGEN_REPO, str(dest)])

    extra = _cxx_flags(job, dest)
    sources = sorted(str(p) for p in (dest / "PackingGeneration").rglob("*.cpp"))
    job.log(f"compiling {len(sources)} sources (one invocation, ~1 min) ...")
    rel = dest / "_Release"
    rel.mkdir(exist_ok=True)
    exe = rel / EXE_NAME
    job.run_streamed([
        "g++", "-std=c++14", *extra,
        "-DBOOST_DISABLE_ASSERTS", "-DNDEBUG",
        f"-I{dest / 'Externals' / 'Eigen'}",
        f"-I{dest / 'Externals' / 'Boost'}",
        f"-I{dest / 'PackingGeneration'}",
        "-O3", "-funroll-loops", "-w",
        *sources, "-o", str(exe),
    ])
    if not exe.exists():
        raise RuntimeError(f"build finished but {exe} not found")
    exe.chmod(exe.stat().st_mode | 0o111)
    job.log(f"built: {exe}")
    return str(exe)


# =====================================================================
# Configuration + run
# =====================================================================

def write_inputs(workdir: Path, *, n: int, box: list[float], d_nominal: float,
                 seed: int, contraction_rate: float, start: int,
                 steps_to_write: int = 1000) -> str:
    """Write ``diameters.txt`` and ``generation.conf``; return the conf text."""
    workdir.mkdir(parents=True, exist_ok=True)
    np.savetxt(workdir / "diameters.txt", np.repeat(float(d_nominal), int(n)))
    conf = CONF_TEMPLATE.format(
        n=int(n), bx=box[0], by=box[1], bz=box[2],
        start=int(start), seed=int(seed),
        steps_to_write=int(steps_to_write),
        contraction_rate=float(contraction_rate),
    )
    (workdir / "generation.conf").write_text(conf)
    return conf


def parse_nfo_porosities(nfo_path: Path) -> tuple[float, float]:
    """Return (theoretical, final) porosity from a .nfo file."""
    text = nfo_path.read_text(errors="ignore")
    mt = re.search(r"Theoretical\s+Porosity:\s*([0-9.eE+-]+)", text)
    mf = re.search(r"Final\s+Porosity:\s*([0-9.eE+-]+)", text)
    if not (mt and mf):
        raise ValueError(f"could not parse porosities from {nfo_path}")
    return float(mt.group(1)), float(mf.group(1))


def apply_issue30_rescale(workdir: Path, job: Job | None = None) -> dict[str, Any]:
    """Rescale packing.xyzd diameters to their TRUE values (Baranau #30).

    Non-destructive bookkeeping: the .nfo is left as the generator wrote
    it; provenance goes to ``packing_meta.json``. Guarded against double
    application via the meta file.
    """
    xyzd = workdir / "packing.xyzd"
    nfo = workdir / "packing.nfo"
    meta_path = workdir / "packing_meta.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("rescale_applied"):
            if job:
                job.log("rescale already applied (packing_meta.json) - skipping")
            return meta

    phi_theor, phi_final = parse_nfo_porosities(nfo)
    scaling = ((1.0 - phi_final) / (1.0 - phi_theor)) ** (1.0 / 3.0)

    packing = np.fromfile(xyzd, dtype="<f8").reshape(-1, 4)
    packing[:, 3] *= scaling
    packing.tofile(xyzd)

    box_vol = None
    phi_true = None
    # recompute true porosity for the meta record (box from generation.conf)
    conf = (workdir / "generation.conf").read_text()
    mb = re.search(r"Packing size:\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)", conf)
    if mb:
        box_vol = float(mb.group(1)) * float(mb.group(2)) * float(mb.group(3))
        phi_true = 1.0 - float(np.sum(np.pi / 6.0 * packing[:, 3] ** 3)) / box_vol

    meta = {
        "rescale_applied": True,
        "scaling_factor": scaling,
        "phi_requested_at_nominal_d": phi_theor,
        "phi_final_reported_by_generator": phi_final,
        "phi_true_after_rescale": phi_true,
        "mean_true_diameter_mm": float(np.mean(packing[:, 3])),
        "n_spheres": int(packing.shape[0]),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    if job:
        if phi_true is not None:
            job.log(f"diameters rescaled to true jammed size "
                    f"(packing-generation issue #30 convention): "
                    f"x{scaling:.6f} -> mean d = "
                    f"{meta['mean_true_diameter_mm']:.4f} mm, "
                    f"phi_true = {phi_true:.4f}")
        else:
            job.log(f"diameters rescaled to true jammed size "
                    f"(issue #30 convention): x{scaling:.6f}")
    return meta


def run_packing(job: Job, workdir: str | Path, exe: str | Path,
                params: dict[str, Any]) -> dict[str, Any]:
    """Full packing simulation: fba -> ls -> lsgd -> rescale."""
    wd = Path(workdir).expanduser()
    exe = str(Path(exe).expanduser())
    p = {**DEFAULTS, **params}
    n = int(p["n_particles"])
    box = [float(x) for x in p["box_mm"]]
    d_nom = float(p["d_nominal_mm"])
    stages = list(p["stages"])

    # stale outputs from a previous run must not survive
    for f in ["packing.xyzd", "packing_init.xyzd", "packing_prev.xyzd",
              "contraction_energies.txt", "packing.nfo", "packing_meta.json"]:
        (wd / f).unlink(missing_ok=True)

    nominal_phi = 1.0 - n * (np.pi / 6.0) * d_nom**3 / (box[0] * box[1] * box[2])
    job.log(f"requested: N={n}, box={box} mm, d={d_nom} mm "
            f"-> nominal phi = {nominal_phi:.4f}")

    job.log(f"diameters.txt: {n} x {d_nom} mm (uniform)")
    first = True
    for stage in stages:
        if job.cancelled:
            raise RuntimeError("cancelled")
        conf = write_inputs(wd, n=n, box=box, d_nominal=d_nom,
                            seed=int(p["seed"]),
                            contraction_rate=float(p["contraction_rate"]),
                            start=1 if first else 0,
                            steps_to_write=int(p.get("steps_to_write", 1000)))
        if not first:
            (wd / "packing.nfo").unlink(missing_ok=True)
        job.log(f"--- stage -{stage} ---")
        job.log("generation.conf:")
        for line in conf.strip().splitlines():
            job.log(f"    {line}")
        job.run_streamed([exe, f"-{stage}"], cwd=str(wd))
        first = False

    meta = apply_issue30_rescale(wd, job)
    meta["workdir"] = str(wd)
    return meta


# =====================================================================
# Preview data
# =====================================================================

def packing_preview(workdir: str | Path) -> dict[str, Any]:
    """Sphere centers + diameters + box for the three.js preview."""
    wd = Path(workdir).expanduser()
    xyzd = wd / "packing.xyzd"
    if not xyzd.exists():
        raise FileNotFoundError(f"no packing.xyzd in {wd} - run the packing first")
    S = np.fromfile(xyzd, dtype="<f8").reshape(-1, 4)

    box = [50.0, 50.0, 50.0]
    conf = wd / "generation.conf"
    if conf.exists():
        mb = re.search(r"Packing size:\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)",
                       conf.read_text())
        if mb:
            box = [float(mb.group(i)) for i in (1, 2, 3)]

    v_box = box[0] * box[1] * box[2]
    phi = 1.0 - float(np.sum(np.pi / 6.0 * S[:, 3] ** 3)) / v_box
    meta_path = wd / "packing_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None

    # "design" option info: porosity at NOMINAL diameters (meta, else nfo
    # Theoretical); None if neither is available (xyzd loaded alone).
    design = None
    phi_target = None
    if meta and meta.get("phi_requested_at_nominal_d") is not None:
        phi_target = float(meta["phi_requested_at_nominal_d"])
    else:
        nfo = wd / "packing.nfo"
        if nfo.exists():
            try:
                phi_target = parse_nfo_porosities(nfo)[0]   # theoretical
            except ValueError:
                phi_target = None
    if phi_target is not None:
        v_solid = float(np.sum(np.pi / 6.0 * S[:, 3] ** 3))
        factor = (((1.0 - phi_target) * v_box) / v_solid) ** (1.0 / 3.0)
        design = {
            "phi": round(phi_target, 6),
            "factor": round(factor, 6),
            "mean_d_mm": round(float(np.mean(S[:, 3])) * factor, 4),
        }

    return {
        "n": int(S.shape[0]),
        "box_mm": box,
        "phi": round(phi, 6),
        "mean_d_mm": round(float(np.mean(S[:, 3])), 6),
        "centers": np.round(S[:, :3], 4).tolist(),
        "diameters": np.round(S[:, 3], 4).tolist(),
        "meta": meta,
        "design": design,
    }
