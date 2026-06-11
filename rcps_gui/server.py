"""Local web server for the RCPS GUI (stdlib http.server, JSON API).

Run with ``rcps-gui`` (console script). Everything stays on-device; the
only network access the GUI itself may trigger is the optional
``git clone`` of the packing generator and the three.js CDN fallback
for the 3D previews.
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from rcps._version import __version__
from rcps_gui import meshjob, packgen
from rcps_gui.jobs import JobManager

STATIC_DIR = Path(__file__).parent / "static"

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}


class AppState:
    """Mutable GUI state, persisted to <project>/.rcps_gui_state.json.

    Run model: every packing (freshly simulated OR loaded from a file)
    gets its own folder under ``<project>/output/run_<timestamp>_<tag>/``
    holding the packing files plus all meshes/exports derived from it.
    ``current_run`` points at the active one.
    """

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.jobs = JobManager()
        self.data: dict[str, Any] = {
            "project_dir": str(project_dir),
            "exe_path": None,
            "current_run": None,
            "pack_params": dict(packgen.DEFAULTS),
            "mesh_params": {"vox_mm": 1.0, "bridge_mode": "cylinders",
                            "radius_frac": 0.15, "export_what": "beads",
                            "diameter": "stored"},
            "export_params": {"vox_mm": 0.1, "grid": [1, 1, 1]},
        }
        self._lock = threading.Lock()
        self._load()

    @property
    def _state_path(self) -> Path:
        return self.project_dir / ".rcps_gui_state.json"

    def _load(self) -> None:
        if self._state_path.exists():
            try:
                saved = json.loads(self._state_path.read_text())
                self.data.update(saved)
                self.data["project_dir"] = str(self.project_dir)
            except (json.JSONDecodeError, OSError):
                pass

    def save(self) -> None:
        with self._lock:
            self.project_dir.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(self.data, indent=2))

    def new_run(self, tag: str) -> Path:
        """Create <project>/output/run_<timestamp>_<tag>/ and make it current."""
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run = self.project_dir / "output" / f"run_{stamp}_{tag}"
        n = 1
        while run.exists():            # same-second collisions
            n += 1
            run = self.project_dir / "output" / f"run_{stamp}_{tag}_{n}"
        run.mkdir(parents=True)
        self.data["current_run"] = str(run)
        self.save()
        return run

    @property
    def current_run_path(self) -> Path | None:
        cr = self.data.get("current_run")
        return Path(cr) if cr else None

    @property
    def packing_path(self) -> Path:
        cr = self.current_run_path
        return (cr / "packing.xyzd") if cr else self.project_dir / "packing.xyzd"

    def list_runs(self) -> list[str]:
        """Run folder names under <project>/output, newest first."""
        out = self.project_dir / "output"
        if not out.exists():
            return []
        return sorted((p.name for p in out.iterdir()
                       if p.is_dir() and p.name.startswith("run_")),
                      reverse=True)

    def select_run(self, name: str) -> Path:
        run = (self.project_dir / "output" / name).resolve()
        if run.parent != (self.project_dir / "output").resolve() or not run.is_dir():
            raise ValueError(f"unknown run: {name}")
        self.data["current_run"] = str(run)
        self.save()
        return run

    def run_status(self) -> dict[str, bool]:
        """What the current run folder already contains (drives the
        sidebar progress dots)."""
        cr = self.current_run_path
        if cr is None or not cr.exists():
            return {"pack": False, "mesh": False, "export": False}
        return {
            "pack": (cr / "packing.xyzd").exists(),
            "mesh": any((cr / "preview").glob("*.3mf")) if (cr / "preview").exists() else False,
            "export": any((cr / "export").rglob("*.3mf"))
                      or (cr / "export" / "facility_map.json").exists()
                      if (cr / "export").exists() else False,
        }


def make_handler(state: AppState):  # noqa: C901 — routing table
    """Build the request-handler class bound to the given state."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"rcps-gui/{__version__}"

        # ---- helpers ----
        def _json(self, obj: Any, code: int = 200) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _bytes(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, msg: str, code: int = 400) -> None:
            self._json({"error": msg}, code)

        def _read_json(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            if n == 0:
                return {}
            return json.loads(self.rfile.read(n).decode())

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            pass

        # ---- GET ----
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            try:
                if path in ("/", "/index.html"):
                    self._static("index.html")
                elif path.startswith("/static/"):
                    self._static(path[len("/static/"):])
                elif path == "/api/state":
                    self._json({**state.data,
                                "packing_exists": state.packing_path.exists(),
                                "run_status": state.run_status(),
                                "runs": state.list_runs(),
                                "version": __version__})
                elif path.startswith("/api/job/"):
                    job = state.jobs.get(path.rsplit("/", 1)[-1])
                    if job is None:
                        self._error("unknown job", 404)
                    else:
                        self._json(job.to_dict())
                elif path == "/api/pack/preview":
                    cr = state.current_run_path
                    if cr is None:
                        self._error("no packing yet - run or load one first", 404)
                    else:
                        self._json(packgen.packing_preview(cr))
                elif path.startswith("/api/mesh/data/"):
                    job = state.jobs.get(path.rsplit("/", 1)[-1])
                    if job is None or job.result_bytes is None:
                        self._error("no mesh buffer for that job", 404)
                    else:
                        self._bytes(job.result_bytes, "application/octet-stream")
                elif path.startswith("/api/sdf/slice/"):
                    from urllib.parse import parse_qs, urlparse

                    q = parse_qs(urlparse(self.path).query)
                    job = state.jobs.get(path.rsplit("/", 1)[-1])
                    if job is None or job.volume is None:
                        self._error("no field for that job", 404)
                    else:
                        self._bytes(
                            meshjob.sdf_slice(job,
                                              int(q.get("axis", ["2"])[0]),
                                              int(q.get("i", ["0"])[0])),
                            "application/octet-stream")
                else:
                    self._error("not found", 404)
            except FileNotFoundError as e:
                self._error(str(e), 404)
            except Exception as e:  # noqa: BLE001
                self._error(str(e), 500)

        def _static(self, rel: str) -> None:
            f = (STATIC_DIR / rel).resolve()
            if not str(f).startswith(str(STATIC_DIR.resolve())) or not f.is_file():
                self._error("not found", 404)
                return
            self._bytes(f.read_bytes(), MIME.get(f.suffix, "application/octet-stream"))

        # ---- POST ----
        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            try:
                body = self._read_json()
                if path == "/api/setup/exe":
                    info = packgen.validate_exe(body["path"])
                    if info["ok"]:
                        state.data["exe_path"] = info["path"]
                        state.save()
                    self._json(info)

                elif path == "/api/setup/clone":
                    # clones into <project>/packing-generation
                    dest = body.get("dest") or str(state.project_dir)

                    def _target(job):
                        exe = packgen.clone_and_build(job, dest)
                        state.data["exe_path"] = exe
                        state.save()
                        job.result = {"exe_path": exe}

                    self._json(state.jobs.start("clone+build", _target).to_dict())

                elif path == "/api/runs/new":
                    run = state.new_run("new")
                    self._json({"ok": True, "run": str(run),
                                "runs": state.list_runs()})

                elif path == "/api/runs/select":
                    run = state.select_run(body["name"])
                    self._json({"ok": True, "run": str(run),
                                "run_status": state.run_status()})

                elif path == "/api/pack/run":
                    exe = state.data.get("exe_path")
                    if not exe:
                        self._error("no packing-generator executable configured")
                        return
                    params = {**state.data["pack_params"], **body}
                    state.data["pack_params"] = params
                    # reuse the current run if it is still empty (e.g. just
                    # created via the + button); otherwise start a fresh one
                    cr = state.current_run_path
                    if cr is not None and cr.exists() and not (cr / "packing.xyzd").exists():
                        run_dir = cr
                        state.save()
                    else:
                        run_dir = state.new_run("packed")

                    def _target(job):
                        job.result = packgen.run_packing(job, run_dir, exe, params)

                    self._json(state.jobs.start("packing", _target).to_dict())

                elif path == "/api/mesh/preview":
                    run_dir = state.current_run_path
                    if run_dir is None:
                        self._error("no packing yet - run or load one first")
                        return
                    params = {**state.data["mesh_params"], **body}
                    params["box_mm"] = state.data["pack_params"]["box_mm"]
                    params["d_nominal_mm"] = state.data["pack_params"]["d_nominal_mm"]
                    state.data["mesh_params"] = {
                        k: params[k] for k in
                        ("vox_mm", "bridge_mode", "radius_frac",
                         "export_what", "diameter", "backend", "iso2mesh")
                        if k in params}
                    state.save()
                    packing = state.packing_path

                    def _target(job):
                        meshjob.preview_job(job, packing, run_dir, params)

                    self._json(state.jobs.start("mesh-preview", _target).to_dict())

                elif path == "/api/sdf/compute":
                    run_dir = state.current_run_path
                    if run_dir is None:
                        self._error("no packing yet - run or load one first")
                        return
                    params = {**state.data["mesh_params"], **body}
                    params["box_mm"] = state.data["pack_params"]["box_mm"]
                    packing = state.packing_path

                    # free the previous field volume (can be hundreds of MB)
                    prev = state.data.get("_last_sdf_job")
                    if prev:
                        pj = state.jobs.get(prev)
                        if pj is not None:
                            pj.volume = None

                    def _target(job):
                        meshjob.sdf_job(job, packing, params)

                    job = state.jobs.start("sdf-field", _target)
                    state.data["_last_sdf_job"] = job.id
                    self._json(job.to_dict())

                elif path == "/api/export/run":
                    run_dir = state.current_run_path
                    if run_dir is None:
                        self._error("no packing yet - run or load one first")
                        return
                    params = {**state.data["mesh_params"],
                              **state.data["export_params"], **body}
                    params["box_mm"] = state.data["pack_params"]["box_mm"]
                    params["d_nominal_mm"] = state.data["pack_params"]["d_nominal_mm"]
                    state.data["export_params"] = {
                        "vox_mm": params["vox_mm"], "grid": params["grid"]}
                    state.save()
                    packing = state.packing_path

                    def _target(job):
                        meshjob.export_job(job, packing, run_dir, params)

                    self._json(state.jobs.start("export", _target).to_dict())

                elif path.startswith("/api/job/") and path.endswith("/cancel"):
                    job = state.jobs.get(path.split("/")[3])
                    if job is None:
                        self._error("unknown job", 404)
                    else:
                        job.cancel()
                        self._json(job.to_dict())

                elif path == "/api/use_example":
                    # load an existing packing.xyzd into a fresh run folder
                    src = body.get("source")
                    if not src:
                        self._error("source path required")
                        return
                    srcp = Path(src).expanduser()
                    if not srcp.exists():
                        self._error(f"not found: {srcp}", 404)
                        return
                    run_dir = state.new_run("loaded")
                    shutil.copy2(srcp, run_dir / "packing.xyzd")
                    nfo = srcp.with_suffix(".nfo")
                    if nfo.exists():
                        shutil.copy2(nfo, run_dir / "packing.nfo")
                    meta = srcp.parent / "packing_meta.json"
                    if meta.exists():
                        shutil.copy2(meta, run_dir / "packing_meta.json")
                    self._json({"ok": True, "packing": str(run_dir / "packing.xyzd"),
                                "run": str(run_dir)})

                else:
                    self._error("not found", 404)
            except KeyError as e:
                self._error(f"missing field {e}")
            except Exception as e:  # noqa: BLE001
                self._error(str(e), 500)

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="rcps-gui",
        description="Local browser GUI for the RCPS packing -> mesh -> .3mf workflow.",
    )
    ap.add_argument("--project", type=Path, default=Path.cwd(),
                    help=("Working folder (default: the current directory). "
                          "Runs go to <folder>/output/run_*, the packing "
                          "generator is cloned to <folder>/packing-generation."))
    ap.add_argument("--port", type=int, default=8421)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--version", action="version",
                    version=f"rcps-3dprint {__version__}")
    args = ap.parse_args(argv)

    project = args.project.expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)
    state = AppState(project)
    state.save()

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"rcps-gui {__version__}")
    print(f"  project: {project}")
    print(f"  serving: {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
