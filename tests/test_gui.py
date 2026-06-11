"""Tests for the rcps_gui backend (stdlib server, jobs, packgen wrapper).

All tests here are dependency-light (numpy + stdlib): the heavy meshing
endpoints are exercised manually / by the e2e suites. The server smoke
test boots the real HTTP server on an ephemeral port and exercises the
JSON API end-to-end (state, example loading, packing preview, dry-run
export via the facility orchestrator, job polling).
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from rcps_gui import packgen
from rcps_gui.jobs import JobManager

# =====================================================================
# jobs
# =====================================================================

class TestJobs:

    def test_job_lifecycle_done(self):
        jm = JobManager()
        job = jm.start("ok", lambda j: j.log("hello") or j.result.update(x=1))
        for _ in range(100):
            if job.status == "done":
                break
            time.sleep(0.01)
        assert job.status == "done"
        assert job.result == {"x": 1}
        assert "hello" in "\n".join(job.log_tail())

    def test_job_error_is_captured(self):
        jm = JobManager()

        def boom(j):
            raise RuntimeError("kaput")

        job = jm.start("bad", boom)
        for _ in range(100):
            if job.status == "error":
                break
            time.sleep(0.01)
        assert job.status == "error"
        assert "kaput" in job.error

    def test_run_streamed_logs_and_raises(self, tmp_path):
        jm = JobManager()

        def target(j):
            j.run_streamed(["python3", "-c", "print('from-subprocess')"])

        job = jm.start("stream", target)
        for _ in range(300):
            if job.status in ("done", "error"):
                break
            time.sleep(0.01)
        assert job.status == "done"
        assert any("from-subprocess" in ln for ln in job.log_tail())


# =====================================================================
# packgen wrapper
# =====================================================================

class TestPackgen:

    def test_conf_writer_matches_notebook_format(self, tmp_path):
        packgen.write_inputs(tmp_path, n=718, box=[50, 50, 50], d_nominal=6.0,
                             seed=809, contraction_rate=1e-3, start=1)
        conf = (tmp_path / "generation.conf").read_text()
        assert "Particles count: 718" in conf
        assert "Packing size: 50 50 50" in conf
        assert "Generation start: 1" in conf
        assert "Seed: 809" in conf
        assert "Boundaries mode: 1" in conf
        assert "Contraction rate: 0.001" in conf
        d = np.loadtxt(tmp_path / "diameters.txt")
        assert d.shape == (718,) and np.allclose(d, 6.0)

    def test_validate_exe(self, tmp_path):
        assert not packgen.validate_exe(tmp_path / "missing")["ok"]
        f = tmp_path / "fake.exe"
        f.write_text("#!/bin/sh\n")
        f.chmod(0o755)
        assert packgen.validate_exe(f)["ok"]

    def test_issue30_rescale_roundtrip(self, tmp_path):
        # synthetic 2-sphere packing at d=6 in a 20mm box; nfo says the
        # generator achieved phi_final (true d smaller than nominal).
        d_nom, box = 6.0, 20.0
        S = np.array([[5, 5, 5, d_nom], [15, 15, 15, d_nom]], dtype="<f8")
        S.tofile(tmp_path / "packing.xyzd")
        v_nom = 2 * np.pi / 6 * d_nom**3
        phi_theor = 1 - v_nom / box**3
        scaling_true = 0.99
        phi_final = 1 - (1 - phi_theor) * scaling_true**3
        (tmp_path / "packing.nfo").write_text(
            f"N: 2\n Dimensions: 20 20 20\n"
            f" Theoretical Porosity: {phi_theor}\n"
            f"Final Porosity: {phi_final} (Tolerance: 1.0001)\n")
        (tmp_path / "generation.conf").write_text(
            "Particles count: 2\nPacking size: 20 20 20\n")

        meta = packgen.apply_issue30_rescale(tmp_path)
        assert abs(meta["scaling_factor"] - scaling_true) < 1e-12
        S2 = np.fromfile(tmp_path / "packing.xyzd", dtype="<f8").reshape(-1, 4)
        assert np.allclose(S2[:, 3], d_nom * scaling_true)
        assert abs(meta["phi_true_after_rescale"] - phi_final) < 1e-9

        # second application must be a no-op (guarded by the meta file)
        meta2 = packgen.apply_issue30_rescale(tmp_path)
        S3 = np.fromfile(tmp_path / "packing.xyzd", dtype="<f8").reshape(-1, 4)
        assert np.allclose(S3[:, 3], S2[:, 3])
        assert meta2["rescale_applied"]

    def test_packing_preview(self, tmp_path):
        S = np.array([[5, 5, 5, 6.0]], dtype="<f8")
        S.tofile(tmp_path / "packing.xyzd")
        (tmp_path / "generation.conf").write_text(
            "Particles count: 1\nPacking size: 20 20 20\n"
            "Generation start: 1\nSeed: 1\n")
        prev = packgen.packing_preview(tmp_path)
        assert prev["n"] == 1
        assert prev["box_mm"] == [20.0, 20.0, 20.0]
        assert abs(prev["phi"] - (1 - (np.pi / 6 * 216) / 8000)) < 1e-6


# =====================================================================
# HTTP API smoke (real server, ephemeral port)
# =====================================================================

@pytest.fixture()
def gui_server(tmp_path):
    from rcps_gui.server import AppState, make_handler

    state = AppState(tmp_path / "project")
    state.save()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", state
    httpd.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode())


def _post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


class TestApiSmoke:

    def test_state_and_static(self, gui_server):
        base, _ = gui_server
        s = _get(f"{base}/api/state")
        assert "project_dir" in s and not s["packing_exists"]
        with urllib.request.urlopen(f"{base}/", timeout=10) as r:
            assert b"RCPS" in r.read()

    def test_example_load_preview_and_dry_run_export(self, gui_server, packing_xyzd_path):
        base, state = gui_server

        r = _post(f"{base}/api/use_example", {"source": str(packing_xyzd_path)})
        assert r["ok"]
        assert state.packing_path.exists()
        # run-folder model: every packing lives in its own output/run_* folder
        assert "output" in state.packing_path.parts
        assert state.packing_path.parent.name.startswith("run_")
        assert "_loaded" in state.packing_path.parent.name

        prev = _get(f"{base}/api/pack/preview")
        assert prev["n"] == 718
        assert abs(prev["phi"] - 0.3633) < 1e-3

        # dry-run facility export through the API (no meshing deps needed)
        job = _post(f"{base}/api/export/run",
                    {"grid": [2, 2, 1], "diameter": "design", "dry_run": True,
                     "vox_mm": 0.1})
        jid = job["id"]
        for _ in range(200):
            j = _get(f"{base}/api/job/{jid}")
            if j["status"] in ("done", "error"):
                break
            time.sleep(0.05)
        assert j["status"] == "done", j.get("error")
        assert j["result"]["n_types"] == 4
        assert Path(j["result"]["map_txt"]).exists()

    def test_unknown_job_404(self, gui_server):
        base, _ = gui_server
        with pytest.raises(urllib.error.HTTPError):
            _get(f"{base}/api/job/nope")
