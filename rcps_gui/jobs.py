"""Threaded background jobs with streamed logs (stdlib only).

Long-running work (clone+build, packing simulation, meshing) runs in
daemon threads; the browser polls ``GET /api/job/<id>`` for status and
the log tail. Subprocess-based jobs are cancellable (the process is
killed); pure-NumPy meshing jobs are not (documented in the UI).
"""

from __future__ import annotations

import itertools
import subprocess
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from typing import Any


class Job:
    """One background job: status, rolling log, optional result."""

    def __init__(self, job_id: str, name: str) -> None:
        self.id = job_id
        self.name = name
        self.status = "pending"          # pending | running | done | error | cancelled
        self.created = time.time()
        self.finished: float | None = None
        self.error: str | None = None
        self.result: dict[str, Any] = {}
        self.result_bytes: bytes | None = None   # binary payloads (mesh buffers)
        self.volume = None                       # 3D float32 ndarray (SDF slicer)
        self._log: deque[str] = deque(maxlen=800)
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._proc: subprocess.Popen | None = None

    # ---- logging ----
    def log(self, line: str) -> None:
        with self._lock:
            self._log.append(line.rstrip("\n"))

    def log_tail(self, n: int = 60) -> list[str]:
        with self._lock:
            return list(self._log)[-n:]

    # ---- cancellation ----
    def cancel(self) -> None:
        self._cancel.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # ---- subprocess helper: run + stream stdout into the log ----
    def run_streamed(self, cmd: list[str], *, cwd: str | None = None,
                     check: bool = True) -> int:
        """Run a subprocess, streaming combined output into the job log."""
        self.log(f"$ {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self._proc = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            self.log(line)
            if self.cancelled:
                proc.kill()
                break
        rc = proc.wait()
        self._proc = None
        if self.cancelled:
            raise RuntimeError("cancelled")
        if check and rc != 0:
            raise RuntimeError(f"command failed with exit code {rc}: {' '.join(cmd)}")
        return rc

    def to_dict(self, tail: int = 60) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "error": self.error,
            "result": self.result,
            "log": self.log_tail(tail),
            "elapsed_s": round((self.finished or time.time()) - self.created, 1),
        }


class JobManager:
    """Create and look up jobs; one daemon thread per job."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._counter = itertools.count(1)
        self._lock = threading.Lock()

    def start(self, name: str, target: Callable[[Job], None]) -> Job:
        with self._lock:
            job = Job(f"job{next(self._counter)}", name)
            self._jobs[job.id] = job

        def _run() -> None:
            job.status = "running"
            job.log(f"[{job.name}] started")
            try:
                target(job)
                job.status = "cancelled" if job.cancelled else "done"
            except Exception as e:  # noqa: BLE001 — surfaced to the UI
                job.status = "cancelled" if job.cancelled else "error"
                job.error = str(e)
                job.log(traceback.format_exc(limit=6))
            finally:
                job.finished = time.time()
                job.log(f"[{job.name}] {job.status}")

        threading.Thread(target=_run, daemon=True, name=f"rcps-gui-{job.id}").start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def any_running(self) -> bool:
        return any(j.status == "running" for j in self._jobs.values())
