"""A per-run Jupyter server, started on demand so the agents can author the
pipeline.ipynb deliverable cell-by-cell via the Jupyter MCP.

ONE server per run (the gate does NOT use this — it executes the notebook
headlessly via nbclient; only interactive *authoring* needs a live server).
Launched with THIS interpreter (so its kernels import f3dasm) and the run's env
(so authoring kernels resolve F3DASM_CANONICAL_STORE — the load-or-create cells
read the run's ledger). Localhost-only, random token, rooted at the run dir.
Context-managed with a hard teardown so it can never outlive the run.
"""
from __future__ import annotations

import contextlib
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

__all__ = ["NotebookServer"]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class NotebookServer:
    """Headless ``jupyter_server`` for one run. Use as a context manager::

        with NotebookServer(run_dir, env=run_env) as nb:
            mcp_env = nb.mcp_env()   # {"JUPYTER_URL": ..., "JUPYTER_TOKEN": ...}
    """

    def __init__(self, root_dir: str | Path, env: dict | None = None) -> None:
        self.root_dir = Path(root_dir)
        self._env = dict(env or os.environ)
        self.port = _free_port()
        self.token = secrets.token_urlsafe(16)
        self._proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def mcp_env(self) -> dict:
        """Env for the jupyter MCP client so it connects to THIS server."""
        return {"JUPYTER_URL": self.url, "JUPYTER_TOKEN": self.token}

    def start(self, timeout: float = 60.0) -> NotebookServer:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "jupyter_server", "--no-browser",
             f"--ServerApp.port={self.port}",
             f"--IdentityProvider.token={self.token}",
             f"--ServerApp.root_dir={self.root_dir}",
             "--ServerApp.disable_check_xsrf=True",
             "--ServerApp.open_browser=False"],
            env=self._env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,   # own process group → clean group teardown
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"jupyter_server exited early (rc={self._proc.returncode})")
            try:
                req = urllib.request.Request(
                    f"{self.url}/api/status",
                    headers={"Authorization": f"token {self.token}"})
                if urllib.request.urlopen(req, timeout=2).status == 200:
                    return self
            except Exception:  # noqa: BLE001 — not up yet
                time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"jupyter_server did not become ready in {timeout:.0f}s")

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        with contextlib.suppress(Exception):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

    def __enter__(self) -> NotebookServer:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
