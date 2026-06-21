"""Jupyter server lifecycle for the MCP-based notebook authoring flow.

One JupyterServer context manager is started per agentic run whenever the
graph contains an agent with ``needs_jupyter_server = True`` (i.e. the
strategizer). It starts ``uv run jupyter server`` in the project env so the
kernel has f3dasm installed, then ``uvx jupyter-mcp-server`` (started by the
Claude SDK per agent invocation) connects to it via ``--runtime-url`` +
``--runtime-token``.

Module-level ``_url`` / ``_token`` are set for the lifetime of the context
manager so ``_make_adapter`` (agent_runtime.py) can read them synchronously.
"""
from __future__ import annotations

import re
import subprocess
import threading

_url: str | None = None
_token: str | None = None

_URL_RE = re.compile(r"http://(?:localhost|127\.0\.0\.1):(\d+)/\?token=(\w+)")


def get_jupyter_mcp_config() -> dict:
    """Return the stdio MCP server config for uvx jupyter-mcp-server.

    Reads the module-level URL/token set by JupyterServer.__enter__.
    Raises RuntimeError if no server is running.
    """
    if _url is None or _token is None:
        raise RuntimeError(
            "No Jupyter server running — wrap the adapter construction "
            "with JupyterServer() first."
        )
    m = _URL_RE.match(_url)
    if not m:
        raise RuntimeError(f"Cannot parse Jupyter server URL: {_url!r}")
    port = m.group(1)
    base_url = f"http://localhost:{port}/"
    return {
        "command": "uvx",
        "args": [
            "jupyter-mcp-server",
            "--runtime-url", base_url,
            "--runtime-token", _token,
        ],
    }


class JupyterServer:
    """Context manager that starts a ``uv run jupyter server`` subprocess.

    Sets the module-level ``_url``/``_token`` on enter so that
    ``get_jupyter_mcp_config()`` can be called from ``_make_adapter`` during
    the lifetime of the context.

    Usage::

        with JupyterServer():
            graph = build_graph(...)
            graph.invoke(...)
    """

    _READY_TIMEOUT = 30  # seconds to wait for the server URL to appear

    def __enter__(self) -> "JupyterServer":
        global _url, _token
        import queue
        self._queue: queue.Queue[str | None] = queue.Queue()

        self._proc = subprocess.Popen(
            ["uv", "run", "jupyter", "server", "--no-browser", "--port=0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        def _reader():
            found = False
            for line in self._proc.stdout:
                if not found:
                    m = _URL_RE.search(line)
                    if m:
                        self._queue.put(m.group(0))
                        found = True
            if not found:
                self._queue.put(None)

        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()

        url = self._queue.get(timeout=self._READY_TIMEOUT)
        if url is None:
            self._proc.terminate()
            raise RuntimeError("Jupyter server exited before printing a URL.")

        m = _URL_RE.match(url)
        _url = url
        _token = m.group(2)
        return self

    def __exit__(self, *_) -> None:
        global _url, _token
        _url = None
        _token = None
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
