"""Phase B — per-run Jupyter server + Jupyter-MCP wiring into authoring agents.

The live server is ONLY for authoring (the gate stays headless via nbclient).
These pin: (1) the server starts/serves/stops cleanly, and (2) the Jupyter MCP
is wired (with a minimal allowlist) into the strategizer + implementer only, and
only when a server is up.
"""
from __future__ import annotations

import urllib.request

import pytest

from f3dasm._src.agentic.agent_runtime import AgenticRun
from f3dasm._src.agentic.agents.critic import AdversarialCritiqueAgent
from f3dasm._src.agentic.agents.implementer import ImplementerAgent
from f3dasm._src.agentic.agents.strategizer import StrategizerAgent
from f3dasm._src.agentic.notebook_server import NotebookServer


def _reachable(url: str, token: str) -> bool:
    try:
        req = urllib.request.Request(
            f"{url}/api/status", headers={"Authorization": f"token {token}"})
        return urllib.request.urlopen(req, timeout=2).status == 200
    except Exception:  # noqa: BLE001
        return False


def test_notebook_server_starts_serves_and_stops(tmp_path):
    srv = NotebookServer(tmp_path / "run")
    srv.start(timeout=60)
    try:
        env = srv.mcp_env()
        assert env["JUPYTER_URL"].startswith("http://127.0.0.1:")
        assert env["JUPYTER_TOKEN"]
        assert _reachable(srv.url, srv.token)
    finally:
        srv.stop()
    assert not _reachable(srv.url, srv.token)  # gone after teardown


class _NoOutgoing:
    def outgoing(self, name):
        return []


class _FakeServer:
    def mcp_env(self):
        return {"JUPYTER_URL": "http://127.0.0.1:9", "JUPYTER_TOKEN": "tok"}


_JUPYTER_TOOLS = {
    "mcp__jupyter__use_notebook", "mcp__jupyter__insert_cell",
    "mcp__jupyter__execute_cell", "mcp__jupyter__read_notebook",
    "mcp__jupyter__edit_cell_source",
}


@pytest.mark.parametrize("name,agent", [
    ("strategizer", StrategizerAgent()), ("implementer", ImplementerAgent())])
def test_jupyter_mcp_wired_to_authoring_agents(tmp_path, name, agent):
    run = AgenticRun(study_dir=tmp_path)
    run._run_dir = None
    run._graph_spec = _NoOutgoing()
    run._notebook_server = _FakeServer()
    a = run._make_adapter(name, agent)
    assert "jupyter" in a.extra_mcp_servers
    assert a.extra_mcp_servers["jupyter"]["command"] == "uvx"
    assert _JUPYTER_TOOLS.issubset(set(a.extra_allowed_tools))
    # execute_code is deliberately NOT granted (oracle-bypass surface)
    assert "mcp__jupyter__execute_code" not in a.extra_allowed_tools


def test_jupyter_mcp_not_wired_to_critic_or_without_server(tmp_path):
    run = AgenticRun(study_dir=tmp_path)
    run._run_dir = None
    run._graph_spec = _NoOutgoing()
    # critic never authors → no jupyter even with a server
    run._notebook_server = _FakeServer()
    assert "jupyter" not in run._make_adapter(
        "critic", AdversarialCritiqueAgent()).extra_mcp_servers
    # no server → no jupyter for anyone
    run._notebook_server = None
    assert "jupyter" not in run._make_adapter(
        "strategizer", StrategizerAgent()).extra_mcp_servers
