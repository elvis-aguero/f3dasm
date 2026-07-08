"""WaitForProcess: a backend-agnostic 'block until job done' closure.

Replaces reliance on the Claude-SDK-only Monitor tool (which has no Ollama
equivalent and would break parity). Declaration-gated like every capability
tool; declared by the implementer/datagenerator/debugger.
"""
from __future__ import annotations

import subprocess
import sys

from f3dasm._src.agentic.nodes.tools.routing import (
    build_declared_shared_closures,
)


class _Node:
    """Minimal node — WaitForProcess needs nothing from the run context."""
    _current_notes_dir = None
    _delegation_log = None
    _ledger = None


def _wait_tool(agent_tools):
    return build_declared_shared_closures(_Node(), frozenset(agent_tools))


def test_waitforprocess_declaration_gated():
    assert "WaitForProcess" in _wait_tool({"WaitForProcess"})
    assert "WaitForProcess" not in _wait_tool({"Read"})


def test_waitforprocess_returns_when_process_exits():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"])
    tool = _wait_tool({"WaitForProcess"})["WaitForProcess"]
    out = tool(proc.pid, timeout_s=10, poll_s=0.05)
    assert "exited" in out.lower(), out
    assert proc.poll() is not None  # actually finished


def test_waitforprocess_reports_not_running_for_dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    tool = _wait_tool({"WaitForProcess"})["WaitForProcess"]
    out = tool(proc.pid, timeout_s=5, poll_s=0.05)
    assert "not running" in out.lower(), out


def test_waitforprocess_times_out_without_killing():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        tool = _wait_tool({"WaitForProcess"})["WaitForProcess"]
        out = tool(proc.pid, timeout_s=1, poll_s=0.1)
        assert "timeout" in out.lower(), out
        assert proc.poll() is None  # NOT killed — still running
    finally:
        proc.kill()
        proc.wait()
