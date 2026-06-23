"""Commit 3: the ResourceBackend abstraction. OS-specifics live ONLY here.
Exercises the real psutil backend against actual child processes (incl. a
new-session grandchild that a process-group kill would miss)."""
from __future__ import annotations

import subprocess
import sys
import time

import pytest

from f3dasm._src.agentic.resource_backend import (
    PsutilBackend,
    ResourceBackend,
    StdlibBackend,
    get_resource_backend,
)


def test_get_backend_is_cached_and_conforms():
    b = get_resource_backend()
    assert isinstance(b, ResourceBackend)
    assert get_resource_backend() is b  # selected once


def test_psutil_available_so_psutil_backend_is_chosen():
    pytest.importorskip("psutil")
    assert isinstance(get_resource_backend(), PsutilBackend)


def test_read_rss_sees_a_child_and_kill_reaps_a_new_session_grandchild():
    pytest.importorskip("psutil")
    b = PsutilBackend()
    # Parent spawns a NEW-SESSION grandchild (start_new_session=True) — exactly
    # the detached-campaign shape that escapes os.killpg. Both sleep.
    parent = subprocess.Popen([
        sys.executable, "-c",
        "import subprocess,sys,time;"
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],"
        "start_new_session=True);"
        "time.sleep(120)",
    ])
    try:
        time.sleep(1.5)  # let the grandchild spawn
        rss = b.read_rss([parent.pid])
        assert rss > 0  # telemetry sees the tree
        n = b.kill([parent.pid])
        assert n >= 1
        for _ in range(40):
            if parent.poll() is not None:
                break
            time.sleep(0.1)
        assert parent.poll() is not None, "parent not reaped"
    finally:
        if parent.poll() is None:
            parent.kill()


def test_read_rss_unknown_pid_is_zero_not_error():
    assert get_resource_backend().read_rss([999_999]) == 0


def test_set_self_limit_returns_bool():
    # Don't actually clamp THIS test process tight; just assert it runs + types.
    b = get_resource_backend()
    assert isinstance(b.set_self_limit(8 * 1024 ** 3), bool)


def test_stdlib_backend_degrades_cleanly():
    s = StdlibBackend()
    assert s.read_rss([999_999]) == 0          # telemetry unavailable, no raise
    assert s.kill([999_999]) == 0              # dead pid → nothing signalled
    assert isinstance(s.set_self_limit(8 * 1024 ** 3), bool)
