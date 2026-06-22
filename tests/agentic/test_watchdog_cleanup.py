"""Watchdog cleanup helpers (BACKLOG #11, #12): reap leftover background jobs and
leave a synthetic post-mortem retrospective on a watchdog kill.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

from f3dasm._src.agentic.watchdog_cleanup import (
    reap_process_group,
    write_watchdog_retrospective,
)


# ── #11: reap leftover background processes ──────────────────────────────────

def test_reap_kills_a_detached_background_process():
    # start_new_session=True puts the child in its OWN session/group (pgid==pid),
    # mirroring a leftover detached campaign — and keeps the test runner's group
    # untouched.
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        start_new_session=True,
    )
    try:
        assert child.poll() is None  # running
        reap_process_group(child.pid)  # pid == pgid for a session leader
        for _ in range(40):  # up to ~4s for SIGTERM to land
            if child.poll() is not None:
                break
            time.sleep(0.1)
        assert child.poll() is not None, "background process was not reaped"
    finally:
        if child.poll() is None:
            child.kill()


def test_reap_unknown_pgid_is_silent():
    # A pgid with no live processes must not raise.
    reap_process_group(999_999)


# ── #12: synthetic watchdog retrospective ────────────────────────────────────

def test_write_watchdog_retrospective_from_disk_state(tmp_path):
    debug = tmp_path / "debug"
    debug.mkdir()
    (debug / "delegation_log.jsonl").write_text(
        json.dumps({"id": "D001", "to_node": "datagenerator", "status": "DONE"}) + "\n"
        + json.dumps({"id": "D004", "to_node": "implementer", "status": "RUNNING"}) + "\n"
    )
    (debug / "diagnostics.jsonl").write_text(
        json.dumps({"error_type": "MILESTONE_BLOCK"}) + "\n"
        + json.dumps({"error_type": "ERROR_RETURN"}) + "\n"
    )
    write_watchdog_retrospective(tmp_path, 3600)
    lines = [l for l in (debug / "retrospectives.jsonl").read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["role"] == "watchdog" and rec["source_id"] == "WATCHDOG"
    assert "WATCHDOG POST-MORTEM" in rec["text"]
    assert "D004" in rec["text"] and "implementer:RUNNING" in rec["text"]
    assert "MILESTONE_BLOCK" in rec["text"]
    assert "strategizer" in rec["text"]  # points to the transcript


def test_write_watchdog_retrospective_appends_not_overwrites(tmp_path):
    debug = tmp_path / "debug"
    debug.mkdir()
    (debug / "retrospectives.jsonl").write_text(
        json.dumps({"source_id": "D001", "role": "datagenerator", "text": "x"}) + "\n"
    )
    write_watchdog_retrospective(tmp_path, 3600)
    lines = [l for l in (debug / "retrospectives.jsonl").read_text().splitlines() if l.strip()]
    assert len(lines) == 2  # original kept, watchdog appended
    assert json.loads(lines[-1])["role"] == "watchdog"


def test_write_watchdog_retrospective_never_raises_on_missing_dir(tmp_path):
    # No debug/ artifacts yet — must still write a (sparse) entry without raising.
    write_watchdog_retrospective(tmp_path, 3600)
    rec = json.loads((tmp_path / "debug" / "retrospectives.jsonl").read_text().splitlines()[0])
    assert rec["role"] == "watchdog" and "(none)" in rec["text"]
