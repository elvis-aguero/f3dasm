"""Best-effort cleanup a study's run watchdog performs before ``os._exit``.

Two jobs, both observed-necessary on the n=5 watchdog post-mortems:
  1. ``reap_process_group`` — kill leftover background jobs the run spawned (e.g.
     an implementer's detached optimisation campaign) so they don't outlive the
     watchdog and keep burning CPU.
  2. ``write_watchdog_retrospective`` — a watchdog kill is abrupt, so the
     strategizer never writes its own end-of-run retrospective and §1 Step 1 is
     blind. Leave a synthetic, clearly-labelled post-mortem in
     ``retrospectives.jsonl`` from disk state.

Both are best-effort and never raise — a watchdog must still exit.
"""
from __future__ import annotations

import json
import os
import signal
from datetime import datetime, timezone
from pathlib import Path


def reap_process_group(pgid: int) -> None:
    """SIGTERM every process in ``pgid`` (the run's process group). run.py makes
    itself a group leader at startup, so its descendants — the SDK ``claude`` CLI,
    the agent's Bash, any backgrounded script — share its group. The caller should
    ``signal.signal(SIGTERM, SIG_IGN)`` in itself FIRST so it survives this group
    signal and still reaches ``os._exit`` with the intended code. Best-effort.

    Residual limit: a descendant that started a NEW session (setsid) escapes the
    group and is not reached; default ``subprocess``/``&`` do not, so the common
    leftover (a detached campaign) is covered.
    """
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def write_watchdog_retrospective(run_dir, watchdog_seconds: int) -> None:
    """Append a synthetic post-mortem to ``<run_dir>/debug/retrospectives.jsonl``.

    The record uses the same shape as a real node retrospective (ts, source_id,
    role, flagged, text) but is honestly labelled ``role="watchdog"`` — it is NOT
    fabricated first-person text. It surfaces the last delegation state and the
    diagnostics tally (disk-only) plus a pointer to the strategizer transcript, so
    a watchdog-killed run gives §1 Step 1 a breadcrumb instead of silence.
    """
    try:
        debug = Path(run_dir) / "debug"
        delg: dict = {}
        dl = debug / "delegation_log.jsonl"
        if dl.exists():
            for ln in dl.read_text().splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                delg[r.get("id")] = f"{r.get('to_node')}:{r.get('status')}"

        diag: dict = {}
        dg = debug / "diagnostics.jsonl"
        if dg.exists():
            for ln in dg.read_text().splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                k = r.get("error_type") or r.get("type") or "other"
                diag[k] = diag.get(k, 0) + 1

        text = (
            "## WATCHDOG POST-MORTEM\n"
            f"Run force-killed at {watchdog_seconds}s — no clean close, so the "
            "strategizer wrote no first-person retrospective. Reconstruct its "
            "reasoning from debug/transcripts/strategizer/ (CLAUDE.md §1 Step 4).\n"
            f"Last delegation state: {delg or '(none)'}\n"
            f"Diagnostics: {diag or '(none)'}"
        )
        rec = {
            "ts": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "source_id": "WATCHDOG",
            "role": "watchdog",
            "flagged": False,
            "text": text,
        }
        debug.mkdir(parents=True, exist_ok=True)
        with (debug / "retrospectives.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
