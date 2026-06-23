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


# ── per-delegation memory watch + recursive reap (resource-governance L2b/#14) ──
# These read the campaign PIDs that get_evaluator self-registered
# (<run_dir>/debug/governor_pids.jsonl) and use the ResourceBackend's RECURSIVE
# tree kill — which catches the new-session/detached campaigns that os.killpg
# misses (the #14 escape) regardless of how the agent's Bash launched them.

def read_governor_pids(run_dir) -> dict[str, list[int]]:
    """Parse governor_pids.jsonl into ``{delegation_id: [pid, ...]}``. Empty on
    any error (best-effort)."""
    out: dict[str, list[int]] = {}
    try:
        reg = Path(run_dir) / "debug" / "governor_pids.jsonl"
        if not reg.exists():
            return out
        for ln in reg.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            did, pid = r.get("delegation_id"), r.get("pid")
            if did is not None and isinstance(pid, int):
                out.setdefault(did, []).append(pid)
    except Exception:
        pass
    return out


def check_memory_and_kill(run_dir, cap_bytes: int, backend=None) -> list[str]:
    """Sample each registered delegation's process-tree RSS; kill any tree over
    ``cap_bytes``. Returns the delegation IDs killed. The HARD memory boundary's
    active enforcer (and the macOS enforcer, where RLIMIT_AS is unreliable).
    Best-effort — never raises."""
    killed: list[str] = []
    if not cap_bytes or cap_bytes <= 0:
        return killed
    try:
        from .resource_backend import get_resource_backend
        be = backend or get_resource_backend()
        for did, pids in read_governor_pids(run_dir).items():
            try:
                if be.read_rss(pids) > cap_bytes:
                    be.kill(pids)
                    killed.append(did)
            except Exception:
                pass
        if killed:
            _log_memory_kill(run_dir, killed, cap_bytes)
    except Exception:
        pass
    return killed


def reap_governor_pids(run_dir, backend=None) -> int:
    """Recursively kill every registered campaign's process tree — the watchdog's
    catch-all for detached/new-session campaigns the process-group kill misses.
    Returns processes signalled. Best-effort."""
    try:
        from .resource_backend import get_resource_backend
        be = backend or get_resource_backend()
        all_pids = [p for pids in read_governor_pids(run_dir).values() for p in pids]
        return be.kill(all_pids) if all_pids else 0
    except Exception:
        return 0


def delegation_rss(run_dir, delegation_id: str, backend=None) -> int:
    """Total process-tree RSS (bytes) of one delegation's registered campaign(s),
    for GetStatus telemetry so the strategizer can SEE a fat delegation. 0 if no
    registered pids / unknown. Best-effort — never raises."""
    try:
        from .resource_backend import get_resource_backend
        pids = read_governor_pids(run_dir).get(delegation_id, [])
        if not pids:
            return 0
        return (backend or get_resource_backend()).read_rss(pids)
    except Exception:
        return 0


def _log_memory_kill(run_dir, killed: list[str], cap_bytes: int) -> None:
    """Append a MEMORY_CAP_KILL diagnostic so the kill is auditable. Best-effort."""
    try:
        debug = Path(run_dir) / "debug"
        if not debug.exists():
            return
        rec = {
            "ts": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "node": ",".join(killed),
            "error_type": "MEMORY_CAP_KILL",
            "message": f"killed {killed} over hard mem cap {cap_bytes} bytes",
        }
        with (debug / "diagnostics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
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
