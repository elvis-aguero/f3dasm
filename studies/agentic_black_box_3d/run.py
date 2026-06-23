"""Run the agentic_black_box_3d study fresh (clears previous artifacts).

A lower-dimensional, faster sibling of agentic_black_box_8d: same black-box
optimisation framing in 3-D, with a deceptive multimodal landscape whose true
global minimum is exactly -1.0 at a randomised (hidden) location.

Topology:
    strategizer → literature_reviewer
    strategizer → datagenerator
    strategizer → implementer
    strategizer → critic
    datagenerator → literature_reviewer
"""

import os
import shutil
import threading
import time
from pathlib import Path

from f3dasm.agentic import (
    AdversarialCritiqueAgent,
    AgenticRun,
    DataGeneratorAgent,
    Edge,
    Graph,
    ImplementerAgent,
    LiteratureReviewAgent,
    StrategizerAgent,
)
from f3dasm._src.agentic.watchdog_cleanup import (
    check_memory_and_kill,
    reap_governor_pids,
    reap_process_group,
    write_watchdog_retrospective,
)

STUDY_DIR = Path(__file__).parent
BUDGET_SECONDS = 45 * 60  # 45 minutes
MODEL = "claude-haiku-4-5-20251001"
# Per-agent strategizer model override; None → use MODEL (Haiku). (Was set to
# Sonnet for the orchestrator A/B; reverted to Haiku for the notebook e2e.)
STRATEGIZER_MODEL = None

# ── dirty-tree guard ─────────────────────────────────────────────────────────
# The ledger stamps the current git SHA as provenance; a dirty working tree
# means that SHA cannot reproduce this run.
import subprocess as _sp

_dirty = _sp.run(
    ["git", "status", "--porcelain"],
    capture_output=True,
    text=True,
    cwd=str(STUDY_DIR),
).stdout.strip()
if _dirty:
    print(
        "ERROR: uncommitted changes detected — refusing to start.\n"
        "Commit or stash all changes before launching a run.\n"
        f"\n{_dirty}",
        flush=True,
    )
    raise SystemExit(1)
del _sp, _dirty

# ── preserve retrospectives across runs ───────────────────────────────────────
# Retrospectives are the highest-signal forensic record (per CLAUDE.md they used
# to be ephemeral — wiped with runs/). Archive each prior run's
# retrospectives.jsonl into a persistent, gitignored dir (NOT in the wipe list
# below) BEFORE cleaning, so they accumulate across runs instead of vanishing.
_retro_archive = STUDY_DIR / "retrospectives"
_runs_dir = STUDY_DIR / "runs"
if _runs_dir.is_dir():
    _retro_archive.mkdir(exist_ok=True)
    for _retro in _runs_dir.glob("*/debug/retrospectives.jsonl"):
        _run_id = _retro.parent.parent.name  # runs/<run_id>/debug/...
        _dest = _retro_archive / f"{_run_id}.jsonl"
        if _retro.stat().st_size > 0 and not _dest.exists():
            shutil.copy2(_retro, _dest)
del _retro_archive, _runs_dir

# ── clean previous artifacts ──────────────────────────────────────────────────
for path in [
    STUDY_DIR / "runs",
    STUDY_DIR / "solution.md",
    STUDY_DIR / "pipeline.py",
    STUDY_DIR / "pipeline.ipynb",
]:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)

# ── graph ─────────────────────────────────────────────────────────────────────
graph = Graph(
    nodes={
        "strategizer":       StrategizerAgent(model=STRATEGIZER_MODEL),
        "literature_reviewer": LiteratureReviewAgent(),
        "datagenerator":     DataGeneratorAgent(),
        "implementer":       ImplementerAgent(),
        "critic":            AdversarialCritiqueAgent(),
    },
    edges=(
        Edge("strategizer", "literature_reviewer"),
        Edge("strategizer", "datagenerator"),
        Edge("strategizer", "implementer"),
        Edge("strategizer", "critic"),
        Edge("datagenerator", "literature_reviewer"),
    ),
    entry="strategizer",
)

# ── process-level watchdog ────────────────────────────────────────────────────
# The runtime's time backstop is only checked BETWEEN strategizer turns, so a
# stall OUTSIDE that loop — startup, inside an LLM/CLI call, inside a tool —
# zombies forever (observed: a claude-CLI startup stall ran 80 min, idle CPU).
# This hard wall-clock watchdog force-exits a stalled run so it can never zombie.
# Set to 2x the in-run time budget: the 2026-06-23 audit showed runs were
# watchdog-killed on TIME (lit review + campaign + multi-attempt gate) before the
# deliverable could close, not on a true stall — so give the work room while still
# bounding a genuine zombie.
WATCHDOG_SECONDS = 2 * BUDGET_SECONDS


def _watchdog() -> None:
    time.sleep(WATCHDOG_SECONDS)
    print(
        f"\nWATCHDOG: run exceeded {WATCHDOG_SECONDS}s wall-clock — force-exiting "
        "(a call stalled outside the turn loop). No clean close.",
        flush=True,
    )
    # Write run_status + ledger row before the hard kill so the run is traceable.
    try:
        import csv as _csv
        import json as _json
        import sys as _sys
        runs_dir = STUDY_DIR / "runs"
        run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir()) if runs_dir.exists() else []
        if run_dirs:
            _rd = run_dirs[-1]
            (_rd / "debug").mkdir(parents=True, exist_ok=True)
            (_rd / "debug" / "run_status.json").write_text(
                _json.dumps({"status": "watchdog_killed"})
            )
            # Leave a synthetic post-mortem so §1 Step 1 isn't blind (the
            # strategizer never wrote its own — the kill is abrupt).
            write_watchdog_retrospective(_rd, WATCHDOG_SECONDS)
            _sys.path.insert(0, str(STUDY_DIR.parent))
            import run_ledger as _rl
            _row = _rl.extract(_rd)
            _row["commit"] = _rl._git_short_sha()
            _new = not _rl.LEDGER.exists()
            with _rl.LEDGER.open("a", newline="") as _f:
                _w = _csv.DictWriter(_f, fieldnames=_rl.COLUMNS)
                if _new:
                    _w.writeheader()
                _w.writerow(_row)
            print(f"WATCHDOG: ledger row appended for {_rd.name}", flush=True)
    except Exception as _e:
        print(f"WATCHDOG: cleanup failed: {_e}", flush=True)
    # Reap leftover background jobs the run spawned (e.g. a detached implementer
    # campaign) so they don't outlive the watchdog. run.py made itself a group
    # leader at startup; ignore SIGTERM in ourselves so we still reach os._exit(2).
    try:
        import signal as _signal
        _signal.signal(_signal.SIGTERM, _signal.SIG_IGN)
        reap_process_group(os.getpgid(0))
        # Catch detached/new-session campaigns the group-kill misses (#14): the
        # recursive backend kill over the self-registered campaign PIDs.
        try:
            runs_dir = STUDY_DIR / "runs"
            _rds = sorted(d for d in runs_dir.iterdir() if d.is_dir()) \
                if runs_dir.exists() else []
            if _rds:
                reap_governor_pids(_rds[-1])
        except Exception:
            pass
        time.sleep(0.5)
    except Exception as _e:
        print(f"WATCHDOG: child reap failed: {_e}", flush=True)
    os._exit(2)


def _memory_watcher() -> None:
    """Daemon: every 5s, kill any delegation whose process tree exceeds the hard
    memory cap. The cap is read from the run's run_config.json (`mem_cap_bytes`,
    sourced from config.yaml) — config is explicit in config.yaml, not env. The
    active enforcer of the one hard boundary; catches between-flush spikes (e.g. a
    GP fit) the per-row governor can't see."""
    import json as _json
    while True:
        time.sleep(5)
        try:
            runs_dir = STUDY_DIR / "runs"
            run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir()) \
                if runs_dir.exists() else []
            if not run_dirs:
                continue
            rd = run_dirs[-1]
            cfg_path = rd / "debug" / "run_config.json"
            if not cfg_path.exists():
                continue  # run not initialised yet
            cap = _json.loads(cfg_path.read_text()).get("mem_cap_bytes")
            if not cap:
                continue
            killed = check_memory_and_kill(rd, int(cap))
            if killed:
                print(f"MEMORY WATCHER: killed over-cap delegation(s) "
                      f"{killed} (cap {cap} bytes)", flush=True)
        except Exception:
            pass


# ── mechanical analysis brief ─────────────────────────────────────────────────
# On a clean close, emit the MECHANICAL half of the CLAUDE.md analysis protocol
# (Step-5 KPI baseline vs the previous run + Step-2 diagnostics tally + verbatim
# ERROR_RETURN events) so the analyst reads ONE artifact instead of grepping the
# ledger, diagnostics.jsonl and run_status by hand. It only COUNTS + points to
# the prose artifacts (retrospectives/critic reviews/delegations) — those need
# judgement and a keyword scan would miss prose-expressed failure modes.
def _emit_analysis_brief() -> None:
    try:
        import sys as _sys
        _sys.path.insert(0, str(STUDY_DIR.parent))
        import run_ledger as _rl
        runs_dir = STUDY_DIR / "runs"
        run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir()) \
            if runs_dir.exists() else []
        if not run_dirs:
            return
        rd = run_dirs[-1]
        brief = _rl.analysis_brief(rd)
        (rd / "debug").mkdir(parents=True, exist_ok=True)
        (rd / "debug" / "analysis_brief.md").write_text(brief, encoding="utf-8")
        print("\n" + "=" * 72)
        print(brief)
        print("=" * 72)
        print(f"[analysis_brief → {rd / 'debug' / 'analysis_brief.md'}]",
              flush=True)
    except Exception as _e:  # never let analysis crash the run's close
        print(f"[analysis_brief skipped: {_e}]", flush=True)


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Become a process-group leader so the watchdog can reap every descendant
    # (SDK CLI → agent Bash → any backgrounded script) in one group kill. Children
    # forked after this inherit the group; default subprocess/`&` don't escape it.
    try:
        os.setpgrp()
    except OSError:
        pass  # already a leader / unsupported — reap falls back to a no-op
    if os.environ.get("F3DASM_DISABLE_WATCHDOG"):
        print("WATCHDOG: disabled via F3DASM_DISABLE_WATCHDOG — wall-clock force-exit OFF "
              "(memory cap watcher stays on for host safety)", flush=True)
    else:
        threading.Thread(target=_watchdog, daemon=True).start()
    threading.Thread(target=_memory_watcher, daemon=True).start()
    result = AgenticRun(
        study_dir=STUDY_DIR,
        graph=graph,
        model=MODEL,
        budget=BUDGET_SECONDS,
        eval_budget=1000,
    ).execute()
    print(result)
    _emit_analysis_brief()
