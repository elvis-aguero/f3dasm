"""Run the agentic_namespace_ring study fresh (clears previous artifacts).

An EASY multi-namespace smoke test (BACKLOG #20, branch exp/open-design-space):
maximise a fixed objective concentrated on a thin ring of the unit disk. The
'cartesian' baseline (x, y) sweep mostly samples near-zero; a 'polar' (r, theta)
reparametrisation of the SAME objective finds the ring trivially. The point is to
exercise the design-namespace machinery end-to-end (datagenerator authors +
registers a namespace oracle; implementer evaluates it via get_evaluator under
F3DASM_NAMESPACE; each namespace gets its own ledger) — comparable-by-construction.

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
BUDGET_SECONDS = 20 * 60  # 20 minutes — this is an easy, fast smoke test
# Model is config-driven: config.yaml `model:` is the source of truth, and the
# runtime default (Haiku, DEFAULT_MODEL) applies when it's absent. Do NOT hardcode
# a model here — AgenticRun resolves `passed-arg → config.yaml → default`, so a
# hardcode would shadow config.yaml. (A 20260623 Sonnet A/B lived here; reverted.)
MODEL = None  # None → let config.yaml / the runtime default decide
# Per-agent override; None → inherit the run model. Set a model id here ONLY for a
# deliberate per-agent A/B (e.g. strategizer-on-Sonnet); leave None for production.
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

# ── process-level watchdog: STALL DETECTOR, not a wall-clock deadline ─────────
# A run is force-exited ONLY when it makes NO progress at all for STALL_SECONDS —
# a genuine hang (a frozen LLM/CLI call or a zombie subprocess writes nothing).
# A run that keeps working is NEVER killed, however long it takes or however many
# experiments it runs in parallel: we must never penalise the agent for
# parallelising or for a slow-but-live campaign (the old flat 2x-budget deadline
# did exactly that). "Progress" = any file written under the run dir (a turn
# streams transcripts, the ledger flushes evals, the logs advance). Memory is
# bounded separately by the memory watcher (the one hard host-safety cap).
STALL_SECONDS = max(15 * 60, BUDGET_SECONDS)  # longer than any single legit LLM
#                                               call / sim; far short of a hang.


def _watchdog() -> None:
    from f3dasm._src.agentic.watchdog_cleanup import seconds_since_last_activity
    runs_dir = STUDY_DIR / "runs"
    # Poll for a stall; only a genuinely idle run (no writes for the whole window)
    # is killed. A progressing run loops here forever, untouched.
    while True:
        time.sleep(60)
        _live = sorted(d for d in runs_dir.iterdir() if d.is_dir()) \
            if runs_dir.exists() else []
        if not _live:
            continue
        _idle = seconds_since_last_activity(_live[-1])
        if _idle < STALL_SECONDS:
            continue  # still making progress — never penalise a live/parallel run
        print(
            f"\nWATCHDOG: run STALLED — no file activity for {int(_idle)}s "
            f"(threshold {STALL_SECONDS}s). Force-exiting a hung run "
            "(not a deadline; a true stall). No clean close.",
            flush=True,
        )
        # Write run_status + ledger row before the hard kill so the run is traceable.
        try:
            import csv as _csv
            import json as _json
            import sys as _sys
            _rd = _live[-1]
            (_rd / "debug").mkdir(parents=True, exist_ok=True)
            (_rd / "debug" / "run_status.json").write_text(
                _json.dumps({"status": "watchdog_killed"})
            )
            # Leave a synthetic post-mortem so §1 Step 1 isn't blind (the
            # strategizer never wrote its own — the kill is abrupt).
            write_watchdog_retrospective(_rd, int(_idle))
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
        # Reap leftover background jobs the run spawned (e.g. a detached
        # implementer campaign) so they don't outlive the watchdog. run.py made
        # itself a group leader at startup; ignore SIGTERM in ourselves so we
        # still reach os._exit(2).
        try:
            import signal as _signal
            _signal.signal(_signal.SIGTERM, _signal.SIG_IGN)
            reap_process_group(os.getpgid(0))
            # Catch detached/new-session campaigns the group-kill misses (#14):
            # the recursive backend kill over the self-registered campaign PIDs.
            try:
                reap_governor_pids(_live[-1])
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
        eval_budget=300,
    ).execute()
    print(result)
    _emit_analysis_brief()
