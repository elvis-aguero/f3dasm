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

STUDY_DIR = Path(__file__).parent
BUDGET_SECONDS = 15 * 60  # 15 minutes
MODEL = "claude-haiku-4-5-20251001"
# Per-agent strategizer model override; None → use MODEL (Haiku). (Was set to
# Sonnet for the orchestrator A/B; reverted to Haiku for the notebook e2e.)
STRATEGIZER_MODEL = None

# ── clean previous artifacts ──────────────────────────────────────────────────
for path in [
    STUDY_DIR / "runs",
    STUDY_DIR / "solution.md",
    STUDY_DIR / "pipeline.py",
    STUDY_DIR / "pipeline.ipynb",
]:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()

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
# Generous (> any healthy run ~22-38 min and the in-run backstop) — it only ever
# fires on a true stall.
WATCHDOG_SECONDS = 60 * 60


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
    os._exit(2)


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=_watchdog, daemon=True).start()
    result = AgenticRun(
        study_dir=STUDY_DIR,
        graph=graph,
        model=MODEL,
        budget=BUDGET_SECONDS,
        eval_budget=1000,
    ).execute()
    print(result)
