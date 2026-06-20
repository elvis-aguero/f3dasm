"""Run the agentic_black_box_8d study fresh (clears previous artifacts).

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
BUDGET_SECONDS = 45 * 60  # 45 minutes
WATCHDOG_SECONDS = 60 * 60
MODEL = "claude-haiku-4-5-20251001"

# ── clean previous artifacts ──────────────────────────────────────────────────
for path in [
    STUDY_DIR / "runs",
    STUDY_DIR / "solution.md",
    STUDY_DIR / "pipeline.py",
    STUDY_DIR / "replicate.py",  # legacy artifact from older runs
]:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)

# ── graph ─────────────────────────────────────────────────────────────────────
graph = Graph(
    nodes={
        "strategizer":       StrategizerAgent(),
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

def _watchdog() -> None:
    time.sleep(WATCHDOG_SECONDS)
    print(
        f"\nWATCHDOG: run exceeded {WATCHDOG_SECONDS}s wall-clock — force-exiting.",
        flush=True,
    )
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
