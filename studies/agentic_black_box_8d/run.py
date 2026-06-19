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
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()

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
