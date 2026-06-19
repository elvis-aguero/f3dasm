"""Full 4-node agentic run for the supercompressible 3D study — all Haiku.

Topology:
    strategizer → implementer
    strategizer → literature_reviewer
    strategizer → critic

Usage:
    uv run python studies/agentic_supercompressible_3d/run_full.py
"""

import os
import shutil
import threading
import time
from pathlib import Path

from f3dasm.agentic import (
    AdversarialCritiqueAgent,
    AgenticRun,
    Edge,
    Graph,
    LiteratureReviewAgent,
    StrategizerAgent,
)
from f3dasm._src.agentic.agents.implementer import F3dasmImplementer

STUDY_DIR = Path(__file__).parent
BUDGET_SECONDS = 45 * 60  # 45 minutes
WATCHDOG_SECONDS = 60 * 60
MODEL = "claude-haiku-4-5-20251001"

# ── clean previous artifacts ──────────────────────────────────────────────────
for path in [
    STUDY_DIR / "runs",
    STUDY_DIR / "solution.md",
    STUDY_DIR / "pipeline.py",
    STUDY_DIR / "pipeline.ipynb",
    STUDY_DIR / "replicate.py",
]:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()

graph = Graph(
    nodes={
        "strategizer":         StrategizerAgent(model=MODEL),
        "implementer":         F3dasmImplementer(model=MODEL),
        "literature_reviewer": LiteratureReviewAgent(model=MODEL),
        "critic":              AdversarialCritiqueAgent(model=MODEL),
    },
    edges=(
        Edge("strategizer", "implementer"),
        Edge("strategizer", "literature_reviewer"),
        Edge("strategizer", "critic"),
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
