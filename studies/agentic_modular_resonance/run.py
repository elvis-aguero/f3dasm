"""Serial agentic run for the modular resonance study — all Haiku.

Topology (serial):
    strategizer → implementer
    strategizer → critic

No parallel datagenerator / literature_reviewer branches — minimal topology
to expose failure modes in the core strategizer → implementer → critic loop.

Usage:
    uv run python studies/agentic_modular_resonance/run.py
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
    ImplementerAgent,
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
    STUDY_DIR / "pipeline.ipynb",
    STUDY_DIR / "replicate.py",
]:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()

# ── graph ─────────────────────────────────────────────────────────────────────
graph = Graph(
    nodes={
        "strategizer": StrategizerAgent(),
        "implementer":  ImplementerAgent(),
        "critic":       AdversarialCritiqueAgent(),
    },
    edges=(
        Edge("strategizer", "implementer"),
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
