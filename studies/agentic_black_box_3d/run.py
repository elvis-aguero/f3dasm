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
