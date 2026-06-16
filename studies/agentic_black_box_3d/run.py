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

import shutil
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

# ── clean previous artifacts ──────────────────────────────────────────────────
for path in [
    STUDY_DIR / "runs",
    STUDY_DIR / "solution.md",
    STUDY_DIR / "pipeline.py",
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

# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = AgenticRun(
        study_dir=STUDY_DIR,
        graph=graph,
        model=MODEL,
        budget=BUDGET_SECONDS,
        eval_budget=1000,
    ).execute()
    print(result)
