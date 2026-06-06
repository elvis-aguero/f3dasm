"""Run the agentic_black_box_8d study fresh (clears previous artifacts).

Topology:
    strategizer → implementer
    strategizer → literature_reviewer
    strategizer → critic
"""

import shutil
from pathlib import Path

from f3dasm.agentic import (
    AdversarialCritiqueAgent,
    AgenticRun,
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
    STUDY_DIR / "replicate.py",
]:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()

# ── graph ─────────────────────────────────────────────────────────────────────
graph = Graph(
    nodes={
        "strategizer":       StrategizerAgent(),
        "implementer":       ImplementerAgent(),
        "literature_reviewer": LiteratureReviewAgent(),
        "critic":            AdversarialCritiqueAgent(),
    },
    edges=(
        Edge("strategizer", "implementer"),
        Edge("strategizer", "literature_reviewer"),
        Edge("strategizer", "critic"),
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
