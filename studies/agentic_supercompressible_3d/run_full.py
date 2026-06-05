"""Full 4-node agentic run for the supercompressible 3D study.

Topology:
    strategizer (Sonnet)  → implementer (Haiku)
    strategizer           → literature_reviewer (Haiku)
    strategizer           → critic (Haiku)

Tests parallel delegation, literature search robustness, and critic Done() gate.

Usage:
    uv run python studies/agentic_supercompressible_3d/run_full.py
"""

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
BUDGET_SECONDS = 30 * 60  # 30 minutes

HAIKU  = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

graph = Graph(
    nodes={
        "strategizer":         StrategizerAgent(model=SONNET),
        "implementer":         F3dasmImplementer(model=HAIKU),
        "literature_reviewer": LiteratureReviewAgent(model=HAIKU),
        "critic":              AdversarialCritiqueAgent(model=HAIKU),
    },
    edges=(
        Edge("strategizer", "implementer"),
        Edge("strategizer", "literature_reviewer"),
        Edge("strategizer", "critic"),
    ),
    entry="strategizer",
)

run = AgenticRun(
    study_dir=STUDY_DIR,
    graph=graph,
    budget=BUDGET_SECONDS,
    interactive=False,
)

if __name__ == "__main__":
    result = run.execute()
    print(result)
