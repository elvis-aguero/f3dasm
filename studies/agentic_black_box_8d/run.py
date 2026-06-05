"""Run the agentic_black_box_8d study.

Topology:
    strategizer → implementer
    strategizer → literature_reviewer
    strategizer → critic         (explicit edge; critic also used by Done() gate)
"""

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

graph = Graph(
    nodes={
        "strategizer": StrategizerAgent(),
        "implementer": ImplementerAgent(),
        "literature_reviewer": LiteratureReviewAgent(),
        "critic": AdversarialCritiqueAgent(),
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
)

if __name__ == "__main__":
    result = run.execute()
    print(result)
