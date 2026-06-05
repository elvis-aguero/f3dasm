"""Default graph factory for f3dasm agentic runs."""

from __future__ import annotations

from ..backends.base import Edge, Graph
from .implementer import ImplementerAgent
from .strategizer import StrategizerAgent


def _default_graph() -> Graph:
    return Graph(
        nodes={
            "strategizer": StrategizerAgent(),
            "implementer": ImplementerAgent(),
        },
        edges=(Edge("strategizer", "implementer"),),
        entry="strategizer",
    )
