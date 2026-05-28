"""Default agent definitions for f3dasm agentic runs."""

from __future__ import annotations

from .agent_prompts import IMPLEMENTER_SYSTEM_PROMPT, STRATEGIZER_SYSTEM_PROMPT
from .backends.base import Agent, Edge, Graph

__all__ = ["StrategizerAgent", "ImplementerAgent", "_default_graph"]


class StrategizerAgent(Agent):
    """Default orchestrator agent for f3dasm agentic runs."""

    system_prompt = STRATEGIZER_SYSTEM_PROMPT
    tools = frozenset({"Done", "Ask", "WriteMarkdown", "ReadNote"})
    reset_on_checkpoint = False
    role = "strategizer"


class ImplementerAgent(Agent):
    """Default worker agent for f3dasm agentic runs."""

    system_prompt = IMPLEMENTER_SYSTEM_PROMPT
    tools = frozenset({
        "Bash", "Edit", "Read", "Write", "Glob", "Grep", "ReportEvals"
    })
    reset_on_checkpoint = True


def _default_graph() -> Graph:
    return Graph(
        nodes={
            "strategizer": StrategizerAgent(),
            "implementer": ImplementerAgent(),
        },
        edges=(Edge("strategizer", "implementer"),),
        entry="strategizer",
    )
