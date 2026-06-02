"""Default agent definitions for f3dasm agentic runs."""

from __future__ import annotations

from .agent_prompts import (
    DEBUGGER_SYSTEM_PROMPT,
    IMPLEMENTER_SYSTEM_PROMPT,
    LITERATURE_REVIEW_SYSTEM_PROMPT,
    STRATEGIZER_SYSTEM_PROMPT,
)
from .backends.base import Agent, Edge, Graph

__all__ = [
    "StrategizerAgent",
    "ImplementerAgent",
    "DebuggerAgent",
    "LiteratureReviewAgent",
    "_default_graph",
]


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


class DebuggerAgent(Agent):
    """Specialist agent for diagnosing and tracing bugs.

    Receives a failing command or traceback from the Strategizer, reproduces
    the failure, traces it to its root cause, and optionally applies a minimal
    fix.  Returns a structured Report with root_cause and fix_applied numbers.
    """

    system_prompt = DEBUGGER_SYSTEM_PROMPT
    tools = frozenset({"Bash", "Read", "Grep", "Edit", "Write"})
    reset_on_checkpoint = True
    description = "Diagnose errors, trace root causes, apply minimal fixes."


class LiteratureReviewAgent(Agent):
    """Specialist agent for searching and summarising scientific literature.

    Retrieves papers relevant to a research question, screens for relevance,
    and returns structured summaries with full citations saved to workspace/.
    Dedicated search and PDF-extraction tools will be added in a future update.
    """

    system_prompt = LITERATURE_REVIEW_SYSTEM_PROMPT
    tools = frozenset({"Read", "Write", "Bash"})
    reset_on_checkpoint = True
    description = "Search, retrieve, and summarise relevant scientific papers."


def _default_graph() -> Graph:
    return Graph(
        nodes={
            "strategizer": StrategizerAgent(),
            "implementer": ImplementerAgent(),
        },
        edges=(Edge("strategizer", "implementer"),),
        entry="strategizer",
    )
