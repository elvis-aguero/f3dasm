"""Public API for the agentic-f3dasm layer.

Architecture: LangGraph StateGraph with two default nodes:
- **StrategizerAgent** — orchestrates via Delegate/Done/Ask closures.
- **ImplementerAgent** — executes tasks using Bash/Read/Write/Edit tools.

The user's only required input is ``<study-dir>/PROBLEM_STATEMENT.md``.
"""

from __future__ import annotations

from .._src.agentic.agent_runtime import (
    AgenticRun,
    AgenticRunError,
    DEFAULT_MODEL,
    Delegation,
    Report,
    StudyConfig,
    Task,
)
from .._src.agentic.agents import (
    DebuggerAgent,
    ImplementerAgent,
    LiteratureReviewAgent,
    StrategizerAgent,
)
from .._src.agentic.backends.base import Agent, Edge, Graph
from .._src.agentic.backends.claude import ClaudeAdapter
from .._src.agentic.backends.ollama import OllamaAdapter
from .._src.agentic.graph_builder import build_graph
from .._src.agentic.graph_state import AgenticState
from .._src.agentic.lookup import LookupDataGenerator
from .._src.agentic.nodes import (
    AgentNode,
    ImplementerNode,
    StrategizerNode,
    WorkerNode,
)
from .._src.agentic.optimizer import AgenticOptimizer

__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"

__all__ = [
    "Agent",
    "AgentNode",
    "AgenticOptimizer",
    "AgenticRun",
    "AgenticRunError",
    "AgenticState",
    "ClaudeAdapter",
    "DEFAULT_MODEL",
    "Delegation",
    "Edge",
    "Graph",
    "DebuggerAgent",
    "ImplementerAgent",
    "LiteratureReviewAgent",
    "ImplementerNode",
    "WorkerNode",
    "LookupDataGenerator",
    "OllamaAdapter",
    "Report",
    "StrategizerAgent",
    "StrategizerNode",
    "StudyConfig",
    "Task",
    "build_graph",
]
