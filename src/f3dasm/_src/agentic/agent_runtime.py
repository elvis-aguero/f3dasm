"""f3dasm agentic runtime — thin LangGraph wrapper."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from .agent_prompts import IMPLEMENTER_SYSTEM_PROMPT, STRATEGIZER_SYSTEM_PROMPT
from .backends.base import Agent, Edge, Graph
from .backends.claude import ClaudeAdapter
from .graph_builder import build_graph
from .graph_state import AgenticState, Delegation, Report, StudyConfig, Task

__all__ = [
    "AgenticRun",
    "AgenticRunError",
    "StrategizerAgent",
    "ImplementerAgent",
]

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AgenticRunError(Exception):
    """Raised when an agentic run fails unrecoverably."""


class StrategizerAgent(Agent):
    """Default orchestrator agent for f3dasm agentic runs."""

    system_prompt = STRATEGIZER_SYSTEM_PROMPT
    tools = frozenset({"Done", "Ask", "WriteMarkdown", "ReadNote"})
    reset_on_checkpoint = False


class ImplementerAgent(Agent):
    """Default worker agent for f3dasm agentic runs."""

    system_prompt = IMPLEMENTER_SYSTEM_PROMPT
    tools = frozenset({"Bash", "Edit", "Read", "Write", "Glob", "Grep"})
    reset_on_checkpoint = True


def _default_graph() -> Graph:
    return Graph(
        nodes={"strategizer": StrategizerAgent(), "implementer": ImplementerAgent()},
        edges=(Edge("strategizer", "implementer"),),
        entry="strategizer",
    )


class AgenticRun:
    """Run an agentic loop over a study directory using LangGraph.

    Parameters
    ----------
    study_dir : Path
        Root of the study tree.  Must contain ``PROBLEM_STATEMENT.md``.
    graph : Graph, optional
        Custom agent graph.  Defaults to a 2-node strategizer→implementer graph.
    model : str, optional
        LLM model identifier.  Defaults to ``DEFAULT_MODEL``.
    budget : float, optional
        Time budget in seconds.  ``None`` means unlimited.
    """

    def __init__(
        self,
        study_dir: Path,
        *,
        graph: Graph | None = None,
        model: str = DEFAULT_MODEL,
        budget: float | None = None,
    ) -> None:
        self.study_dir = Path(study_dir)
        self._model = model
        self._budget = budget
        self._graph_spec = graph or _default_graph()
        self._graph = build_graph(self._graph_spec, self._make_adapter)

    def execute(self) -> str:
        """Run the agentic loop; return the final report text.

        Reads ``PROBLEM_STATEMENT.md`` from the study directory and passes it
        as the initial user message to the entry node.
        """
        problem_path = self.study_dir / "PROBLEM_STATEMENT.md"
        if not problem_path.exists():
            raise AgenticRunError(
                f"PROBLEM_STATEMENT.md not found in {self.study_dir}"
            )
        problem = problem_path.read_text()

        config: dict[str, Any] = {"configurable": {"thread_id": str(uuid.uuid4())}}
        initial_state = AgenticState(
            messages=[HumanMessage(content=problem)],
            study_dir=str(self.study_dir),
            done=False,
            last_report=None,
            total_delegations=0,
            budget_seconds=self._budget,
        )
        result = self._graph.invoke(initial_state, config=config)
        return result.get("last_report") or ""

    def _make_adapter(self, name: str, agent: Agent) -> ClaudeAdapter:
        return ClaudeAdapter(
            model=agent.model or self._model,
            system_prompt=agent.system_prompt,
            study_dir=self.study_dir,
            native_tools=list(agent.tools),
        )
