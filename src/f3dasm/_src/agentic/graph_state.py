"""State types for the LangGraph-based agentic runtime."""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import MessagesState


class AgenticState(MessagesState):
    """LangGraph state for one agentic run.

    Inherits ``messages: Annotated[list[AnyMessage], add_messages]``
    from :class:`~langgraph.graph.MessagesState`.
    """

    study_dir: str
    done: bool
    last_report: str | None
    total_delegations: int
    budget_seconds: float | None
    run_dir: str | None          # absolute path to runs/<timestamp>/
    eval_budget: int | None      # max function evaluations (from config)
    evals_used: int              # running count across delegations
    start_time: float | None     # time.time() at run start, for budget enforcement
    return_to: str | None
    required_deliverables: list | None  # paths relative to study_dir; checked before Done accepted


# ---------------------------------------------------------------------------
# Data-transfer objects (moved from agent_runtime)
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """A unit of work delegated from one agent to another."""

    intent: str
    expected_report: str
    target: str


@dataclass
class Report:
    """The result returned by an implementer agent."""

    content: str
    target: str | None = None


@dataclass
class Delegation:
    """A delegation request parsed from an agent's response."""

    target: str
    task: str
    expected_report: str = ""


@dataclass
class StudyConfig:
    """Lightweight config loaded from the study directory."""

    study_dir: str
    model: str = "claude-opus-4-5"
    budget_seconds: float | None = None
