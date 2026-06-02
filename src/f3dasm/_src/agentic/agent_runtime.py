"""f3dasm agentic runtime — thin LangGraph wrapper."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # available via hydra-core
from langchain_core.messages import HumanMessage

from .agent_prompts import (
    RUN_PATHS_PREAMBLE_TEMPLATE,
    WORKSPACE_PREAMBLE_TEMPLATE,
)
from .agents import ImplementerAgent, StrategizerAgent, _default_graph
from .backends.base import Agent, Graph
from .backends.claude import ClaudeAdapter
from .graph_builder import build_graph
from .graph_state import AgenticState, Delegation, Report, StudyConfig, Task

# Real tool names recognised by the claude-agent-sdk as native CLI tools.
_CLAUDE_NATIVE_TOOLS = frozenset({
    "Bash", "Edit", "Read", "Write", "Glob", "Grep",
    "Task", "WebFetch", "WebSearch", "computer",
})

__all__ = [
    "AgenticRun",
    "AgenticRunError",
    "DEFAULT_MODEL",
    "Delegation",
    "ImplementerAgent",
    "Report",
    "StrategizerAgent",
    "StudyConfig",
    "Task",
]

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AgenticRunError(Exception):
    """Raised when an agentic run fails unrecoverably."""


def _load_study_config(study_dir: Path) -> dict:
    """Read study_dir/config.yaml if present; return empty dict otherwise."""
    cfg_path = study_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_budget_str(value) -> float | None:
    """Parse budget: float seconds passthrough, or 'HH:MM:SS' string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    return float(value)


class AgenticRun:
    """Run an agentic loop over a study directory using LangGraph.

    Parameters
    ----------
    study_dir : Path
        Root of the study tree.  Must contain ``PROBLEM_STATEMENT.md``.
    graph : Graph, optional
        Custom agent graph.  Defaults to a 2-node strategizer→implementer
        graph.
    model : str, optional
        LLM model identifier.  Defaults to ``DEFAULT_MODEL``.
    budget : float, optional
        Time budget in seconds.  ``None`` means unlimited.
    eval_budget : int, optional
        Maximum function evaluations across all delegations.
    """

    def __init__(
        self,
        study_dir: Path,
        *,
        graph: Graph | None = None,
        model: str | None = None,
        budget: float | None = None,
        eval_budget: int | None = None,
        interactive: bool = False,
    ) -> None:
        self.study_dir = Path(study_dir).resolve()
        cfg = _load_study_config(self.study_dir)

        self._model = model or cfg.get("model") or DEFAULT_MODEL
        self._backend = cfg.get("backend", "claude")
        self._eval_budget = (
            eval_budget if eval_budget is not None else cfg.get("eval_budget")
        )
        self._required_deliverables = cfg.get("required_deliverables") or []

        # budget from config is HH:MM:SS string or seconds float
        if budget is not None:
            self._budget = budget
        elif "budget" in cfg:
            self._budget = _parse_budget_str(cfg["budget"])
        else:
            self._budget = None

        self._graph_spec = graph or _default_graph()
        self._interactive = interactive
        self._run_dir = None  # set in execute()

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
        problem = problem_path.read_text(encoding="utf-8")

        # Create run directory
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_dir = self.study_dir / "runs" / ts
        notes_dir = run_dir / "strategizer_notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir

        # Set up run.log
        log = logging.getLogger(f"f3dasm.agentic.{ts}")
        log.setLevel(logging.INFO)
        handler = logging.FileHandler(run_dir / "run.log")
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        log.addHandler(handler)
        log.info(f"Run starting: model={self._model}, study={self.study_dir}")

        start_time = time.time()

        # Build the graph now (after _run_dir is set) so _make_adapter sees it
        # If a pre-built graph was injected (e.g. in tests), use it directly.
        graph = getattr(self, "_graph", None) or build_graph(
            self._graph_spec, self._make_adapter, study_dir=self.study_dir,
            interactive=self._interactive,
        )

        config: dict[str, Any] = {
            "configurable": {"thread_id": str(uuid.uuid4())}
        }
        initial_state = AgenticState(
            messages=[HumanMessage(content=problem)],
            study_dir=str(self.study_dir),
            done=False,
            last_report=None,
            total_delegations=0,
            budget_seconds=getattr(self, "_budget", None),
            run_dir=str(run_dir),
            eval_budget=getattr(self, "_eval_budget", None),
            evals_used=0,
            start_time=start_time,
            return_to=None,
            required_deliverables=(
                getattr(self, "_required_deliverables", None) or None
            ),
        )

        log.info("Invoking graph")
        result = graph.invoke(initial_state, config=config)
        report = result.get("last_report") or ""
        evals = result.get("evals_used", 0)

        # Write solution.md
        solution_path = run_dir / "solution.md"
        now_ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        solution_path.write_text(
            f"# Solution\n\n"
            f"{report}\n\n"
            f"## Run metadata\n\n"
            f"- timestamp: {now_ts}\n"
            f"- model: {self._model}\n"
            f"- total_delegations: {result.get('total_delegations', 0)}\n"
            f"- evals_used: {evals}\n"
            f"- run_dir: {run_dir}\n",
            encoding="utf-8",
        )
        log.info(f"Run complete. Evals used: {evals}. solution.md written.")
        log.removeHandler(handler)
        handler.close()

        return report

    def _make_adapter(self, name: str, agent: Agent):
        run_dir = self._run_dir
        workspace_dir = self.study_dir / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        has_outgoing = (
            run_dir
            and hasattr(self._graph_spec, "outgoing")
            and self._graph_spec.outgoing(name)
        )
        if has_outgoing:
            notes_dir = Path(run_dir) / "strategizer_notes"
            preamble = RUN_PATHS_PREAMBLE_TEMPLATE.format(
                study_dir=self.study_dir,
                notes_dir=notes_dir,
            )
            system_prompt = preamble + agent.system_prompt
            cwd = self.study_dir
        else:
            preamble = WORKSPACE_PREAMBLE_TEMPLATE.format(
                workspace_dir=workspace_dir,
            )
            system_prompt = preamble + agent.system_prompt
            cwd = workspace_dir

        model = agent.model or self._model
        backend = agent.backend or self._backend

        if backend == "ollama":
            from .backends.ollama import OllamaAdapter
            _closure_tool_names = {
                "Done", "Ask", "WriteMarkdown", "ReadNote", "ReportEvals"
            }
            ollama_native = [
                t for t in agent.tools if t not in _closure_tool_names
            ]
            return OllamaAdapter(
                model=model,
                system_prompt=system_prompt,
                study_dir=cwd,
                native_tools=ollama_native,
            )

        native = [t for t in agent.tools if t in _CLAUDE_NATIVE_TOOLS]
        return ClaudeAdapter(
            model=model,
            system_prompt=system_prompt,
            study_dir=cwd,
            native_tools=native,
        )
