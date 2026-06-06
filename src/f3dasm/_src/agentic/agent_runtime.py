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
from .container_runner import ContainerRunner
from .delegation_log import DelegationLog
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
    "DEFAULT_OLLAMA_MODEL",
    "Delegation",
    "ImplementerAgent",
    "Report",
    "StrategizerAgent",
    "StudyConfig",
    "Task",
]

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"


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
        max_ask: int = 1,
        container: bool = False,
        container_image: str = "f3dasm-agentic:latest",
    ) -> None:
        self.study_dir = Path(study_dir).resolve()
        cfg = _load_study_config(self.study_dir)

        _backend_cfg = cfg.get("backend", "claude")
        self._backend = _backend_cfg
        self._model = model or cfg.get("model") or (
            DEFAULT_OLLAMA_MODEL if _backend_cfg == "ollama" else DEFAULT_MODEL
        )
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
        self._max_ask = max_ask
        self._container = container
        self._container_image = container_image
        self._run_dir = None  # set in execute()

    def execute(self) -> str:
        """Run the agentic loop; return the final report text.

        Reads ``PROBLEM_STATEMENT.md`` from the study directory and passes it
        as the initial user message to the entry node.
        """
        if getattr(self, "_container", False):
            runner = ContainerRunner(
                self.study_dir,
                model=self._model,
                budget=getattr(self, "_budget", None),
                backend=getattr(self, "_backend", "claude"),
                image=getattr(self, "_container_image", "f3dasm-agentic:latest"),
            )
            exit_code = runner.run()
            if exit_code != 0:
                raise AgenticRunError(f"Container exited with code {exit_code}")
            return runner._latest_solution()

        problem_path = self.study_dir / "PROBLEM_STATEMENT.md"
        if not problem_path.exists():
            raise AgenticRunError(
                f"PROBLEM_STATEMENT.md not found in {self.study_dir}"
            )
        problem = problem_path.read_text(encoding="utf-8")

        # Create run directory
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_dir = self.study_dir / "runs" / ts
        debug_dir = run_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        notes_dir = debug_dir / "strategizer_notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        lit_reviewer_notes_dir = debug_dir / "lit_reviewer_notes"
        # lit_reviewer_notes_dir is created by LiteratureCorpus.__init__
        self._run_dir = run_dir

        # Set up run.log
        log = logging.getLogger(f"f3dasm.agentic.{ts}")
        log.setLevel(logging.INFO)
        handler = logging.FileHandler(debug_dir / "run.log")
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        log.addHandler(handler)
        log.info(f"Run starting: model={self._model}, study={self.study_dir}")

        start_time = time.time()

        # Create graph-wide delegation log for episodic memory.
        delegation_log_path = debug_dir / "delegation_log.jsonl"
        delegation_log = DelegationLog(delegation_log_path)

        # workspace_dir for worker delegations
        workspace_dir = debug_dir / "delegations"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Build the graph now (after _run_dir is set) so _make_adapter sees it
        # If a pre-built graph was injected (e.g. in tests), use it directly.
        graph = getattr(self, "_graph", None) or build_graph(
            self._graph_spec, self._make_adapter, study_dir=self.study_dir,
            interactive=self._interactive, max_ask=self._max_ask,
            notes_dir=notes_dir,
            lit_reviewer_notes_dir=lit_reviewer_notes_dir,
            workspace_dir=workspace_dir,
            delegation_log=delegation_log,
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
        tokens = result.get("token_totals") or {}
        error_counts = result.get("error_counts") or {}

        # Write solution.md — primary output at study root, visible to user
        solution_path = self.study_dir / "solution.md"
        now_ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        elapsed = time.time() - start_time
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)

        tokens_in = tokens.get("input_tokens", 0) or 0
        tokens_out = tokens.get("output_tokens", 0) or 0
        cache_read = tokens.get("cache_read_input_tokens", 0) or 0
        cache_create = tokens.get("cache_creation_input_tokens", 0) or 0
        cost = tokens.get("total_cost_usd")
        cost_str = f"${cost:.4f}" if cost is not None else "n/a"

        solution_path.write_text(
            f"# Solution\n\n"
            f"{report}\n\n"
            f"## Run metadata\n\n"
            f"- timestamp: {now_ts}\n"
            f"- model: {self._model}\n"
            f"- total_delegations: {result.get('total_delegations', 0)}\n"
            f"- evals_used: {evals}\n"
            f"- run_dir: {run_dir}\n"
            f"- time_used: {h:02d}:{m:02d}:{s:02d}\n\n"
            f"## Token usage\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| input_tokens | {tokens_in:,} |\n"
            f"| output_tokens | {tokens_out:,} |\n"
            f"| cache_read_tokens | {cache_read:,} |\n"
            f"| cache_creation_tokens | {cache_create:,} |\n"
            f"| total_tokens | {tokens_in + tokens_out:,} |\n"
            f"| estimated_cost | {cost_str} |\n"
            + (
                f"\n## Tool-call errors per node\n\n"
                + "| node | error_count |\n"
                + "|------|-------------|\n"
                + "".join(
                    f"| {node} | {count} |\n"
                    for node, count in sorted(error_counts.items())
                )
                if error_counts else ""
            ),
            encoding="utf-8",
        )
        # Inject provenance header into replicate.py if the agent wrote one
        # via WriteDeliverable (which writes directly to study_dir/).
        replicate_path = self.study_dir / "replicate.py"
        if replicate_path.exists():
            now_iso = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
            header = (
                f"# Auto-generated by agentic-f3dasm\n"
                f"# Timestamp : {now_iso}\n"
                f"# Model     : {self._model}\n"
                f"# Run       : {run_dir}\n"
                f"#\n"
            )
            existing = replicate_path.read_text(encoding="utf-8")
            if not existing.startswith("# Auto-generated"):
                replicate_path.write_text(header + existing, encoding="utf-8")

        log.info(
            f"Run complete. Evals: {evals}. "
            f"Tokens in/out: {tokens_in}/{tokens_out}. "
            f"Cost: {cost_str}. solution.md written."
        )
        log.removeHandler(handler)
        handler.close()

        return report

    def _make_adapter(self, name: str, agent: Agent):
        run_dir = self._run_dir

        has_outgoing = (
            run_dir
            and hasattr(self._graph_spec, "outgoing")
            and self._graph_spec.outgoing(name)
        )
        if has_outgoing:
            notes_dir = Path(run_dir) / "debug" / "strategizer_notes"
            debug_dir = Path(run_dir) / "debug"
            preamble = RUN_PATHS_PREAMBLE_TEMPLATE.format(
                study_dir=self.study_dir,
                run_dir=run_dir,
                debug_dir=debug_dir,
                notes_dir=notes_dir,
            )
            system_prompt = preamble + agent.system_prompt
            cwd = self.study_dir
        else:
            workspace_dir = Path(self._run_dir) / "debug" / "delegations"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            preamble = WORKSPACE_PREAMBLE_TEMPLATE.format(
                workspace_dir=workspace_dir,
                study_dir=self.study_dir,
            )
            system_prompt = preamble + agent.system_prompt
            cwd = workspace_dir

        model = agent.model or self._model
        backend = agent.backend or self._backend

        _persistent = not agent.reset_on_checkpoint
        _max_history_pairs = getattr(agent, "max_history_pairs", 5)

        lit_reviewer_notes_dir = Path(self._run_dir) / "debug" / "lit_reviewer_notes"

        if backend == "ollama":
            import os
            from .backends.ollama import OllamaAdapter
            _closure_tool_names = {
                "Done", "FollowUp", "WriteNote", "ReadNote", "ReportEvals"
            }
            ollama_native = [
                t for t in agent.tools if t not in _closure_tool_names
            ]
            adapter = OllamaAdapter(
                model=model,
                system_prompt=system_prompt,
                study_dir=cwd,
                native_tools=ollama_native,
                extra_mcp_servers=dict(getattr(agent, "mcp_servers", {})),
                extra_allowed_tools=list(getattr(agent, "extra_allowed_tools", frozenset())),
                base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                persistent=_persistent,
                max_history_pairs=_max_history_pairs,
            )
            extra_closures = agent.build_closure_tools(
                self.study_dir,
                lit_reviewer_notes_dir=lit_reviewer_notes_dir,
            )
            if extra_closures:
                adapter.closure_tools.update(extra_closures)
            return adapter

        native = [t for t in agent.tools if t in _CLAUDE_NATIVE_TOOLS]
        adapter = ClaudeAdapter(
            model=model,
            system_prompt=system_prompt,
            study_dir=cwd,
            native_tools=native,
            extra_mcp_servers=dict(getattr(agent, "mcp_servers", {})),
            extra_allowed_tools=list(getattr(agent, "extra_allowed_tools", frozenset())),
            persistent=_persistent,
            max_history_pairs=_max_history_pairs,
        )
        extra_closures = agent.build_closure_tools(
            self.study_dir,
            lit_reviewer_notes_dir=lit_reviewer_notes_dir,
        )
        if extra_closures:
            adapter.closure_tools.update(extra_closures)
        return adapter
