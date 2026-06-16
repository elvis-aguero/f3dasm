"""f3dasm agentic runtime — thin LangGraph wrapper."""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # available via hydra-core
from langchain_core.messages import HumanMessage

from . import settings
from .agent_prompts import (
    RUN_PATHS_PREAMBLE_TEMPLATE,
    WORKSPACE_PREAMBLE_TEMPLATE,
)
from .agents import ImplementerAgent, StrategizerAgent, _default_graph
from .backends.base import Agent, Graph
from .container_runner import ContainerRunner
from .delegation_log import DelegationLog
from .graph_builder import build_graph
from .graph_state import AgenticState, Delegation, Report, StudyConfig, Task

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


def _init_canonical_store(
    run_dir: Path,
    study_dir: Path,
    evaluator_config: dict | None = None,
) -> dict:
    """Create canonical store dirs and write run_config.json sidecar.

    Creates:
    - ``<run_dir>/experiment_data/`` — shared ExperimentData project_dir
    - ``<run_dir>/debug/run_config.json`` — config read by get_evaluator()

    Parameters
    ----------
    run_dir : Path
        The timestamped run directory.
    study_dir : Path
        The study root (contains config.yaml, PROBLEM_STATEMENT.md, …).
    evaluator_config : dict or None, optional
        Parsed ``evaluator:`` block from config.yaml.  Keys:

        - ``entrypoint`` (str) — ``"path/to/file.py:attr"``
        - ``output_names`` (list[str]) — required for bare-fn entrypoints
        - ``lookup`` (dict) — ``{"pool": ..., "input_columns": ...,
          "output_columns": ...}``
        - ``fidelity_column`` (str or None)

    Returns the config dict that was written.
    """
    import json as _json

    store_dir = run_dir / "experiment_data"
    store_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = evaluator_config or {}

    config: dict = {
        "store_dir": str(store_dir),
        # Co-locate the lock with the data (store_dir/experiment_data/) so
        # D000 ingestion (_ingest_precomputed_pool) and D001+ evaluations
        # (InstrumentedDataGenerator default) all lock the SAME file.
        "lock_path": str(store_dir / "experiment_data" / ".lock"),
        "evaluator_name": study_dir.name,
        "study_dir": str(study_dir),
        "fidelity_column": eval_cfg.get("fidelity_column"),
        # Extensible, oracle-stamped provenance columns ({col: value}) — an
        # open schema so any study can carry the metadata its science needs
        # (fidelity, regime, seed, mesh, …). Stamped on every ledger row by
        # InstrumentedDataGenerator, never authored by the agent.
        "provenance": eval_cfg.get("provenance"),
        "evaluator_entrypoint": eval_cfg.get("entrypoint"),
        "evaluator_output_names": eval_cfg.get("output_names"),
        "evaluator_lookup": eval_cfg.get("lookup"),
    }
    (run_dir / "debug" / "run_config.json").write_text(
        _json.dumps(config, indent=2), encoding="utf-8"
    )
    return config


def register_evaluator_entrypoint(
    run_config_path: Path,
    generator_file: Path | str,
    attr: str,
    output_names: list | None = None,
) -> str:
    """Register an agent-authored DataGenerator as the canonical oracle.

    Atomically updates ``run_config.json`` so the next ``get_evaluator()``
    call (which re-reads the config on every invocation) resolves the
    authored generator with no manual config edit.  This is the runtime
    side of the datagenerator → oracle handoff: the agent writes the
    generator file (+ a registration manifest); the runtime points the
    canonical entrypoint at it.

    Parameters
    ----------
    run_config_path : Path
        Path to the run's ``run_config.json``.
    generator_file : Path or str
        Path to the authored generator ``.py``.  An absolute path is made
        relative to the study root (``study_dir`` in the config); a relative
        path is assumed already study-relative and kept as-is.
        ``load_inner_evaluator`` resolves it as ``study_dir / file_part``.
    attr : str
        Name of the callable or ``DataGenerator`` subclass inside that file.
    output_names : list or None, optional
        Output column names — required when ``attr`` is a bare callable.

    Returns
    -------
    str
        The ``"file:attr"`` entrypoint that was written.
    """
    import json as _json
    import os as _os

    run_config_path = Path(run_config_path)
    config = _json.loads(run_config_path.read_text(encoding="utf-8"))

    gen_path = Path(generator_file)
    if gen_path.is_absolute():
        study_dir = Path(config["study_dir"]).resolve()
        file_part = str(gen_path.resolve().relative_to(study_dir))
    else:
        file_part = str(gen_path)

    entrypoint = f"{file_part}:{attr}"
    config["evaluator_entrypoint"] = entrypoint
    config["evaluator_output_names"] = output_names
    config["evaluator_lookup"] = None  # entrypoint takes precedence

    tmp = run_config_path.with_suffix(".json.tmp")
    tmp.write_text(_json.dumps(config, indent=2), encoding="utf-8")
    _os.replace(tmp, run_config_path)
    return entrypoint


def _ingest_precomputed_pool(
    store_dir: Path,
    study_dir: Path,
    lookup_cfg: dict,
) -> int:
    """Ingest a precomputed pool into the canonical store as D000 rows.

    Loads the pool ExperimentData from
    ``study_dir / lookup_cfg["pool"]`` and writes its rows into
    *store_dir* with provenance stamped as
    ``_delegation_id='D000'``, ``source='precomputed_pool'``.

    D000 rows are ground-truth data — they are *never* counted as
    evaluations.  ``_resolve_delegation_evals`` is called only for
    real delegation IDs (D001+) so D000 will never be counted.

    Parameters
    ----------
    store_dir : Path
        The run-level canonical store directory
        (``run_dir/experiment_data``).
    study_dir : Path
        Study root; pool path is resolved relative to this.
    lookup_cfg : dict
        The ``evaluator.lookup`` block from config.yaml.  Must have
        a ``"pool"`` key.

    Returns
    -------
    int
        Number of rows ingested.
    """
    from datetime import datetime, timezone

    from filelock import FileLock

    from ..design.domain import Domain
    from ..errors import EmptyFileError, ReachMaximumTriesError
    from ..experimentdata import ExperimentData
    from ..experimentsample import ExperimentSample, JobStatus

    pool_project = study_dir / lookup_cfg["pool"]
    pool = ExperimentData.from_file(project_dir=pool_project)

    ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    n_rows = len(pool)

    df_in, df_out = pool.to_pandas()

    # Stamp provenance onto every output row.
    batch_samples: dict = {}
    for i in range(n_rows):
        row_in = (
            {} if df_in is None
            else {k: df_in.iloc[i][k] for k in df_in.columns}
        )
        row_out = (
            {} if df_out is None
            else {k: df_out.iloc[i][k] for k in df_out.columns}
        )
        row_out["_delegation_id"] = "D000"
        row_out["_source"] = "precomputed_pool"
        row_out["_ts"] = ts
        batch_samples[i] = ExperimentSample(
            _input_data=row_in,
            _output_data=row_out,
            job_status=JobStatus.FINISHED,
        )

    # Build batch domain: copy pool domain + provenance outputs.
    batch_domain = Domain()
    all_out_keys: set[str] = set()
    for s in batch_samples.values():
        all_out_keys.update(s._output_data.keys())
    for key in sorted(all_out_keys):
        batch_domain.add_output(key, exist_ok=True)

    batch = ExperimentData.from_data(
        data=batch_samples, domain=batch_domain
    )

    lock_path = store_dir / "experiment_data" / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path)):
        try:
            canon = ExperimentData.from_file(project_dir=store_dir)
        except (
            FileNotFoundError,
            EmptyFileError,
            ReachMaximumTriesError,
        ):
            canon = ExperimentData(domain=batch_domain)
        for col in ("_delegation_id", "_source", "_ts"):
            canon._domain.add_output(col, exist_ok=True)
        merged = canon + batch
        merged.store(project_dir=store_dir)

    return n_rows


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
        budget_usd: float | None = None,
        eval_budget: int | None = None,
        interactive: bool = True,
        max_ask: int = 1,
        container: bool = False,
        container_image: str = "f3dasm-agentic:latest",
        resume_from: Path | None = None,
        review_statement: bool = True,
    ) -> None:
        self.study_dir = Path(study_dir).resolve()
        cfg = _load_study_config(self.study_dir)
        # config.yaml is the source of truth for run knobs (debug, timeouts,
        # retry, backstop, recursion_limit, …). Install the `runtime:` block so
        # scattered read sites resolve via settings (env still overrides).
        settings.configure(cfg.get("runtime") or {})

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

        # Hard USD cost ceiling (None = no ceiling). Honoured only when the
        # backend reports per-call cost (claude); ollama has no cost data.
        self._budget_usd = (
            budget_usd if budget_usd is not None else cfg.get("budget_usd")
        )

        self._graph_spec = graph or _default_graph()
        self._interactive = interactive
        self._max_ask = max_ask
        self._container = container
        self._container_image = container_image
        self._resume_from = (
            Path(resume_from) if resume_from is not None else None
        )
        # Pre-run problem-statement review (Item B). User-controllable; cfg can
        # also disable it. Default on.
        self._review_statement = (
            review_statement
            if review_statement is not None
            else cfg.get("review_statement", True)
        )
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

        # Create run directory (or reuse an existing one when resuming).
        # getattr default: some tests build AgenticRun via __new__.
        _resume = getattr(self, "_resume_from", None)
        if _resume is not None:
            run_dir = _resume.resolve()
            if not (run_dir / "debug" / "thread_id").exists():
                raise AgenticRunError(
                    f"resume_from={run_dir} is not a resumable run dir "
                    "(no debug/thread_id)"
                )
            ts = run_dir.name
        else:
            ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
            run_dir = self.study_dir / "runs" / ts
        debug_dir = run_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        notes_dir = debug_dir / "strategizer_notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        lit_reviewer_notes_dir = debug_dir / "lit_reviewer_notes"
        # lit_reviewer_notes_dir is created by LiteratureCorpus.__init__
        self._run_dir = run_dir

        # Canonical store: experiment_data/ + run_config.json
        _full_cfg = _load_study_config(self.study_dir)
        _eval_cfg = _full_cfg.get("evaluator")
        canonical_cfg = _init_canonical_store(
            run_dir, self.study_dir, evaluator_config=_eval_cfg
        )

        # Ingest a precomputed pool as D000 ground-truth rows. Two sources,
        # one ingestion path — D000 rows are never counted as evaluations
        # (_resolve_delegation_evals runs only for real delegations D001+):
        #   evaluator.lookup.pool  → pool IS the oracle (queried via
        #                            LookupDataGenerator) AND training data.
        #   training_data          → pool is ONLY training data; there is NO
        #                            live oracle (e.g. surrogate-only studies
        #                            where new evaluations cannot be run).
        _lookup_cfg = (_eval_cfg or {}).get("lookup")
        _training_data = _full_cfg.get("training_data")
        _pool_cfg = _lookup_cfg or (
            {"pool": _training_data} if _training_data else None
        )
        if _pool_cfg:
            _store_dir = Path(canonical_cfg["store_dir"])
            try:
                _n_ingested = _ingest_precomputed_pool(
                    _store_dir, self.study_dir, _pool_cfg
                )
                log_ingested = _n_ingested  # captured for log below
            except Exception as _exc:  # noqa: BLE001
                log_ingested = None
                _exc_msg = str(_exc)
        else:
            log_ingested = None
            _exc_msg = None

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
        if log_ingested is not None:
            log.info(
                f"D000: ingested {log_ingested} precomputed pool rows"
                f" from {_pool_cfg.get('pool', '?')}"
            )
        elif _exc_msg is not None:
            log.warning(f"D000 pool ingest failed: {_exc_msg}")

        start_time = time.time()

        # Create graph-wide delegation log for episodic memory.
        delegation_log_path = debug_dir / "delegation_log.jsonl"
        delegation_log = DelegationLog(delegation_log_path)

        # workspace_dir for worker delegations
        workspace_dir = debug_dir / "delegations"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Stable thread_id: persisted so a crashed run can be resumed against
        # the same LangGraph checkpoint. Resume reads it back; a fresh run
        # mints and stores it (own file: run_config.json is rewritten).
        _tid_path = debug_dir / "thread_id"
        if _resume is not None:
            thread_id = _tid_path.read_text().strip()
        else:
            thread_id = str(uuid.uuid4())
            _tid_path.write_text(thread_id)

        # Pre-run problem-statement review (advisory; interactive-refine when
        # enabled). Fresh runs only — a resume replays the checkpoint and must
        # not re-prompt. Skipped when a graph is injected programmatically
        # (a test affordance — _run_dir is set above so _make_adapter can build
        # the ephemeral reviewer session on real runs).
        if (
            _resume is None
            and getattr(self, "_review_statement", True)
            and getattr(self, "_graph", None) is None
        ):
            problem = self._review_problem_statement(problem, debug_dir)

        initial_state = AgenticState(
            messages=[HumanMessage(content=problem)],
            study_dir=str(self.study_dir),
            done=False,
            last_report=None,
            total_delegations=0,
            budget_seconds=getattr(self, "_budget", None),
            budget_usd=getattr(self, "_budget_usd", None),
            run_dir=str(run_dir),
            eval_budget=getattr(self, "_eval_budget", None),
            evals_used=0,
            start_time=start_time,
            return_to=None,
            required_deliverables=(
                getattr(self, "_required_deliverables", None) or None
            ),
            experiment_data_dir=canonical_cfg["store_dir"],
        )

        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            # 2000 ≈ hundreds of delegations; the old 500 (and the legacy 25 on
            # some branches) could crash a long multi-delegation run mid-flight
            # (GraphRecursionError). Knob: recursion_limit (config.yaml runtime
            # block; F3DASM_RECURSION_LIMIT overrides).
            "recursion_limit": settings.get_int("recursion_limit", 2000),
        }

        # Durable checkpoint to disk so the run survives a crash; resume passes
        # None as input (LangGraph convention: replay from last checkpoint).
        from langgraph.checkpoint.sqlite import SqliteSaver
        ckpt_path = debug_dir / "checkpoints.sqlite"
        log.info("Invoking graph")
        with SqliteSaver.from_conn_string(str(ckpt_path)) as saver:
            graph = getattr(self, "_graph", None) or build_graph(
                self._graph_spec, self._make_adapter, study_dir=self.study_dir,
                interactive=self._interactive, max_ask=self._max_ask,
                notes_dir=notes_dir,
                lit_reviewer_notes_dir=lit_reviewer_notes_dir,
                workspace_dir=workspace_dir,
                delegation_log=delegation_log,
                checkpointer=saver,
            )
            graph_input = None if _resume is not None else initial_state
            # On resume, the checkpointed state still carries the OLD budgets
            # and start_time. Re-seed them from this AgenticRun so a run that
            # halted on a budget can actually make progress after the user
            # raises it (cumulative token_totals persist in the checkpoint, so
            # the spend-so-far is still counted against the new ceiling).
            if _resume is not None and hasattr(graph, "update_state"):
                try:
                    graph.update_state(config, {
                        "budget_seconds": getattr(self, "_budget", None),
                        "budget_usd": getattr(self, "_budget_usd", None),
                        "eval_budget": getattr(self, "_eval_budget", None),
                        "start_time": start_time,
                    })
                except Exception:  # noqa: BLE001
                    log.warning("resume state refresh failed", exc_info=True)
            try:
                result = graph.invoke(graph_input, config=config)
            except BaseException as _exc:  # noqa: BLE001
                # Any unhandled crash (GraphRecursionError, KeyboardInterrupt,
                # OOM, …): record a resumable status so resume_from is always
                # an option after a break, then re-raise (we do not swallow).
                try:
                    (debug_dir / "run_status.json").write_text(
                        json.dumps({
                            "status": "crashed",
                            "reason": f"{type(_exc).__name__}: {_exc}"[:500],
                            "resumable": True,
                            "thread_id": thread_id,
                        }, indent=2),
                        encoding="utf-8",
                    )
                except OSError:
                    pass
                raise
        # Merge per-call telemetry into an analysis-ready summary.json (additive,
        # off the decision path — a failure here must not fail the run).
        try:
            from .telemetry import Telemetry
            Telemetry.merge(debug_dir)
        except Exception:  # noqa: BLE001
            log.warning("telemetry merge failed", exc_info=True)
        report = result.get("last_report") or ""
        # Authoritative eval count = provenance-stamped rows in the canonical
        # ledger, NOT the run-state counter. evals_used is summed from a
        # registry that clears Done entries on loop-back, so it under-reports
        # (0) on any run that re-prompts (e.g. every UNGATED run). The ledger
        # never loses rows — and it also captures cancelled-but-completed
        # delegations whose evals are real.
        evals = result.get("evals_used", 0)
        try:
            from .instrumented import RunStateSummary
            _summary = RunStateSummary.from_store(
                debug_dir.parent / "experiment_data")
            if _summary is not None:
                evals = sum(_summary.n_per_delegation.values())
        except Exception:  # noqa: BLE001
            log.warning("ledger eval-count failed; using state counter",
                        exc_info=True)
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
                "\n## Tool-call errors per node\n\n"
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
        # Inject provenance header into pipeline.py if the agent wrote one
        # via WriteDeliverable (which writes directly to study_dir/).
        pipeline_path = self.study_dir / "pipeline.py"
        if pipeline_path.exists():
            now_iso = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
            header = (
                f"# Auto-generated by agentic-f3dasm\n"
                f"# Timestamp : {now_iso}\n"
                f"# Model     : {self._model}\n"
                f"# Run       : {run_dir}\n"
                f"#\n"
            )
            existing = pipeline_path.read_text(encoding="utf-8")
            if not existing.startswith("# Auto-generated"):
                pipeline_path.write_text(header + existing, encoding="utf-8")

        # Append a KPI row to the longitudinal ledger automatically (best
        # effort). The extraction logic lives in studies/run_ledger.py (the one
        # source of truth, writing studies/run_ledger.csv); we invoke it as a
        # subprocess when present so a run is always recorded without a manual
        # step. Absent (e.g. a non-studies install) → silently skipped.
        try:
            import subprocess
            import sys as _sys
            ledger_script = self.study_dir.parent / "run_ledger.py"
            if ledger_script.exists():
                proc = subprocess.run(
                    [_sys.executable, str(ledger_script), str(run_dir)],
                    capture_output=True, text=True, timeout=60,
                )
                if proc.returncode == 0:
                    log.info("KPI ledger: %s", proc.stdout.strip())
                else:
                    log.warning(
                        "KPI ledger append failed (rc=%s): %s",
                        proc.returncode, proc.stderr.strip())
        except Exception:
            log.warning("KPI ledger append errored", exc_info=True)

        log.info(
            f"Run complete. Evals: {evals}. "
            f"Tokens in/out: {tokens_in}/{tokens_out}. "
            f"Cost: {cost_str}. solution.md written."
        )
        log.removeHandler(handler)
        handler.close()

        return report

    def _review_problem_statement(
        self, problem: str, debug_dir: Path, *, adapter=None
    ) -> str:
        """Advisory pre-run well-posedness review (Item B).

        Always writes ``debug/problem_statement_review.md``.  When the run is
        interactive and gaps are found, offers a per-gap refine via the same
        ``input()`` channel the in-graph FollowUp uses, appending accepted
        clarifications to the statement (and to a saved addendum).  Returns the
        (possibly augmented) problem text.  NEVER blocks an autonomous run: any
        reviewer failure falls back to the original statement unchanged.
        """
        from .reviewer import (
            ProblemStatementReviewerAgent,
            format_review_markdown,
            parse_review,
            review_gaps,
        )

        try:
            if adapter is None:
                adapter = self._make_adapter(
                    "problem_statement_reviewer",
                    ProblemStatementReviewerAgent(),
                )
            raw = adapter.invoke([{"role": "user", "content": problem}])
            review = parse_review(raw)
        except Exception:  # noqa: BLE001 — advisory, never blocks
            return problem

        try:
            (debug_dir / "problem_statement_review.md").write_text(
                format_review_markdown(review, problem), encoding="utf-8"
            )
        except OSError:
            pass

        gaps = review_gaps(review)
        if not (gaps and self._interactive):
            return problem

        clarifications: list[tuple[str, str]] = []
        print(
            "\nThe problem statement may be under-specified. For each gap, "
            "type a clarification (or leave blank to skip):"
        )
        for g in gaps:
            try:
                ans = input(f"  [{g['element']}] {g['note']}\n  > ").strip()
            except (EOFError, KeyboardInterrupt):
                ans = ""
            if ans:
                clarifications.append((g["element"], ans))

        if not clarifications:
            return problem

        addendum = "\n\n## Clarifications (added pre-run via HITL review)\n" + (
            "\n".join(f"- **{el}**: {ans}" for el, ans in clarifications)
        )
        try:
            (debug_dir / "problem_statement_addendum.md").write_text(
                addendum.strip() + "\n", encoding="utf-8"
            )
        except OSError:
            pass
        return problem + addendum

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
                experiment_data_dir=Path(run_dir) / "experiment_data",
            )
            system_prompt = preamble + agent.system_prompt
            cwd = self.study_dir
        else:
            if run_dir is not None:
                workspace_dir = Path(run_dir) / "debug" / "delegations"
                workspace_dir.mkdir(parents=True, exist_ok=True)
            else:
                workspace_dir = self.study_dir
            preamble = WORKSPACE_PREAMBLE_TEMPLATE.format(
                workspace_dir=workspace_dir,
                study_dir=self.study_dir,
            )
            system_prompt = preamble + agent.system_prompt
            # Critics read from the study tree, not from a delegation subfolder.
            if getattr(agent, "role", None) == "critic":
                cwd = self.study_dir
            else:
                cwd = workspace_dir

        model = agent.model or self._model
        backend = agent.backend or self._backend

        _persistent = not agent.reset_on_checkpoint
        _max_history_pairs = getattr(agent, "max_history_pairs", 5)

        lit_reviewer_notes_dir = (
            Path(self._run_dir) / "debug" / "lit_reviewer_notes"
            if self._run_dir is not None
            else self.study_dir / "lit_reviewer_notes"
        )

        # Registry-driven, forward-compatible dispatch: resolve the adapter
        # class by backend name and let it choose its own native tools. Adding
        # a backend to backends/registry.py makes it dispatchable here with no
        # change to this method. Backend-specific endpoint/auth (base_url,
        # api_key) is resolved inside each adapter from env/defaults, so the
        # construction kwargs are common to every backend.
        from .backends.registry import get_adapter_class

        adapter_cls = get_adapter_class(backend)
        native = adapter_cls.select_native_tools(agent.tools)
        adapter = adapter_cls(
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
