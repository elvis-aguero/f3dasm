"""StrategizerNode: orchestrator node for the f3dasm agentic runtime.

``inspect.getsource(StrategizerNode.__call__)`` reads the full routing logic.
"""

from __future__ import annotations

import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..graph_state import AgenticState

from ..delegation_log import DelegationLog
from ..hypothesis_ledger import HypothesisLedger
from ..science_monitor import ScienceMonitor
from ._constants import run_backstop_multiple
from .base import AgentNode
from .critic_gate import CriticGateMixin
from .lifecycle import LifecycleMixin
from .parsing import _to_adapter_messages
from .recording import RecordingMixin


class StrategizerNode(RecordingMixin, CriticGateMixin, LifecycleMixin, AgentNode):
    """Orchestrator: reads Reports, decides next Delegation or Done/Ask."""

    def __init__(
        self,
        adapter: Any,
        name: str,
        outgoing: list[str],
        spec: Any,
        study_dir: Any = None,
        interactive: bool = False,
        max_ask: int = 1,
        worker_adapters: dict | None = None,
        notes_dir: Any = None,
        workspace_dir: Any = None,
        delegation_log: DelegationLog | None = None,
    ) -> None:
        super().__init__(adapter)
        self._name = name
        self._outgoing = list(outgoing)
        self._spec = spec
        self._route: dict = {}
        self._study_dir = study_dir
        self._workspace_dir = Path(workspace_dir) if workspace_dir is not None else None
        self._interactive = interactive
        self._max_ask = max_ask
        self._ask_count = 0
        self._current_notes_dir: Path | None = None
        # Parallel delegation registry: id → {"status", "result", "evals", "hypothesis_ids", "started_at"}
        self._worker_adapters: dict[str, Any] = worker_adapters or {}
        self._registry: dict[str, dict] = {}
        self._registry_lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        # Push notifications: background threads append here; tool calls drain it.
        self._notifications: list[str] = []
        self._notifications_lock = threading.Lock()
        # Budget state — set at the start of each __call__ from AgenticState
        self._budget_seconds: float | None = None
        self._run_start: float | None = None
        # Hard USD cost ceiling (None = inactive). Set each __call__ from state.
        self._budget_usd: float | None = None
        # True once any LLM call reports a real cost (claude). Stays False under
        # ollama (cost is None) → the USD ceiling is treated as inactive.
        self._cost_observed: bool = False
        self._usd_inactive_warned: bool = False
        # Consecutive Errored delegations per target (reset on that target's
        # next success). Drives the repeated-errors resumable halt.
        self._consecutive_errors: dict[str, int] = {}
        # Per-delegation pending messages (budget warnings) to prepend to
        # worker tool results.  Keyed by delegation_id; drained on next call.
        self._pending_worker_msgs: dict[str, list[str]] = {}
        self._pending_worker_msgs_lock = threading.Lock()
        # Tracks which budget % thresholds (80, 90, 100, 110 …) have already
        # been broadcast to workers so each is sent exactly once.
        self._budget_notified_pcts: set[int] = set()
        # Hypothesis ledger — persists hypotheses.json
        self._ledger: HypothesisLedger | None = (
            HypothesisLedger(Path(notes_dir)) if notes_dir is not None else None
        )
        # Milestone ledger (process policy) — persists milestones.json. Seeded
        # with the config default gates unless disabled. DISTINCT from the
        # hypothesis ledger (epistemics): process vs what's-true.
        from ..milestones import MilestoneLedger
        from ..settings import get_bool
        self._milestones: MilestoneLedger | None = None
        if notes_dir is not None and get_bool("milestones_enabled", True):
            self._milestones = MilestoneLedger(Path(notes_dir))
            # C3 switchable: the draft-pipeline gate seeds only when the
            # pipeline-deliverable knob is on (off = byte-identical to today).
            self._milestones.seed_defaults(
                include_pipeline=get_bool("pipeline_deliverable", True))
        # Graph-wide delegation log (demand-driven episodic memory)
        self._delegation_log: DelegationLog | None = delegation_log
        # Science drift monitor — active when both ledger and log present
        self._science_monitor: ScienceMonitor | None = None
        if self._ledger is not None and delegation_log is not None:
            self._science_monitor = ScienceMonitor(
                self._ledger,
                delegation_log,
                diagnostics_writer=self._record_science_drift,
            )
        # Running total of delegations at the START of the current __call__
        # Used as a seed for the delegation sequence counter.
        self._state_total_delegations: int = 0
        # Monotonic per-node delegation counter — never reset within a
        # run.  Seeded from _state_total_delegations on first __call__
        # so checkpoint-resumed runs continue from the correct offset.
        # Because it never resets, it avoids the ID collision that
        # occurs when completed delegations are pruned from the registry
        # but _state_total_delegations has not yet accumulated them.
        self._delegation_seq: int = 0
        # Two-shot Done() gate: first call warns, second call closes.
        # Resets to False whenever a new Delegate() fires.
        self._done_warned: bool = False
        # Post-Done exit interview: set after the critic accepts; the next
        # Done() carries only the retrospective. _final_summary holds the real
        # conclusion so the recorded summary is the science, not the interview.
        self._awaiting_retro: bool = False
        self._final_summary: str | None = None
        # Consecutive non-PASS critic verdicts; after 3, the gate closes
        # gracefully UNGATED (bounded escape) instead of looping forever.
        self._revise_count: int = 0
        # Eval budget for this run (stashed each turn from state).
        self._eval_budget: int | None = None
        # Cumulative cap on the "no canonical source registered" nudge (soft).
        self._no_source_nudges: int = 0
        # Bounded re-prompt counter: incremented each time the node loops back
        # due to an unaccepted termination (no Done or refused Done).  NOT reset
        # in the A1/A2 per-turn block — it persists across loopbacks within one
        # run.  After 3 loopbacks the run terminates UNGATED.
        self._finish_attempts: int = 0
        # Accumulated token usage across strategizer + all workers this run.
        self._token_totals: dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "total_cost_usd": 0.0,
        }
        # Per-node raw tool-call error count: any ERROR: return or raised
        # exception from any injected closure counts as one error for that node.
        self._error_counts: dict[str, int] = {}
        # Separable per-call telemetry — additive, off the decision path. Lives
        # under debug/telemetry/ (notes_dir is debug/strategizer_notes).
        from ..telemetry import Telemetry
        self._telemetry: Telemetry | None = (
            Telemetry(Path(notes_dir).parent) if notes_dir is not None else None
        )
        self.adapter.closure_tools.update(self._build_routing_closures())
        self.adapter.route_watcher = lambda: self._route.get("kind") == "done"

    def _find_datagenerator_name(self) -> str | None:
        """Name of the first connected datagenerator worker, or None.

        Its presence means a canonical ground-truth source CAN be authored
        and registered for this study.
        """
        spec = self._spec
        if spec is None or not hasattr(spec, "nodes"):
            return None
        for target in self._outgoing:
            agent = spec.nodes.get(target)
            if (
                agent is not None
                and getattr(agent, "role", None) == "datagenerator"
                and target in self._worker_adapters
            ):
                return target
        return None

    def _canonical_source_registered(self) -> bool:
        """True if a canonical ground-truth source is resolvable.

        Reads run_config.json: an evaluator entrypoint OR a lookup pool counts
        as a registered source. Best-effort — on any read failure, assume not
        registered (the nudge is soft, so a false 'no' just costs one notice).
        """
        notes = self._current_notes_dir
        if notes is None:
            return False
        try:
            import json as _json
            cfg = _json.loads(
                (Path(notes).parent / "run_config.json").read_text())
            return bool(
                cfg.get("evaluator_entrypoint")
                or cfg.get("evaluator_lookup")
            )
        except Exception:  # noqa: BLE001
            return False

    def _drain_notifications(self) -> str:
        """Return and clear any pending push notifications, or empty
        string."""
        with self._notifications_lock:
            if not self._notifications:
                text = ""
            else:
                msgs = list(self._notifications)
                self._notifications.clear()
                text = "\n".join(msgs) + "\n\n"
        if self._science_monitor is not None:
            offenders = self._science_monitor.escalation_due()
            _critic_name = self._find_critic_name()
            if offenders and _critic_name is not None:
                # Escalation fires: perform bookkeeping-only drain (discard
                # text) so the critic findings are the sole corrective
                # payload — regular drift messages would pollute context.
                self._science_monitor.drain()
                task_msg = self._build_feedback_task_msg(offenders)
                findings = self._invoke_critic(task_msg)
                self._science_monitor.note_escalated()
                if self._delegation_log is not None:
                    _fb_id = (
                        "FB"
                        + datetime.now(
                            tz=timezone.utc
                        ).strftime("%H%M%S")
                    )
                    self._delegation_log.record(
                        id=_fb_id,
                        from_node=self._name,
                        to_node=_critic_name,
                        task="ScienceMonitor escalation audit",
                        deliverable=findings,
                        hypothesis_ids=offenders,
                        started_at=datetime.now(
                            tz=timezone.utc
                        ).isoformat(timespec="seconds"),
                        completed_at=datetime.now(
                            tz=timezone.utc
                        ).isoformat(timespec="seconds"),
                        status="FEEDBACK",
                        tokens_in=0,
                        tokens_out=0,
                        cost_usd=None,
                    )
                text += (
                    "[SCIENCE MONITOR — ESCALATION] Repeated drift "
                    f"on {', '.join(offenders)}. Critic audit "
                    f"findings:\n{findings}\n"
                )
            else:
                # No escalation: inject regular drift messages normally.
                drift = self._science_monitor.drain()
                if drift:
                    text += drift
        return text

    def _build_routing_closures(self) -> dict:
        from .tools.routing import build_routing_tools
        return build_routing_tools(self)

    def _wrap_closure(self, fn: Any, node_name: str) -> Any:
        """Return a version of *fn* that records ERROR returns and exceptions.

        Uses functools.wraps so inspect.signature() follows __wrapped__ to the
        original function — _infer_schema_from_callable must see the real
        parameter names, not (*args, **kwargs).

        Also coerces string-typed arguments to int/float/bool when the
        function annotation requests it (handles Ollama passing "5" for
        an int parameter).
        """
        import functools as _functools
        import inspect as _inspect
        import typing as _typing

        node = self
        tool_name = getattr(fn, "__name__", repr(fn))

        # Resolve type hints once; fall back to {} if any forward ref
        # cannot be resolved (e.g. "DelegationLog | None").
        try:
            _hints = _typing.get_type_hints(fn)
        except Exception:  # noqa: BLE001
            _hints = {}
        _COERCIBLE = {int, float, bool}

        def _coerce(name: str, value: Any) -> Any:
            target = _hints.get(name)
            if target not in _COERCIBLE or not isinstance(value, str):
                return value
            if target is bool:
                low = value.strip().lower()
                if low in ("true", "1", "yes"):
                    return True
                if low in ("false", "0", "no"):
                    return False
                return value
            try:
                return target(value)
            except ValueError:
                return value

        @_functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            # Coerce string args before calling the real function.
            try:
                bound = _inspect.signature(fn).bind_partial(
                    *args, **kwargs
                )
                for pname in list(bound.arguments):
                    bound.arguments[pname] = _coerce(
                        pname, bound.arguments[pname]
                    )
                args, kwargs = bound.args, bound.kwargs
            except TypeError:
                pass  # signature mismatch: let fn raise its own error

            try:
                result = fn(*args, **kwargs)
                if (
                    isinstance(result, str)
                    and result.lstrip().startswith("ERROR:")
                ):
                    node._record_tool_error(
                        node_name,
                        tool_name,
                        "ERROR_RETURN",
                        result[:300],
                    )
                return result
            except Exception as exc:
                node._record_tool_error(
                    node_name,
                    tool_name,
                    type(exc).__name__,
                    str(exc)[:300],
                    tb=traceback.format_exc(),
                )
                raise

        return _wrapped

    def _build_hypothesis_closures(self) -> dict:
        """Build HypothesisPropose/Update/List/Get closures."""
        node = self

        from ..tool_catalog import tool_examples

        @tool_examples(
            "HypothesisPropose('Optimal t/L is near 0.08 — thin walls maximise "
            "buckling', 'a point with t/L in [0.10,0.14] beats "
            "buckling_load_norm 1.47', 'best at t/L ~ 0.08 ± 0.02', 0.55)",
        )
        def HypothesisPropose(
            statement: str,
            falsification_criterion: str,
            prediction: str,
            prior: float,
        ) -> str:
            """Propose a hypothesis. Returns its ID (H1, H2, …) or ERROR.

            statement: ONE falsifiable claim (no compound claims).
            falsification_criterion: what observation would kill it.
            prediction: the measurable outcome you expect.
            prior: plausibility in (0, 1). Max 3 OPEN at any time."""
            if node._ledger is None:
                return (
                    "ERROR: hypothesis ledger not available in this run."
                )
            return node._ledger.propose(
                statement=statement,
                falsification_criterion=falsification_criterion,
                prediction=prediction,
                prior=prior,
                proposed_by=node._name,
            )

        @tool_examples(
            "HypothesisUpdate('H1', 'SUPPORTED', 'sweep top t/L=0.09 beats "
            "threshold', 0.80, evidence={'delegation': 'D001', 'numbers': "
            "{'top_tL': 0.09, 'buckling_load_norm': 1.47}})",
        )
        def HypothesisUpdate(
            hypothesis_id: str,
            status: str,
            comment: str,
            posterior: float,
            evidence: dict | None = None,
        ) -> str:
            """Update hypothesis status with evidence and updated belief.

            status: OPEN | SUPPORTED | FALSIFIED | INCONCLUSIVE.
            posterior: your updated belief in [0, 1] — always required.
            evidence: {"delegation": "D###", "numbers": {key: value}}
              required for closing statuses; numbers must come from
              that delegation's report.
            triggered_by is auto-injected from last completed
            delegation."""
            if node._ledger is None:
                return (
                    "ERROR: hypothesis ledger not available in this run."
                )
            if isinstance(evidence, str):
                import json as _json
                try:
                    evidence = _json.loads(evidence)
                except _json.JSONDecodeError:
                    return (
                        "ERROR: evidence must be a JSON object like "
                        '{"delegation": "D004", "numbers": {...}}.'
                    )
            triggered_by: str | None = (
                node._delegation_log.last_completed_id(node._name)
                if node._delegation_log is not None else None
            )
            if triggered_by is None:
                with node._registry_lock:
                    done_entries = [
                        (d_id, entry)
                        for d_id, entry in node._registry.items()
                        if entry.get("status") in ("Done", "Errored")
                    ]
                if done_entries:
                    triggered_by = done_entries[-1][0]
            result = node._ledger.update(
                hypothesis_id,
                status,
                comment,
                evidence,
                posterior,
                triggered_by,
            )
            if (
                node._science_monitor is not None
                and not result.startswith("ERROR:")
            ):
                inline = node._science_monitor.on_hypothesis_update(
                    hypothesis_id
                )
                if inline:
                    result += "\n\n" + "\n".join(inline)
            return result

        def LinkFalsificationAttempt(
            delegation_id: str, hypothesis_id: str
        ) -> str:
            """Retroactively mark a completed delegation as a falsification
            ATTEMPT of a registered hypothesis (the read-time safety net for
            when the attempt was not declared up front at Delegate time).

            Links ONLY — it does NOT record a verdict and CANNOT change the
            hypothesis's pre-registered prediction. You must still call
            HypothesisUpdate to record the verdict, judged against that
            immutable prediction. Link only if the delegation genuinely tested
            the prediction — never retrofit an exploratory result."""
            if node._ledger is None:
                return (
                    "ERROR: hypothesis ledger not available in this run."
                )
            h_entry = node._ledger.get(hypothesis_id)
            if h_entry is None:
                return f"ERROR: hypothesis {hypothesis_id!r} not found."
            with node._registry_lock:
                entry = node._registry.get(delegation_id)
                if entry is None:
                    return (
                        f"ERROR: unknown delegation {delegation_id!r}. "
                        f"Known: {list(node._registry)}"
                    )
                if entry.get("status") != "Done":
                    return (
                        f"ERROR: {delegation_id} is not a completed (Done) "
                        "delegation; cannot link it as a falsification "
                        "attempt."
                    )
                entry["is_falsification_attempt"] = True
                hids = entry.get("hypothesis_ids") or []
                if hypothesis_id not in hids:
                    hids = [*hids, hypothesis_id]
                entry["hypothesis_ids"] = hids
                entry["reconciled"] = True
            if node._delegation_log is not None:
                node._delegation_log.mark_attempt(
                    delegation_id, hypothesis_id)
            pred = (
                h_entry.get("prediction")
                or h_entry.get("falsification_criterion")
                or "(no prediction on record)"
            )
            return (
                f"Linked {delegation_id} as a falsification attempt of "
                f"{hypothesis_id} (post-hoc). Pre-registered prediction: "
                f"\"{pred}\". Now record the VERDICT: "
                f"HypothesisUpdate('{hypothesis_id}', "
                "status=SUPPORTED|FALSIFIED|INCONCLUSIVE, posterior=…, "
                f"evidence={{'delegation': '{delegation_id}', "
                "'numbers': {…}}), judging THIS report against that "
                "prediction. Linking does NOT record a verdict."
            )

        def HypothesisList(hypothesis_ids: list | None = None) -> str:
            """List all hypotheses with id, status, belief, statement.

            Takes no real arguments — it always lists ALL hypotheses. The
            optional `hypothesis_ids` is accepted-and-ignored so a stray kwarg
            (agents confuse this with Delegate/AskForFeedback) returns the list
            instead of crashing the turn with a TypeError.
            """
            if node._ledger is None:
                return (
                    "ERROR: hypothesis ledger not available in this run."
                )
            items = node._ledger.list_all()
            if not items:
                return "No hypotheses proposed yet."
            lines = [
                f"- {h['id']} [{h['current_status']}]"
                f" (belief {h['belief']}): {h['statement']}"
                for h in items
            ]
            return "\n".join(lines)

        def HypothesisGet(hypothesis_id: str) -> str:
            """Get full hypothesis entry including status_log."""
            if node._ledger is None:
                return (
                    "ERROR: hypothesis ledger not available in this run."
                )
            import json as _json
            entry = node._ledger.get(hypothesis_id)
            if entry is None:
                return f"ERROR: hypothesis {hypothesis_id!r} not found."
            return _json.dumps(entry, indent=2)

        return {
            "HypothesisPropose": HypothesisPropose,
            "HypothesisUpdate": HypothesisUpdate,
            "HypothesisList": HypothesisList,
            "HypothesisGet": HypothesisGet,
            "LinkFalsificationAttempt": LinkFalsificationAttempt,
        }

    def _build_milestone_closures(self) -> dict:
        """Build MilestoneList/Propose/Complete/Skip closures (process policy)."""
        node = self

        def MilestoneList() -> str:
            """List process milestones with id, status, gate/phase, description.

            Milestones are PROCESS steps (do X before Y; get Z ready), distinct
            from hypotheses (epistemics). Default gates self-resolve when their
            condition is met; you author your own with MilestonePropose."""
            if node._milestones is None:
                return "Milestone ledger not available in this run."
            return node._milestones.format()

        def MilestonePropose(description: str, phase: str | None = None,
                             gate: bool = False) -> str:
            """Add your own process milestone. Returns its id (M1, M2, …).

            phase: optional f3dasm phase it relates to. gate=True makes it nudge
            when that phase is entered while still pending."""
            if node._milestones is None:
                return "ERROR: milestone ledger not available in this run."
            return node._milestones.propose(description, phase, gate)

        def MilestoneComplete(milestone_id: str, note: str) -> str:
            """Mark a milestone DONE. A brief `note` (one line on WHY it's
            satisfied — what was done / which delegation) is REQUIRED, so
            ticking is a deliberate, auditable act, not a rubber stamp."""
            if node._milestones is None:
                return "ERROR: milestone ledger not available in this run."
            if not note or not note.strip():
                return (
                    "ERROR: a brief note is required to complete a milestone — "
                    "one line on why it's satisfied (what you did / which "
                    "delegation). This keeps ticking honest and auditable."
                )
            return node._milestones.complete(milestone_id, note)

        def MilestoneSkip(milestone_id: str, reason: str) -> str:
            """Skip a milestone this study legitimately doesn't need (give a
            reason). The escape hatch so soft gates never deadlock you."""
            if node._milestones is None:
                return "ERROR: milestone ledger not available in this run."
            return node._milestones.skip(milestone_id, reason)

        return {
            "MilestoneList": MilestoneList,
            "MilestonePropose": MilestonePropose,
            "MilestoneComplete": MilestoneComplete,
            "MilestoneSkip": MilestoneSkip,
        }

    def _missing_deliverables(self, state: AgenticState) -> list[str]:
        """Return required deliverable paths not present at study_dir yet.

        pipeline.py is always required — the single deliverable, written via
        WriteDeliverable() before Done() is accepted. It is the human-readable
        f3dasm Pipeline AND the reproduction: the runtime executes it lazily
        (see _reproduction_gate) to verify the headline re-derives from the
        ledger with zero new evals. Additional paths can be declared in
        state['required_deliverables'].
        """
        study_dir = Path(state.get("study_dir", "."))
        # WriteDeliverable writes BARE names to study_dir/ (it rejects path
        # separators), and solution.md lands at study_dir/ too. Normalise any
        # configured path to its basename so a stray 'workspace/…' prefix in a
        # study config can't spuriously flag a present deliverable as missing.
        required = ["pipeline.py"] + list(
            state.get("required_deliverables") or [])
        seen: set[str] = set()
        missing: list[str] = []
        for p in required:
            name = Path(p).name
            if name in seen:
                continue
            seen.add(name)
            if not (study_dir / name).exists():
                missing.append(name)
        return missing

    def _reproduction_gate(self, state: AgenticState | None = None) -> str | None:
        """Execute pipeline.py under a CONTROLLED reproduction gate.

        The binding reproducibility check. The pipeline must:
          (a) finish cleanly within a time ceiling (no heavy from-scratch run);
          (b) add ZERO new oracle rows (lazy: skip FINISHED evals);
          (c) NOT modify/delete existing ledger rows (integrity — no faking the
              zero-delta by delete+re-add or value rewrite);
          (d) print ``REPRODUCED: <value>`` that the runtime INDEPENDENTLY
              confirms is grounded in the ledger (an extremum of the objective),
              so the headline cannot be hardcoded/fabricated.
        Returns None on PASS (and stashes ``self._repro_ok_detail``), else a
        problem string. Skips silently when there is no run context. Callable
        without ``state`` — study dir comes from ``self._study_dir``.

        On PASS the deliverable is a faithful, lightweight, lazy reproduction —
        not a script doing "sneaky stuff" unrelated to validating the pipeline.
        """
        import json as _json
        import os
        import re
        import shutil
        import subprocess
        import sys
        import tempfile

        study_dir = (
            Path(self._study_dir) if getattr(self, "_study_dir", None) is not None
            else Path((state or {}).get("study_dir", "."))
        )
        pipeline_py = study_dir / "pipeline.py"
        if not pipeline_py.exists():
            return None  # absence is handled by _missing_deliverables
        notes = self._current_notes_dir
        if notes is None:
            return None  # no run dir context (e.g. non-debug) — skip the gate
        run_dir = notes.parent.parent              # …/runs/<id>
        store_dir = run_dir / "experiment_data"
        run_config = run_dir / "debug" / "run_config.json"

        def _ledger_snapshot(store: Path) -> tuple[int, str, list[float]]:
            """(row_count, content_hash, objective_extrema) for a store dir.

            content_hash is order-independent (sorted rounded values) so a
            faithful lazy re-store doesn't false-trip it; objective_extrema are
            the min/max used for the independent headline check.
            """
            import hashlib
            try:
                from f3dasm import ExperimentData
                data = ExperimentData.from_file(project_dir=store)
                _, out = data.to_pandas()
            except Exception:  # noqa: BLE001
                return 0, "", []
            cols = [c for c in out.columns if not str(c).startswith("_")]
            if not cols:
                return len(out), "", []
            vals = out[cols].round(10)
            rows = sorted(tuple(r) for r in vals.to_numpy().tolist())
            h = hashlib.sha256(repr(rows).encode()).hexdigest()
            obj_cols = cols
            try:
                if run_config.exists():
                    _on = _json.loads(run_config.read_text()).get(
                        "evaluator_output_names")
                    if _on and _on[0] in cols:
                        obj_cols = [_on[0]]
            except Exception:  # noqa: BLE001
                pass
            extrema: list[float] = []
            for c in obj_cols:
                try:
                    extrema += [float(out[c].min()), float(out[c].max())]
                except Exception:  # noqa: BLE001
                    pass
            return len(out), h, extrema

        # ── HERMETIC SANDBOX ──────────────────────────────────────────────────
        # CRITICAL: run pipeline.py against a COPY of the canonical store, never
        # the live one. A faithful lazy pipeline adds nothing; a NON-lazy one
        # (re-evaluating) writes its evals into the THROWAWAY copy — we detect
        # that as "not lazy" while the real ledger stays pristine. Without this,
        # checking a non-lazy pipeline pollutes + inflates the canonical store
        # (and CheckDeliverable could be looped to balloon it without bound).
        before_n, before_hash, extrema = _ledger_snapshot(store_dir)
        sandbox = Path(tempfile.mkdtemp(prefix="f3dasm_repro_"))
        try:
            sb_store = sandbox / "experiment_data"
            if store_dir.exists():
                shutil.copytree(store_dir, sb_store)
            else:
                sb_store.mkdir(parents=True, exist_ok=True)
            # A sandbox run_config so get_evaluator() also writes to the COPY
            # (it resolves the store from run_config["store_dir"], not the env).
            sb_run_config = sandbox / "run_config.json"
            if run_config.exists():
                _cfg = _json.loads(run_config.read_text())
            else:
                _cfg = {}
            _cfg["store_dir"] = str(sb_store)
            _cfg["lock_path"] = str(sb_store / "experiment_data" / ".lock")
            sb_run_config.write_text(_json.dumps(_cfg))

            env = dict(os.environ)
            env["F3DASM_CANONICAL_STORE"] = str(sb_store)
            env["F3DASM_RUN_CONFIG"] = str(sb_run_config)
            env.setdefault("F3DASM_DELEGATION_ID", "D999")
            _timeout = (
                max(0.1 * self._budget_seconds, 180.0)
                if self._budget_seconds else 300.0
            )
            try:
                proc = subprocess.run(
                    [sys.executable, str(pipeline_py)],
                    cwd=str(sandbox), env=env,
                    capture_output=True, text=True, timeout=_timeout,
                )
            except subprocess.TimeoutExpired:
                return (
                    f"pipeline.py did not finish within {_timeout:.0f}s. A "
                    "reproduction must be lightweight — load the ledger and skip "
                    "finished evals and heavy refits (cache-or-load surrogates). "
                    "Make it lazy.")
            after_n, after_hash, _ = _ledger_snapshot(sb_store)
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

        # (a) clean exit — surface a generous stderr tail for sighted debugging.
        if proc.returncode != 0:
            return (
                f"pipeline.py FAILED to run (exit {proc.returncode}). It must "
                "load the ledger and derive the headline cleanly. Stderr:\n"
                + (proc.stderr or "")[-3000:]
                + ("\n\nStdout tail:\n" + proc.stdout[-800:]
                   if proc.stdout else ""))
        # (b) zero new evals (lazy).
        if after_n != before_n:
            return (
                f"pipeline.py is NOT lazy: re-running it changed the ledger row "
                f"count ({before_n} → {after_n}). It must LOAD the ledger "
                "(ExperimentData.from_file) and reach the oracle only via "
                "get_evaluator() so FINISHED rows are skipped — zero new evals.")
        # (c) integrity — existing rows unchanged.
        if before_hash and after_hash and before_hash != after_hash:
            return (
                "pipeline.py MODIFIED existing ledger rows. A reproduction must "
                "read the ledger READ-ONLY (it may re-store identical rows, but "
                "must not rewrite values or delete+re-add). Do not tamper with "
                "the canonical store.")
        # (d) independent headline check — the printed REPRODUCED value must be
        # grounded in the ledger, not hardcoded/fabricated.
        m = re.search(r"REPRODUCED:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
                      proc.stdout or "")
        if m is None:
            return (
                "pipeline.py did not print a headline the runtime can verify. "
                "Its analyze step must derive the result FROM the ledger and "
                "print exactly 'REPRODUCED: <value>' so the runtime can confirm "
                "it independently (this is how a fabricated/hardcoded headline "
                "is caught).")
        claimed = float(m.group(1))
        if extrema:
            tol = 1e-6 + 1e-6 * max(abs(x) for x in extrema)
            if not any(abs(claimed - x) <= tol for x in extrema):
                return (
                    f"pipeline.py printed REPRODUCED: {claimed}, which is NOT "
                    "grounded in the ledger (objective extrema in the canonical "
                    f"store: {sorted(set(round(x, 6) for x in extrema))}). "
                    "Derive the headline from the loaded rows — do not hardcode "
                    "or fabricate it.")
        self._repro_ok_detail = (
            f"verified REPRODUCED={claimed} against the ledger "
            f"({before_n} rows, unchanged, 0 new evals, "
            f"ran in <{_timeout:.0f}s)")
        return None

    def __call__(self, state: AgenticState) -> Any:
        import time

        from langchain_core.messages import AIMessage, HumanMessage
        from langgraph.graph import END
        from langgraph.types import Command

        # Update notes_dir from current state run_dir
        run_dir = state.get("run_dir")
        if run_dir:
            self._current_notes_dir = (
                Path(run_dir) / "debug" / "strategizer_notes"
            )
            if self._ledger is None:
                self._ledger = HypothesisLedger(self._current_notes_dir)
            # Wire canonical store dir into ScienceMonitor lazily.
            # store_dir is the ExperimentData *project_dir* (run_dir/
            # experiment_data), NOT the folder holding the CSVs. ExperimentData
            # appends its own EXPERIMENTDATA_SUBFOLDER ("experiment_data"), so
            # the rows live one level deeper at
            # run_dir/experiment_data/experiment_data/output.csv — hence the
            # apparent double directory is correct, not a typo.
            if self._science_monitor is not None:
                self._science_monitor.store_dir = (
                    self._current_notes_dir.parent.parent
                    / "experiment_data"
                )

        # Eval budget → available to the Done() critic gate for budget-aware
        # framing (judge the best honest conclusion within evals spent).
        self._eval_budget = state.get("eval_budget")

        # Capture total_delegations so Delegate() can seed the counter.
        self._state_total_delegations = state.get("total_delegations", 0)
        # Seed the monotonic counter from state on first __call__ (or
        # after a checkpoint rebuild).  Never decremented — ensures IDs
        # are unique even when the registry is pruned between turns.
        if self._delegation_seq < self._state_total_delegations:
            self._delegation_seq = self._state_total_delegations

        # Time budget is a SOFT constraint — warnings only; the run is
        # never force-terminated for exceeding it. A separate run-level
        # backstop (RUN_BACKSTOP_MULTIPLE x budget) bounds runaway cost.
        budget_warnings: list[dict] = []
        budget = state.get("budget_seconds")
        start = state.get("start_time")
        # Store on node so GetStatus() can compute delegation timeout
        self._budget_seconds = budget
        self._run_start = start
        self._budget_usd = state.get("budget_usd")
        if budget is not None and start is not None:
            elapsed = time.time() - start
            pct = elapsed / budget
            if pct >= 1.0:
                budget_warnings.append({
                    "role": "user",
                    "content": (
                        f"Time budget fully consumed "
                        f"({elapsed:.0f}s / {budget:.0f}s). This is an "
                        "advisory soft limit — the run is NOT terminated. "
                        "Wind down: finish the experiment in flight, then "
                        "wrap up and call Done(); avoid starting new "
                        "delegations. A hard cost backstop applies only at "
                        f"{int(run_backstop_multiple())}x budget. Do NOT cancel "
                        "a delegation that is still progressing to save time — "
                        "its ledgered evals already persist, so cancelling only "
                        "throws away its report; let it finish and read it."
                    ),
                })
            elif pct >= 0.95:
                budget_warnings.append({
                    "role": "user",
                    "content": (
                        f"Warning: time budget at {pct*100:.0f}% "
                        f"({elapsed:.0f}s / {budget:.0f}s). "
                        "Begin wrapping up — call Done() soon. Don't cancel a "
                        "progressing delegation under time pressure; its evals "
                        "are already ledgered and cancelling only loses its "
                        "report (GetStatus shows whether it's progressing)."
                    ),
                })

        eval_budget = state.get("eval_budget")
        evals_used = state.get("evals_used", 0)
        if eval_budget is not None and evals_used >= eval_budget:
            budget_warnings.append({
                "role": "user",
                "content": (
                    f"Warning: eval budget exceeded"
                    f" ({evals_used} used / {eval_budget} budget)."
                    f" Do not run further evaluations."
                ),
            })

        _halt = self._check_unrecoverable(state, budget, start)
        if _halt is not None:
            return _halt

        # A1/A2: reset per-turn state so a reused node starts clean each call.
        # Working/FollowUp entries are preserved so loopbacks don't orphan live
        # delegations whose background threads are still running.
        self._route.clear()
        self._ask_count = 0
        self._done_warned = False
        with self._registry_lock:
            self._registry = {
                d: e for d, e in self._registry.items()
                if e["status"] in ("Working", "FollowUp")
            }
            self._threads = {
                d: t for d, t in self._threads.items()
                if d in self._registry
            }
        with self._notifications_lock:
            self._notifications.clear()

        # ── No-canonical-source nudge (soft, ≤3×) ─────────────────────────
        # If this graph has a datagenerator (so a canonical ground-truth
        # source CAN be authored) but none is registered, recommend
        # delegating to it. Without a registered source every evaluation
        # lands off-ledger and nothing is reproducible from the canonical
        # store. Soft and capped — never blocks; the strategizer may ignore
        # it for a genuinely source-free study.
        registration_nudge: list[dict] = []
        if (
            self._no_source_nudges < 3
            and self._find_datagenerator_name() is not None
            and not self._canonical_source_registered()
        ):
            self._no_source_nudges += 1
            _dg = self._find_datagenerator_name()
            registration_nudge.append({
                "role": "user",
                "content": (
                    "[SETUP] No canonical ground-truth source is registered "
                    "for this study (no evaluator entrypoint or lookup pool). "
                    f"A '{_dg}' agent is available — delegate to it to author "
                    "and register the source, so evaluations flow through "
                    "get_evaluator(), land in the canonical store, and the "
                    "result is reproducible. If this is intentionally a "
                    "source-free (surrogate-only) study, disregard this. "
                    f"(notice {self._no_source_nudges}/3)"
                ),
            })
            self._record_intervention(
                "NO_SOURCE_NUDGE", self._name,
                "No canonical source registered; recommended delegating to "
                f"'{_dg}'.",
                notice=self._no_source_nudges, cap=3,
            )

        # Announce the process backlog ONCE, at the start, as a conversation
        # message — so the agent cannot claim it didn't know these gate the
        # implementer. Injected the first time the strategizer is invoked.
        backlog_announce: list = []
        if (self._milestones is not None
                and not getattr(self, "_backlog_announced", False)):
            from ..milestones import render_backlog
            _bl = render_backlog(self._milestones)
            if _bl:
                backlog_announce = [{"role": "user", "content": _bl}]
            self._backlog_announced = True

        messages = (
            _to_adapter_messages(state["messages"])
            + budget_warnings + registration_nudge + backlog_announce
        )
        # DEBUG: stream this strategizer turn's full reasoning + tool-calls
        # to debug/transcripts/strategizer/turn_NNN.jsonl.
        from ..backends.base import (
            debug_enabled as _dbg,
        )
        from ..backends.base import (
            set_transcript_sink as _set_sink,
        )
        self._turn_count = getattr(self, "_turn_count", 0) + 1
        if _dbg() and self._current_notes_dir is not None:
            _set_sink(str(
                self._current_notes_dir.parent / "transcripts"
                / "strategizer" / f"turn_{self._turn_count:03d}.jsonl"))
        text = self.adapter.invoke(messages)
        # Accumulate strategizer's own token usage.
        self._record_usage(
            getattr(self.adapter, "last_usage", {}) or {},
            role=self._role_of(self._name),
            model=getattr(self.adapter, "model", None),
            phase="strategizer_turn",
            delegation_id=None,
        )
        ai_msg = AIMessage(content=text)

        route = self._route
        accepted = route.get("kind") == "done"
        missing = self._missing_deliverables(state)
        # Reproduction is owned entirely by the Done() gate now (it runs the
        # controlled gate before any close and declares a FAILED run after a
        # bounded number of sighted attempts — see CheckDeliverable). So there is
        # no separate post-accept repro check here; this branch handles only
        # deliverable presence and un-accepted termination.

        # ── Bounded re-prompt on unaccepted termination ───────────────────────
        if (not accepted or missing) and self._finish_attempts < 3:
            self._finish_attempts += 1
            problems: list[str] = []
            if missing:
                missing_list = "\n".join(f"- {p}" for p in missing)
                problems.append(
                    "Required deliverables are missing from the"
                    f" study directory:\n{missing_list}\n"
                    "Write them via WriteDeliverable() before"
                    " calling Done()."
                )
            if not accepted:
                with self._registry_lock:
                    working = [
                        d for d, e in self._registry.items()
                        if e["status"] in ("Working", "FollowUp")
                    ]
                if working:
                    problems.append(
                        f"Delegations still running: {working}."
                        " Poll them with GetStatus() and call Done()"
                        " once they finish."
                    )
                else:
                    problems.append(
                        "You ended your turn without an accepted"
                        " Done(). If Done() was refused (critic"
                        " verdict, two-shot confirmation, or another"
                        " gate), address the refusal and call Done()"
                        " again. A run only closes through an"
                        " accepted Done()."
                    )
            return Command(
                goto=self._name,
                update={
                    "messages": [
                        ai_msg,
                        HumanMessage(content=(
                            "Run cannot complete"
                            f" (attempt {self._finish_attempts}/3):\n"
                            + "\n\n".join(problems)
                        )),
                    ],
                },
            )

        # ── Terminal branch ───────────────────────────────────────────────────
        # Accumulate delegation counts and evals from registry
        with self._registry_lock:
            total_new = len(self._registry)
            evals_new = sum(e["evals"] for e in self._registry.values())

        summary = route.get("summary") or text

        # Prepend UNGATED banner if the run ends without an accepted Done().
        # (A FAILED-reproduction close carries its own ⛔ banner in route summary
        # and IS accepted=done, so it is not re-banner'd here.)
        if not accepted or missing:
            flags = []
            if not accepted:
                flags.append(
                    "the run terminated WITHOUT an accepted Done() —"
                    " the final conclusions did NOT pass the"
                    " adversarial critic gate"
                )
            if missing:
                flags.append(
                    f"required deliverables missing: {missing}"
                )
            summary = (
                "## ⚠ UNGATED RUN\n\n"
                "This run is NOT validated: " + "; ".join(flags) +
                ".\nTreat all conclusions below as unaudited.\n\n---\n\n"
                + summary
            )

        return Command(
            goto=END,
            update={
                "messages": [ai_msg],
                "done": True,
                "last_report": summary,
                "total_delegations": state["total_delegations"] + total_new,
                "evals_used": state.get("evals_used", 0) + evals_new,
                "token_totals": dict(self._token_totals),
                "error_counts": dict(self._error_counts),
            },
        )


