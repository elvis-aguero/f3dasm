"""LangGraph node classes for the f3dasm agentic runtime.

Each node class implements ``__call__(state: AgenticState) -> Command``,
which is the ADAS-inspectable topology entry point.
``inspect.getsource(StrategizerNode.__call__)`` reads the full routing logic.
"""

from __future__ import annotations

import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .graph_state import AgenticState

from .delegation_log import DelegationLog
from .hypothesis_ledger import HypothesisLedger
from .science_monitor import ScienceMonitor

__all__ = [
    "AgentNode", "StrategizerNode", "WorkerNode", "ImplementerNode",
    "_to_adapter_messages",
    "_resolve_delegation_evals",
]


def _resolve_delegation_evals(
    store_dir: Path | None,
    delegation_id: str,
    reported: int,
) -> int:
    """Return the eval count for a delegation.

    Prefers the row count from the canonical evaluation ledger
    (authoritative) over the honour-system ReportEvals self-report.
    Falls back to *reported* for delegations that bypassed the ledger
    (wrote no rows) — e.g. delegations using lookup tables directly.

    Parameters
    ----------
    store_dir:
        The run-level directory that contains ``experiment_data/``
        (i.e. ``run_config["store_dir"]``).  ``None`` disables ledger
        counting and always returns *reported*.
    delegation_id:
        E.g. ``"D003"``.
    reported:
        The value from ``ReportEvals`` (honour-system fallback).
    """
    if store_dir is None:
        return reported
    try:
        from .instrumented import RunStateSummary
        summary = RunStateSummary.from_store(store_dir)
        if (
            summary is not None
            and summary.n_per_delegation.get(delegation_id, 0) > 0
        ):
            return summary.n_per_delegation[delegation_id]
    except Exception:  # noqa: BLE001
        pass
    return reported


def _stamped_eval_count(store_dir: Path | None, delegation_id: str) -> int:
    """Rows in the canonical store stamped with this delegation_id (0 if none).

    Unlike _resolve_delegation_evals (which falls back to the honour-system
    count), this reports ONLY provenance-stamped rows — so a caller can detect
    a delegation that evaluated but bypassed get_evaluator().
    """
    if store_dir is None:
        return 0
    try:
        from .instrumented import RunStateSummary
        summary = RunStateSummary.from_store(store_dir)
        if summary is not None:
            return int(summary.n_per_delegation.get(delegation_id, 0))
    except Exception:  # noqa: BLE001
        pass
    return 0

# Time budget is a SOFT constraint (warnings only). This multiple is the
# run-level cost backstop: a run is aborted once it exceeds
# RUN_BACKSTOP_MULTIPLE x the time budget, to bound runaway cost.
# Wall-clock is a poor proxy for cost when a single oracle eval can take days
# (the SOTA problems), so the cap is configurable and can be DISABLED: set
# F3DASM_RUN_BACKSTOP_MULTIPLE <= 0 to turn the hard cap off entirely.
import os as _os  # noqa: E402

RUN_BACKSTOP_MULTIPLE = float(
    _os.environ.get("F3DASM_RUN_BACKSTOP_MULTIPLE", "2.0")
)
_BACKSTOP_ENABLED = RUN_BACKSTOP_MULTIPLE > 0

_REQUIRED_SUBSECTIONS = [
    "### Actions taken",
    "### Files touched",
    "### Conclusions",
    "### Numbers",
]

# Post-Done exit interview for the strategizer. Asked as a SEPARATE turn only
# after the critic accepted the conclusion — so the strategizer never carries
# the interview in its working context (no pollution). It answers with one more
# Done() whose summary is just a ### Retrospective block.
_EXIT_INTERVIEW = (
    "Your conclusion has been accepted by the critic and recorded — the run "
    "is effectively closed. One last thing before we finalise: a quick "
    "question about the SYSTEM you worked within (its rules, tools, and the "
    "monitor/critic feedback), NOT the science. Call Done() ONE more time "
    "with a summary containing only a ### Retrospective block:\n"
    "- CONSISTENCY: ok | flagged — did any rule, tool, monitor message, or "
    "critic finding contradict another, or contradict what you were told "
    "elsewhere (e.g. a rule that rejected evidence you believe was correct)? "
    "Write 'flagged' and QUOTE both sides; otherwise 'ok'. (Most important.)\n"
    "- DECISION: the one strategic choice you were least sure the system "
    "wanted, and why you made it.\n"
    "- FRICTION: anything counterintuitive about the rules/tools, or 'none'.\n"
    "This will NOT reopen the run."
)
_CAPABILITY_PHRASES = [
    "i cannot", "i can't", "i don't have access",
    "unable to", "not able to", "i am unable",
]

# The critic emits exactly these three (agents/critic.py); no others.
_VALID_VERDICTS = {"PASS", "REVISE", "REJECT"}


def _parse_verdict(text: str) -> str:
    """Extract the critic's GATE verdict (PASS/REVISE/REJECT/…) from its text.

    Tolerant of markdown emphasis and punctuation around the token — critics
    write ``### Verdict\\n\\n**PASS**`` — so the leading ``**`` no longer makes
    a bare ``(\\w+)`` capture the asterisk and fall through to UNKNOWN (the bug
    that turned an earned PASS into an infinite revise loop). Falls back to a
    ``verdict: X`` line (e.g. in a ### Numbers block). Returns the UPPER token
    or ``"UNKNOWN"``.
    """
    import re as _re
    m = _re.search(
        r"###\s*Verdict\b[\s:>*_`\"'\-]*([A-Za-z]+)", text, _re.IGNORECASE)
    if m and m.group(1).upper() in _VALID_VERDICTS:
        return m.group(1).upper()
    for mm in _re.finditer(
        r"verdict\s*[:=]\s*[*_`\"']*([A-Za-z]+)", text, _re.IGNORECASE
    ):
        if mm.group(1).upper() in _VALID_VERDICTS:
            return mm.group(1).upper()
    return "UNKNOWN"


def _to_adapter_messages(lc_messages: list) -> list[dict]:
    """Convert LangChain message objects to adapter-format dicts."""
    from langchain_core.messages import AIMessage, HumanMessage

    result: list[dict] = []
    for msg in lc_messages:
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            result.append({"role": "user", "content": str(content)})
        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            result.append({"role": "ai", "content": str(content)})
    return result


def _classify_response(
    text: str,
    required_sections: list[str] | None = None,
) -> str | None:
    """Return a REFLECT diagnosis string if text is malformed, else None.

    Parameters
    ----------
    text : str
        The raw response text from a worker agent.
    required_sections : list[str] or None
        Subsection headers that must appear in the ``## Report`` block.
        When ``None`` the four default sections from
        ``_REQUIRED_SUBSECTIONS`` are used.
    """
    from .agent_prompts import (
        REFLECT_DIAGNOSIS_CAPABILITY_LIMIT,
        REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE,
        REFLECT_DIAGNOSIS_NO_REPORT_HEADING,
        REFLECT_DIAGNOSIS_SHORT,
    )

    sections = _REQUIRED_SUBSECTIONS if required_sections is None else required_sections

    if len(text.strip()) < 100:
        return REFLECT_DIAGNOSIS_SHORT
    low = text.lower()
    if any(p in low for p in _CAPABILITY_PHRASES):
        return REFLECT_DIAGNOSIS_CAPABILITY_LIMIT
    if "## report" not in low:
        return REFLECT_DIAGNOSIS_NO_REPORT_HEADING
    missing = [s for s in sections if s.lower() not in low]
    if missing:
        return REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE.format(
            missing_subsections=", ".join(f"'{s}'" for s in missing)
        )
    return None


def _extract_report_section(text: str, name: str) -> str:
    """Return the body under a ``### <name>`` report heading, or '' if absent.

    Captures from the heading to the next ``###``/``##`` heading, a
    horizontal rule, or end of text. Best-effort and tolerant of trailing
    free-form content.
    """
    import re as _re
    m = _re.search(
        rf"(?mis)^###\s+{_re.escape(name)}\s*\n(.*?)"
        r"(?=^\s*###\s|^\s*##\s|^---\s*$|\Z)",
        text,
    )
    return m.group(1).strip() if m else ""


def _consult_handbook(query: str = "") -> str:
    """ConsultHandbook tool: browse the curated handbook of project conventions.

    Call with NO argument to get the table of contents (every chapter's id +
    title). Pass a chapter id (e.g. "falsification-charter") to read that one
    chapter in full. Pass free-text keywords to search when you don't know the
    id. Read-only and best-effort — never raises into the agent loop.
    """
    try:
        from .knowledge import KnowledgeBase
        kb = KnowledgeBase.load()
    except Exception as exc:  # noqa: BLE001
        return f"(handbook unavailable: {exc})"
    q = str(query).strip()
    if not q:
        return kb.toc()
    entry = kb.get(q)  # exact chapter id → full chapter
    if entry is not None:
        return entry.render()
    hits = kb.search(q, k=3)  # otherwise keyword search
    if not hits:
        return (
            "No chapter id or keyword matched. Call ConsultHandbook() with no "
            "argument to list the available chapters."
        )
    return "\n\n---\n\n".join(e.render() for e in hits)


class AgentNode:
    """Base class for ADAS-inspectable LangGraph nodes.

    Subclasses override __call__ to define routing topology.
    inspect.getsource(MyNode.__call__) reads the full routing logic.
    """

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def __call__(self, state: AgenticState) -> Any:
        raise NotImplementedError


class StrategizerNode(AgentNode):
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
        self.adapter.closure_tools.update(self._build_routing_closures())
        self.adapter.route_watcher = lambda: self._route.get("kind") == "done"

    def _find_critic_name(self) -> str | None:
        """Name of the first connected critic worker, or None."""
        spec = self._spec
        if spec is None or not hasattr(spec, "nodes"):
            return None
        for target in self._outgoing:
            agent = spec.nodes.get(target)
            if (
                agent is not None
                and getattr(agent, "role", None) == "critic"
                and target in self._worker_adapters
            ):
                return target
        return None

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

    def _invoke_critic(self, task_msg: str) -> str:
        """Synchronously invoke the connected critic; returns its
        text or an ERROR string.

        The critic is a worker too: under F3DASM_DEBUG its full transcript is
        streamed to disk, its verdict/review is ALWAYS persisted (the PASS
        branch doesn't echo it to the strategizer, so this is the only place
        the deciding verdict is auditable), and its ### Retrospective is
        recorded like every other node's (#7).
        """
        critic_name = self._find_critic_name()
        if critic_name is None:
            return "ERROR: no critic connected."
        adapter = self._worker_adapters[critic_name]
        worker = (
            adapter.copy() if hasattr(adapter, "copy") else adapter
        )
        self._critic_calls = getattr(self, "_critic_calls", 0) + 1
        _n = self._critic_calls
        from .backends.base import (
            debug_enabled as _dbg,
        )
        from .backends.base import (
            get_transcript_sink as _get_sink,
        )
        from .backends.base import (
            set_transcript_sink as _set_sink,
        )
        _notes = self._current_notes_dir
        _prev_sink = _get_sink()
        if _dbg() and _notes is not None:
            _set_sink(str(
                _notes.parent / "transcripts" / "critic"
                / f"call_{_n:03d}.jsonl"))
        try:
            critique = worker.invoke(
                [{"role": "user", "content": task_msg}]
            )
        except Exception:  # noqa: BLE001
            critique = (
                "ERROR: critic invocation failed:\n"
                f"{traceback.format_exc()}"
            )
        finally:
            # Restore the strategizer's own sink (same thread-local).
            if _dbg() and _notes is not None:
                _set_sink(_prev_sink)
        # Always-on: persist the verdict/review to disk + record retrospective.
        self._persist_critic_review(_n, critique)
        self._record_retrospective("critic", f"critic-{_n}", critique)
        return critique

    def _persist_critic_review(self, n: int, critique_text: str) -> None:
        """Write the critic's full review to debug/critic_reviews/ so the
        deciding verdict is auditable regardless of PASS/REVISE. Best-effort."""
        try:
            notes = self._current_notes_dir
            if notes is None:
                return
            d = Path(notes).parent / "critic_reviews"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"call_{n:03d}.md").write_text(
                critique_text or "", encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    def _build_feedback_task_msg(self, h_ids: list) -> str:
        """<mode>FEEDBACK</mode> task message with paths block."""
        notes_path = str(self._current_notes_dir or "")
        _notes_dir = self._current_notes_dir
        _study_dir = self._study_dir
        _debug_dir = (
            _notes_dir.parent if _notes_dir is not None else None
        )
        return (
            "<mode>FEEDBACK</mode>\n\n"
            "Perform a synchronous find-only adversarial audit.  "
            "PASS is not an available verdict — return REVISE or "
            "REJECT with your findings.\n\n"
            "<paths>\n"
            f"study_dir             = {_study_dir}\n"
            f"debug_dir             = {_debug_dir}\n"
            f"delegation_log        = {_debug_dir}/delegation_log.jsonl\n"
            f"diagnostics           = {_debug_dir}/diagnostics.jsonl\n"
            f"strategizer_notes     = {notes_path}\n"
            f"delegations_workspace = {_debug_dir}/delegations/\n"
            f"deliverable            = {_study_dir}/replicate.py "
            "(solution.md is written by the runtime AFTER this gate, from the "
            "accepted summary — do NOT flag it as missing)\n"
            "</paths>\n\n"
            f"Focus hypotheses: {h_ids if h_ids else 'all'}"
        )

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
        """Return routing closure tools that write to self._route."""
        node = self
        route = self._route
        outgoing = self._outgoing
        study_dir = self._study_dir
        interactive = self._interactive

        # Build target hints from each connected agent's description.
        _target_hints = "\n  ".join(
            f"{t}: {node._spec.nodes[t].description}"
            for t in outgoing
            if t in node._spec.nodes
        )
        _delegate_doc = (
            "Fire a task to a connected agent.\n\n"
            "wait=False (default): returns a D### ID immediately; poll with\n"
            "  GetStatus(id) to retrieve the result.\n"
            "wait=True: blocks until the worker finishes and returns the report\n"
            "  directly. Use this for sequential tasks where you do not need\n"
            "  parallelism — eliminates all GetStatus() polling.\n\n"
            "CONTEXT PACKAGING: workers start each delegation with no memory of\n"
            "prior delegations. Include in the task message everything the worker\n"
            "needs: relevant paths, key findings from prior delegations, and the\n"
            "precise question to answer.\n\n"
            "hypothesis_ids must be non-empty when the ledger is active.\n"
            "The worker writes exclusively to {id}/ (relative to their workspace\n"
            "in debug/delegations/).\n\n"
            "Set is_falsification_attempt=True when this delegation attacks"
            " a hypothesis's stated falsification criterion.\n\n"
            f"Available targets:\n  {_target_hints}"
        )

        def Delegate(
            target: str,
            intent: str,
            expected_report: str,
            hypothesis_ids: list | None = None,
            wait: bool = False,
            is_falsification_attempt: bool = False,
        ) -> str:
            if target not in outgoing:
                return (
                    f"ERROR: unknown target {target!r}."
                    f" Valid targets: {outgoing}"
                )
            worker_template = node._worker_adapters.get(target)
            if worker_template is None:
                return (
                    f"ERROR: no worker adapter for target {target!r}."
                    f" Available: {list(node._worker_adapters)}"
                )

            # Enforce hypothesis linkage when ledger is active.
            # LLMs pass strings in several shapes — decode all of
            # them: '["H1","H2"]' (JSON), 'H1, H2' (joined), 'H1'.
            if isinstance(hypothesis_ids, str):
                raw = hypothesis_ids.strip()
                if raw.startswith("["):
                    import json as _json
                    try:
                        decoded = _json.loads(raw)
                        h_ids: list[str] = [
                            str(h) for h in decoded
                        ] if isinstance(decoded, list) else [raw]
                    except _json.JSONDecodeError:
                        h_ids = [raw]
                elif "," in raw:
                    h_ids = [
                        p.strip() for p in raw.split(",") if p.strip()
                    ]
                else:
                    h_ids = [raw]
            else:
                h_ids = [str(h) for h in (hypothesis_ids or [])]
            if node._ledger is not None:
                if not h_ids:
                    return (
                        "ERROR: hypothesis_ids must not be empty. "
                        "Every delegation must be linked to at "
                        "least one hypothesis. Call "
                        "HypothesisList() to see open hypotheses."
                    )
                known = {
                    h["id"] for h in node._ledger.list_all()
                }
                unknown = [h for h in h_ids if h not in known]
                if unknown:
                    return (
                        f"ERROR: unknown hypothesis IDs {unknown}."
                        f" Valid IDs: "
                        f"{sorted(known) or '(none proposed yet)'}."
                    )

            # Delegation ID: globally unique when a shared DelegationLog
            # is present (multiple orchestrating nodes share one log, so
            # IDs must be unique across all of them).  Falls back to the
            # per-node monotonic counter when no log is attached.
            if node._delegation_log is not None:
                delegation_id = node._delegation_log.next_id()
                # Keep per-node seq in sync so checkpoint/WorkerNode
                # paths that read _delegation_seq stay consistent.
                with node._registry_lock:
                    try:
                        node._delegation_seq = int(
                            delegation_id[1:]
                        )
                    except (ValueError, IndexError):
                        pass
            else:
                with node._registry_lock:
                    node._delegation_seq += 1
                    delegation_id = f"D{node._delegation_seq:03d}"

            start_time_mono = time.monotonic()
            started_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

            _followup_event = threading.Event()
            with node._registry_lock:
                node._registry[delegation_id] = {
                    "status": "Working",
                    "result": None,
                    "evals": 0,
                    "start_time": start_time_mono,
                    "hypothesis_ids": h_ids,
                    "is_falsification_attempt": bool(
                        is_falsification_attempt
                    ),
                    "started_at": started_at,
                    "target": target,
                    "followup_question": None,
                    "followup_answer": None,
                    "followup_event": _followup_event,
                    "getstatus_count": 0,
                    "followup_count": 0,
                }

            # Build task message
            edge = node._spec.edge(node._name, target)
            preamble = edge.preamble if edge else ""
            task_msg = (
                f"<workspace_subfolder>{delegation_id}/</workspace_subfolder>\n\n"
                + intent
            )
            if expected_report:
                task_msg += (
                    f"\n\n**Required deliverables / acceptance"
                    f" criteria:**\n{expected_report}"
                )
            if preamble:
                task_msg = preamble + "\n\n" + task_msg

            # Prepend time-budget banner so every worker starts
            # time-aware.  GetStatus handles mid-run updates.
            _b = node._budget_seconds
            _rs = node._run_start
            if _b is not None and _rs is not None:
                _el = time.time() - _rs
                _pct = (_el / _b) * 100
                _banner = (
                    f"[Time budget: {_el:.0f}s / {_b:.0f}s used"
                    f" ({_pct:.0f}%). Work efficiently and"
                    f" return a report promptly.]\n\n"
                )
                task_msg = _banner + task_msg

            # Inject PROBLEM_STATEMENT for agents that request it
            _target_agent = node._spec.nodes.get(target) if node._spec else None
            if getattr(_target_agent, "inject_problem_statement", False) and node._study_dir:
                _ps_path = Path(node._study_dir) / "PROBLEM_STATEMENT.md"
                if _ps_path.exists():
                    _ps_text = _ps_path.read_text(encoding="utf-8")
                    task_msg = (
                        f"<problem_statement>\n{_ps_text}\n</problem_statement>\n\n"
                        + task_msg
                    )

            # Each delegation gets its OWN adapter copy (D1/D2 concurrency fix).
            worker = worker_template.copy() if hasattr(worker_template, "copy") else worker_template

            # Inject per-delegation Write sandbox: writes go to {delegation_id}/ only.
            _study = node._study_dir
            if _study is not None:
                _workspace = (
                    node._workspace_dir.resolve()
                    if node._workspace_dir is not None
                    else (Path(node._study_dir or ".") / "debug" / "delegations").resolve()
                )
                _delegation_ws = (_workspace / delegation_id).resolve()

                def _sandboxed_write(path: str, body: str, _ws=_delegation_ws, _did=delegation_id) -> str:
                    """Write restricted to {delegation_id}/."""
                    try:
                        candidate = (_ws / path).resolve()
                    except Exception as exc:  # noqa: BLE001
                        return f"ERROR: invalid path {path!r}: {exc}"
                    try:
                        candidate.relative_to(_ws)
                    except ValueError:
                        return (
                            f"ERROR: write rejected — {candidate} is outside "
                            f"{_ws}. Write only to {_did}/."
                        )
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    candidate.write_text(body, encoding="utf-8")
                    return f"Written: {candidate}"

                worker.closure_tools["Write"] = node._wrap_closure(_sandboxed_write, target)

            def _run() -> None:
                evals_box: dict = {"count": 0}

                def ReportEvals(count: int) -> str:
                    """Report the number of function evaluations used."""
                    evals_box["count"] = int(count)
                    # Drain any queued budget warnings for this delegation.
                    with node._pending_worker_msgs_lock:
                        msgs = node._pending_worker_msgs.pop(delegation_id, [])
                    prefix = ("\n".join(msgs) + "\n\n") if msgs else ""
                    return prefix + f"Recorded {count} evaluations."

                def FollowUp(question: str) -> str:
                    """Ask your delegating party one clarifying question before proceeding.

                    Routes to whoever sent you this task: the agent that delegated
                    to you.  One FollowUp per delegation.  The answer is injected
                    directly into your context.  If no answer arrives, proceed with
                    best judgment.
                    """
                    with node._registry_lock:
                        entry = node._registry.get(delegation_id, {})
                        if entry.get("followup_count", 0) >= 1:
                            return (
                                "FollowUp limit reached (1 per delegation). "
                                "Proceed with best judgment."
                            )
                        node._registry[delegation_id]["followup_question"] = question
                        node._registry[delegation_id]["followup_count"] = 1
                        node._registry[delegation_id]["status"] = "FollowUp"
                        evt = node._registry[delegation_id]["followup_event"]
                    with node._notifications_lock:
                        node._notifications.append(
                            f"[{delegation_id} FollowUp: {question!r} "
                            f"→ call Reply('{delegation_id}', answer)]"
                        )
                    evt.wait(timeout=300)  # 5-minute patience; proceed if no reply
                    with node._registry_lock:
                        answer = node._registry[delegation_id].get("followup_answer")
                        node._registry[delegation_id]["status"] = "Working"
                    # Drain any queued budget warnings alongside the answer.
                    with node._pending_worker_msgs_lock:
                        msgs = node._pending_worker_msgs.pop(delegation_id, [])
                    budget_prefix = ("\n".join(msgs) + "\n\n") if msgs else ""
                    base = answer or "No answer received. Proceed with best judgment."
                    return budget_prefix + base

                worker.closure_tools["ReportEvals"] = ReportEvals  # not wrapped: never errors
                worker.closure_tools["FollowUp"] = node._wrap_closure(FollowUp, target)
                # On-demand handbook lookup, available to every worker.
                worker.closure_tools["ConsultHandbook"] = _consult_handbook
                try:
                    from .agent_prompts import IMPLEMENTER_REPORT_RETRY_PROMPT
                    from .backends.base import (
                        debug_enabled as _dbg,
                    )
                    from .backends.base import (
                        set_delegation_id as _set_did,
                    )
                    from .backends.base import (
                        set_transcript_sink as _set_sink,
                    )

                    # Bind the delegation id for this worker thread so the
                    # backend can inject F3DASM_DELEGATION_ID into the session
                    # env → get_evaluator() resolves without a mandatory cd
                    # into D### (audit Finding 2).
                    _set_did(delegation_id)

                    # DEBUG: stream this worker's full reasoning + tool-calls
                    # to debug/transcripts/{delegation_id}.jsonl (thread-local;
                    # this _run is the worker's own thread).
                    if _dbg() and node._current_notes_dir is not None:
                        _set_sink(str(
                            node._current_notes_dir.parent / "transcripts"
                            / f"{delegation_id}.jsonl"))

                    messages = [{"role": "user", "content": task_msg}]
                    text = worker.invoke(messages)

                    # Validate against THIS agent's declared report_sections
                    # (audit Finding 4 — report_sections is now the single
                    # source of truth, not a hardcoded list), so e.g. a missing
                    # ### Retrospective earns one corrective retry.
                    _req_sections = list(
                        getattr(_target_agent, "report_sections", None) or []
                    ) or None
                    diagnosis = _classify_response(text, _req_sections)
                    if diagnosis is not None:
                        retry_messages = messages + [
                            {"role": "ai", "content": text},
                            {
                                "role": "user",
                                "content": (
                                    f"{IMPLEMENTER_REPORT_RETRY_PROMPT}"
                                    f"\n\nDiagnosis: {diagnosis}"
                                ),
                            },
                        ]
                        text = worker.invoke(retry_messages)

                    # ── Unledgered-evals bounce (soft, ≤3×) ──────────────
                    # The worker reported evaluations but wrote no
                    # provenance-stamped rows to the canonical store — it
                    # bypassed get_evaluator(), so its numbers can't anchor a
                    # headline. Bounce it back to re-run through get_evaluator,
                    # in-place, instead of making the strategizer spend a whole
                    # new delegation re-ledgering. ONLY when a canonical source
                    # is registered (else get_evaluator can't work and the fix
                    # is upstream — the strategizer source nudge handles that).
                    # Soft: after 3 tries, accept anyway.
                    _bounce_store = (
                        node._current_notes_dir.parent.parent
                        / "experiment_data"
                        if node._current_notes_dir is not None else None
                    )
                    if node._canonical_source_registered():
                        from .agent_prompts import (
                            UNLEDGERED_EVALS_RETRY_PROMPT,
                        )
                        _bounces = 0
                        _stamped_before = _stamped_eval_count(
                            _bounce_store, delegation_id)
                        while (
                            _bounces < 3
                            and evals_box["count"] > 0
                            and _stamped_eval_count(
                                _bounce_store, delegation_id) == 0
                        ):
                            _bounces += 1
                            text = worker.invoke(messages + [
                                {"role": "ai", "content": text},
                                {"role": "user", "content": (
                                    UNLEDGERED_EVALS_RETRY_PROMPT
                                    + f"\n\n(notice {_bounces}/3)"
                                )},
                            ])
                        if _bounces > 0:
                            # Direct evidence of whether the bounce worked:
                            # stamped rows before vs after the re-runs.
                            _stamped_after = _stamped_eval_count(
                                _bounce_store, delegation_id)
                            node._record_intervention(
                                "UNLEDGERED_BOUNCE", target,
                                f"{delegation_id} reported "
                                f"{evals_box['count']} evals off-ledger; "
                                f"bounced to re-run via get_evaluator().",
                                bounces=_bounces,
                                stamped_before=_stamped_before,
                                stamped_after=_stamped_after,
                                corrected=bool(_stamped_after > 0),
                            )

                    # Direct evidence: log any raw-oracle nudge firings from
                    # this delegation (drained from the adapter's budget).
                    _onb = getattr(worker, "_oracle_nudge", None)
                    _evs = list(getattr(_onb, "events", []) or [])
                    for _ev in _evs:
                        node._record_intervention(
                            "RAW_ORACLE_NUDGE", target,
                            f"{delegation_id}: a {_ev.get('tool')} call "
                            "reached the oracle directly; nudged toward "
                            "get_evaluator().",
                            snippet=_ev.get("snip", ""),
                        )
                    if _onb is not None:
                        _onb.events = []

                    # Accumulate token usage from this worker invocation.
                    _usage = getattr(worker, "last_usage", {}) or {}
                    node._accumulate_usage(_usage)

                    # Scan worker report for MCP tool errors (infrastructure, not agent fault).
                    # MCP errors appear as lines containing "error" near tool names in the report.
                    import re as _re
                    _MCP_ERROR_PATTERNS = [
                        r"(mcp__\w+__\w+)[^\n]*?(429|rate.?limit|timeout|timed.?out|unavailable|connection.?error)",
                        r"(HTTP\s+(?:429|500|502|503))[^\n]*",
                        r"(rate.?limit(?:ed|ing)?)[^\n]*",
                    ]
                    for _pat in _MCP_ERROR_PATTERNS:
                        for _m in _re.finditer(_pat, text, _re.IGNORECASE):
                            node._record_tool_error(
                                target,
                                _m.group(1) if _m.lastindex and _m.lastindex >= 1 else "mcp_tool",
                                "MCP_REPORTED",
                                _m.group(0)[:200],
                            )
                            # Override fault to "system" for MCP infrastructure errors
                            # (already classified correctly by _record_tool_error)
                            break  # one log entry per pattern match per delegation

                    # Ledger row count preferred over self-reported
                    # ReportEvals: count rows per delegation_id in the
                    # canonical store (authoritative); fall back to
                    # the honour-system evals_box when the delegation
                    # wrote no rows (bypassed get_evaluator).
                    _notes = node._current_notes_dir
                    _store_dir = (
                        _notes.parent.parent / "experiment_data"
                        if _notes is not None else None
                    )
                    _evals = _resolve_delegation_evals(
                        _store_dir,
                        delegation_id,
                        evals_box["count"],
                    )
                    with node._registry_lock:
                        node._registry[delegation_id].update({
                            "status": "Done",
                            "result": text,
                            "evals": _evals,
                            "usage": _usage,
                        })
                    with node._notifications_lock:
                        node._notifications.append(
                            f"[Delegation {delegation_id} Done]"
                        )
                    # Write delegation record to graph-wide delegation log
                    if node._delegation_log is not None:
                        node._delegation_log.record(
                            id=delegation_id,
                            from_node=node._name,
                            to_node=target,
                            task=intent,
                            deliverable=text,
                            hypothesis_ids=h_ids,
                            started_at=started_at,
                            completed_at=datetime.now(
                                tz=timezone.utc
                            ).isoformat(timespec="seconds"),
                            status="DONE",
                            tokens_in=(
                                _usage.get("input_tokens", 0) or 0
                            ),
                            tokens_out=(
                                _usage.get("output_tokens", 0) or 0
                            ),
                            cost_usd=_usage.get("total_cost_usd"),
                            is_falsification_attempt=bool(
                                is_falsification_attempt
                            ),
                            evals=_evals,
                        )
                        if node._science_monitor is not None:
                            try:
                                node._science_monitor\
                                    .on_delegation_complete(
                                    delegation_id
                                )
                            except Exception:  # noqa: BLE001
                                pass

                    # Registration handoff: when a datagenerator delegation
                    # authors an oracle, it drops a registration.json manifest
                    # in its workspace. Point the canonical entrypoint at it so
                    # the next get_evaluator() (re-reads config each call)
                    # resolves it — no manual config edit. Best-effort: never
                    # fail a delegation over registration.
                    try:
                        _tgt = node._spec.nodes.get(target)
                        if (
                            _notes is not None
                            and getattr(_tgt, "role", None) == "datagenerator"
                        ):
                            _run_dir = _notes.parent.parent
                            _ws = (
                                _run_dir / "debug" / "delegations"
                                / delegation_id / "generators"
                            )
                            _manifest = _ws / "registration.json"
                            if _manifest.exists():
                                import json as _json

                                from .agent_runtime import (
                                    register_evaluator_entrypoint,
                                )
                                _m = _json.loads(_manifest.read_text())
                                _gf = _m["generator_file"]
                                _gf_path = Path(_gf)
                                if not _gf_path.is_absolute():
                                    # Resolve against the manifest dir, then
                                    # the run dir; take the first that exists.
                                    for _base in (_ws, _run_dir):
                                        _cand = (_base / _gf).resolve()
                                        if _cand.exists():
                                            _gf_path = _cand
                                            break
                                _ep = register_evaluator_entrypoint(
                                    _run_dir / "debug" / "run_config.json",
                                    _gf_path,
                                    _m["attr"],
                                    output_names=_m.get("output_names"),
                                )
                                with node._notifications_lock:
                                    node._notifications.append(
                                        f"[Evaluator registered: {_ep}]"
                                    )
                    except Exception:  # noqa: BLE001
                        pass

                    # Worker retrospective (every node has a 'job done'
                    # moment — see _record_retrospective).
                    _role = (
                        getattr(node._spec.nodes.get(target), "role", None)
                        or target
                    )
                    node._record_retrospective(_role, delegation_id, text)
                except Exception:  # noqa: BLE001
                    tb = traceback.format_exc()
                    _usage = getattr(worker, "last_usage", {}) or {}
                    node._accumulate_usage(_usage)
                    with node._registry_lock:
                        node._registry[delegation_id].update({
                            "status": "Errored",
                            "result": tb,
                            "evals": evals_box["count"],
                            "usage": _usage,
                        })
                    with node._notifications_lock:
                        node._notifications.append(
                            f"[Delegation {delegation_id} Errored]"
                        )
                    if node._delegation_log is not None:
                        node._delegation_log.record(
                            id=delegation_id,
                            from_node=node._name,
                            to_node=target,
                            task=intent,
                            # Keep the TAIL: the root exception is on
                            # the last line of a traceback.
                            deliverable="ERROR: " + tb[-2000:],
                            hypothesis_ids=h_ids,
                            started_at=started_at,
                            completed_at=datetime.now(
                                tz=timezone.utc
                            ).isoformat(timespec="seconds"),
                            status="FAILED",
                            tokens_in=(
                                _usage.get("input_tokens", 0) or 0
                            ),
                            tokens_out=(
                                _usage.get("output_tokens", 0) or 0
                            ),
                            cost_usd=_usage.get("total_cost_usd"),
                            is_falsification_attempt=bool(
                                is_falsification_attempt
                            ),
                        )

            t = threading.Thread(target=_run, daemon=True, name=delegation_id)
            with node._registry_lock:
                node._threads[delegation_id] = t  # A3: inside lock
            t.start()

            # Reset two-shot Done() gate so the next Done() warns again.
            node._done_warned = False

            if wait:
                # Synchronous mode: block until the delegation finishes.
                t.join()
                with node._registry_lock:
                    entry = dict(node._registry.get(delegation_id, {}))
                status = entry.get("status", "Errored")
                if status == "Done":
                    return f"Done\n\n{entry['result']}"
                return f"Errored:\n{entry.get('result', '(no details)')}"

            return (
                f"Delegation started. ID: {delegation_id!r}. "
                f"Use GetStatus('{delegation_id}') to poll for completion."
            )

        Delegate.__doc__ = _delegate_doc

        def GetStatus(delegation_id: str) -> str:
            """Poll a background delegation; also delivers push notifications.

            Returns one of:
              'Working (running for Xs, polled N times)' — still running
              'Done\\n\\n<full report>'                  — completed
              'Errored:\\n<traceback>'                   — failed

            Tip: use Delegate(wait=True) when you do not need parallelism —
            it blocks until the result is ready without any polling.
            """
            prefix = node._drain_notifications()

            # Drain any budget warnings queued for this delegation.
            with node._pending_worker_msgs_lock:
                worker_msgs = node._pending_worker_msgs.pop(delegation_id, [])
            if worker_msgs:
                prefix += "\n".join(worker_msgs) + "\n\n"

            with node._registry_lock:
                entry = node._registry.get(delegation_id)
                if entry is None:
                    known = list(node._registry)
                    return (
                        prefix +
                        f"ERROR: unknown delegation ID {delegation_id!r}. "
                        f"Known IDs: {known}"
                    )
                status = entry["status"]
                if status in ("Working", "FollowUp"):
                    # Increment poll count and record timing.
                    entry["getstatus_count"] = entry.get("getstatus_count", 0) + 1
                    poll_count = entry["getstatus_count"]
                    last_poll = entry.get("last_getstatus_time")
                    now_mono = time.monotonic()
                    entry["last_getstatus_time"] = now_mono
                    elapsed = int(now_mono - entry["start_time"])

            # Status token FIRST (contract: callers dispatch on the
            # leading word); queued notifications follow the report.
            _tail = (
                ("\n\n" + prefix.rstrip()) if prefix.strip() else ""
            )
            if status == "Done":
                return f"Done\n\n{entry['result']}" + _tail
            if status not in ("Working", "FollowUp"):
                return f"Errored:\n{entry['result']}" + _tail

            # --- Still working: build informative response ---
            hints: list[str] = []

            # Rate warning: polled too recently.
            if last_poll is not None and (now_mono - last_poll) < 30:
                hints.append(
                    f"NOTE: you polled {delegation_id} only "
                    f"{now_mono - last_poll:.0f}s ago. "
                    "The worker runs in a background thread — polling faster "
                    "does not make it finish sooner. Do other work in the "
                    "meantime."
                )

            # Poll-count escalation.
            if poll_count >= 30:
                hints.append(
                    f"WARNING: polled {poll_count} times ({elapsed}s elapsed). "
                    "This delegation is taking very long. Strongly consider "
                    "proceeding without this result or using Delegate(wait=True) "
                    "for future sequential tasks."
                )
            elif poll_count >= 15:
                hints.append(
                    f"This delegation has been polled {poll_count} times "
                    f"({elapsed}s elapsed). Consider working on other tasks "
                    "rather than polling in a tight loop."
                )
            elif poll_count >= 5:
                hints.append(
                    f"Still running after {poll_count} polls ({elapsed}s). "
                    "Work on other tasks and poll less frequently."
                )

            # Budget broadcast: check if a new 10%-overbudget threshold is reached.
            budget = node._budget_seconds
            run_start = node._run_start
            if budget is not None and run_start is not None:
                elapsed_wall = time.time() - run_start
                pct = (elapsed_wall / budget) * 100
                # Thresholds: 80, 90, 100, 110, 120, …
                threshold = int(pct // 10) * 10
                if threshold >= 80:
                    with node._pending_worker_msgs_lock:
                        already_sent = node._budget_notified_pcts
                        if threshold not in already_sent:
                            already_sent.add(threshold)
                            _over = (
                                _BACKSTOP_ENABLED
                                and pct >= RUN_BACKSTOP_MULTIPLE * 100
                            )
                            msg = (
                                f"BACKSTOP IMMINENT: {pct:.0f}% of time "
                                "budget — past the "
                                f"{int(RUN_BACKSTOP_MULTIPLE)}x cost "
                                "backstop. Stop polling and call Done() "
                                "NOW with a partial report."
                            ) if _over else (
                                f"BUDGET: {pct:.0f}% of time budget consumed. "
                                "Wrap up your current work and return a partial "
                                "report as soon as possible."
                            )
                            # Queue for all currently Working delegations.
                            with node._registry_lock:
                                active = [
                                    did for did, e in node._registry.items()
                                    if e["status"] in ("Working", "FollowUp")
                                    and did != delegation_id
                                ]
                            for did in active:
                                node._pending_worker_msgs.setdefault(did, []).append(msg)
                            # Include in this response too.
                            hints.append(msg)

            # Status token FIRST (documented contract: callers may
            # dispatch on the leading word); hints and queued
            # notifications follow.
            hint_str = ("\n\n" + "\n".join(hints)) if hints else ""
            tail = ("\n\n" + prefix.rstrip()) if prefix.strip() else ""
            return (
                f"Working (running for {elapsed}s, "
                f"polled {poll_count} times)" + hint_str + tail
            )

        def Reply(delegation_id: str, answer: str) -> str:
            """Answer a worker's FollowUp question and unblock it.

            Call this after GetStatus returns 'FollowUp: <question>'.
            The answer is injected into the worker's context and it resumes.
            """
            prefix = node._drain_notifications()
            with node._registry_lock:
                entry = node._registry.get(delegation_id)
                if entry is None:
                    return prefix + f"ERROR: unknown delegation {delegation_id!r}."
                if entry.get("status") != "FollowUp":
                    return (
                        prefix +
                        f"ERROR: delegation {delegation_id!r} is not awaiting a "
                        f"FollowUp (status: {entry.get('status')!r})."
                    )
                entry["followup_answer"] = answer
                evt = entry["followup_event"]
            evt.set()
            return prefix + f"Reply sent to {delegation_id}. Worker resuming."

        def Done(summary: str) -> str:
            """Signal end of run with a summary of findings (two-shot).

            First call: issues a WARNING and lists any open delegations or
            unmet conditions; does NOT close.
            Second call: closes the run.

            Refused if any delegation is still Working — call GetStatus()
            on all pending delegations first, or use Delegate(wait=True)
            for sequential execution.
            """
            prefix = node._drain_notifications()
            with node._registry_lock:
                pending = [
                    did for did, e in node._registry.items()
                    if e["status"] == "Working"
                ]
            if pending:
                return (
                    prefix +
                    f"ERROR: {len(pending)} delegation(s) still running: "
                    f"{pending}. "
                    "Wait for all delegations to complete (Done or Errored) "
                    "before calling Done(). Use Delegate(wait=True) next time "
                    "to avoid this."
                )
            # Exit-interview capture (final stage): the conclusion is already
            # accepted + recorded; this Done() carries ONLY the retrospective.
            # Capture it, then actually close. The strategizer hears about the
            # interview ONLY after the critic accepted — never during its
            # working turns, so its orchestration context stays clean.
            if node._awaiting_retro:
                node._awaiting_retro = False
                node._record_retrospective("strategizer", "DONE", summary)
                node._done_warned = False
                route["kind"] = "done"
                route["summary"] = node._final_summary
                return prefix + "Run complete."
            # Two-shot gate: first call warns, second call closes.
            if not node._done_warned:
                node._done_warned = True
                open_hypotheses: list[str] = []
                if node._ledger is not None:
                    open_hypotheses = [
                        h["id"]
                        for h in node._ledger.list_all()
                        if h.get("current_status") == "OPEN"
                    ]
                warn_parts = [
                    "WARNING: first Done() call — confirm you are ready to close.",
                    "Call Done() again to confirm and end the run.",
                ]
                if open_hypotheses:
                    warn_parts.append(
                        f"Open hypotheses still in OPEN state: "
                        f"{open_hypotheses}. "
                        "Consider updating their status before closing."
                    )
                # WARNING goes first; then any pending notifications.
                return "  ".join(warn_parts) + (("\n\n" + prefix.rstrip()) if prefix.strip() else "")
            # Second call — run critic gate if critic is in the graph.
            if node._find_critic_name() is not None:
                # Synchronous critic gate
                _notes_dir = node._current_notes_dir
                _study_dir = node._study_dir
                _debug_dir = (
                    _notes_dir.parent
                    if _notes_dir is not None else None
                )
                notes_path = str(_notes_dir or "")
                task_msg = (
                    "<mode>GATE</mode>\n\n"
                    "Final gate check before run closes. PASS to accept the "
                    "conclusion; REVISE/REJECT only for a CRITICAL or MAJOR "
                    "objection.\n\n"
                    "<paths>\n"
                    f"study_dir             = {_study_dir}\n"
                    f"debug_dir             = {_debug_dir}\n"
                    "delegation_log        = "
                    f"{_debug_dir}/delegation_log.jsonl\n"
                    f"diagnostics           = "
                    f"{_debug_dir}/diagnostics.jsonl\n"
                    f"strategizer_notes     = {notes_path}\n"
                    "delegations_workspace = "
                    f"{_debug_dir}/delegations/\n"
                    f"deliverable           = {_study_dir}/replicate.py "
                    "(solution.md is written by the runtime AFTER this gate, "
                    "from the accepted summary — do NOT flag it as missing)\n"
                    "</paths>\n\n"
                    # FULL conclusion — never truncate what the adversarial gate
                    # must validate (a head-excerpt would let an over-claim in
                    # the body pass unseen). A Done() summary is small; context
                    # is not a concern, and final_summary.md is also available.
                    f"Proposed conclusion:\n{summary}"
                )
                import json as _json
                ledger_dump = "(no hypotheses)"
                if node._ledger is not None:
                    ledger_dump = _json.dumps(
                        {
                            h["id"]: node._ledger.get(h["id"])
                            for h in node._ledger.list_all()
                        },
                        indent=2,
                    )
                attempts: list[str] = []
                if node._delegation_log is not None:
                    attempts = [
                        f"{r['id']}: "
                        f"hypotheses={r.get('hypothesis_ids')} "
                        f"is_falsification_attempt="
                        f"{r.get('is_falsification_attempt', False)}"
                        for r in node._delegation_log.query_all()
                    ]
                # Budget-aware framing: tell the critic what was actually
                # spent so it judges the BEST HONEST conclusion reachable
                # within budget, rather than demanding falsification work the
                # budget no longer allows (which strands the close).
                _spent = 0
                if node._delegation_log is not None:
                    _spent = sum(
                        (r.get("evals") or 0)
                        for r in node._delegation_log.query_all()
                    )
                _bud = getattr(node, "_eval_budget", None)
                _exhausted = _bud is not None and _spent >= _bud
                task_msg += (
                    "\n\n<hypothesis_ledger>\n" + ledger_dump
                    + "\n</hypothesis_ledger>\n\n"
                    "<delegation_flags>\n"
                    + "\n".join(attempts)
                    + "\n</delegation_flags>\n\n"
                    "<budget>\n"
                    f"Evaluation budget: {_bud if _bud else 'unspecified'}; "
                    f"ledgered evaluations spent: {_spent}"
                    + (" (EXHAUSTED)." if _exhausted else ".") + "\n"
                    "If the budget is exhausted, judge the BEST HONEST "
                    "conclusion reachable within the evals actually spent: an "
                    "honest INCONCLUSIVE/negative result whose falsification "
                    "attempts were adequate FOR THE REMAINING BUDGET can PASS. "
                    "Do NOT REVISE solely to demand evaluations the budget no "
                    "longer allows — note them as future work instead.\n"
                    "</budget>\n\n"
                    "For each hypothesis, judge whether its stated "
                    "falsification_criterion was actually tested by "
                    "a delegation flagged is_falsification_attempt "
                    "— adequacy of the test (given the budget), not mere "
                    "presence of the flag."
                )
                critique_text = node._invoke_critic(task_msg)
                verdict = _parse_verdict(critique_text)

                if verdict == "PASS":
                    # Conclusion accepted + recorded. Now — and only now —
                    # ask the exit interview as a separate turn.
                    node._done_warned = False
                    node._revise_count = 0
                    node._awaiting_retro = True
                    node._final_summary = summary
                    return prefix + _EXIT_INTERVIEW

                # Non-PASS: reset the two-shot and count the revision. After
                # 3 unsatisfiable verdicts, close GRACEFULLY UNGATED rather
                # than looping to recursion-limit — record the objections so
                # the run terminates honestly. (Bounded escape, N=3.) This is
                # a SILENT internal failsafe — it is deliberately NOT disclosed
                # to the strategizer (advertising "call Done() 3x to close"
                # teaches it to exhaust the critic instead of earning a PASS).
                node._done_warned = False
                node._revise_count = getattr(node, "_revise_count", 0) + 1
                if node._revise_count >= 3:
                    banner = (
                        "## ⚠ UNGATED RUN\n\n"
                        "This run is NOT validated: the conclusion did not "
                        f"earn a critic PASS after {node._revise_count} "
                        "revision attempts. Closing honestly with the critic's "
                        "outstanding objections recorded below rather than "
                        "looping.\n\n### Last critic findings\n"
                        + critique_text.strip() + "\n\n---\n\n"
                    )
                    node._revise_count = 0
                    route["kind"] = "done"
                    route["summary"] = banner + summary
                    return prefix + (
                        "Run complete (UNGATED — closed after "
                        "3 unsatisfiable critic revisions; objections "
                        "recorded in solution.md)."
                    )
                return (
                    prefix +
                    f"Critic verdict: {verdict}. Address the findings below "
                    "and revise your conclusion, then call Done() again — a "
                    "run closes only on a critic PASS.\n\n" + critique_text
                )

            # No critic — no review stage, so no exit interview; close
            # directly (the interview is "your work was reviewed, now a
            # question", which only applies when a critic gate ran).
            node._done_warned = False
            route["kind"] = "done"
            route["summary"] = summary
            return prefix + "Run complete."

        def FollowUp(question: str) -> str:
            """Ask your delegating party one clarifying question before proceeding.

            Routes to whoever sent you this task: the human operator if you are
            the entry node, or the agent that delegated to you if you are a worker.
            One FollowUp per delegation.  The answer is injected directly into
            your context.  If no answer is available, proceed with best judgment.
            """
            if node._ask_count >= node._max_ask:
                return (
                    f"FollowUp limit reached ({node._max_ask} per run). "
                    "Proceed autonomously with the information you have."
                )
            node._ask_count += 1
            if interactive:
                print(f"\n[Node {node._name}] {question}\nAnswer: ", end="", flush=True)
                return input()
            # Non-interactive: auto-respond so run proceeds autonomously.
            return (
                "No operator is present. Proceed autonomously "
                "using only information available in the task message "
                "and files in the study directory."
            )

        def WriteNote(path: str, body: str) -> str:
            """Write a Markdown (.md) note to strategizer_notes/."""
            prefix = node._drain_notifications()
            notes_dir = node._current_notes_dir
            if notes_dir is None:
                return "ERROR: notes_dir not set (run_dir missing from state)."
            bare = Path(path).name
            if not bare.endswith(".md"):
                bare = bare + ".md"
            target = Path(notes_dir) / bare
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
            return prefix + f"Written: {target}"

        def ReadNote(path: str) -> str:
            """Read a file from the study directory."""
            prefix = node._drain_notifications()
            if study_dir is None:
                return "ERROR: study_dir not set."
            target = Path(study_dir) / path
            if not target.exists():
                return prefix + f"NOT FOUND: {target}"
            if target.is_dir():
                return prefix + (
                    f"ERROR: {target} is a directory."
                    " Pass a file path."
                )
            return prefix + target.read_text(encoding="utf-8")

        # RecallHistory: demand-driven episodic memory via delegation log.
        if node._delegation_log is not None:
            _dlog = node._delegation_log

            def RecallHistory(n: int = 5) -> str:
                """Return the last n delegations received by this node as (task, deliverable) pairs.
                Call at the start of a delegation to recall prior work. Returns oldest-first."""
                records = _dlog.query_received(node._name, n)
                if not records:
                    return "No prior delegations found."
                parts = []
                for i, r in enumerate(records, 1):
                    parts.append(
                        f"== Prior delegation {i} ==\n"
                        f"Task: {r['task']}\n\n"
                        f"Deliverable:\n{r['deliverable']}"
                    )
                return "\n\n---\n\n".join(parts)

        # ------------------------------------------------------------------
        # RecallStore / QueryStore: canonical ledger read tools.
        # notes_dir = run_dir/debug/strategizer_notes
        #   → parent       = run_dir/debug/
        #   → parent.parent = run_dir
        #   → store_dir    = run_dir/experiment_data
        # RunStateSummary.from_store(store_dir) reads
        #   store_dir/experiment_data/output.csv  ✓
        # ------------------------------------------------------------------

        def _derive_store_dir() -> Path | None:
            nd = node._current_notes_dir
            if nd is None:
                return None
            return nd.parent.parent / "experiment_data"

        def RecallStore() -> str:
            """Summary of the run's canonical evaluation ledger: rows per
            delegation/source, output ranges. Call before deciding the next
            delegation."""
            from .instrumented import RunStateSummary
            sd = _derive_store_dir()
            if sd is None:
                return (
                    "Canonical store is empty — no instrumented "
                    "evaluations recorded yet."
                )
            summary = RunStateSummary.from_store(sd)
            if summary is None:
                return (
                    "Canonical store is empty — no instrumented "
                    "evaluations recorded yet."
                )
            return summary.format()

        def QueryStore(
            delegation_ids: str | list | None = None,
            source: str | None = None,
            n_best: int | None = None,
            output_name: str | None = None,
        ) -> str:
            """Filtered view of the evaluation ledger (e.g. rows from D001+D003
            only). Use to ground claims or to select training subsets; cite row
            values from here as evidence."""
            import json as _json

            from ..errors import EmptyFileError, ReachMaximumTriesError
            from ..experimentdata import ExperimentData
            from .instrumented import _PROVENANCE_COLS

            sd = _derive_store_dir()
            if sd is None:
                return (
                    "Canonical store is empty — no instrumented "
                    "evaluations recorded yet."
                )
            try:
                data = ExperimentData.from_file(project_dir=sd)
                df_in, df_out = data.to_pandas()
            except (FileNotFoundError, EmptyFileError,
                    ReachMaximumTriesError):
                return (
                    "Canonical store is empty — no instrumented "
                    "evaluations recorded yet."
                )
            if df_out.empty:
                return (
                    "Canonical store is empty — no instrumented "
                    "evaluations recorded yet."
                )

            # Decode delegation_ids: JSON / comma / bare / list
            d_ids: list[str] | None = None
            if delegation_ids is not None:
                if isinstance(delegation_ids, list):
                    d_ids = [str(x) for x in delegation_ids]
                elif isinstance(delegation_ids, str):
                    raw = delegation_ids.strip()
                    if raw.startswith("["):
                        try:
                            decoded = _json.loads(raw)
                            d_ids = (
                                [str(h) for h in decoded]
                                if isinstance(decoded, list)
                                else [raw]
                            )
                        except _json.JSONDecodeError:
                            d_ids = [raw]
                    elif "," in raw:
                        d_ids = [
                            p.strip() for p in raw.split(",")
                            if p.strip()
                        ]
                    else:
                        d_ids = [raw]

            # Apply filters (read-only: build a boolean mask)
            import pandas as _pd
            mask = _pd.Series([True] * len(df_out), index=df_out.index)
            if d_ids is not None and "_delegation_id" in df_out.columns:
                mask &= df_out["_delegation_id"].isin(d_ids)
            if source is not None and "source" in df_out.columns:
                mask &= df_out["source"] == source

            filtered = df_out[mask]
            filtered_in = df_in[mask] if df_in is not None else None

            if filtered.empty:
                return "No rows match the given filters."

            # n_best: return the n rows with smallest output_name value.
            # MCP string-in tools may pass n_best as a string ("5") — coerce
            # to int (pandas nsmallest does `if n <= 0`, which TypeErrors on
            # a str). Same string-arg-decoding discipline as delegation_ids.
            if n_best is not None:
                try:
                    n_best = int(n_best)
                except (TypeError, ValueError):
                    return (
                        f"ERROR: n_best must be an integer, got "
                        f"{n_best!r}."
                    )
            if n_best is not None and output_name is not None:
                if output_name not in filtered.columns:
                    return (
                        f"ERROR: output column {output_name!r} not found. "
                        f"Available: {list(filtered.columns)}"
                    )
                import pandas as _pd2
                col_num = _pd2.to_numeric(
                    filtered[output_name], errors="coerce"
                )
                best_idx = col_num.nsmallest(n_best).index
                best_rows = filtered.loc[best_idx]
                # Include input columns + output_name + _delegation_id
                show_cols = []
                if filtered_in is not None:
                    input_cols = [
                        c for c in filtered_in.columns
                        if c not in _PROVENANCE_COLS
                    ]
                    show_cols.extend(input_cols)
                show_cols.append(output_name)
                if "_delegation_id" in best_rows.columns:
                    show_cols.append("_delegation_id")
                show_cols = [c for c in show_cols if c in best_rows.columns]

                if filtered_in is not None:
                    best_in = filtered_in.loc[best_idx, [
                        c for c in filtered_in.columns
                        if c in show_cols
                    ]]
                    combined = _pd2.concat(
                        [best_in, best_rows[
                            [c for c in show_cols
                             if c not in best_in.columns]
                        ]], axis=1
                    )
                else:
                    combined = best_rows[
                        [c for c in show_cols if c in best_rows.columns]
                    ]
                return combined.to_string(index=False)

            # Default: return count + first 20 rows
            n_shown = min(20, len(filtered))
            subset = filtered.iloc[:n_shown]
            return (
                f"{len(filtered)} rows match. "
                f"Showing first {n_shown}:\n"
                + subset.to_string(index=False)
            )

        # Topology-injected tools go to every orchestrating node.
        closures: dict = {
            "Delegate": Delegate,
            "GetStatus": GetStatus,
            "Reply": Reply,
            "FollowUp": FollowUp,
            "RecallStore": RecallStore,
            "QueryStore": QueryStore,
        }

        if node._delegation_log is not None:
            closures["RecallHistory"] = RecallHistory

        # Agent-declared closure tools: inject only what the subclass opted in to.
        # Done / WriteNote / ReadNote are declared in StrategizerAgent.tools;
        # an ImplementerAgent or DebuggerAgent that gains outgoing edges does not
        # declare them and therefore does not receive them.
        _agent_tools: frozenset = frozenset()
        if self._spec is not None:
            _ag = self._spec.nodes.get(self._name)
            if _ag is not None:
                _agent_tools = _ag.tools

        def WriteDeliverable(filename: str, content: str) -> str:
            """Write a final deliverable file to runs/<timestamp>/ (alongside solution.md).

            Use to produce replicate.py or other top-level artifacts.
            filename must end in .py or .md. Content is written verbatim.
            """
            prefix = node._drain_notifications()
            if node._study_dir is None:
                return "ERROR: study_dir not available."

            allowed_exts = {".py", ".md"}
            from pathlib import Path as _Path
            p = _Path(filename)
            if p.suffix not in allowed_exts:
                return (
                    f"ERROR: filename must end in {allowed_exts}, got {filename!r}."
                )
            if "/" in filename or "\\" in filename:
                return "ERROR: filename must be a bare name (no path separators)."

            # Write directly to study_dir/ — the user-visible output location.
            target = _Path(node._study_dir) / p.name
            target.write_text(content, encoding="utf-8")
            return prefix + f"Written: {target}"

        if "Done" in _agent_tools:
            closures["Done"] = Done
        if "WriteNote" in _agent_tools:
            closures["WriteNote"] = WriteNote
        if "ReadNote" in _agent_tools:
            closures["ReadNote"] = ReadNote
        if "WriteDeliverable" in _agent_tools:
            closures["WriteDeliverable"] = WriteDeliverable
        # On-demand handbook lookup, available to the strategizer too.
        closures["ConsultHandbook"] = _consult_handbook

        # Hypothesis closures: always built but functionally inert without a
        # ledger (notes_dir only provided to the entry node).
        closures.update(self._build_hypothesis_closures())

        # AskForFeedback is only injected when a critic node is
        # connected AND this is the entry node (only the entry node
        # gates Done).
        critic_name: str | None = self._find_critic_name()
        spec = self._spec

        if critic_name is not None:
            _critic_name = critic_name
            _node = node
            _critic_desc = spec.nodes[critic_name].description if spec else ""

            def AskForFeedback(hypothesis_ids: list | None = None) -> str:
                """Placeholder — __doc__ overridden below."""
                # Resolve hypothesis IDs
                h_ids: list[str] = []
                if hypothesis_ids is not None:
                    h_ids = list(hypothesis_ids)
                elif _node._ledger is not None:
                    h_ids = [h["id"] for h in _node._ledger.list_all()]

                started_at = datetime.now(tz=timezone.utc).isoformat(
                    timespec="seconds"
                )
                task_msg = _node._build_feedback_task_msg(h_ids)
                text = _node._invoke_critic(task_msg)

                # Log to delegation log
                if _node._delegation_log is not None:
                    _node._delegation_log.record(
                        id=f"FB{datetime.now(tz=timezone.utc).strftime('%H%M%S')}",
                        from_node=_node._name,
                        to_node=_critic_name,
                        task="AskForFeedback (synchronous audit)",
                        deliverable=text,
                        hypothesis_ids=h_ids,
                        started_at=started_at,
                        completed_at=datetime.now(
                            tz=timezone.utc
                        ).isoformat(timespec="seconds"),
                        status="FEEDBACK",
                        tokens_in=0,
                        tokens_out=0,
                        cost_usd=None,
                    )

                return text

            AskForFeedback.__doc__ = (
                f"Synchronous find-only audit by: {_critic_desc} "
                "PASS is not a valid verdict — only REVISE or REJECT. "
                "hypothesis_ids: H-ids to focus on; None = all hypotheses auto-injected. "
                "Use Done() for the final gate check."
            )
            closures["AskForFeedback"] = AskForFeedback

        # Wrap every closure so ERROR returns and exceptions are counted.
        _node_name = self._name
        return {k: self._wrap_closure(v, _node_name) for k, v in closures.items()}

    def _record_tool_error(
        self,
        node_name: str,
        tool_name: str,
        error_type: str,
        message: str,
        tb: str | None = None,
    ) -> None:
        """Increment error counter and append to diagnostics.jsonl (thread-safe)."""
        import json as _json

        # Classify fault: system (rate limit / network / API) vs agent (bad args / wrong usage)
        _SYSTEM_EXC_TYPES = {
            "ConnectionError", "Timeout", "ReadTimeout", "ConnectTimeout",
            "HTTPError", "ChunkedEncodingError", "ProxyError", "SSLError",
        }
        _SYSTEM_MSG_PATTERNS = [
            "429", "rate limit", "rate-limit", "timeout", "timed out",
            "connection", "503", "502", "500", "network", "unavailable",
            "temporary", "retry",
        ]
        msg_lower = (message or "").lower()
        if error_type in _SYSTEM_EXC_TYPES or any(p in msg_lower for p in _SYSTEM_MSG_PATTERNS):
            fault = "system"
        else:
            fault = "agent"

        with self._registry_lock:
            self._error_counts[node_name] = self._error_counts.get(node_name, 0) + 1
        notes = self._current_notes_dir
        if notes is None:
            return
        # _current_notes_dir is debug/strategizer_notes/; parent is debug/
        debug_dir = Path(notes).parent
        record: dict = {
            "ts": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "node": node_name,
            "tool": tool_name,
            "error_type": error_type,
            "fault": fault,
            "message": message,
        }
        if tb:
            record["traceback"] = tb
        try:
            with (debug_dir / "diagnostics.jsonl").open("a", encoding="utf-8") as f:
                f.write(_json.dumps(record) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def _record_retrospective(
        self, role: str, source_id: str, report_text: str
    ) -> None:
        """Capture a node's end-of-life ### Retrospective.

        Every node has a 'my job is done' moment — workers at delegation
        completion, the strategizer (and any future orchestrator) at Done().
        Parse the section, persist to retrospectives.jsonl, and surface a
        diagnostic + strategizer notification the instant it flags
        contradictory system instructions (the cheapest, highest-value
        failure mode to catch). Best-effort; never raises.
        """
        try:
            retro = _extract_report_section(report_text or "", "Retrospective")
            if not retro:
                return
            notes = self._current_notes_dir
            if notes is None:
                return
            import json as _json
            import re as _re
            debug_dir = Path(notes).parent
            flagged = bool(_re.search(r"CONSISTENCY:\s*flagged", retro, _re.I))
            now = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
            rec = {
                "ts": now, "source_id": source_id, "role": role,
                "flagged": flagged, "text": retro[:2000],
            }
            with (debug_dir / "retrospectives.jsonl").open(
                    "a", encoding="utf-8") as f:
                f.write(_json.dumps(rec) + "\n")
            if flagged:
                drec = {
                    "ts": now, "node": role, "tool": "Retrospective",
                    "error_type": "CONSISTENCY_FLAG", "fault": "system",
                    "message": retro[:300],
                }
                with (debug_dir / "diagnostics.jsonl").open(
                        "a", encoding="utf-8") as f:
                    f.write(_json.dumps(drec) + "\n")
                with self._notifications_lock:
                    self._notifications.append(
                        f"[CONSISTENCY FLAG — {role} ({source_id}) reported "
                        "contradictory system instructions; see "
                        "debug/retrospectives.jsonl]"
                    )
        except Exception:  # noqa: BLE001
            pass

    def _record_intervention(
        self, kind: str, target: str, message: str, **extra
    ) -> None:
        """Log a scientific-correction event (a nudge/bounce firing) to
        diagnostics.jsonl — direct evidence the self-healing layer acted,
        and, when ``extra`` carries before/after state, whether it worked.

        Neutral classification: fault='nudge' — a nudge is a correction, not
        an agent error, so this must NOT bump the error/escalation counters
        (unlike _record_tool_error). Best-effort; never raises.
        """
        notes = self._current_notes_dir
        if notes is None:
            return
        import json as _json
        debug_dir = Path(notes).parent
        record: dict = {
            "ts": datetime.now(tz=timezone.utc).isoformat(
                timespec="seconds"),
            "node": target,
            "tool": kind,
            "error_type": kind,
            "fault": "nudge",
            "message": message,
        }
        record.update(extra)
        try:
            with (debug_dir / "diagnostics.jsonl").open(
                    "a", encoding="utf-8") as f:
                f.write(_json.dumps(record) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def _record_science_drift(self, payload: dict) -> None:
        """Append a SCIENCE_DRIFT record to diagnostics.jsonl."""
        import json as _json
        notes = self._current_notes_dir
        if notes is None:
            return
        record = {
            "ts": datetime.now(tz=timezone.utc).isoformat(
                timespec="seconds"),
            "node": self._name,
            "error_type": "SCIENCE_DRIFT",
            **payload,
        }
        try:
            path = Path(notes).parent / "diagnostics.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(record) + "\n")
        except Exception:  # noqa: BLE001
            pass

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

    def _accumulate_usage(self, usage: dict) -> None:
        """Thread-safe accumulation of token counts from adapter.last_usage."""
        with self._registry_lock:
            self._token_totals["input_tokens"] += usage.get("input_tokens", 0) or 0
            self._token_totals["output_tokens"] += usage.get("output_tokens", 0) or 0
            self._token_totals["cache_read_input_tokens"] += usage.get("cache_read_input_tokens", 0) or 0
            self._token_totals["cache_creation_input_tokens"] += usage.get("cache_creation_input_tokens", 0) or 0
            cost = usage.get("total_cost_usd")
            if cost is not None:
                self._token_totals["total_cost_usd"] += cost

    def _build_hypothesis_closures(self) -> dict:
        """Build HypothesisPropose/Update/List/Get closures."""
        node = self

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

        def HypothesisList() -> str:
            """List all hypotheses with id, status, belief, statement."""
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
        }

    def _missing_deliverables(self, state: AgenticState) -> list[str]:
        """Return required deliverable paths not present at study_dir yet.

        replicate.py is always required — it must be written via
        WriteDeliverable() before Done() is accepted.  Additional paths
        can be declared in state['required_deliverables'].
        """
        study_dir = Path(state.get("study_dir", "."))
        # WriteDeliverable writes BARE names to study_dir/ (it rejects path
        # separators), and solution.md lands at study_dir/ too. Normalise any
        # configured path to its basename so a stray 'workspace/…' prefix in a
        # study config can't spuriously flag a present deliverable as missing.
        required = ["replicate.py"] + list(
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
                        f"{int(RUN_BACKSTOP_MULTIPLE)}x budget."
                    ),
                })
            elif pct >= 0.95:
                budget_warnings.append({
                    "role": "user",
                    "content": (
                        f"Warning: time budget at {pct*100:.0f}% "
                        f"({elapsed:.0f}s / {budget:.0f}s). "
                        "Begin wrapping up — call Done() soon."
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

        # ── Run-level cost backstop ───────────────────────────────────────────
        # Time budget itself is SOFT (warnings only). This is a separate
        # outermost guard: if the run blows past RUN_BACKSTOP_MULTIPLE x the
        # budget it is aborted to bound runaway cost — a backstop, not the
        # budget being a hard constraint. Checked between turns; a turn stuck
        # polling is nudged toward Done() via GetStatus (see budget broadcast).
        if _BACKSTOP_ENABLED and budget is not None and start is not None:
            _elapsed_now = time.time() - start
            if _elapsed_now > budget * RUN_BACKSTOP_MULTIPLE:
                # Collect any abandoned delegations for reporting
                with self._registry_lock:
                    _abandoned = [
                        d for d, e in self._registry.items()
                        if e["status"] in ("Working", "FollowUp")
                    ]
                    _total_new_budget = len(self._registry)
                    _evals_new_budget = sum(
                        e["evals"] for e in self._registry.values()
                    )
                # Best-effort: grab last AI text from state messages
                _prior_text = ""
                for _m in reversed(state["messages"]):
                    if isinstance(_m, AIMessage):
                        _prior_text = str(_m.content)
                        break
                # Log the backstop event to diagnostics.jsonl so it
                # is traceable, but do NOT stamp the banner into
                # solution.md — the deliverable carries the conclusion.
                self._record_science_drift({
                    "error_type": "RUN_BACKSTOP",
                    "elapsed": _elapsed_now,
                    "budget": budget,
                    "multiple": RUN_BACKSTOP_MULTIPLE,
                    "abandoned": _abandoned,
                })
                return Command(
                    goto=END,
                    update={
                        "messages": [],
                        "done": True,
                        "last_report": (
                            _prior_text or "(no prior report)"
                        ),
                        "total_delegations": (
                            state["total_delegations"]
                            + _total_new_budget
                        ),
                        "evals_used": (
                            state.get("evals_used", 0)
                            + _evals_new_budget
                        ),
                        "token_totals": dict(self._token_totals),
                        "error_counts": dict(self._error_counts),
                    },
                )

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

        messages = (
            _to_adapter_messages(state["messages"])
            + budget_warnings + registration_nudge
        )
        # DEBUG: stream this strategizer turn's full reasoning + tool-calls
        # to debug/transcripts/strategizer/turn_NNN.jsonl.
        from .backends.base import (
            debug_enabled as _dbg,
        )
        from .backends.base import (
            set_transcript_sink as _set_sink,
        )
        self._turn_count = getattr(self, "_turn_count", 0) + 1
        if _dbg() and self._current_notes_dir is not None:
            _set_sink(str(
                self._current_notes_dir.parent / "transcripts"
                / "strategizer" / f"turn_{self._turn_count:03d}.jsonl"))
        text = self.adapter.invoke(messages)
        # Accumulate strategizer's own token usage.
        self._accumulate_usage(getattr(self.adapter, "last_usage", {}) or {})
        ai_msg = AIMessage(content=text)

        route = self._route
        accepted = route.get("kind") == "done"
        missing = self._missing_deliverables(state)

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


class WorkerNode(AgentNode):
    """Generic worker node: executes tasks, writes Reports, returns to caller.

    Used for any non-orchestrator agent (Implementer, Debugger,
    LiteratureReviewer, etc.).  The node name in the graph is what
    distinguishes agents — not the node class.
    """

    def __init__(
        self,
        adapter: Any,
        study_dir: Any = None,
        workspace_dir: Any = None,
        delegation_log: DelegationLog | None = None,
        name: str = "worker",
    ) -> None:
        super().__init__(adapter)
        self._name = name
        self._delegation_log = delegation_log
        self._evals_reported: dict = {}
        self._workspace_dir = Path(workspace_dir) if workspace_dir else None
        self._setup_sandboxed_write()
        self.adapter.closure_tools.update(self._build_eval_closures())
        if delegation_log is not None:
            self.adapter.closure_tools["RecallHistory"] = self._make_recall_history()

    def _make_recall_history(self) -> Any:
        """Build the RecallHistory closure for this worker node."""
        node = self

        def RecallHistory(n: int = 5) -> str:
            """Return the last n delegations received by this node as (task, deliverable) pairs.
            Call at the start of a delegation to recall prior work. Returns oldest-first."""
            if node._delegation_log is None:
                return "No delegation log available."
            records = node._delegation_log.query_received(node._name, n)
            if not records:
                return "No prior delegations found."
            parts = []
            for i, r in enumerate(records, 1):
                parts.append(
                    f"== Prior delegation {i} ==\n"
                    f"Task: {r['task']}\n\n"
                    f"Deliverable:\n{r['deliverable']}"
                )
            return "\n\n---\n\n".join(parts)

        return RecallHistory

    def _setup_sandboxed_write(self) -> None:
        """Replace native Write with a workspace-sandboxed closure.

        Removes 'Write' from native_tools so the SDK doesn't expose it,
        then injects a closure that hard-rejects any path that resolves
        outside the workspace.  Path.resolve() collapses '..' and symlinks,
        so traversal attacks are blocked at the tool level, not just the prompt.

        Bash is kept native but cwd is already set to study_dir by the adapter;
        the prompt further constrains it to the workspace.
        """
        if self._workspace_dir is None:
            return  # no sandboxing if study_dir unknown (e.g. tests)

        workspace = self._workspace_dir.resolve()

        # Remove native Write so the SDK doesn't expose an unrestricted version
        if hasattr(self.adapter, "native_tools") and "Write" in self.adapter.native_tools:
            self.adapter.native_tools = [
                t for t in self.adapter.native_tools if t != "Write"
            ]

        def Write(path: str, body: str) -> str:
            """Write a file.  Restricted to workspace — no exceptions."""
            try:
                # Resolve against workspace so relative paths land there
                candidate = (workspace / path).resolve()
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: invalid path {path!r}: {exc}"

            # Reject anything that escapes the delegations tree
            try:
                candidate.relative_to(workspace)
            except ValueError:
                return (
                    f"ERROR: write rejected — path resolves to {candidate}, "
                    f"which is outside the workspace ({workspace}). "
                    "Only paths inside the workspace are permitted."
                )

            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text(body, encoding="utf-8")
            return f"Written: {candidate}"

        self.adapter.closure_tools["Write"] = Write

    def _build_eval_closures(self) -> dict:
        evals = self._evals_reported

        def ReportEvals(count: int) -> str:
            """Report the number of function evaluations used in this task."""
            evals["count"] = int(count)
            return f"Recorded {count} evaluations."

        return {"ReportEvals": ReportEvals}

    def __call__(self, state: AgenticState) -> Any:
        from langchain_core.messages import AIMessage
        from langgraph.types import Command

        from .agent_prompts import IMPLEMENTER_REPORT_RETRY_PROMPT

        self._evals_reported.clear()
        messages = _to_adapter_messages(state["messages"])
        text = self.adapter.invoke(messages)

        diagnosis = _classify_response(text)
        if diagnosis is not None:
            # One retry with correction prompt
            retry_messages = messages + [
                {"role": "ai", "content": text},
                {
                    "role": "user",
                    "content": (
                        f"{IMPLEMENTER_REPORT_RETRY_PROMPT}"
                        f"\n\nDiagnosis: {diagnosis}"
                    ),
                },
            ]
            text = self.adapter.invoke(retry_messages)

        ai_msg = AIMessage(content=text)
        evals_delta = self._evals_reported.get("count", 0)
        return_to = state.get("return_to")
        return Command(
            goto=return_to,
            update={
                "messages": [ai_msg],
                "last_report": text,
                "evals_used": state.get("evals_used", 0) + evals_delta,
            },
        )


# Backward-compatible alias — ImplementerNode is the WorkerNode used in the
# canonical 2-node topology.  New code should use WorkerNode directly.
ImplementerNode = WorkerNode
