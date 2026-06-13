"""The strategizer's tool closures (Delegate/Parallel/GetStatus/Done/FollowUp/
WriteNote/ReadNote/WriteDeliverable/RecallStore/QueryStore/AskForFeedback +
hypothesis tools). Built per-node; the node is passed in so closures reach its
state. Extracted verbatim from StrategizerNode._build_routing_closures."""
from __future__ import annotations

import re
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .._constants import _BACKSTOP_ENABLED, RUN_BACKSTOP_MULTIPLE
from ..parsing import (
    _classify_response,
    _consult_handbook,
    _parse_verdict,
    _resolve_delegation_evals,
    _stamped_eval_count,
)

# Forward-compatible delegation-target resolution. Agents repeatedly name a
# target by CAPABILITY rather than the exact graph node name — e.g.
# "pipeline"/"pipeline_executor" for the implementer (the "pipeline executor"),
# "data_generation" for the datagenerator — and bounce off "unknown target".
# Resolve in order: exact node name -> normalized name (case/separator-
# insensitive) -> normalized role. Resolution is by live node name or role
# only — there is NO hardcoded capability-synonym table. The strategizer prompt
# names targets by their hint/role, so it does not invent capability words like
# 'pipeline'/'oracle'; an unresolvable target returns None and the caller errors
# with the valid-target list (the agent then self-corrects). Resolving by role
# (not node name) keeps it forward-compatible across node renames.


def _norm_target(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# f3dasm marks infeasible / failed designs with a large-magnitude sentinel
# output (e.g. ±1e9; the resonance study treats resonance <= -1e8 as
# infeasible). "Best" must never return such a placeholder.
_INFEASIBLE_SENTINEL_MAG = 1e8


def _select_best_index(values, n_best, minimize=True, sentinel_mag=_INFEASIBLE_SENTINEL_MAG):
    """Index of the n best *feasible* values.

    Coerces to numeric, drops NaN and large-magnitude infeasibility
    placeholders, then picks the smallest (minimize) or largest (maximize).
    Returns a possibly-empty index when no feasible values remain.
    """
    import pandas as pd
    s = pd.to_numeric(values, errors="coerce").dropna()
    s = s[s.abs() < sentinel_mag]
    if s.empty:
        return s.index
    chosen = s.nsmallest(n_best) if minimize else s.nlargest(n_best)
    return chosen.index


def resolve_target(
    requested: str, outgoing: list[str], roles: dict[str, str]
) -> str | None:
    """Map a requested delegation target to a valid outgoing node name, or None.

    ``roles`` maps node name -> its configured role. Resolution is exact-name →
    normalized-name → normalized-role. Only a unique, confident match resolves;
    anything else returns None (caller errors)."""
    if requested in outgoing:
        return requested
    rn = _norm_target(requested)
    if not rn:
        return None
    for t in outgoing:  # normalized node name (case/separator-insensitive)
        if _norm_target(t) == rn:
            return t
    for t in outgoing:  # normalized role
        if _norm_target(roles.get(t, "")) == rn:
            return t
    return None


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


def build_routing_tools(node) -> dict:
    route = node._route
    outgoing = node._outgoing
    study_dir = node._study_dir
    interactive = node._interactive

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
        _resolved = resolve_target(
            target, outgoing,
            {t: getattr(node._spec.nodes.get(t), "role", "") for t in outgoing},
        )
        if _resolved is None:
            return (
                f"ERROR: unknown target {target!r}."
                f" Valid targets: {outgoing}"
            )
        if _resolved != target:
            # Forward-compatible alias resolution: the agent named the target by
            # capability (e.g. 'pipeline_executor' -> 'implementer'). Proceed and
            # record it for observability instead of bouncing the agent.
            node._record_intervention(
                "TARGET_ALIAS", _resolved,
                f"delegation target {target!r} resolved to {_resolved!r}.",
            )
            target = _resolved
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
                from ...agent_prompts import IMPLEMENTER_REPORT_RETRY_PROMPT
                from ...backends.base import (
                    debug_enabled as _dbg,
                )
                from ...backends.base import (
                    set_delegation_id as _set_did,
                )
                from ...backends.base import (
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
                    from ...agent_prompts import (
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

                # Accumulate token usage from this worker invocation
                # (_usage is also consumed downstream by the delegation log).
                _usage = getattr(worker, "last_usage", {}) or {}
                node._record_worker_usage(worker, target, delegation_id)

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
                    _detached = (
                        node._registry[delegation_id].get("status")
                        == "Cancelled"
                    )
                    if _detached:
                        # CancelDelegation detached this while it ran: keep it
                        # Cancelled and DISCARD the deliverable. Usage was
                        # already recorded above (the worker did spend tokens).
                        node._registry[delegation_id]["evals"] = _evals
                    else:
                        node._registry[delegation_id].update({
                            "status": "Done",
                            "result": text,
                            "evals": _evals,
                            "usage": _usage,
                        })
                        # This target made progress → clear its consecutive
                        # error streak (the repeated-errors halt is for a
                        # target stuck failing, not one that recovers).
                        node._consecutive_errors[target] = 0
                with node._notifications_lock:
                    node._notifications.append(
                        f"[Delegation {delegation_id} "
                        + ("completed after cancellation — result discarded]"
                           if _detached else "Done]")
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

                            from ...agent_runtime import (
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
                node._record_retrospective(
                    node._role_of(target), delegation_id, text
                )
            except Exception:  # noqa: BLE001
                tb = traceback.format_exc()
                _usage = getattr(worker, "last_usage", {}) or {}
                node._record_worker_usage(worker, target, delegation_id)
                with node._registry_lock:
                    node._registry[delegation_id].update({
                        "status": "Errored",
                        "result": tb,
                        "evals": evals_box["count"],
                        "usage": _usage,
                    })
                    node._consecutive_errors[target] = (
                        node._consecutive_errors.get(target, 0) + 1
                    )
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

    def CancelDelegation(delegation_id: str) -> str:
        """Detach a still-running delegation so it no longer blocks Done().

        Marks it Cancelled: it is excluded from the active-delegation check and
        its eventual result is discarded, though its token usage is still
        accounted. The background worker is not force-killed (it finishes on
        its own and is ignored). Use when a delegation is no longer needed —
        e.g. it has run long enough, or you want to conclude without it."""
        prefix = node._drain_notifications()
        with node._registry_lock:
            entry = node._registry.get(delegation_id)
            if entry is None:
                return (
                    prefix + f"No delegation {delegation_id!r}. "
                    f"Known: {list(node._registry)}"
                )
            st = entry.get("status")
            if st not in ("Working", "FollowUp"):
                return (
                    prefix + f"Delegation {delegation_id} is {st!r}, not "
                    "running — nothing to cancel."
                )
            entry["status"] = "Cancelled"
        with node._notifications_lock:
            node._notifications.append(
                f"[Delegation {delegation_id} cancelled — detached; its "
                "result will be ignored]"
            )
        return (
            prefix + f"Delegation {delegation_id} cancelled (detached): "
            "excluded from the run, its result will be ignored. You may "
            "proceed (e.g. call Done() if nothing else is running) or start "
            "other work."
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
            # Soft nudge (NOT an "ERROR:" return, so it isn't counted as a
            # tool error): closing now is premature, but offer the three real
            # ways forward instead of a dead-end bounce.
            return (
                prefix +
                f"{len(pending)} delegation(s) still running: {pending}. "
                "Closing now is premature — you have three options:\n"
                "  (a) keep working: inspect results so far, write notes, or "
                "start another delegation while these finish;\n"
                "  (b) wait, then GetStatus(<id>) on each and interpret its "
                "results before you conclude;\n"
                "  (c) CancelDelegation(<id>) if a delegation is no longer "
                "needed — it detaches and stops blocking Done().\n"
                "Re-call Done() once none are still running. "
                "(Tip: Delegate(wait=True) avoids this for sequential tasks.)"
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
        from ...instrumented import RunStateSummary
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
        minimize: bool = True,
    ) -> str:
        """Filtered view of the evaluation ledger (e.g. rows from D001+D003
        only). Use to ground claims or to select training subsets; cite row
        values from here as evidence.

        n_best returns the best rows by output_name: smallest when minimize=True
        (default), largest when minimize=False (set this for MAXIMIZATION
        objectives). Infeasible/placeholder rows (large-magnitude sentinel
        outputs) are never returned as 'best'."""
        import json as _json

        from ....errors import EmptyFileError, ReachMaximumTriesError
        from ....experimentdata import ExperimentData
        from ...instrumented import _PROVENANCE_COLS

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
        if source is not None and "_source" in df_out.columns:
            mask &= df_out["_source"] == source

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
        if isinstance(minimize, str):  # MCP string-in tools may pass "false"
            minimize = minimize.strip().lower() not in (
                "false", "0", "no", "max", "maximize"
            )
        if n_best is not None and output_name is not None:
            if output_name not in filtered.columns:
                return (
                    f"ERROR: output column {output_name!r} not found. "
                    f"Available: {list(filtered.columns)}"
                )
            best_idx = _select_best_index(
                filtered[output_name], n_best, minimize=minimize
            )
            if len(best_idx) == 0:
                return (
                    f"No feasible rows to rank by {output_name!r} "
                    "(all matching rows are infeasibility placeholders)."
                )
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
                combined = _pd.concat(
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
        "CancelDelegation": CancelDelegation,
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
    if node._spec is not None:
        _ag = node._spec.nodes.get(node._name)
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
        p = Path(filename)
        if p.suffix not in allowed_exts:
            return (
                f"ERROR: filename must end in {allowed_exts}, got {filename!r}."
            )
        if "/" in filename or "\\" in filename:
            return "ERROR: filename must be a bare name (no path separators)."

        # Write directly to study_dir/ — the user-visible output location.
        target = Path(node._study_dir) / p.name
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
    closures.update(node._build_hypothesis_closures())

    # AskForFeedback is only injected when a critic node is
    # connected AND this is the entry node (only the entry node
    # gates Done).
    critic_name: str | None = node._find_critic_name()
    spec = node._spec

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
    _node_name = node._name
    return {k: node._wrap_closure(v, _node_name) for k, v in closures.items()}
