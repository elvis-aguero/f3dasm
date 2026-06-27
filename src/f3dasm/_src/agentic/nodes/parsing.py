"""Pure helper functions for parsing and classifying agentic node responses.

These functions have no side effects, no class coupling, and no ``self``
parameter — they are extracted here from ``core.py`` to keep that module
focused on node class definitions.
"""

from __future__ import annotations

from pathlib import Path

_REQUIRED_SUBSECTIONS = [
    "### Actions taken",
    "### Files touched",
    "### Conclusions",
    "### Numbers",
]

_CAPABILITY_PHRASES = [
    "i cannot", "i can't", "i don't have access",
    "unable to", "not able to", "i am unable",
]

# The critic emits exactly these three (agents/critic.py); no others.
_VALID_VERDICTS = {"PASS", "REVISE", "REJECT"}


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
        from ..instrumented import RunStateSummary
        summary = RunStateSummary.from_store(store_dir)
        if (
            summary is not None
            and summary.n_per_delegation.get(delegation_id, 0) > 0
        ):
            return summary.n_per_delegation[delegation_id]
    except Exception:  # noqa: BLE001
        pass
    return reported


def delegation_eval_store(
    run_exp_dir: Path | None, namespace: str | None
) -> Path | None:
    """Resolve the ExperimentData store a delegation actually writes to.

    A namespaced delegation (design-namespace, Axis 3a) writes to its OWN store
    at ``<run_exp_dir>/<namespace>``, NOT the canonical ``<run_exp_dir>``. The
    unledgered-evals guard and stamped-row counts MUST look there — otherwise a
    namespaced worker that correctly used get_evaluator() reads as 0 rows in the
    canonical store and is falsely bounced as having bypassed the oracle
    (observed: run 20260626T231202 D003 re-ran 6x chasing a check it could never
    satisfy, inflating the polar ledger to 600 rows). ``namespace=None`` → the
    canonical store, unchanged.
    """
    if run_exp_dir is None:
        return None
    return (run_exp_dir / namespace) if namespace else run_exp_dir


def _stamped_eval_count(store_dir: Path | None, delegation_id: str) -> int:
    """Rows in the canonical store stamped with this delegation_id (0 if none).

    Unlike _resolve_delegation_evals (which falls back to the honour-system
    count), this reports ONLY provenance-stamped rows — so a caller can detect
    a delegation that evaluated but bypassed get_evaluator().
    """
    if store_dir is None:
        return 0
    try:
        from ..instrumented import RunStateSummary
        summary = RunStateSummary.from_store(store_dir)
        if summary is not None:
            return int(summary.n_per_delegation.get(delegation_id, 0))
    except Exception:  # noqa: BLE001
        pass
    return 0


def _reconcile_delegation_evals(
    store_dir: Path | None,
    delegation_id: str,
    claimed: int,
    source_registered: bool,
) -> tuple[int, bool, int]:
    """Reconcile a worker's claimed eval count against the canonical store.

    The canonical store is the single source of truth. When a ground-truth
    source is registered and the worker CLAIMED evaluations but NONE are
    provenance-stamped in the store, the delegation evaluated off-ledger: the
    truthful count is 0. Otherwise fall back to the usual resolution (ledger
    rows if any, else the honour-system claim — e.g. lookup-direct studies).

    Used on BOTH the normal-return and the cancel/detach path so an off-ledger
    delegation cannot silently keep a claimed-but-unledgered eval count.

    Returns
    -------
    (evals, off_ledger, stamped)
        ``evals`` — the truthful count to record; ``off_ledger`` — True when the
        claim could not be backed by stamped rows; ``stamped`` — stamped row
        count (for diagnostics).
    """
    stamped = _stamped_eval_count(store_dir, delegation_id)
    if source_registered and claimed > 0 and stamped == 0:
        return 0, True, stamped
    return (
        _resolve_delegation_evals(store_dir, delegation_id, claimed),
        False,
        stamped,
    )


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
    from ..agent_prompts import (
        REFLECT_DIAGNOSIS_CAPABILITY_LIMIT,
        REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE,
        REFLECT_DIAGNOSIS_NO_REPORT_HEADING,
        REFLECT_DIAGNOSIS_SHORT,
    )

    sections = _REQUIRED_SUBSECTIONS if required_sections is None else required_sections

    if len(text.strip()) < 100:
        return REFLECT_DIAGNOSIS_SHORT
    low = text.lower()
    # A capability-limit only counts when the worker produced NO report at all
    # (it gave up). A valid report may legitimately say "I cannot reproduce X"
    # in its Conclusions without being a capability failure (audit O41: the
    # phrase-match used to fire on correct reports).
    if "## report" not in low:
        if any(p in low for p in _CAPABILITY_PHRASES):
            return REFLECT_DIAGNOSIS_CAPABILITY_LIMIT
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
        from ..knowledge import KnowledgeBase
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
