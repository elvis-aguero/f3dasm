"""Pure helper functions for parsing and classifying agentic node responses.

These functions have no side effects, no class coupling, and no ``self``
parameter — they are extracted here from ``core.py`` to keep that module
focused on node class definitions.
"""

from __future__ import annotations

from pathlib import (
    Path,  # noqa: F401  (used by _resolve_delegation_evals / _stamped_eval_count)
)

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
