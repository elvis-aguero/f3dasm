"""run_ledger.analysis_brief: mechanical post-run digest.

Asserts it (a) offloads the MECHANICAL facts (KPI row, diagnostics tally,
verbatim ERROR_RETURN) and (b) RELOCATES the prose (retrospectives verbatim)
rather than classifying it — the line we must not cross is turning a
judgement (reading prose for varied failure-mode wording) into a proxy grep.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_STUDIES = Path(__file__).resolve().parents[2] / "studies"
if str(_STUDIES) not in sys.path:
    sys.path.insert(0, str(_STUDIES))

import run_ledger  # noqa: E402


def _make_run(tmp_path: Path) -> Path:
    # studies/<study>/runs/<id>/debug/...
    run_dir = tmp_path / "studyX" / "runs" / "20260101T000000"
    debug = run_dir / "debug"
    debug.mkdir(parents=True)
    (debug / "run_status.json").write_text(json.dumps({"status": "GATED"}))
    (debug / "diagnostics.jsonl").write_text(
        json.dumps({"type": "ERROR_RETURN", "source_id": "D002",
                    "detail": "tool raised"}) + "\n"
        + json.dumps({"type": "RAW_ORACLE_NUDGE", "source_id": "D003"}) + "\n"
    )
    (debug / "retrospectives.jsonl").write_text(
        json.dumps({"source_id": "D001", "role": "implementer",
                    "flagged": True,
                    "text": "FRICTION: the catalog named a tool I could not "
                            "call. BLOCKED: none."}) + "\n"
    )
    cr = debug / "critic_reviews"
    cr.mkdir()
    (cr / "call_001.md").write_text("verdict: PASS")
    return run_dir


def test_brief_offloads_mechanical_kpis_and_diagnostics(tmp_path):
    brief = run_ledger.analysis_brief(_make_run(tmp_path))
    # Step-5 KPI baseline + Step-2 tally are computed for us
    assert "## KPIs — this run vs previous (Step 5)" in brief
    assert "outcome:" in brief
    assert "critic_consults: 1" in brief          # one critic call = gate attempt
    assert "## ScienceMonitor diagnostics tally (Step 2)" in brief
    assert "ERROR_RETURN: 1" in brief and "RAW_ORACLE_NUDGE: 1" in brief
    # ERROR_RETURN (KPI target 0) surfaced verbatim, structured — not a prose grep
    assert "ERROR_RETURN events verbatim (1; target 0)" in brief
    assert "tool raised" in brief


def test_brief_relocates_prose_but_does_not_classify(tmp_path):
    brief = run_ledger.analysis_brief(_make_run(tmp_path))
    # prose artifacts are pointed to with counts...
    assert "Step 1 retrospectives: 1 entries" in brief
    assert "Step 3 critic_reviews: 1 calls" in brief
    assert "read FIRST" in brief and "judge, don't grep" in brief
    # ...and the retrospective is surfaced VERBATIM (relocation, not a verdict):
    # the brief must contain the exact prose, leaving the judgement to the reader.
    assert ("FRICTION: the catalog named a tool I could not call. BLOCKED: none."
            in brief)
    # and it must NOT invent a classification the agent never wrote
    assert "is a bug" not in brief.lower()
    assert "root cause" not in brief.lower()
