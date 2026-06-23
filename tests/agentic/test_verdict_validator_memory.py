"""#1 fix: the live verdict validator was STATELESS — build_judge_prompt got the
current rationale but not the hypothesis's prior rulings, so on a borderline case
the verdict oscillated (FALSIFIED↔INCONCLUSIVE) between calls (observed in the
Haiku run 20260623T194849). The fix feeds the judge its own earlier rulings on
the SAME hypothesis (from the ledger status_log) with a justify-any-reversal
guard — the proven pattern the gate critic already uses.
"""
from __future__ import annotations

from f3dasm._src.agentic.verdict_validator import build_judge_prompt
from f3dasm._src.agentic.nodes.critic_gate import _prior_rulings_digest


def _base_kwargs(**over):
    kw = dict(
        statement="basin is unique", prediction="f <= -0.95",
        criterion="250+ diverse evals find none < -0.85", status="FALSIFIED",
        comment="result -0.9190 missed threshold", evidence={"delegation": "D002"},
        delegation_report=None,
    )
    kw.update(over)
    return kw


def test_no_prior_rulings_keeps_judge_stateless():
    # First verdict on a hypothesis: no history → no prior block, behaviour as before.
    prompt = build_judge_prompt(**_base_kwargs())
    assert "PRIOR RULINGS" not in prompt
    assert "reverse a prior ruling" not in prompt


def test_prior_rulings_inject_history_and_consistency_guard():
    digest = "- ruled INCONCLUSIVE: zone-based, between thresholds\n    [your ruling then: §3 allows it]"
    prompt = build_judge_prompt(**_base_kwargs(prior_rulings=digest))
    # the judge now SEES its earlier ruling ...
    assert "INCONCLUSIVE: zone-based" in prompt
    assert "your ruling then" in prompt
    # ... and is told to stay consistent / justify any reversal (the anti-oscillation guard)
    assert "CONSISTENTLY" in prompt
    assert "reverse a prior ruling ONLY by explicitly naming" in prompt
    assert "must not flip the verdict between calls" in prompt


def test_digest_excludes_current_entry_and_formats_prior():
    # status_log: the LAST entry is the verdict under judgement now; earlier ones are prior.
    h = {"status_log": [
        {"status": "OPEN", "comment": "initial proposal", "validator_note": None},
        {"status": "FALSIFIED", "comment": "missed by 0.032", "validator_note": "§3: adequate+contradicted"},
        {"status": "INCONCLUSIVE", "comment": "near-miss, zone-based", "validator_note": None},  # <- current
    ]}
    digest = _prior_rulings_digest(h)
    assert "ruled OPEN" in digest
    assert "ruled FALSIFIED: missed by 0.032" in digest
    assert "§3: adequate+contradicted" in digest  # prior validator_note echoed back
    assert "zone-based" not in digest             # the current entry is excluded


def test_digest_empty_for_single_entry_log():
    # Only the entry under judgement exists → no prior ruling → empty digest.
    h = {"status_log": [{"status": "FALSIFIED", "comment": "x", "validator_note": None}]}
    assert _prior_rulings_digest(h) == ""
    assert _prior_rulings_digest({}) == ""
    assert _prior_rulings_digest({"status_log": []}) == ""
