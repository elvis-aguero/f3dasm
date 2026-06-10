"""Tests for ScienceMonitor — drift rules, hygiene, escalation."""

from __future__ import annotations

import json
from pathlib import Path

from f3dasm._src.agentic.delegation_log import DelegationLog
from f3dasm._src.agentic.hypothesis_ledger import HypothesisLedger
from f3dasm._src.agentic.science_monitor import (
    ESCALATE_AFTER_VIOLATIONS,
    ScienceMonitor,
)


def make_world(tmp_path):
    ledger = HypothesisLedger(tmp_path)
    dlog = DelegationLog(tmp_path / "delegation_log.jsonl")
    drift_records = []
    mon = ScienceMonitor(
        ledger, dlog, diagnostics_writer=drift_records.append)
    return ledger, dlog, mon, drift_records


def propose(ledger, statement="Claim below 1.0"):
    return ledger.propose(
        statement=statement,
        falsification_criterion="any point below 0.5",
        prediction="sweep finds nothing below 0.5",
        prior=0.5, proposed_by="strategizer")


def record_done(dlog, d_id, h_ids, deliverable, falsify=False):
    dlog.record(
        id=d_id, from_node="strategizer", to_node="implementer",
        task="t", deliverable=deliverable, hypothesis_ids=h_ids,
        started_at="x", completed_at="y", status="DONE",
        is_falsification_attempt=falsify)


REPORT = (
    "## Report\n\n### Actions taken\n- ran\n\n### Files touched\n"
    "- none\n\n### Conclusions\nok\n\n### Numbers\nbest_y: 1.47\n"
)


# ---------------------------------------------------------------------------
# Test 1: EVIDENCE_DELEGATION_EXISTS — cites non-existent delegation
# ---------------------------------------------------------------------------

def test_evidence_delegation_exists(tmp_path):
    ledger, dlog, mon, _ = make_world(tmp_path)
    h_id = propose(ledger)
    # Update H1 citing delegation D999 which doesn't exist in the log
    ledger.update(
        h_id, "SUPPORTED", "looks good",
        evidence={"delegation": "D999", "numbers": {}},
        posterior=0.9, triggered_by=None)
    violations = mon.evaluate()
    rules = {v.rule for v in violations}
    assert "EVIDENCE_DELEGATION_EXISTS" in rules, (
        f"Expected EVIDENCE_DELEGATION_EXISTS, got: {rules}")
    v = next(v for v in violations if v.rule == "EVIDENCE_DELEGATION_EXISTS")
    assert v.h_id == h_id
    assert v.severity == "error"


# ---------------------------------------------------------------------------
# Test 2: EVIDENCE_NUMBERS_MATCH — cited numbers not in report
# ---------------------------------------------------------------------------

def test_evidence_numbers_match(tmp_path, monkeypatch):
    import f3dasm._src.agentic.science_monitor as sm
    ledger, dlog, mon, _ = make_world(tmp_path)
    h_id = propose(ledger)
    record_done(dlog, "D001", [h_id], REPORT)
    # Update with wrong number (9.99 not in REPORT which has 1.47)
    ledger.update(
        h_id, "SUPPORTED", "found it",
        evidence={"delegation": "D001", "numbers": {"best_y": 9.99}},
        posterior=0.9, triggered_by=None)

    # DEFAULT: the rule is OFF — a mismatch must NOT fire (the critic
    # catches fabricated numbers; the verbatim match stalled wrap-up).
    monkeypatch.setattr(sm, "EVIDENCE_NUMBERS_MATCH_ENABLED", False)
    rules = {v.rule for v in mon.evaluate()}
    assert "EVIDENCE_NUMBERS_MATCH" not in rules, (
        f"rule is disabled by default; got: {rules}")

    # When explicitly re-enabled, it still fires on a true mismatch.
    monkeypatch.setattr(sm, "EVIDENCE_NUMBERS_MATCH_ENABLED", True)
    rules = {v.rule for v in mon.evaluate()}
    assert "EVIDENCE_NUMBERS_MATCH" in rules, (
        f"Expected EVIDENCE_NUMBERS_MATCH when enabled, got: {rules}")

    # The dormant matcher itself is unchanged.
    assert mon._numbers_match({"best_y": 1.4700000001}, REPORT)
    assert not mon._numbers_match({"x": 9.99}, REPORT)


def test_evidence_anchored_tolerates_derived_numbers(tmp_path):
    """The fix: an update whose RAW measurement traces to the report must
    NOT flag just because it also cites DERIVED/interpretive quantities the
    worker never reported (counts, classifications, reformatted coords).
    Only a fully-ungrounded update (no cited value in the report) flags."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    h_id = propose(ledger)
    record_done(dlog, "D001", [h_id], REPORT)  # REPORT has best_y: 1.47
    ledger.update(
        h_id, "SUPPORTED", "anchored + derived",
        evidence={"delegation": "D001", "numbers": {
            "best_y": 1.47,           # raw — present in REPORT (anchor)
            "n_interior_dims": 7,     # derived — absent from REPORT
            "scattered": "True",      # derived classification
        }},
        posterior=0.9, triggered_by=None)
    rules = {v.rule for v in mon.evaluate()}
    assert "EVIDENCE_NUMBERS_MATCH" not in rules, (
        "anchored evidence with derived extras must not flag")
    # Direct: anchored because best_y matches, despite derived extras.
    assert mon._numbers_match(
        {"best_y": 1.47, "n_interior_dims": 7, "scattered": "True"}, REPORT)
    # Fully ungrounded (no cited value in the report) still flags.
    assert not mon._numbers_match({"n_interior_dims": 7, "k": 9.99}, REPORT)


def test_evidence_self_heals_after_correct_citation(tmp_path):
    """A superseded early bad citation must NOT nag forever: once a later
    entry cites a real delegation, EVIDENCE_DELEGATION_EXISTS clears."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    h_id = propose(ledger)
    # First (bad) citation: D999 does not exist → would fire.
    ledger.update(
        h_id, "INCONCLUSIVE", "premature",
        evidence={"delegation": "D999", "numbers": {}},
        posterior=0.5, triggered_by=None)
    assert "EVIDENCE_DELEGATION_EXISTS" in {v.rule for v in mon.evaluate()}
    # Later correct citation: D001 exists. Latest entry is valid → clears.
    record_done(dlog, "D001", [h_id], REPORT, falsify=True)
    ledger.update(
        h_id, "FALSIFIED", "corrected",
        evidence={"delegation": "D001", "numbers": {"best_y": 1.47}},
        posterior=0.05, triggered_by=None)
    assert "EVIDENCE_DELEGATION_EXISTS" not in {v.rule for v in mon.evaluate()}


def test_d000_is_valid_evidence_anchor(tmp_path):
    """D000 (the precomputed ground-truth pool) is a legitimate evidence
    anchor in lookup studies — citing it must not fire DELEGATION_EXISTS,
    and its numbers (from the store, not a delegation report) are not
    checked against a delegation deliverable."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    h_id = propose(ledger)
    ledger.update(
        h_id, "FALSIFIED", "falsified by exhaustive pool search",
        evidence={"delegation": "D000", "numbers": {"best_y": 0.07}},
        posterior=0.02, triggered_by=None)
    rules = {v.rule for v in mon.evaluate()}
    assert "EVIDENCE_DELEGATION_EXISTS" not in rules
    assert "EVIDENCE_NUMBERS_MATCH" not in rules


# ---------------------------------------------------------------------------
# Test 3: string fallback — value in prose without ### Numbers section
# ---------------------------------------------------------------------------

def test_numbers_match_string_fallback(tmp_path):
    ledger, dlog, mon, _ = make_world(tmp_path)
    # Deliverable WITHOUT a ### Numbers section
    deliverable = "## Report\n\nWe observed rmse=0.042 in our runs.\n"
    # "0.042" appears in prose → should match via full-text fallback
    assert mon._numbers_match({"rmse": 0.042}, deliverable)
    # Value not present → no match
    assert not mon._numbers_match({"rmse": 9.99}, deliverable)


# ---------------------------------------------------------------------------
# Test 4: SUPPORTED_WITHOUT_ATTACK — self-healing after falsify=True
# ---------------------------------------------------------------------------

def test_supported_without_attack(tmp_path):
    ledger, dlog, mon, _ = make_world(tmp_path)
    h_id = propose(ledger)
    record_done(dlog, "D001", [h_id], REPORT, falsify=False)
    # Close h as SUPPORTED with a non-falsification delegation
    ledger.update(
        h_id, "SUPPORTED", "looks supported",
        evidence={"delegation": "D001", "numbers": {"best_y": 1.47}},
        posterior=0.85, triggered_by=None)
    violations = mon.evaluate()
    rules = {v.rule for v in violations}
    assert "SUPPORTED_WITHOUT_ATTACK" in rules, (
        f"Expected SUPPORTED_WITHOUT_ATTACK, got: {rules}")

    # Now add a falsification attempt delegation targeting h_id
    record_done(dlog, "D002", [h_id], REPORT, falsify=True)
    violations2 = mon.evaluate()
    rules2 = {v.rule for v in violations2}
    assert "SUPPORTED_WITHOUT_ATTACK" not in rules2, (
        "Rule should self-heal after is_falsification_attempt=True delegation")


# ---------------------------------------------------------------------------
# Test 5: STALE_OPEN — fires only with ≥3 completed delegations
# ---------------------------------------------------------------------------

def test_stale_open(tmp_path):
    ledger, dlog, mon, _ = make_world(tmp_path)
    h1 = propose(ledger, statement="Hypothesis one claim")
    h2 = propose(ledger, statement="Hypothesis two claim")

    # With <3 completed — no STALE_OPEN at all
    record_done(dlog, "D001", [h1], REPORT)
    record_done(dlog, "D002", [h1], REPORT)
    violations = mon.evaluate()
    stale_rules = [v for v in violations if v.rule == "STALE_OPEN"]
    assert len(stale_rules) == 0, (
        "STALE_OPEN should not fire with <3 completed delegations")

    # 3 completed, all linked to h1 — h2 should be stale, h1 should not
    record_done(dlog, "D003", [h1], REPORT)
    violations2 = mon.evaluate()
    stale_ids = {v.h_id for v in violations2 if v.rule == "STALE_OPEN"}
    assert h2 in stale_ids, f"Expected {h2} to be stale, got: {stale_ids}"
    assert h1 not in stale_ids, f"h1 should not be stale, got: {stale_ids}"


# ---------------------------------------------------------------------------
# Test 6: UNANCHORED_DELEGATION — delegation with no ### Numbers section;
#         self-healing when a newer anchored delegation lands for same H.
# ---------------------------------------------------------------------------

def test_unanchored_delegation(tmp_path):
    ledger, dlog, mon, _ = make_world(tmp_path)
    h_id = propose(ledger)
    # Deliverable with no ### Numbers section
    bare_report = (
        "## Report\n\n### Actions taken\n- ran sweep\n\n"
        "### Conclusions\nNothing interesting found.\n"
    )
    record_done(dlog, "D001", [h_id], bare_report)
    violations = mon.evaluate()
    rules = {v.rule for v in violations}
    assert "UNANCHORED_DELEGATION" in rules, (
        f"Expected UNANCHORED_DELEGATION, got: {rules}")
    v = next(v for v in violations if v.rule == "UNANCHORED_DELEGATION")
    assert v.severity == "warn"
    assert v.h_id == h_id
    # Violation names the most-recent linked delegation (D001)
    assert "D001" in v.message


def test_unanchored_delegation_self_heals(tmp_path):
    """Once an anchored delegation D002 lands for H1, the bare D001
    stops nagging — UNANCHORED_DELEGATION for H1 disappears."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    h_id = propose(ledger)
    bare_report = (
        "## Report\n\n### Actions taken\n- ran sweep\n\n"
        "### Conclusions\nNothing interesting found.\n"
    )
    # D001 is bare — violation fires for H1
    record_done(dlog, "D001", [h_id], bare_report)
    violations = mon.evaluate()
    assert any(
        v.rule == "UNANCHORED_DELEGATION" and v.h_id == h_id
        for v in violations
    ), "UNANCHORED_DELEGATION should fire for H1 after bare D001"

    # D002 is anchored (has ### Numbers) and also linked to H1
    record_done(dlog, "D002", [h_id], REPORT)
    violations2 = mon.evaluate()
    unanchored = [
        v for v in violations2
        if v.rule == "UNANCHORED_DELEGATION" and v.h_id == h_id
    ]
    assert len(unanchored) == 0, (
        "UNANCHORED_DELEGATION for H1 should self-heal once anchored "
        f"D002 lands; violations: {violations2}"
    )


# ---------------------------------------------------------------------------
# Test 7: POSTERIOR_INERTIA — barely moved posterior triggers, big move doesn't
# ---------------------------------------------------------------------------

def test_posterior_inertia(tmp_path, monkeypatch):
    import f3dasm._src.agentic.science_monitor as sm
    ledger, dlog, mon, _ = make_world(tmp_path)
    h_id = propose(ledger)  # prior=0.5, initial log entry posterior=0.5
    record_done(dlog, "D001", [h_id], REPORT)
    # Close with posterior barely different from prior (0.52)
    ledger.update(
        h_id, "SUPPORTED", "barely moved",
        evidence={"delegation": "D001", "numbers": {"best_y": 1.47}},
        posterior=0.52, triggered_by=None)
    # DEFAULT: OFF — trust the agent to calibrate; no nag on a small move.
    monkeypatch.setattr(sm, "POSTERIOR_INERTIA_ENABLED", False)
    assert "POSTERIOR_INERTIA" not in {v.rule for v in mon.evaluate()}
    # When explicitly re-enabled, it still fires on a barely-moved update.
    monkeypatch.setattr(sm, "POSTERIOR_INERTIA_ENABLED", True)
    rules = {v.rule for v in mon.evaluate()}
    assert "POSTERIOR_INERTIA" in rules, (
        f"Expected POSTERIOR_INERTIA for 0.5→0.52 when enabled, got: {rules}")

    # New world: close with a bigger jump (0.05 from prior 0.5 = big enough)
    ledger2, dlog2, mon2, _ = make_world(tmp_path / "sub")
    h2 = propose(ledger2)
    record_done(dlog2, "D001", [h2], REPORT)
    ledger2.update(
        h2, "SUPPORTED", "big jump",
        evidence={"delegation": "D001", "numbers": {"best_y": 1.47}},
        posterior=0.05, triggered_by=None)
    violations2 = mon2.evaluate()
    rules2 = {v.rule for v in violations2}
    assert "POSTERIOR_INERTIA" not in rules2, (
        f"No POSTERIOR_INERTIA expected for 0.5→0.05, got: {rules2}")


# ---------------------------------------------------------------------------
# Test 8: drain() caps at max_inject with digest for overflow
# ---------------------------------------------------------------------------

def test_drain_caps_and_digests(tmp_path):
    # Use a monitor with max_inject=2 explicitly and engineer ≥3
    # unique-keyed violations so we get 2 full lines + 1 digest line.
    #
    # Violations engineered:
    #   - STALE_OPEN(H1): delegations D001-D003 all linked to H2/H3
    #   - STALE_OPEN(H2): delegations D001-D003 NOT linked to H2 ...
    #   Wait — H2 IS linked to D001-D003, so it is NOT stale.
    #
    # Strategy: create 3 hypotheses, link D001-D003 to H3 only.
    #   → H1 and H2 become stale (2 STALE_OPEN violations, distinct keys)
    #   → H3 is linked in last 3 delegations but bare → UNANCHORED fires
    #   → Total unique violations: 3 (STALE H1, STALE H2, UNANCHORED H3)
    #   → With max_inject=2: 2 full lines + 1 digest line ✓
    ledger, dlog, mon, _ = make_world(tmp_path)
    h1 = propose(ledger, statement="First open claim")
    h2 = propose(ledger, statement="Second open claim")
    h3 = propose(ledger, statement="Third open claim")
    bare = "## Report\n\n### Conclusions\nno numbers here.\n"
    # Link all 3 delegations to h3 only → h1 and h2 are stale
    # h3 bare report → UNANCHORED fires for h3
    record_done(dlog, "D001", [h3], bare)
    record_done(dlog, "D002", [h3], bare)
    record_done(dlog, "D003", [h3], bare)

    violations = mon.evaluate()
    # Dedupe by key: STALE_OPEN(H1), STALE_OPEN(H2), UNANCHORED(H3)
    # But UNANCHORED fires 3 times (D001, D002, D003) with key
    # (UNANCHORED_DELEGATION, H3) — same key → only 1 unique
    unique_keys = {v.key for v in violations}
    assert len(unique_keys) >= 3, (
        f"Need ≥3 unique violation keys, got {len(unique_keys)}: "
        f"{[(v.rule, v.h_id) for v in violations]}")

    result = mon.drain()
    full_lines = [
        ln for ln in result.splitlines()
        if ln.startswith("[SCIENCE MONITOR — ")]
    assert len(full_lines) == mon._max_inject, (
        f"Expected {mon._max_inject} full messages, "
        f"got {len(full_lines)}: {result!r}")
    digest_lines = [
        ln for ln in result.splitlines()
        if ln.startswith("[SCIENCE MONITOR] +")]
    assert len(digest_lines) == 1, (
        f"Expected 1 digest line, got {len(digest_lines)}: {result!r}")


# ---------------------------------------------------------------------------
# Test 9: drain() drops resolved violations (self-healing)
# ---------------------------------------------------------------------------

def test_drain_drops_resolved_violations(tmp_path):
    ledger, dlog, mon, _ = make_world(tmp_path)
    h_id = propose(ledger)
    record_done(dlog, "D001", [h_id], REPORT, falsify=False)
    ledger.update(
        h_id, "SUPPORTED", "looks good",
        evidence={"delegation": "D001", "numbers": {"best_y": 1.47}},
        posterior=0.85, triggered_by=None)
    result1 = mon.drain()
    assert "SUPPORTED_WITHOUT_ATTACK" in result1, (
        f"Expected rule in first drain, got: {result1!r}")

    # Now record a falsification attempt → rule resolves
    record_done(dlog, "D002", [h_id], REPORT, falsify=True)
    result2 = mon.drain()
    assert "SUPPORTED_WITHOUT_ATTACK" not in result2, (
        f"Rule should vanish after fix, got: {result2!r}")


# ---------------------------------------------------------------------------
# Test 10: drain() returns empty string when world is healthy
# ---------------------------------------------------------------------------

def test_drain_returns_empty_when_healthy(tmp_path):
    ledger, dlog, mon, _ = make_world(tmp_path)
    h_id = propose(ledger)
    # Anchored delegation (has ### Numbers) linked to the hypothesis
    record_done(dlog, "D001", [h_id], REPORT)
    result = mon.drain()
    assert result == "", (
        f"Expected empty drain for healthy world, got: {result!r}")


# ---------------------------------------------------------------------------
# Test 11: diagnostics written once per live violation (deduplication)
# ---------------------------------------------------------------------------

def test_diagnostics_written_once_per_live_violation(tmp_path):
    ledger, dlog, mon, drift_records = make_world(tmp_path)
    h_id = propose(ledger)
    bare = "## Report\n\n### Conclusions\nno numbers.\n"
    record_done(dlog, "D001", [h_id], bare)
    mon.drain()
    count_after_first = sum(
        1 for r in drift_records if r["rule"] == "UNANCHORED_DELEGATION")
    assert count_after_first == 1

    # Second drain — violation still live but should not re-log
    mon.drain()
    count_after_second = sum(
        1 for r in drift_records if r["rule"] == "UNANCHORED_DELEGATION")
    assert count_after_second == 1, (
        "Diagnostics should not re-log an already-logged live violation")


# ---------------------------------------------------------------------------
# Test 12: escalation after 3 distinct rules on one hypothesis
# ---------------------------------------------------------------------------

def test_escalation_after_three_violations_one_hypothesis(tmp_path, monkeypatch):
    """Three distinct rules accumulate in _h_rule_seen for one h_id.

    Phase 1: h_id OPEN + bare delegations → UNANCHORED_DELEGATION
    Phase 2: Close with tiny posterior shift → POSTERIOR_INERTIA
    Phase 3: SUPPORTED but no falsification → SUPPORTED_WITHOUT_ATTACK
    After 3 distinct rules for h_id → escalation_due() returns [h_id].
    """
    import f3dasm._src.agentic.science_monitor as sm
    monkeypatch.setattr(sm, "POSTERIOR_INERTIA_ENABLED", True)  # needed as rule 2
    ledger, dlog, mon, _ = make_world(tmp_path)
    h_id = propose(ledger)
    # Need a second h so the ledger has room and STALE_OPEN doesn't
    # interfere — link all delegations to h_id to avoid h_id going stale
    h2 = propose(ledger, statement="Second anchor hypothesis claim")
    bare = "## Report\n\n### Conclusions\nno numbers.\n"

    # Phase 1: 3 bare delegations linked to h_id only
    # → UNANCHORED_DELEGATION(h_id) fires (rule 1)
    # → h2 becomes stale (not h_id)
    record_done(dlog, "D001", [h_id], bare)
    record_done(dlog, "D002", [h_id], bare)
    record_done(dlog, "D003", [h_id], bare)
    mon.drain()
    assert "UNANCHORED_DELEGATION" in mon._h_rule_seen.get(h_id, set()), \
        f"Expected UNANCHORED in h_rule_seen, got: {mon._h_rule_seen}"

    # Phase 2: Close h_id as SUPPORTED with barely-moved posterior
    # prior=0.5, initial log posterior=0.5 → close at 0.52 (delta < 0.05)
    # → POSTERIOR_INERTIA(h_id) fires (rule 2)
    # → SUPPORTED_WITHOUT_ATTACK(h_id) also fires (no falsification yet)
    #    (rule 3 — we get both in one drain!)
    record_done(dlog, "D004", [h_id], REPORT, falsify=False)
    ledger.update(
        h_id, "SUPPORTED", "evidence from D004",
        evidence={"delegation": "D004", "numbers": {"best_y": 1.47}},
        posterior=0.52, triggered_by=None)
    mon.drain()

    seen = mon._h_rule_seen.get(h_id, set())
    assert len(seen) >= ESCALATE_AFTER_VIOLATIONS, (
        f"Expected ≥{ESCALATE_AFTER_VIOLATIONS} rules for {h_id}, "
        f"got {seen}")

    esc = mon.escalation_due()
    assert esc is not None, "escalation_due() should not be None"
    assert h_id in esc, (
        f"Expected {h_id} in escalation list, got: {esc}")

    mon.note_escalated()
    assert mon.escalation_due() is None, (
        "escalation_due() should be None after note_escalated()")


# ---------------------------------------------------------------------------
# Test 13: escalation capped at escalation_cap
# ---------------------------------------------------------------------------

def test_escalation_capped(tmp_path):
    ledger, dlog, mon, _ = make_world(tmp_path)
    mon._escalation_cap = 1
    h_id = propose(ledger)
    record_done(dlog, "D001", [h_id], REPORT, falsify=False)
    ledger.update(
        h_id, "SUPPORTED", "supported",
        evidence={"delegation": "D001", "numbers": {"best_y": 1.47}},
        posterior=0.85, triggered_by=None)

    # First escalation cycle — trigger 3 distinct rules
    # SUPPORTED_WITHOUT_ATTACK (1st rule)
    mon.drain()
    # Trigger POSTERIOR_INERTIA — need another update
    # but h is already SUPPORTED, so we reopen and re-close
    # Actually easier: use separate h for 3 rules; or just verify
    # that once cap is hit, future escalations don't fire.
    # Trigger escalation via error_streak (EVIDENCE_DELEGATION_EXISTS x2)
    h2 = propose(ledger, statement="Second open claim")
    ledger.update(
        h2, "SUPPORTED", "bad evidence",
        evidence={"delegation": "D999", "numbers": {}},
        posterior=0.8, triggered_by=None)
    mon.drain()
    mon.drain()  # second evaluation of same error → streak=2 → escalation
    assert mon.escalation_due() is not None
    mon.note_escalated()  # cap=1, this uses the 1 allowed escalation

    # Re-trigger qualifying state — should return None now (capped)
    mon.drain()
    assert mon.escalation_due() is None, (
        "escalation_due() must be None after cap is reached")


# ---------------------------------------------------------------------------
# Test 14: error streak escalates after 2 evaluations without fix
# ---------------------------------------------------------------------------

def test_error_streak_no_double_count_within_turn(tmp_path):
    """on_hypothesis_update + one drain must not count the streak twice.

    If _bookkeep incremented the streak in BOTH on_hypothesis_update and
    drain(), a single turn would jump to streak=2 and trigger escalation
    prematurely.  Only drain() should increment; so after one turn the
    streak is 1 and escalation_due() returns None.
    """
    ledger, dlog, mon, _ = make_world(tmp_path)
    h_id = propose(ledger)
    # Cite non-existent delegation D999 → EVIDENCE_DELEGATION_EXISTS fires
    ledger.update(
        h_id, "SUPPORTED", "fake evidence",
        evidence={"delegation": "D999", "numbers": {}},
        posterior=0.8, triggered_by=None)

    # Simulate one agent turn: update hook then drain
    mon.on_hypothesis_update(h_id)
    mon.drain()

    # Streak should be 1 (only drain counted), not 2
    key = ("EVIDENCE_DELEGATION_EXISTS", h_id)
    assert mon._error_streak.get(key, 0) == 1, (
        f"Streak should be 1 after one turn, got "
        f"{mon._error_streak.get(key, 0)}")
    assert mon.escalation_due() is None, (
        "No escalation expected after only 1 drain (streak==1)"
    )


def test_error_streak_escalates(tmp_path):
    ledger, dlog, mon, _ = make_world(tmp_path)
    h_id = propose(ledger)
    # Cite non-existent delegation D999 → EVIDENCE_DELEGATION_EXISTS persists
    ledger.update(
        h_id, "SUPPORTED", "fake evidence",
        evidence={"delegation": "D999", "numbers": {}},
        posterior=0.8, triggered_by=None)

    # First evaluation — violation fires, streak=1
    mon.drain()
    assert mon.escalation_due() is None, (
        "No escalation after only 1 occurrence")

    # Second evaluation — streak=2 → meets ESCALATE_AFTER_ERROR_REPEATS=2
    mon.drain()
    esc = mon.escalation_due()
    assert esc is not None, (
        "escalation_due() should fire after 2 consecutive error violations")
    assert h_id in esc, (
        f"Expected {h_id} in escalation list, got: {esc}")


# ---------------------------------------------------------------------------
# Test 17: POSTERIOR_INERTIA boundary — delta 0.049 fires, delta 0.06 does not
# ---------------------------------------------------------------------------

def test_posterior_inertia_boundary(tmp_path, monkeypatch):
    """Epsilon boundary (when ENABLED): |delta| < 0.05 fires; >= 0.05 does not.

    World A: prior=0.5, close at 0.54 → delta=0.04 < 0.05 → fires.
    World B: prior=0.5, close at 0.56 → delta=0.06 >= 0.05 → silent.
    """
    import f3dasm._src.agentic.science_monitor as sm
    monkeypatch.setattr(sm, "POSTERIOR_INERTIA_ENABLED", True)
    # World A: delta = 0.04 — should fire POSTERIOR_INERTIA
    ledger_a, dlog_a, mon_a, _ = make_world(tmp_path / "a")
    h_a = propose(ledger_a)   # prior=0.5
    record_done(dlog_a, "D001", [h_a], REPORT)
    ledger_a.update(
        h_a, "SUPPORTED", "barely moved",
        evidence={"delegation": "D001", "numbers": {"best_y": 1.47}},
        posterior=0.54, triggered_by=None)
    violations_a = mon_a.evaluate()
    rules_a = {v.rule for v in violations_a}
    assert "POSTERIOR_INERTIA" in rules_a, (
        f"Expected POSTERIOR_INERTIA for delta=0.04 (0.5→0.54), "
        f"got: {rules_a}")

    # World B: delta = 0.06 — should NOT fire POSTERIOR_INERTIA
    ledger_b, dlog_b, mon_b, _ = make_world(tmp_path / "b")
    h_b = propose(ledger_b)   # prior=0.5
    record_done(dlog_b, "D001", [h_b], REPORT)
    ledger_b.update(
        h_b, "SUPPORTED", "meaningful jump",
        evidence={"delegation": "D001", "numbers": {"best_y": 1.47}},
        posterior=0.56, triggered_by=None)
    violations_b = mon_b.evaluate()
    rules_b = {v.rule for v in violations_b}
    assert "POSTERIOR_INERTIA" not in rules_b, (
        f"No POSTERIOR_INERTIA expected for delta=0.06 (0.5→0.56), "
        f"got: {rules_b}")


def test_numbers_section_with_markdown_bold_keys(tmp_path):
    """Workers format Numbers keys in bold — must still anchor.

    Observed live: '- **questions_addressed**: 3' was not matched by
    the key regex, firing a false UNANCHORED_DELEGATION.
    """
    ledger, dlog, mon, _ = make_world(tmp_path)
    h = propose(ledger)
    bold_report = (
        "## Report\n\n### Conclusions\nok\n\n### Numbers\n"
        "- **questions_addressed**: 3\n- `papers_consulted`: 9\n"
    )
    record_done(dlog, "D001", [h], bold_report)
    rules = {v.rule for v in mon.evaluate()}
    assert "UNANCHORED_DELEGATION" not in rules


# ---------------------------------------------------------------------------
# UNLEDGERED_EVALS: path-bug regression tests (Bug #A)
# ---------------------------------------------------------------------------


def test_unledgered_does_not_fire_when_store_has_rows(tmp_path):
    """UNLEDGERED_EVALS must NOT fire when the store has rows for the
    delegation.

    Bug #A: when store_dir was one level too shallow (run_dir instead of
    run_dir/experiment_data), RunStateSummary returned None even for a
    full store, causing false-positive UNLEDGERED fires.
    This test verifies the rule is silent when rows exist.
    """
    from unittest.mock import patch
    from f3dasm._src.agentic.instrumented import RunStateSummary

    ledger, dlog, mon, _ = make_world(tmp_path)
    h = propose(ledger)

    # D001 reported evals=5 in the delegation log
    dlog.record(
        id="D001", from_node="strategizer", to_node="implementer",
        task="t", deliverable=REPORT, hypothesis_ids=[h],
        started_at="x", completed_at="y", status="DONE",
        evals=5,
    )

    # Attach a store_dir so UNLEDGERED_EVALS rule runs
    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()
    mon.store_dir = str(store_dir)

    # Stub the summary so the store appears full for D001
    stub = RunStateSummary(
        n_rows=5,
        n_per_delegation={"D001": 5},
        n_per_source={},
        n_per_fidelity=None,
        output_stats={},
    )
    with patch.object(RunStateSummary, "from_store", return_value=stub):
        violations = mon.evaluate()

    rules = {v.rule for v in violations}
    assert "UNLEDGERED_EVALS" not in rules, (
        "UNLEDGERED_EVALS fired as a false positive when store had "
        f"rows for D001; violations: {violations}"
    )


def test_unledgered_fires_when_store_has_no_rows(tmp_path):
    """UNLEDGERED_EVALS fires when delegation reported evals but wrote
    no rows to the canonical store."""
    from unittest.mock import patch
    from f3dasm._src.agentic.instrumented import RunStateSummary

    ledger, dlog, mon, _ = make_world(tmp_path)
    h = propose(ledger)

    dlog.record(
        id="D001", from_node="strategizer", to_node="implementer",
        task="t", deliverable=REPORT, hypothesis_ids=[h],
        started_at="x", completed_at="y", status="DONE",
        evals=10,
    )

    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()
    mon.store_dir = str(store_dir)

    # Store is empty — summary returns None
    with patch.object(RunStateSummary, "from_store", return_value=None):
        violations = mon.evaluate()

    rules = {v.rule for v in violations}
    assert "UNLEDGERED_EVALS" in rules, (
        "UNLEDGERED_EVALS should fire when delegation reported evals "
        f"but store has no rows; violations: {violations}"
    )
