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

def test_evidence_numbers_match(tmp_path):
    ledger, dlog, mon, _ = make_world(tmp_path)
    h_id = propose(ledger)
    record_done(dlog, "D001", [h_id], REPORT)
    # Update with wrong number (9.99 not in REPORT which has 1.47)
    ledger.update(
        h_id, "SUPPORTED", "found it",
        evidence={"delegation": "D001", "numbers": {"best_y": 9.99}},
        posterior=0.9, triggered_by=None)
    violations = mon.evaluate()
    rules = {v.rule for v in violations}
    assert "EVIDENCE_NUMBERS_MATCH" in rules, (
        f"Expected EVIDENCE_NUMBERS_MATCH, got: {rules}")

    # Tolerance check: 1.4700000001 should match 1.47
    assert mon._numbers_match({"best_y": 1.4700000001}, REPORT)
    # Mismatch check: 9.99 is not in REPORT
    assert not mon._numbers_match({"x": 9.99}, REPORT)


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
# Test 6: UNANCHORED_DELEGATION — delegation with no ### Numbers section
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


# ---------------------------------------------------------------------------
# Test 7: POSTERIOR_INERTIA — barely moved posterior triggers, big move doesn't
# ---------------------------------------------------------------------------

def test_posterior_inertia(tmp_path):
    ledger, dlog, mon, _ = make_world(tmp_path)
    h_id = propose(ledger)  # prior=0.5, initial log entry posterior=0.5
    record_done(dlog, "D001", [h_id], REPORT)
    # Close with posterior barely different from prior (0.52)
    ledger.update(
        h_id, "SUPPORTED", "barely moved",
        evidence={"delegation": "D001", "numbers": {"best_y": 1.47}},
        posterior=0.52, triggered_by=None)
    violations = mon.evaluate()
    rules = {v.rule for v in violations}
    assert "POSTERIOR_INERTIA" in rules, (
        f"Expected POSTERIOR_INERTIA for 0.5→0.52, got: {rules}")

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

def test_escalation_after_three_violations_one_hypothesis(tmp_path):
    """Three distinct rules accumulate in _h_rule_seen for one h_id.

    Phase 1: h_id OPEN + bare delegations → UNANCHORED_DELEGATION
    Phase 2: Close with tiny posterior shift → POSTERIOR_INERTIA
    Phase 3: SUPPORTED but no falsification → SUPPORTED_WITHOUT_ATTACK
    After 3 distinct rules for h_id → escalation_due() returns [h_id].
    """
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
