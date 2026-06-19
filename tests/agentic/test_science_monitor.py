"""Tests for ScienceMonitor.

Only UNLEDGERED_EVALS remains as a runtime rule. Hard invariants
(EVIDENCE_DELEGATION_EXISTS, SUPPORTED_WITHOUT_ATTACK) have moved to
HypothesisUpdate in strategizer.py. Soft rules (STALE_OPEN,
UNANCHORED_DELEGATION, POSTERIOR_INERTIA, EVIDENCE_NUMBERS_MATCH) removed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from f3dasm._src.agentic.delegation_log import DelegationLog
from f3dasm._src.agentic.hypothesis_ledger import HypothesisLedger
from f3dasm._src.agentic.instrumented import RunStateSummary
from f3dasm._src.agentic.science_monitor import ScienceMonitor


def make_world(tmp_path):
    ledger = HypothesisLedger(tmp_path)
    dlog = DelegationLog(tmp_path / "delegation_log.jsonl")
    drift_records = []
    mon = ScienceMonitor(
        ledger, dlog, diagnostics_writer=drift_records.append)
    return ledger, dlog, mon, drift_records


REPORT = (
    "## Report\n\n### Actions taken\n- ran\n\n### Conclusions\nok\n\n"
    "### Numbers\nbest_y: 1.47\n"
)


# ---------------------------------------------------------------------------
# UNLEDGERED_EVALS — must NOT fire when store has rows
# ---------------------------------------------------------------------------

def test_unledgered_does_not_fire_when_store_has_rows(tmp_path):
    """False-positive guard: rule is silent when rows exist for delegation."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    ledger.propose("claim", "criterion", "pred", 0.5, proposed_by="s")
    dlog.record(
        id="D001", from_node="strategizer", to_node="implementer",
        task="t", deliverable=REPORT, hypothesis_ids=[],
        started_at="x", completed_at="y", status="DONE", evals=5)
    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()
    mon.store_dir = str(store_dir)
    stub = RunStateSummary(
        n_rows=5, n_per_delegation={"D001": 5},
        n_per_source={}, n_per_fidelity=None, output_stats={})
    with patch.object(RunStateSummary, "from_store", return_value=stub):
        violations = mon.evaluate()
    assert "UNLEDGERED_EVALS" not in {v.rule for v in violations}


def test_unledgered_fires_when_store_has_no_rows(tmp_path):
    """UNLEDGERED_EVALS fires when delegation reported evals but wrote nothing."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    ledger.propose("claim", "criterion", "pred", 0.5, proposed_by="s")
    dlog.record(
        id="D001", from_node="strategizer", to_node="implementer",
        task="t", deliverable=REPORT, hypothesis_ids=[],
        started_at="x", completed_at="y", status="DONE", evals=10)
    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()
    mon.store_dir = str(store_dir)
    with patch.object(RunStateSummary, "from_store", return_value=None):
        violations = mon.evaluate()
    assert "UNLEDGERED_EVALS" in {v.rule for v in violations}


def test_unledgered_does_not_fire_when_no_evals(tmp_path):
    """UNLEDGERED_EVALS must NOT fire for delegations that reported 0 evals."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    dlog.record(
        id="D001", from_node="strategizer", to_node="implementer",
        task="t", deliverable=REPORT, hypothesis_ids=[],
        started_at="x", completed_at="y", status="DONE", evals=0)
    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()
    mon.store_dir = str(store_dir)
    with patch.object(RunStateSummary, "from_store", return_value=None):
        violations = mon.evaluate()
    assert "UNLEDGERED_EVALS" not in {v.rule for v in violations}


# ---------------------------------------------------------------------------
# drain() basics
# ---------------------------------------------------------------------------

def test_drain_returns_empty_when_healthy(tmp_path):
    """drain() returns empty string when no violations fire."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    ledger.propose("claim", "criterion", "pred", 0.5, proposed_by="s")
    dlog.record(
        id="D001", from_node="strategizer", to_node="implementer",
        task="t", deliverable=REPORT, hypothesis_ids=[],
        started_at="x", completed_at="y", status="DONE")
    assert mon.drain() == ""


def test_drain_formats_unledgered_violation(tmp_path):
    """drain() formats the UNLEDGERED_EVALS violation into an injected message."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    dlog.record(
        id="D001", from_node="strategizer", to_node="implementer",
        task="t", deliverable=REPORT, hypothesis_ids=[],
        started_at="x", completed_at="y", status="DONE", evals=5)
    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()
    mon.store_dir = str(store_dir)
    with patch.object(RunStateSummary, "from_store", return_value=None):
        result = mon.drain()
    assert "UNLEDGERED_EVALS" in result
    assert "D001" in result


def test_drain_caps_output_to_max_inject(tmp_path):
    """drain() caps at max_inject=2 full lines + a digest line for the rest."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()
    mon.store_dir = str(store_dir)
    for i in range(1, 4):
        dlog.record(
            id=f"D00{i}", from_node="strategizer", to_node="implementer",
            task="t", deliverable=REPORT, hypothesis_ids=[],
            started_at="x", completed_at="y", status="DONE", evals=5)
    # 3 unledgered delegations → 3 distinct violation keys
    with patch.object(RunStateSummary, "from_store", return_value=None):
        result = mon.drain()
    full_lines = [
        ln for ln in result.splitlines()
        if ln.startswith("[SCIENCE MONITOR — ")]
    assert len(full_lines) == mon._max_inject
    digest_lines = [
        ln for ln in result.splitlines()
        if ln.startswith("[SCIENCE MONITOR] +")]
    assert len(digest_lines) == 1


def test_diagnostics_written_once_per_live_violation(tmp_path):
    """diagnostics_writer called once per new violation, not on re-drain."""
    ledger, dlog, mon, drift_records = make_world(tmp_path)
    dlog.record(
        id="D001", from_node="strategizer", to_node="implementer",
        task="t", deliverable=REPORT, hypothesis_ids=[],
        started_at="x", completed_at="y", status="DONE", evals=5)
    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()
    mon.store_dir = str(store_dir)
    with patch.object(RunStateSummary, "from_store", return_value=None):
        mon.drain()
        count1 = sum(1 for r in drift_records if r["rule"] == "UNLEDGERED_EVALS")
        assert count1 == 1
        mon.drain()
        count2 = sum(1 for r in drift_records if r["rule"] == "UNLEDGERED_EVALS")
        assert count2 == 1, "should not re-log a still-live violation"


# ---------------------------------------------------------------------------
# Removed rules must NOT fire (regression guard)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", [
    "EVIDENCE_NUMBERS_MATCH",
    "STALE_OPEN",
    "UNANCHORED_DELEGATION",
    "POSTERIOR_INERTIA",
])
def test_removed_rule_does_not_fire(tmp_path, rule):
    """Removed rules must never appear in evaluate()."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    h = ledger.propose("claim", "criterion", "pred", 0.5, proposed_by="s")
    bare = "## Report\n\n### Conclusions\nno numbers.\n"
    for i in range(1, 5):
        dlog.record(
            id=f"D00{i}", from_node="strategizer", to_node="implementer",
            task="t", deliverable=bare, hypothesis_ids=[h],
            started_at="x", completed_at="y", status="DONE")
    violations = mon.evaluate()
    fired = {v.rule for v in violations}
    assert rule not in fired, f"{rule} was expected removed but fired: {fired}"
