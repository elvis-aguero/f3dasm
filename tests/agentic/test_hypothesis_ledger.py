"""Tests for HypothesisLedger — written before implementation (TDD)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from f3dasm._src.agentic.hypothesis_ledger import HypothesisLedger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fresh_ledger(tmp_path: Path) -> HypothesisLedger:
    return HypothesisLedger(tmp_path)


# ---------------------------------------------------------------------------
# HypothesisPropose
# ---------------------------------------------------------------------------


def test_propose_creates_open_entry(tmp_path):
    ledger = fresh_ledger(tmp_path)
    h_id = ledger.propose("Thin longerons buckle first", "strategizer")
    entry = ledger.get(h_id)
    assert entry is not None
    assert entry["statement"] == "Thin longerons buckle first"
    assert entry["proposed_by"] == "strategizer"
    assert entry["status_log"][-1]["status"] == "OPEN"
    assert entry["status_log"][-1]["triggered_by"] is None


def test_propose_assigns_sequential_ids(tmp_path):
    ledger = fresh_ledger(tmp_path)
    h1 = ledger.propose("H1 statement", "strategizer")
    h2 = ledger.propose("H2 statement", "strategizer")
    h3 = ledger.propose("H3 statement", "strategizer")
    assert h1 == "H1"
    assert h2 == "H2"
    assert h3 == "H3"


def test_propose_rejects_fourth_when_three_open(tmp_path):
    ledger = fresh_ledger(tmp_path)
    ledger.propose("A", "strategizer")
    ledger.propose("B", "strategizer")
    ledger.propose("C", "strategizer")
    result = ledger.propose("D", "strategizer")
    assert result.startswith("ERROR:")
    assert ledger.get("H4") is None


def test_propose_allows_fourth_after_close(tmp_path):
    ledger = fresh_ledger(tmp_path)
    ledger.propose("A", "strategizer")
    ledger.propose("B", "strategizer")
    ledger.propose("C", "strategizer")
    ledger.update("H1", "FALSIFIED", "disproved", None)
    h4 = ledger.propose("D", "strategizer")
    assert h4 == "H4"
    assert not h4.startswith("ERROR:")


# ---------------------------------------------------------------------------
# HypothesisUpdate
# ---------------------------------------------------------------------------


def test_update_appends_to_status_log(tmp_path):
    ledger = fresh_ledger(tmp_path)
    h_id = ledger.propose("Statement", "strategizer")
    ledger.update(h_id, "FALSIFIED", "disproved by D001", "D001")
    entry = ledger.get(h_id)
    assert len(entry["status_log"]) == 2
    last = entry["status_log"][-1]
    assert last["status"] == "FALSIFIED"
    assert last["comment"] == "disproved by D001"


def test_update_injects_triggered_by(tmp_path):
    ledger = fresh_ledger(tmp_path)
    h_id = ledger.propose("Statement", "strategizer")
    ledger.update(h_id, "SUPPORTED", "confirmed", "D002")
    last = ledger.get(h_id)["status_log"][-1]
    assert last["triggered_by"] == "D002"


def test_update_triggered_by_nullable(tmp_path):
    ledger = fresh_ledger(tmp_path)
    h_id = ledger.propose("Statement", "strategizer")
    ledger.update(h_id, "INCONCLUSIVE", "ambiguous", None)
    last = ledger.get(h_id)["status_log"][-1]
    assert last["triggered_by"] is None


def test_update_rejects_invalid_status(tmp_path):
    ledger = fresh_ledger(tmp_path)
    h_id = ledger.propose("Statement", "strategizer")
    result = ledger.update(h_id, "BANANA", "bad", None)
    assert result.startswith("ERROR:")
    # status_log still has only the OPEN entry
    assert len(ledger.get(h_id)["status_log"]) == 1


def test_update_rejects_unknown_hypothesis_id(tmp_path):
    ledger = fresh_ledger(tmp_path)
    result = ledger.update("H99", "FALSIFIED", "nope", None)
    assert result.startswith("ERROR:")


def test_update_ts_is_present(tmp_path):
    ledger = fresh_ledger(tmp_path)
    h_id = ledger.propose("Statement", "strategizer")
    ledger.update(h_id, "SUPPORTED", "confirmed", "D001")
    last = ledger.get(h_id)["status_log"][-1]
    assert "ts" in last
    assert last["ts"]  # non-empty string


# ---------------------------------------------------------------------------
# HypothesisList
# ---------------------------------------------------------------------------


def test_list_returns_only_summary_fields(tmp_path):
    ledger = fresh_ledger(tmp_path)
    ledger.propose("First hypothesis", "strategizer")
    ledger.propose("Second hypothesis", "strategizer")
    items = ledger.list_all()
    assert len(items) == 2
    for item in items:
        assert set(item.keys()) == {"id", "statement", "current_status"}
        assert "status_log" not in item
        assert "proposed_by" not in item


def test_list_reflects_current_status(tmp_path):
    ledger = fresh_ledger(tmp_path)
    h_id = ledger.propose("To be falsified", "strategizer")
    ledger.update(h_id, "FALSIFIED", "gone", "D001")
    items = ledger.list_all()
    assert items[0]["current_status"] == "FALSIFIED"


def test_list_empty_when_no_hypotheses(tmp_path):
    ledger = fresh_ledger(tmp_path)
    assert ledger.list_all() == []


# ---------------------------------------------------------------------------
# HypothesisGet
# ---------------------------------------------------------------------------


def test_get_returns_full_entry(tmp_path):
    ledger = fresh_ledger(tmp_path)
    h_id = ledger.propose("Full entry", "D001")
    entry = ledger.get(h_id)
    required_keys = {"id", "statement", "proposed_by", "proposed_at", "status_log"}
    assert required_keys.issubset(set(entry.keys()))
    assert entry["proposed_by"] == "D001"


def test_get_returns_none_for_missing(tmp_path):
    ledger = fresh_ledger(tmp_path)
    assert ledger.get("H99") is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_persists_and_reloads_from_disk(tmp_path):
    ledger = fresh_ledger(tmp_path)
    h_id = ledger.propose("Persistent statement", "strategizer")
    ledger.update(h_id, "SUPPORTED", "confirmed", "D001")

    # new ledger instance reads same file
    ledger2 = HypothesisLedger(tmp_path)
    entry = ledger2.get(h_id)
    assert entry is not None
    assert entry["statement"] == "Persistent statement"
    assert entry["status_log"][-1]["status"] == "SUPPORTED"


def test_hypotheses_json_is_valid_json(tmp_path):
    ledger = fresh_ledger(tmp_path)
    ledger.propose("Test", "strategizer")
    raw = (tmp_path / "hypotheses.json").read_text()
    data = json.loads(raw)
    assert "H1" in data


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_propose_is_thread_safe(tmp_path):
    """Two threads propose concurrently; max 3 OPEN must be respected."""
    ledger = fresh_ledger(tmp_path)
    errors = []
    results = []
    lock = threading.Lock()

    def propose(stmt):
        r = ledger.propose(stmt, "strategizer")
        with lock:
            results.append(r)

    threads = [threading.Thread(target=propose, args=(f"H {i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    open_count = sum(
        1 for r in results if not r.startswith("ERROR:")
    )
    assert open_count <= 3
