"""ROOT 1 fix: the canonical agentic ledger must be un-clobberable.

A protected store (marked with PROTECTED_STORE_SENTINEL by the agentic runtime)
must refuse any ExperimentData.store() that would SHRINK it — so a stray agent
`.store(canonical_dir)` on a partial table cannot destroy the metered rows that
get_evaluator() accumulated (the v9 360→100 loss). Unprotected stores keep their
normal overwrite behavior (no blast radius for non-agentic users).
"""
from __future__ import annotations

import pytest

from f3dasm._src._io import PROTECTED_STORE_SENTINEL
from f3dasm._src.design.domain import Domain
from f3dasm._src.experimentdata import ExperimentData
from f3dasm._src.experimentsample import ExperimentSample, JobStatus


def _ed(n: int) -> ExperimentData:
    domain = Domain()
    domain.add_float("a", 0.0, 100.0)
    domain.add_output("z", exist_ok=True)
    rows = {
        i: ExperimentSample(
            _input_data={"a": float(i)},
            _output_data={"z": float(-i)},
            job_status=JobStatus.FINISHED,
        )
        for i in range(n)
    }
    return ExperimentData.from_data(data=rows, domain=domain)


def test_protected_store_refuses_shrinking_write(tmp_path):
    store = tmp_path / "canonical"
    store.mkdir()
    (store / PROTECTED_STORE_SENTINEL).touch()
    _ed(5).store(project_dir=store)                       # 0 → 5 (growth) OK
    with pytest.raises(RuntimeError, match="PROTECTED"):  # 5 → 2 would clobber
        _ed(2).store(project_dir=store)
    # the ledger was NOT truncated
    assert len(ExperimentData.from_file(project_dir=store)) == 5


def test_protected_store_allows_equal_and_growth(tmp_path):
    store = tmp_path / "canonical"
    store.mkdir()
    (store / PROTECTED_STORE_SENTINEL).touch()
    _ed(3).store(project_dir=store)
    _ed(3).store(project_dir=store)   # equal count (e.g. a full re-store) → OK
    _ed(6).store(project_dir=store)   # superset (instrumented merge) → OK
    assert len(ExperimentData.from_file(project_dir=store)) == 6


def test_unprotected_store_truncates_normally(tmp_path):
    """No sentinel → core behavior unchanged (overwrite, no guard)."""
    store = tmp_path / "scratch"
    store.mkdir()
    _ed(5).store(project_dir=store)
    _ed(2).store(project_dir=store)   # would-shrink, but unprotected → allowed
    assert len(ExperimentData.from_file(project_dir=store)) == 2
