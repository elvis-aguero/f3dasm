"""Namespace-aware eval accounting (fixes the run 20260626T231202 cascade).

Two namespace-blind defects, one root: per-namespace stores are isolated, but
the run-level eval accounting and the unledgered-evals guard both keyed on the
single canonical store.

1. `total_ledgered_evals` must sum rows across the canonical store AND every
   namespace store (so `evals_used` / the soft budget see all real evals).
2. `delegation_eval_store` must resolve a namespaced delegation's store, so the
   unledgered-evals guard checks the store the worker actually wrote to (not the
   canonical store, which falsely reads 0 → false off-ledger bounce → re-run
   thrash → duplicate rows).
"""
from __future__ import annotations

from pathlib import Path

from f3dasm._src.design.domain import Domain
from f3dasm._src.experimentdata import ExperimentData
from f3dasm._src.experimentsample import ExperimentSample, JobStatus


def _seed_store(store_dir: Path, n: int, delegation_id: str = "D001") -> None:
    """Write `n` finished rows to the ExperimentData project at store_dir."""
    dom = Domain()
    dom.add_float("x", 0.0, 1.0)
    dom.add_output("score", exist_ok=True)
    dom.add_output("_delegation_id", exist_ok=True)
    rows = {
        i: ExperimentSample(
            _input_data={"x": i / max(n, 1)},
            _output_data={"score": 0.5, "_delegation_id": delegation_id},
            job_status=JobStatus.FINISHED,
        )
        for i in range(n)
    }
    ExperimentData.from_data(data=rows, domain=dom).store(project_dir=store_dir)


def test_total_ledgered_evals_sums_across_namespaces(tmp_path):
    from f3dasm._src.agentic.instrumented import total_ledgered_evals

    # canonical store at <store>/experiment_data; namespace 'polar' at
    # <store>/polar/experiment_data — the on-disk layout get_evaluator uses.
    store = tmp_path / "experiment_data"
    _seed_store(store, 100)                 # canonical (cartesian)
    _seed_store(store / "polar", 100)       # namespace
    # Canonical-only count is the OLD (buggy) view.
    from f3dasm._src.agentic.instrumented import RunStateSummary
    assert RunStateSummary.from_store(store).n_rows == 100
    # Aggregate count sees both.
    assert total_ledgered_evals(store) == 200


def test_total_ledgered_evals_canonical_only_is_unchanged(tmp_path):
    """No namespaces → identical to the canonical row count (back-compat)."""
    from f3dasm._src.agentic.instrumented import total_ledgered_evals

    store = tmp_path / "experiment_data"
    _seed_store(store, 42)
    assert total_ledgered_evals(store) == 42


def test_delegation_eval_store_resolves_namespace(tmp_path):
    from f3dasm._src.agentic.nodes.parsing import delegation_eval_store

    exp = tmp_path / "experiment_data"
    # None → the canonical store; a namespace → its subdir.
    assert delegation_eval_store(exp, None) == exp
    assert delegation_eval_store(exp, "polar") == exp / "polar"
