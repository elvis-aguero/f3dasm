"""The reported eval count == the guarded eval count == provenance-stamped rows
attributed to a delegation. D000 pool rows and unstamped rows are NOT evals.

Root cause (post-mortem 989e7daa): the headline count summed RAW rows
(len(df_out), incl D000 + unstamped) while the UNLEDGERED_EVALS guard summed
stamped-per-delegation rows — two different sets, never reconciled, so every
past fix patched a consumer and a new backdoor appeared. This pins the
unification: total_ledgered_evals reads the SAME set the guard reads.
"""
from __future__ import annotations

from pathlib import Path

from f3dasm._src.agentic.instrumented import (
    delegation_evals,
    total_ledgered_evals,
)
from f3dasm._src.design.domain import Domain
from f3dasm._src.experimentdata import ExperimentData
from f3dasm._src.experimentsample import ExperimentSample, JobStatus


def _build_store(store_dir: Path, rows) -> None:
    domain = Domain()
    domain.add_float("x0", 0.0, 10.0)
    for k in ("f", "_delegation_id", "_source", "_ts"):
        domain.add_output(k, exist_ok=True)
    samples = {}
    for i, (x0, f, did, src) in enumerate(rows):
        samples[i] = ExperimentSample(
            _input_data={"x0": x0},
            _output_data={"f": f, "_delegation_id": did, "_source": src,
                          "_ts": "2026-01-01T00:00:00+00:00"},
            job_status=JobStatus.FINISHED,
        )
    ExperimentData.from_data(data=samples, domain=domain).store(
        project_dir=store_dir)


def test_headline_count_excludes_d000_and_unstamped(tmp_path):
    store_root = tmp_path / "experiment_data"
    _build_store(store_root, [
        (0.1, 1.0, "D000", "precomputed_pool"),   # pool — ground truth, not evals
        (0.2, 2.0, "D000", "precomputed_pool"),
        (0.3, 3.0, "D001", "oracle"),             # real
        (0.4, 4.0, "D001", "oracle"),             # real
        (0.5, 5.0, "D002", "oracle"),             # real
        (0.6, 6.0, "", "rogue_append"),           # unstamped — not an eval
    ])
    # 3 real evals: D001 x2 + D002 x1. NOT the 6 raw rows.
    assert total_ledgered_evals(store_root) == 3


def test_headline_equals_guard_set(tmp_path):
    store_root = tmp_path / "experiment_data"
    _build_store(store_root, [
        (0.1, 1.0, "D000", "precomputed_pool"),
        (0.3, 3.0, "D001", "oracle"),
        (0.5, 5.0, "D002", "oracle"),
    ])
    # the headline count is exactly the sum of the guard's per-delegation counts
    guard_sum = delegation_evals(store_root, "D001") + delegation_evals(store_root, "D002")
    assert total_ledgered_evals(store_root) == guard_sum == 2
    # D000 is still bucketed for provenance, it just isn't an eval
    assert delegation_evals(store_root, "D000") == 1
