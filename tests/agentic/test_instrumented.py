"""Tests for InstrumentedDataGenerator and get_evaluator.

TDD: these tests were written BEFORE the implementation.
Run with:
    uv run pytest tests/agentic/test_instrumented.py -v --no-cov
"""
from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from pathlib import Path

import pytest

from f3dasm._src.core import DataGenerator, datagenerator
from f3dasm._src.design.domain import Domain
from f3dasm._src.experimentdata import ExperimentData
from f3dasm._src.experimentsample import ExperimentSample, JobStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_domain() -> Domain:
    d = Domain()
    d.add_float("x0", 0.0, 1.0)
    return d


def _make_sample(x0: float = 0.5) -> ExperimentSample:
    return ExperimentSample(
        _input_data={"x0": x0},
        _output_data={},
        job_status=JobStatus.OPEN,
    )


class _SumGenerator(DataGenerator):
    """Trivial DataGenerator: f = sum of inputs."""

    def execute(
        self, experiment_sample: ExperimentSample, **kwargs
    ) -> ExperimentSample:
        val = sum(experiment_sample._input_data.values())
        experiment_sample._output_data["f"] = val
        experiment_sample.job_status = JobStatus.FINISHED
        return experiment_sample


# ---------------------------------------------------------------------------
# 1. Provenance stamping
# ---------------------------------------------------------------------------


def test_execute_stamps_provenance(tmp_path):
    from f3dasm._src.agentic.instrumented import InstrumentedDataGenerator

    gen = InstrumentedDataGenerator(
        inner=_SumGenerator(),
        store_dir=tmp_path,
        delegation_id="D001",
        source="test_source",
        flush_every=1,
    )
    sample = _make_sample(0.3)
    gen.execute(sample)

    data = ExperimentData.from_file(project_dir=tmp_path)
    df_in, df_out = data.to_pandas()

    assert "_delegation_id" in df_out.columns, df_out.columns.tolist()
    assert "_source" in df_out.columns, df_out.columns.tolist()
    assert "_ts" in df_out.columns, df_out.columns.tolist()
    assert "f" in df_out.columns, df_out.columns.tolist()

    row = df_out.iloc[0]
    assert row["_delegation_id"] == "D001"
    assert row["_source"] == "test_source"
    # _ts should be a non-empty string
    assert isinstance(row["_ts"], str) and len(row["_ts"]) > 0


def test_to_numpy_excludes_underscore_provenance(tmp_path):
    """All provenance columns are now underscore-prefixed (_delegation_id,
    _source, _ts), so core to_numpy() drops them and returns a clean numeric
    array instead of an object-dtype array contaminated by the metadata."""
    import numpy as np
    from f3dasm._src.agentic.instrumented import InstrumentedDataGenerator

    gen = InstrumentedDataGenerator(
        inner=_SumGenerator(),
        store_dir=tmp_path,
        delegation_id="D001",
        source="test_source",
        flush_every=1,
    )
    gen.execute(_make_sample(0.3))

    data = ExperimentData.from_file(project_dir=tmp_path)
    _, out_arr = data.to_numpy()

    # only the real output "f" survives → numeric, not object dtype
    assert out_arr.shape[1] == 1, out_arr
    assert np.issubdtype(out_arr.dtype, np.floating), out_arr.dtype


# ---------------------------------------------------------------------------
# 2. Concurrent appends — no rows lost
# ---------------------------------------------------------------------------


def _worker(store_dir, delegation_id, n_samples, lock_path):
    """Run in a thread; each gets its own InstrumentedDataGenerator."""
    from f3dasm._src.agentic.instrumented import InstrumentedDataGenerator

    gen = InstrumentedDataGenerator(
        inner=_SumGenerator(),
        store_dir=store_dir,
        delegation_id=delegation_id,
        source="concurrent_test",
        lock_path=lock_path,
        flush_every=1,
    )
    for i in range(n_samples):
        gen.execute(_make_sample(float(i) * 0.05))
    gen.flush()


def test_concurrent_appends_no_loss(tmp_path):
    lock_path = tmp_path / "experiment_data" / ".lock"
    delegation_ids = ["D001", "D002", "D003"]
    K = 10  # samples per delegation

    threads = [
        threading.Thread(
            target=_worker,
            args=(tmp_path, did, K, lock_path),
        )
        for did in delegation_ids
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = ExperimentData.from_file(project_dir=tmp_path)
    df_in, df_out = data.to_pandas()

    total = len(df_out)
    assert total == len(delegation_ids) * K, (
        f"Expected {len(delegation_ids) * K} rows, got {total}"
    )

    for did in delegation_ids:
        count = (df_out["_delegation_id"] == did).sum()
        assert count == K, (
            f"Expected {K} rows for {did}, got {count}"
        )


# ---------------------------------------------------------------------------
# 4. Provenance survives + reindex
# ---------------------------------------------------------------------------


def test_provenance_survives_plus_reindex(tmp_path):
    from f3dasm._src.agentic.instrumented import InstrumentedDataGenerator

    lock_path = tmp_path / "experiment_data" / ".lock"

    gen1 = InstrumentedDataGenerator(
        inner=_SumGenerator(),
        store_dir=tmp_path,
        delegation_id="D001",
        source="s1",
        lock_path=lock_path,
        flush_every=2,
    )
    gen2 = InstrumentedDataGenerator(
        inner=_SumGenerator(),
        store_dir=tmp_path,
        delegation_id="D002",
        source="s2",
        lock_path=lock_path,
        flush_every=2,
    )

    gen1.execute(_make_sample(0.1))
    gen1.execute(_make_sample(0.2))  # triggers flush

    gen2.execute(_make_sample(0.3))
    gen2.execute(_make_sample(0.4))  # triggers flush

    data = ExperimentData.from_file(project_dir=tmp_path)
    _, df_out = data.to_pandas()

    assert len(df_out) == 4
    d1_rows = df_out[df_out["_delegation_id"] == "D001"]
    d2_rows = df_out[df_out["_delegation_id"] == "D002"]
    assert len(d1_rows) == 2
    assert len(d2_rows) == 2


# ---------------------------------------------------------------------------
# 5. get_evaluator binds delegation_id from cwd
# ---------------------------------------------------------------------------


def test_get_evaluator_binds_delegation_id_from_cwd(
    tmp_path, monkeypatch
):
    from f3dasm._src.agentic.instrumented import get_evaluator

    # Build the workspace: .../runs/ts/debug/delegations/D007
    debug_dir = tmp_path / "runs" / "ts" / "debug"
    delegation_dir = debug_dir / "delegations" / "D007"
    delegation_dir.mkdir(parents=True)

    store_dir = tmp_path / "store"
    store_dir.mkdir()

    run_config = {
        "store_dir": str(store_dir),
        "lock_path": str(store_dir / "experiment_data" / ".lock"),
        "source": "test_eval",
        "evaluator_name": "test_eval",
        "fidelity_column": None,
        "evaluator_entrypoint": None,
    }
    run_config_path = debug_dir / "run_config.json"
    run_config_path.write_text(json.dumps(run_config))

    monkeypatch.chdir(delegation_dir)
    monkeypatch.delenv("F3DASM_DELEGATION_ID", raising=False)

    gen = get_evaluator(inner=_SumGenerator())
    assert gen.delegation_id == "D007"


# ---------------------------------------------------------------------------
# 6. get_evaluator raises outside delegation workspace
# ---------------------------------------------------------------------------


def test_get_evaluator_raises_outside_delegation(
    tmp_path, monkeypatch
):
    from f3dasm._src.agentic.instrumented import get_evaluator

    # cwd is not a D### directory
    bad_dir = tmp_path / "not_a_delegation"
    bad_dir.mkdir()
    monkeypatch.chdir(bad_dir)
    monkeypatch.delenv("F3DASM_DELEGATION_ID", raising=False)

    with pytest.raises(ValueError, match="get_evaluator"):
        get_evaluator(inner=_SumGenerator())


# ---------------------------------------------------------------------------
# 7. flush_every batching
# ---------------------------------------------------------------------------


def test_flush_every_batches(tmp_path):
    from f3dasm._src.agentic.instrumented import InstrumentedDataGenerator

    gen = InstrumentedDataGenerator(
        inner=_SumGenerator(),
        store_dir=tmp_path,
        delegation_id="D001",
        source="batch_test",
        flush_every=5,
    )

    for i in range(4):
        gen.execute(_make_sample(float(i) * 0.1))

    # Store should not exist yet (or be empty if dir exists)
    exp_data_dir = tmp_path / "experiment_data"
    if exp_data_dir.exists():
        # If store was written it would have CSV files
        input_csv = exp_data_dir / "input_data.csv"
        output_csv = exp_data_dir / "output_data.csv"
        if output_csv.exists():
            import pandas as pd
            df = pd.read_csv(output_csv, index_col=0)
            assert len(df) == 0, (
                f"Expected 0 rows before flush_every=5 triggered, "
                f"got {len(df)}"
            )

    # 5th execute triggers flush
    gen.execute(_make_sample(0.4))

    data = ExperimentData.from_file(project_dir=tmp_path)
    _, df_out = data.to_pandas()
    assert len(df_out) == 5, f"Expected 5 rows after flush, got {len(df_out)}"


# ---------------------------------------------------------------------------
# 8. Public API: importable from f3dasm.agentic
# ---------------------------------------------------------------------------


def test_public_api_importable():
    from f3dasm.agentic import (  # noqa: F401
        InstrumentedDataGenerator,
        get_evaluator,
    )


def test_store_rows_accumulate_across_generator_instances(tmp_path):
    """Two generators in the SAME delegation accumulate rows in store.

    Observed live: a worker built one generator per phase; the store
    accumulated rows (600) correctly while a counter undercounted (300).
    The store is now the single source of truth for eval counts.
    """
    from f3dasm._src.agentic.instrumented import (
        InstrumentedDataGenerator,
        RunStateSummary,
    )

    store_dir = tmp_path / "store"
    store_dir.mkdir()

    def make_gen():
        @datagenerator(output_names=["f"])
        def inner(**kw):
            return float(sum(kw.values()))

        return InstrumentedDataGenerator(
            inner, store_dir, "D001",
            source="s", flush_every=1,
        )

    g1 = make_gen()
    g1.execute(_make_sample(0.1))
    g1.execute(_make_sample(0.2))

    g2 = make_gen()  # new instance, same delegation
    g2.execute(_make_sample(0.3))

    summary = RunStateSummary.from_store(store_dir)
    assert summary is not None
    # Store accumulates across both generator instances
    assert summary.n_per_delegation.get("D001", 0) == 3
