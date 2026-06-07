"""End-to-end seam test: real _init_canonical_store consumed by real
get_evaluator from a simulated worker cwd, through append + counter +
the runtime's eval-resolution helper. No LLM, deterministic.
"""

from __future__ import annotations

import os
from pathlib import Path

from f3dasm import ExperimentData, datagenerator
from f3dasm._src.samplers import RandomUniform
from f3dasm.design import Domain
from f3dasm._src.agentic.agent_runtime import _init_canonical_store
from f3dasm._src.agentic.instrumented import get_evaluator
from f3dasm._src.agentic.nodes import _resolve_delegation_evals


def test_runtime_config_to_store_roundtrip(tmp_path, monkeypatch):
    # 1. runtime sets up the canonical store + run_config sidecar
    run_dir = tmp_path / "runs" / "T0"
    (run_dir / "debug" / "delegations" / "D001").mkdir(parents=True)
    study_dir = tmp_path / "agentic_demo_study"
    study_dir.mkdir()
    cfg = _init_canonical_store(run_dir, study_dir)

    # 2. a worker, cwd == its delegation folder, obtains the evaluator
    monkeypatch.chdir(run_dir / "debug" / "delegations" / "D001")

    @datagenerator(output_names=["f"])
    def black_box(**kw):
        return float(sum(kw.values()))

    gen = get_evaluator(inner=black_box)
    assert gen.delegation_id == "D001"
    assert gen.source == "agentic_demo_study"  # evaluator_name → source

    # 3. evaluate through the instrumented path
    d = Domain()
    d.add_float("x", low=0.0, high=1.0)
    d.add_float("y", low=0.0, high=1.0)
    data = ExperimentData(domain=d)
    data = RandomUniform(seed=0).call(data, n_samples=4)
    data = gen.call(data, mode="sequential")
    gen.flush()

    # 4. canonical store has the rows with provenance
    canon = ExperimentData.from_file(project_dir=cfg["store_dir"])
    _, out = canon.to_pandas()
    assert len(out) == 4
    assert (out["_delegation_id"] == "D001").all()
    assert (out["source"] == "agentic_demo_study").all()
    assert out["_ts"].notna().all()
    assert "f" in out.columns

    # 5. mechanical counter file written, and the runtime helper reads it
    counter_dir = Path(cfg["counter_dir"])
    assert int((counter_dir / "D001.count").read_text()) == 4
    # honor-system "reported" value is overridden by the mechanical count
    assert _resolve_delegation_evals(counter_dir, "D001", reported=999) == 4
