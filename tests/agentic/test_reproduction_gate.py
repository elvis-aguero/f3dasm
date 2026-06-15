"""Unit tests for StrategizerNode._reproduction_gate (§B of the pipeline
deliverable redesign).

The gate EXECUTES pipeline.py lazily against the canonical ledger and PASSES
(returns None) iff the script exits 0 (its own derive-from-ledger assert held)
AND adds ZERO new oracle rows (lazy: a reproduction must skip finished evals).
These tests drive each control-flow branch with real subprocess execution.
"""
from __future__ import annotations

from pathlib import Path

from f3dasm._src.agentic.backends.base import Agent, Edge, Graph
from f3dasm._src.agentic.instrumented import InstrumentedDataGenerator
from f3dasm._src.agentic.nodes import StrategizerNode
from f3dasm._src.core import DataGenerator
from f3dasm._src.experimentsample import ExperimentSample, JobStatus


class _StubAdapter:
    def __init__(self) -> None:
        self.closure_tools: dict = {}

    def invoke(self, messages):  # pragma: no cover - not exercised
        return "ok"


class _Sum(DataGenerator):
    def execute(self, s, **k):
        s._output_data["f"] = sum(s._input_data.values())
        s.job_status = JobStatus.FINISHED
        return s


def _spec() -> Graph:
    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "WriteNote"})
        description = "s"

    class B(Agent):
        description = "i"

    return Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),),
        entry="strategizer",
    )


def _seed_store(store_dir: Path, n: int = 2) -> None:
    """Write n provenance-stamped rows to store_dir (= the project_dir)."""
    gen = InstrumentedDataGenerator(
        inner=_Sum(), store_dir=store_dir, delegation_id="D001", flush_every=1)
    for i in range(n):
        gen.execute(ExperimentSample(
            _input_data={"x0": float(i)}, _output_data={},
            job_status=JobStatus.OPEN))
    gen.flush()


def _setup(tmp_path: Path):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    run_dir = tmp_path / "runs" / "T0"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
    _seed_store(run_dir / "experiment_data", n=2)  # before == 2 rows
    node = StrategizerNode(
        _StubAdapter(), name="strategizer", outgoing=["implementer"],
        spec=_spec(), study_dir=study_dir)
    node._current_notes_dir = run_dir / "debug" / "strategizer_notes"
    return node, study_dir


def test_gate_passes_for_clean_lazy_pipeline(tmp_path):
    node, study_dir = _setup(tmp_path)
    (study_dir / "pipeline.py").write_text("print('REPRODUCED: 1.0')\n")
    assert node._reproduction_gate({"study_dir": str(study_dir)}) is None


def test_gate_fails_for_erroring_pipeline(tmp_path):
    node, study_dir = _setup(tmp_path)
    (study_dir / "pipeline.py").write_text("import sys\nsys.exit(2)\n")
    problem = node._reproduction_gate({"study_dir": str(study_dir)})
    assert problem is not None and "failed" in problem.lower()


def test_gate_fails_when_pipeline_adds_evals(tmp_path):
    """A NON-lazy pipeline that re-evaluates stamps a new row → not lazy."""
    node, study_dir = _setup(tmp_path)
    (study_dir / "pipeline.py").write_text(
        "import os\n"
        "from f3dasm._src.agentic.instrumented import InstrumentedDataGenerator\n"
        "from f3dasm._src.core import DataGenerator\n"
        "from f3dasm._src.experimentsample import ExperimentSample, JobStatus\n"
        "class G(DataGenerator):\n"
        "    def execute(self, s, **k):\n"
        "        s._output_data['f'] = 1.0\n"
        "        s.job_status = JobStatus.FINISHED\n"
        "        return s\n"
        "store = os.environ['F3DASM_CANONICAL_STORE']\n"
        "g = InstrumentedDataGenerator(inner=G(), store_dir=store,\n"
        "                              delegation_id='D777', flush_every=1)\n"
        "g.execute(ExperimentSample(_input_data={'x0': 9.0}, _output_data={},\n"
        "                           job_status=JobStatus.OPEN))\n"
        "g.flush()\n"
    )
    problem = node._reproduction_gate({"study_dir": str(study_dir)})
    assert problem is not None and "lazy" in problem.lower()


def test_gate_skips_without_run_context(tmp_path):
    node, study_dir = _setup(tmp_path)
    (study_dir / "pipeline.py").write_text("print('ok')\n")
    node._current_notes_dir = None
    assert node._reproduction_gate({"study_dir": str(study_dir)}) is None
