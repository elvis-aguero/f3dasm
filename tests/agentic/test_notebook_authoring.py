"""Structured notebook-authoring closures (SetNotebookIntro / AddPipelineCell).

These make the four-pillar structure + the WHY-explainer UNFORGEABLE: the pillar
name and the rationale are required arguments, so the agent cannot author a
structureless notebook. They replace the live Jupyter MCP server (ripped out).
"""
from __future__ import annotations

import nbformat
import pytest

from f3dasm._src.agentic.backends.base import Agent, Edge, Graph
from f3dasm._src.agentic.nodes import StrategizerNode


class _Stub:
    def __init__(self):
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""


def _node(study_dir):
    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "SetNotebookIntro", "AddPipelineCell"})
        description = "strategizer"

    class B(Agent):
        role = "implementer"
        description = "implementer"

    spec = Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),), entry="strategizer",
    )
    return StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()}, study_dir=str(study_dir),
    )


def _read(study_dir):
    return nbformat.read(str(study_dir / "pipeline.ipynb"), as_version=4)


def _named(nb):
    return {c.metadata.get("name"): c for c in nb.cells if c.metadata.get("name")}


def test_closures_present_only_when_declared(tmp_path):
    n = _node(tmp_path)
    assert "SetNotebookIntro" in n.adapter.closure_tools
    assert "AddPipelineCell" in n.adapter.closure_tools


def test_set_intro_then_add_pillars_canonical_order(tmp_path):
    n = _node(tmp_path)
    tools = n.adapter.closure_tools
    # Add pillars OUT of order — the notebook must still come out canonical.
    tools["AddPipelineCell"]("analysis", "derive headline", "print('REPRODUCED: 1.0')")
    tools["AddPipelineCell"]("doe", "LHS over the box", "domain = ...; sampler = ...")
    tools["SetNotebookIntro"]("minimise f over the 3-box.", "H1: ... H2: ...")
    tools["AddPipelineCell"]("data_generation", "evaluate via get_evaluator", "data = ...")

    nb = _read(tmp_path)
    names = [c.metadata.get("name") for c in nb.cells if c.metadata.get("name")]
    # canonical: problem, hypotheses, then doe(+why), data_generation(+why),
    # ... analysis(+why) — phases present appear in pillar order regardless of
    # the call order above.
    assert names == [
        "problem", "hypotheses",
        "doe__why", "doe",
        "data_generation__why", "data_generation",
        "analysis__why", "analysis",
    ]
    # code cell carries name + tag metadata (machine-checkable pillar presence)
    by = _named(nb)
    assert by["doe"].cell_type == "code"
    assert by["doe"].metadata.get("tags") == ["doe"]
    assert by["doe__why"].cell_type == "markdown"


def test_add_pillar_replaces_on_recall(tmp_path):
    n = _node(tmp_path)
    tools = n.adapter.closure_tools
    tools["AddPipelineCell"]("doe", "first", "v = 1")
    tools["AddPipelineCell"]("doe", "second", "v = 2")
    nb = _read(tmp_path)
    does = [c for c in nb.cells if c.metadata.get("name") == "doe"]
    assert len(does) == 1 and "v = 2" in does[0].source


def test_add_pillar_rejects_unknown_phase(tmp_path):
    n = _node(tmp_path)
    out = n.adapter.closure_tools["AddPipelineCell"]("surrogate", "why", "code")
    assert out.startswith("ERROR:") and "phase must be one of" in out
    assert not (tmp_path / "pipeline.ipynb").exists()


def test_add_pillar_requires_why_and_code(tmp_path):
    n = _node(tmp_path)
    tools = n.adapter.closure_tools
    assert tools["AddPipelineCell"]("doe", "  ", "code").startswith("ERROR:")
    assert tools["AddPipelineCell"]("doe", "why", "").startswith("ERROR:")


def test_intro_recall_replaces_not_duplicates(tmp_path):
    n = _node(tmp_path)
    tools = n.adapter.closure_tools
    tools["SetNotebookIntro"]("first problem", "h")
    tools["SetNotebookIntro"]("second problem", "h2")
    nb = _read(tmp_path)
    problems = [c for c in nb.cells if c.metadata.get("name") == "problem"]
    assert len(problems) == 1 and "second problem" in problems[0].source


def test_authored_notebook_passes_the_gate(tmp_path):
    """A notebook authored purely through the closures runs through the
    reproduction gate (a real, executable deliverable)."""
    run_dir = tmp_path / "runs" / "A0"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
    (run_dir / "experiment_data").mkdir()
    n = _node(tmp_path)
    n._study_dir = tmp_path
    n._current_notes_dir = run_dir / "debug" / "strategizer_notes"
    tools = n.adapter.closure_tools
    tools["SetNotebookIntro"]("minimise f", "H1")
    tools["AddPipelineCell"]("analysis", "derive", "print('REPRODUCED: 1.0')")
    assert n._reproduction_gate({"study_dir": str(tmp_path)}) is None
