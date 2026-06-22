"""Headless tests for the notebook-manipulation tools added so the strategizer
can run/edit/inspect/delete deliverable cells (not just append via
AddPipelineCell): EditPipelineCell, DeletePipelineCell, RunScratch.
"""
from __future__ import annotations

from pathlib import Path

import nbformat

from f3dasm._src.agentic.backends.base import Agent, Edge, Graph
from f3dasm._src.agentic.nodes import StrategizerNode


class _Stub:
    def __init__(self) -> None:
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""

    def copy(self):
        s = self.__class__.__new__(self.__class__)
        _Stub.__init__(s)
        s.closure_tools = dict(self.closure_tools)
        return s


_NB_TOOLS = frozenset({
    "SetNotebookIntro", "AddPipelineCell", "EditPipelineCell",
    "DeletePipelineCell", "RunScratch",
})


def _node(tmp_path):
    class A(Agent):
        role = "strategizer"
        tools = _NB_TOOLS
        description = "strategizer"

    class B(Agent):
        role = "implementer"
        description = "implementer"

    spec = Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),),
        entry="strategizer",
    )
    n = StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()}, study_dir=str(tmp_path),
    )
    return n


def _tool(n, name):
    return n.adapter.closure_tools[name]


def _read_nb(tmp_path):
    return nbformat.read(str(tmp_path / "pipeline.ipynb"), as_version=4)


def _cell(nb, name):
    for c in nb.cells:
        if (c.get("metadata", {}) or {}).get("name") == name:
            return c
    return None


# ── EditPipelineCell ─────────────────────────────────────────────────────────

def test_edit_patches_code_without_touching_why(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="original rationale", code="x = 1")
    out = _tool(n, "EditPipelineCell")("doe", code="x = 2")
    assert "Edited doe" in out
    nb = _read_nb(tmp_path)
    assert _cell(nb, "doe")["source"] == "x = 2"
    assert "original rationale" in _cell(nb, "doe__why")["source"]


def test_edit_patches_why_without_touching_code(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("ml", why="old why", code="fit()")
    _tool(n, "EditPipelineCell")("ml", why="new why")
    nb = _read_nb(tmp_path)
    assert _cell(nb, "ml")["source"] == "fit()"
    assert "new why" in _cell(nb, "ml__why")["source"]


def test_edit_missing_phase_errors(tmp_path):
    n = _node(tmp_path)
    out = _tool(n, "EditPipelineCell")("analysis", code="y = 1")
    assert "not in pipeline.ipynb" in out


def test_edit_nothing_to_change_errors(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="w", code="c")
    out = _tool(n, "EditPipelineCell")("doe")
    assert "nothing to edit" in out.lower()


# ── DeletePipelineCell ───────────────────────────────────────────────────────

def test_delete_removes_cell_and_why(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="w", code="c")
    _tool(n, "AddPipelineCell")("ml", why="w2", code="c2")
    out = _tool(n, "DeletePipelineCell")("ml")
    assert "Deleted ml" in out
    nb = _read_nb(tmp_path)
    assert _cell(nb, "ml") is None
    assert _cell(nb, "ml__why") is None
    # doe survives
    assert _cell(nb, "doe") is not None


def test_delete_absent_phase_is_noop(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="w", code="c")
    out = _tool(n, "DeletePipelineCell")("analysis")
    assert "Nothing to delete" in out


# ── RunScratch ───────────────────────────────────────────────────────────────

def test_run_scratch_executes_and_returns_stdout(tmp_path):
    n = _node(tmp_path)
    # RunScratch needs a run context (notes dir → run_dir/experiment_data).
    notes = tmp_path / "runs" / "T1" / "debug" / "strategizer_notes"
    notes.mkdir(parents=True)
    n._current_notes_dir = notes
    out = _tool(n, "RunScratch")("print(6 * 7)")
    assert "exit 0" in out
    assert "42" in out


def test_run_scratch_surfaces_errors(tmp_path):
    n = _node(tmp_path)
    notes = tmp_path / "runs" / "T1" / "debug" / "strategizer_notes"
    notes.mkdir(parents=True)
    n._current_notes_dir = notes
    out = _tool(n, "RunScratch")("raise ValueError('boom')")
    assert "exit 1" in out
    assert "boom" in out


def test_run_scratch_empty_code_errors(tmp_path):
    n = _node(tmp_path)
    out = _tool(n, "RunScratch")("   ")
    assert "ERROR" in out
