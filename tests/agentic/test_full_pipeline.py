"""End-to-end integration test: AgenticRun.execute() with mock adapters."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from f3dasm._src.agentic.agent_runtime import AgenticRun
from f3dasm._src.agentic.backends.base import Agent, Edge, Graph

from .fixtures import MockWorkerAdapter, ScriptedStrategistAdapter


PROBLEM_MD = """\
# Test Problem

Maximise sigma_crit for coilable designs.

## Design space
ratio_d: [0.004, 0.073]

## Objective
Find highest sigma_crit with coilable == 1.
"""


def _make_study(tmp_path: Path) -> Path:
    study = tmp_path / "study"
    study.mkdir()
    (study / "PROBLEM_STATEMENT.md").write_text(PROBLEM_MD)
    return study


class _StrategistSpec(Agent):
    role = "strategizer"
    description = "Test strategizer."
    tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote"})


class _WorkerSpec(Agent):
    description = "Test implementer."


def _graph_spec() -> Graph:
    return Graph(
        nodes={"strategizer": _StrategistSpec(), "implementer": _WorkerSpec()},
        edges=(Edge("strategizer", "implementer"),),
        entry="strategizer",
    )


@pytest.fixture
def pipeline_run(tmp_path):
    """AgenticRun with mock adapters; returns (run, study_dir) after execute()."""
    study = _make_study(tmp_path)
    strat = ScriptedStrategistAdapter()
    worker = MockWorkerAdapter()

    run = AgenticRun(study_dir=study, graph=_graph_spec())

    def _mock_make_adapter(name, agent):
        return strat if name == "strategizer" else worker

    run._make_adapter = _mock_make_adapter
    run.execute()
    return run, study


# ---------------------------------------------------------------------------
# Output directory structure
# ---------------------------------------------------------------------------


def test_run_dir_created(pipeline_run, tmp_path):
    """execute() creates runs/<timestamp>/ directory; solution.md at study root."""
    _, study = pipeline_run
    runs = list((study / "runs").iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / "debug" / "strategizer_notes").is_dir()
    assert (run_dir / "debug" / "run.log").exists()
    # solution.md is written to the study root, not inside run_dir
    assert (study / "solution.md").exists()


def test_workspace_subfolders_created(pipeline_run, tmp_path):
    """Worker delegations create debug/delegations/D001/ and D002/."""
    _, study = pipeline_run
    run_dir = next((study / "runs").iterdir())
    workspace = run_dir / "debug" / "delegations"
    assert workspace.is_dir()
    # Two delegations → two subfolders
    subfolders = [p.name for p in workspace.iterdir() if p.is_dir()]
    assert "D001" in subfolders
    assert "D002" in subfolders


# ---------------------------------------------------------------------------
# Hypothesis ledger
# ---------------------------------------------------------------------------


def test_hypotheses_json_created(pipeline_run):
    """hypotheses.json exists under strategizer_notes/."""
    _, study = pipeline_run
    notes = next((study / "runs").iterdir()) / "debug" / "strategizer_notes"
    assert (notes / "hypotheses.json").exists()


def test_hypotheses_json_has_two_entries(pipeline_run):
    """H1 and H2 are recorded in hypotheses.json."""
    _, study = pipeline_run
    notes = next((study / "runs").iterdir()) / "debug" / "strategizer_notes"
    data = json.loads((notes / "hypotheses.json").read_text())
    assert "H1" in data
    assert "H2" in data


def test_h1_falsified_h2_supported(pipeline_run):
    """H1 ends FALSIFIED and H2 ends SUPPORTED."""
    _, study = pipeline_run
    notes = next((study / "runs").iterdir()) / "debug" / "strategizer_notes"
    data = json.loads((notes / "hypotheses.json").read_text())
    assert data["H1"]["status_log"][-1]["status"] == "FALSIFIED"
    assert data["H2"]["status_log"][-1]["status"] == "SUPPORTED"


def test_triggered_by_links_delegation(pipeline_run):
    """H1's FALSIFIED entry has triggered_by = D001."""
    _, study = pipeline_run
    notes = next((study / "runs").iterdir()) / "debug" / "strategizer_notes"
    data = json.loads((notes / "hypotheses.json").read_text())
    falsified_entry = data["H1"]["status_log"][-1]
    assert falsified_entry["triggered_by"] == "D001"


# ---------------------------------------------------------------------------
# Delegation log
# ---------------------------------------------------------------------------


def test_delegation_log_created(pipeline_run):
    """delegation_log.jsonl exists under debug/ with two records."""
    _, study = pipeline_run
    debug = next((study / "runs").iterdir()) / "debug"
    lines = (debug / "delegation_log.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_delegation_records_have_token_fields(pipeline_run):
    """Each delegation record has non-zero tokens_in and tokens_out."""
    _, study = pipeline_run
    debug = next((study / "runs").iterdir()) / "debug"
    lines = (debug / "delegation_log.jsonl").read_text().strip().splitlines()
    for line in lines:
        rec = json.loads(line)
        assert rec["tokens_in"] > 0, f"tokens_in missing in {rec['id']}"
        assert rec["tokens_out"] > 0, f"tokens_out missing in {rec['id']}"
        assert rec["cost_usd"] is not None


def test_delegation_records_link_hypotheses(pipeline_run):
    """D001 links to H1 and D002 links to H2."""
    _, study = pipeline_run
    debug = next((study / "runs").iterdir()) / "debug"
    records = [
        json.loads(l)
        for l in (debug / "delegation_log.jsonl").read_text().strip().splitlines()
    ]
    by_id = {r["id"]: r for r in records}
    assert "H1" in by_id["D001"]["hypothesis_ids"]
    assert "H2" in by_id["D002"]["hypothesis_ids"]


# ---------------------------------------------------------------------------
# solution.md token table
# ---------------------------------------------------------------------------


def test_solution_md_has_token_table(pipeline_run):
    """solution.md contains a ## Token usage section with token counts."""
    _, study = pipeline_run
    text = (study / "solution.md").read_text()
    assert "## Token usage" in text
    assert "input_tokens" in text
    assert "output_tokens" in text
    assert "estimated_cost" in text


def test_solution_md_nonzero_tokens(pipeline_run):
    """Token counts in solution.md are non-zero (mock adapters return fake usage)."""
    _, study = pipeline_run
    text = (study / "solution.md").read_text()
    # Extract total_tokens row value
    m = re.search(r"\| total_tokens \| ([\d,]+) \|", text)
    assert m, "total_tokens row not found in solution.md"
    total = int(m.group(1).replace(",", ""))
    assert total > 0


# ---------------------------------------------------------------------------
# replicate.py provenance header
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_run_with_replicate(tmp_path):
    """Like pipeline_run but worker writes a replicate.py to its sandbox."""
    study = _make_study(tmp_path)
    strat = ScriptedStrategistAdapter()

    class ReplicateWritingWorker(MockWorkerAdapter):
        def invoke(self, messages):
            write = self.closure_tools.get("Write")
            if write is not None:
                write("results.csv", "ratio_d,sigma_crit\n0.01,120.5\n")
                write("replicate.py", "import f3dasm\nprint('hello')\n")
            return self._report

        def copy(self):
            fresh = ReplicateWritingWorker(self._report)
            fresh.closure_tools = dict(self.closure_tools)
            return fresh

    worker = ReplicateWritingWorker()
    run = AgenticRun(study_dir=study, graph=_graph_spec())

    def _mock_make_adapter(name, agent):
        return strat if name == "strategizer" else worker

    run._make_adapter = _mock_make_adapter
    run.execute()
    return run, study


def test_replicate_py_has_provenance_header(pipeline_run_with_replicate):
    """replicate.py at study root starts with the auto-generated provenance header."""
    _, study = pipeline_run_with_replicate
    replicate = study / "replicate.py"
    assert replicate.exists(), "replicate.py not written to study root"
    content = replicate.read_text(encoding="utf-8")
    assert content.startswith("# Auto-generated by agentic-f3dasm"), (
        f"replicate.py does not start with provenance header; got: {content[:100]!r}"
    )
    assert "# Timestamp :" in content
    assert "# Model     :" in content
    assert "# Run       :" in content


# ---------------------------------------------------------------------------
# run.log
# ---------------------------------------------------------------------------


def test_run_log_mentions_tokens(pipeline_run):
    """run.log final line includes token counts."""
    _, study = pipeline_run
    run_dir = next((study / "runs").iterdir())
    log_text = (run_dir / "debug" / "run.log").read_text()
    assert "Tokens" in log_text or "tokens" in log_text
