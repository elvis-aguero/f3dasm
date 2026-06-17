"""Wet integration test for LiteratureReviewAgent.

Runs a real delegation with actual LLM + arxiv MCP tools.
Marked `integration` — not run in CI, requires claude-agent-sdk + network.

Research question: Gaussian Process surrogate modeling for high-dimensional
design-of-experiments — directly relevant to f3dasm's core use case.

Run with:
    uv run pytest tests/agentic/test_literature_wet.py -v -s --no-cov
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from f3dasm._src.agentic.agent_runtime import AgenticRun
from f3dasm._src.agentic.agents import LiteratureReviewAgent, StrategizerAgent
from f3dasm._src.agentic.backends.base import Edge, Graph


# ---------------------------------------------------------------------------
# Research question and study setup
# ---------------------------------------------------------------------------

PROBLEM_STATEMENT = """\
# Research Problem: Surrogate Modelling for Design-of-Experiments

## Background
We are running a design-of-experiments campaign for a structural mechanics
optimisation problem. The design space is three-dimensional (continuous) and
evaluations are expensive (FEM simulations). We want to guide the search
using a surrogate model.

## Research questions for literature review
1. What Gaussian Process surrogate modelling strategies have been proposed
   for high-dimensional or expensive design-of-experiments problems?
2. What acquisition functions are recommended when the objective landscape
   has multiple local optima?
3. Has any prior work combined GP surrogates with physics-based constraints
   or coilability-type feasibility filters?

## Scope
Focus on methods published since 2015. Prefer papers with open-source
implementations or reproducible benchmarks.
"""

DELEGATION_TASK = """\
Answer the following research questions from primary literature.
Cite exact passages with page and paragraph references.
Do NOT answer from memory — every claim must come from a paper in the corpus.

Build a corpus of AT MOST 4 full-text papers (the most relevant ones).
Stop adding papers once 4 are in the corpus.
Prefer arXiv/open-access PDFs.

Questions:
1. What Gaussian Process surrogate modelling strategies have been proposed
   for high-dimensional or expensive design-of-experiments problems?
2. What acquisition functions are recommended when the objective landscape
   has multiple local optima?
3. Has any prior work combined GP surrogates with physics-based constraints
   or feasibility filters?
"""


@pytest.mark.integration
def test_literature_review_wet(tmp_path):
    """End-to-end wet test: LiteratureReviewAgent with real LLM + arxiv MCP."""

    # --- Study setup -------------------------------------------------------
    study = tmp_path / "study"
    study.mkdir()
    (study / "PROBLEM_STATEMENT.md").write_text(PROBLEM_STATEMENT)

    # --- Scripted strategist -----------------------------------------------
    # Delegates once to literature_reviewer then calls Done().
    # Does NOT use two-shot Done() guard since no critic in this graph.
    from tests.agentic.test_nodes import StubAdapter
    import re, time as _time

    class LitReviewStrategist(StubAdapter):
        def invoke(self, messages):
            # Propose a hypothesis to satisfy the ledger (if active).
            # On re-invocation (graph loop), reuse the existing open
            # hypothesis instead of proposing duplicates until the
            # MAX_OPEN cap rejects and the ID fallback loops forever.
            h_id = None
            if "HypothesisPropose" in self.closure_tools:
                listing = self.closure_tools["HypothesisList"]()
                m_existing = re.search(r"\bH\d+\b", listing or "")
                if m_existing:
                    h_id = m_existing.group()
                else:
                    h_id = self.closure_tools["HypothesisPropose"](
                        statement="GP surrogates with acquisition"
                                  " functions are the dominant approach"
                                  " for expensive DOE.",
                        falsification_criterion=(
                            "a non-GP method outperforms GP on benchmark"
                        ),
                        prediction="GP wins on majority of DOE benchmarks",
                        prior=0.7,
                    )

            # Delegate to literature reviewer
            if not h_id or h_id.startswith("ERROR"):
                raise AssertionError(
                    f"could not obtain a hypothesis ID: {h_id!r}"
                )
            h_ids = [h_id]
            result = self.closure_tools["Delegate"](
                target="literature_reviewer",
                intent=DELEGATION_TASK,
                expected_report=(
                    "### Papers reviewed\n"
                    "### Key findings\n"
                    "### Conclusions\n"
                    "### Numbers"
                ),
                hypothesis_ids=h_ids,
            )

            # Poll until Done/Errored — up to 20 min (budget is 15)
            worker_report = ""
            m = re.search(r"D\d{3}", result)
            if m:
                d_id = m.group()
                for _ in range(2400):  # 2400 × 0.5s = 20 min ceiling
                    status = self.closure_tools["GetStatus"](d_id)
                    if not status.strip().startswith("Working"):
                        worker_report = re.sub(r"^Done\s*\n+", "", status, flags=re.DOTALL)
                        break
                    _time.sleep(0.5)

            summary = worker_report or "Literature review complete."
            # pipeline.ipynb is a hard Done() requirement — author it via the
            # structured tools (a minimal self-asserting analysis cell).
            if "AddPipelineCell" in self.closure_tools:
                self.closure_tools["SetNotebookIntro"](
                    "Literature review of the problem.", "n/a")
                self.closure_tools["AddPipelineCell"](
                    "analysis", "re-run the literature review delegation",
                    "print('REPRODUCED: 0.0')")
            # Two-shot Done(): first call → warning; second → closes.
            # If delegation is still working, wait briefly and retry.
            r1 = self.closure_tools["Done"](summary=summary)
            if "ERROR" in r1 and "still running" in r1:
                _time.sleep(10)
                self.closure_tools["Done"](summary=summary)
                self.closure_tools["Done"](summary=summary)
            else:
                self.closure_tools["Done"](summary=summary)
            return "Done."

    # --- Graph -------------------------------------------------------------
    class _StratSpec(StrategizerAgent):
        description = "Orchestrates the literature review run."

    class _LitSpec(LiteratureReviewAgent):
        pass  # inherits everything

    graph = Graph(
        nodes={
            "strategizer": _StratSpec(),
            "literature_reviewer": _LitSpec(),
        },
        edges=(Edge("strategizer", "literature_reviewer"),),
        entry="strategizer",
    )

    # --- Run ---------------------------------------------------------------
    run = AgenticRun(
        study_dir=study,
        graph=graph,
        budget=20 * 60,  # 20-minute wall-clock cap
    )

    # Inject scripted strategist adapter
    strat_adapter = LitReviewStrategist()

    def _mock_make_adapter(name, agent):
        if name == "strategizer":
            return strat_adapter
        # LiteratureReviewAgent: build the real adapter with corpus tools
        from f3dasm._src.agentic.backends.claude import ClaudeAdapter
        native = [t for t in agent.tools
                  if t in {"Bash", "Edit", "Read", "Write", "Glob", "Grep"}]
        # The adapter spawns the claude CLI with cwd=study_dir; a
        # nonexistent cwd makes subprocess.Popen fail instantly.
        (study / "workspace").mkdir(parents=True, exist_ok=True)
        adapter = ClaudeAdapter(
            model="claude-haiku-4-5-20251001",
            system_prompt=agent.system_prompt,
            study_dir=study / "workspace",
            native_tools=native,
            extra_mcp_servers=dict(agent.mcp_servers),
            extra_allowed_tools=list(agent.extra_allowed_tools),
        )
        extra = agent.build_closure_tools(study)
        if extra:
            adapter.closure_tools.update(extra)
        return adapter

    run._make_adapter = _mock_make_adapter

    start = time.time()
    report = run.execute()
    elapsed = time.time() - start

    # --- Assertions --------------------------------------------------------
    run_dir = next((study / "runs").iterdir())
    # build_closure_tools(study) without lit_reviewer_notes_dir uses
    # the study/delegations/literature fallback.
    corpus_dir = study / "delegations" / "literature"

    # 1. Report was produced and is non-trivial
    assert report and len(report) > 100, (
        f"Report too short or empty: {report!r}"
    )

    # 2. Corpus infrastructure was initialised (dir created by build_closure_tools)
    assert corpus_dir.exists(), (
        "delegations/literature/ not created by build_closure_tools"
    )

    # 3. corpus.csv check — may or may not exist depending on whether agent
    # used CorpusAdd vs. arxiv read_paper directly; both are acceptable.
    corpus_csv = corpus_dir / "corpus.csv"
    corpus_paper_count = 0
    if corpus_csv.exists():
        rows = corpus_csv.read_text().strip().splitlines()
        corpus_paper_count = max(0, len(rows) - 1)

    # 4. Report contains literature-related content
    report_lower = report.lower()
    literature_signals = [
        "paper", "arxiv", "gaussian process", "surrogate", "acquisition",
        "doi", "##", "###", "et al", "bayesian",
    ]
    matched = [s for s in literature_signals if s in report_lower]
    assert len(matched) >= 2, (
        f"Report doesn't look like a literature review — "
        f"matched only {matched} from {literature_signals}\n\nReport:\n{report[:500]}"
    )

    # 5. token table stamped into the deliverable notebook (there is no
    #    solution.md — the notebook's markdown IS the writeup).
    import nbformat
    nb = nbformat.read(str(study / "pipeline.ipynb"), as_version=4)
    md = "\n\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "## Token usage" in md

    # 6. delegation log has the literature review delegation
    jsonl = run_dir / "debug" / "delegation_log.jsonl"
    assert jsonl.exists(), "debug/delegation_log.jsonl missing"
    records = [
        json.loads(l) for l in jsonl.read_text().strip().splitlines()
    ]
    lit_records = [
        r for r in records if r.get("to_node") == "literature_reviewer"
    ]
    assert lit_records, "No delegation to literature_reviewer in JSONL"

    print(f"\n✓ Wet test passed in {elapsed:.0f}s")
    print(f"  Report length: {len(report)} chars")
    print(f"  Papers in corpus CSV: {corpus_paper_count}")
    print(f"  Literature signals matched: {matched}")
    if corpus_csv.exists():
        print(f"  corpus.csv: {corpus_paper_count} papers")
    print(f"\n--- Report (first 800 chars) ---\n{report[:800]}")
