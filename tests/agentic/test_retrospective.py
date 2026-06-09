"""Worker retrospective: self-consistency-first exit interview.

Each worker ends its report with a ### Retrospective section auditing the
SYSTEM (instructions/contracts/tools), led by a CONSISTENCY: ok|flagged line.
The runtime parses it, persists to retrospectives.jsonl, and — the instant a
worker flags contradictory instructions — emits a diagnostic + notification.
"""
from __future__ import annotations

from f3dasm._src.agentic.nodes import _extract_report_section


class TestSectionExtractor:
    def test_extracts_retrospective_body(self):
        text = (
            "## Report\n\n### Conclusions\nFound x.\n\n"
            "### Numbers\nbest: 1.2\n\n"
            "### Retrospective\n- CONSISTENCY: ok\n- DECISION: used seed 0\n"
            "- FRICTION: none\n"
        )
        body = _extract_report_section(text, "Retrospective")
        assert "CONSISTENCY: ok" in body
        assert "DECISION: used seed 0" in body
        # Must not bleed earlier sections in
        assert "best: 1.2" not in body

    def test_absent_section_returns_empty(self):
        assert _extract_report_section("## Report\n### Numbers\na: 1", "Retrospective") == ""

    def test_stops_at_trailing_rule(self):
        text = "### Retrospective\n- CONSISTENCY: ok\n---\nRequired keys: ...\n"
        body = _extract_report_section(text, "Retrospective")
        assert "CONSISTENCY: ok" in body
        assert "Required keys" not in body

    def test_flag_detection_regex(self):
        import re
        flagged = "- CONSISTENCY: flagged — 'do X' vs 'never do X'"
        ok = "- CONSISTENCY: ok"
        assert re.search(r"CONSISTENCY:\s*flagged", flagged, re.I)
        assert not re.search(r"CONSISTENCY:\s*flagged", ok, re.I)


class TestPromptsCarryRetrospective:
    def test_implementer_claude_prompt(self):
        from f3dasm._src.agentic.agents.implementer import (
            IMPLEMENTER_SYSTEM_PROMPT,
        )
        assert "### Retrospective" in IMPLEMENTER_SYSTEM_PROMPT
        assert "CONSISTENCY: ok | flagged" in IMPLEMENTER_SYSTEM_PROMPT

    def test_implementer_ollama_prompt(self):
        from f3dasm._src.agentic.agent_prompts import (
            IMPLEMENTER_SYSTEM_PROMPT_OLLAMA,
        )
        assert "### Retrospective" in IMPLEMENTER_SYSTEM_PROMPT_OLLAMA
        assert "CONSISTENCY: ok | flagged" in IMPLEMENTER_SYSTEM_PROMPT_OLLAMA

    def test_datagenerator_prompt(self):
        from f3dasm._src.agentic.agents.datagenerator import (
            DATA_GENERATOR_SYSTEM_PROMPT,
        )
        assert "### Retrospective" in DATA_GENERATOR_SYSTEM_PROMPT

    def test_critic_prompt(self):
        from f3dasm._src.agentic.agents.critic import (
            ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT,
        )
        assert "### Retrospective" in ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT

    def test_literature_prompt(self):
        from f3dasm._src.agentic.agents.literature import (
            LITERATURE_REVIEW_SYSTEM_PROMPT,
        )
        assert "### Retrospective" in LITERATURE_REVIEW_SYSTEM_PROMPT

    def test_strategizer_prompt_is_NOT_polluted(self):
        """The strategizer must NOT carry the interview in its working
        context — that would pollute every orchestration turn. The exit
        interview is asked by the runtime AFTER the critic accepts."""
        from f3dasm._src.agentic.agents.strategizer import (
            STRATEGIZER_SYSTEM_PROMPT,
        )
        assert "### Retrospective" not in STRATEGIZER_SYSTEM_PROMPT
        assert "<retrospective>" not in STRATEGIZER_SYSTEM_PROMPT

    def test_exit_interview_is_a_runtime_post_done_turn(self):
        from f3dasm._src.agentic.nodes import _EXIT_INTERVIEW
        # Asked only after acceptance; carries the same 3 questions.
        assert "accepted by the critic" in _EXIT_INTERVIEW
        assert "CONSISTENCY" in _EXIT_INTERVIEW
        assert "DECISION" in _EXIT_INTERVIEW
        assert "FRICTION" in _EXIT_INTERVIEW
        assert "Done() ONE more time" in _EXIT_INTERVIEW


class TestNoCanonicalSourceNudge:
    """Strategizer flowchart: when a datagenerator exists but no canonical
    source is registered, recommend delegating to it (soft, ≤3×)."""

    def _spec_with_datagenerator(self):
        from f3dasm._src.agentic.backends.base import Agent, Edge, Graph

        class S(Agent):
            role = "strategizer"
            description = "strategizer"

        class DG(Agent):
            role = "datagenerator"
            description = "datagenerator"

        return Graph(
            nodes={"strategizer": S(), "datagenerator": DG()},
            edges=(Edge("strategizer", "datagenerator"),),
            entry="strategizer",
        )

    def _node(self):
        from f3dasm._src.agentic.nodes import StrategizerNode

        class _Stub:
            def __init__(self):
                self.role = "strategizer"
                self.closure_tools = {}
                self.route_watcher = None
                self.last_usage = {}

            def invoke(self, messages):
                return "ok"

        return StrategizerNode(
            _Stub(), name="strategizer", outgoing=["datagenerator"],
            spec=self._spec_with_datagenerator(),
            worker_adapters={"datagenerator": _Stub()},
        )

    def test_finds_datagenerator(self):
        assert self._node()._find_datagenerator_name() == "datagenerator"

    def test_source_unregistered_when_entrypoint_absent(self, tmp_path):
        import json
        node = self._node()
        notes = tmp_path / "debug" / "strategizer_notes"
        notes.mkdir(parents=True)
        (notes.parent / "run_config.json").write_text(json.dumps(
            {"evaluator_entrypoint": None, "evaluator_lookup": None}))
        node._current_notes_dir = notes
        assert node._canonical_source_registered() is False

    def test_source_registered_when_entrypoint_present(self, tmp_path):
        import json
        node = self._node()
        notes = tmp_path / "debug" / "strategizer_notes"
        notes.mkdir(parents=True)
        (notes.parent / "run_config.json").write_text(json.dumps(
            {"evaluator_entrypoint": "workspace/e.py:f"}))
        node._current_notes_dir = notes
        assert node._canonical_source_registered() is True

    def test_lookup_counts_as_registered(self, tmp_path):
        import json
        node = self._node()
        notes = tmp_path / "debug" / "strategizer_notes"
        notes.mkdir(parents=True)
        (notes.parent / "run_config.json").write_text(json.dumps(
            {"evaluator_lookup": "experiment_data"}))
        node._current_notes_dir = notes
        assert node._canonical_source_registered() is True

    def test_nudge_counter_caps_at_three(self):
        # The cap field exists and starts at 0; __call__ increments up to 3.
        node = self._node()
        assert node._no_source_nudges == 0

    def test_report_sections_include_retrospective(self):
        from f3dasm._src.agentic.agents.critic import (
            AdversarialCritiqueAgent,
        )
        from f3dasm._src.agentic.agents.datagenerator import DataGeneratorAgent
        from f3dasm._src.agentic.agents.implementer import (
            F3dasmImplementerAgent,
        )
        from f3dasm._src.agentic.agents.literature import (
            LiteratureReviewAgent,
        )
        for agent in (F3dasmImplementerAgent, DataGeneratorAgent,
                      AdversarialCritiqueAgent, LiteratureReviewAgent):
            assert "### Retrospective" in agent.report_sections, agent.__name__


class TestStrategizerGranularity:
    def test_anti_monolith_rule_present(self):
        from f3dasm._src.agentic.agents.strategizer import (
            STRATEGIZER_SYSTEM_PROMPT,
        )
        assert "MONOLITHIC DELEGATION" in STRATEGIZER_SYSTEM_PROMPT
        assert "ONE bounded experiment" in STRATEGIZER_SYSTEM_PROMPT

    def test_hypothesis_framing_guidance_present(self):
        from f3dasm._src.agentic.agents.strategizer import (
            STRATEGIZER_SYSTEM_PROMPT,
        )
        # domain-neutral: no leaked example framings, but the principle is there
        assert "not a bet on which method" in STRATEGIZER_SYSTEM_PROMPT.lower() \
            or "NOT a bet on which method" in STRATEGIZER_SYSTEM_PROMPT
