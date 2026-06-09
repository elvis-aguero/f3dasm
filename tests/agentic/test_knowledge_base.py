"""Draft agentic knowledge base: corpus integrity + search API contract."""
from __future__ import annotations

from f3dasm._src.agentic.knowledge import KBEntry, KnowledgeBase


class TestCorpusIntegrity:
    def test_loads_seeded_entries(self):
        kb = KnowledgeBase.load()
        assert len(kb.entries) >= 8, "expected the seeded corpus to load"

    def test_every_entry_has_required_frontmatter(self):
        kb = KnowledgeBase.load()
        for e in kb.entries:
            assert e.id and e.id != e.body, f"missing id: {e}"
            assert e.title, f"missing title: {e.id}"
            assert e.tags, f"entry {e.id} has no tags (needed for search)"
            assert e.body.strip(), f"entry {e.id} has empty body"

    def test_entry_ids_are_unique(self):
        kb = KnowledgeBase.load()
        ids = [e.id for e in kb.entries]
        assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"

    def test_get_by_id(self):
        kb = KnowledgeBase.load()
        e = kb.get("evaluate-through-get-evaluator")
        assert e is not None
        assert "get_evaluator()" in e.body


class TestSearch:
    def test_finds_evaluation_entry(self):
        kb = KnowledgeBase.load()
        hits = kb.search("how do I make my evaluations count in the ledger")
        assert hits
        assert hits[0].id == "evaluate-through-get-evaluator"

    def test_finds_replicate_entry(self):
        kb = KnowledgeBase.load()
        hits = kb.search("replicate reproducibility headline from store")
        assert any(h.id == "replicate-reproduces-from-store" for h in hits)

    def test_respects_k(self):
        kb = KnowledgeBase.load()
        assert len(kb.search("evaluation store provenance", k=2)) <= 2

    def test_empty_query_returns_nothing(self):
        kb = KnowledgeBase.load()
        assert kb.search("   ") == []

    def test_irrelevant_query_returns_nothing(self):
        kb = KnowledgeBase.load()
        assert kb.search("zzzznonsensetokenqqq") == []


class TestParser:
    def test_frontmatter_lists_parse(self, tmp_path):
        d = tmp_path / "entries"
        d.mkdir()
        (d / "x.md").write_text(
            "---\nid: t\ntitle: T\ntags: [a, b, c]\naudience: [impl]\n"
            "---\nbody here\n"
        )
        kb = KnowledgeBase.load(d)
        e = kb.get("t")
        assert e is not None
        assert e.tags == ["a", "b", "c"]
        assert e.audience == ["impl"]
        assert "body here" in e.body

    def test_render_is_human_readable(self):
        kb = KnowledgeBase.load()
        out = kb.entries[0].render()
        assert out.startswith("## ")
        assert "[tags:" in out


class TestConsultHandbookTool:
    def test_returns_relevant_entry(self):
        from f3dasm._src.agentic.nodes import _consult_handbook
        out = _consult_handbook("how do I make my evaluations count in the ledger")
        assert "get_evaluator" in out

    def test_no_match_is_graceful(self):
        from f3dasm._src.agentic.nodes import _consult_handbook
        out = _consult_handbook("zzzznonsenseqqq")
        assert "No handbook entry matched" in out

    def test_never_raises(self):
        from f3dasm._src.agentic.nodes import _consult_handbook
        # odd inputs must not raise into the agent loop
        for q in ("", "   ", 123):
            assert isinstance(_consult_handbook(q), str)
