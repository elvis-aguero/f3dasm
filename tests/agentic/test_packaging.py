"""Item D — lean `agentic` vs heavy `agentic-extra` extras split.

The lean `agentic` extra must be torch-free yet still run a *functional*
literature reviewer (pymupdf PDF→MD + BM25 retrieval).  The torch-pullers
(docling, fastembed) live in `agentic-extra` as a quality upgrade.  These
tests pin the dependency partition and the graceful lean-image degradation.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import tomllib

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _dep_names(reqs: list[str]) -> set[str]:
    """Normalised distribution names from a list of requirement strings."""
    names = set()
    for r in reqs:
        # strip environment marker, extras, and version spec
        head = re.split(r"[;\[]", r, maxsplit=1)[0]
        name = re.split(r"[<>=!~ ]", head, maxsplit=1)[0].strip().lower()
        if name:
            names.add(name.replace("_", "-"))
    return names


@pytest.fixture(scope="module")
def extras() -> dict:
    data = tomllib.loads(_PYPROJECT.read_text())
    return {
        k: _dep_names(v)
        for k, v in data["project"]["optional-dependencies"].items()
    }


def test_lean_agentic_keeps_a_working_reviewer(extras):
    """Lean reviewer = pymupdf + BM25 + (lightweight) paper search, all in the
    lean extra."""
    assert "pymupdf" in extras["agentic"]
    assert "rank-bm25" in extras["agentic"]
    # semanticscholar is lightweight (httpx + tenacity), so it stays in core
    assert "semanticscholar" in extras["agentic"]


def test_lean_agentic_is_torch_free_no_heavy_deps(extras):
    """The torch-pullers (and ONLY those) must NOT be in the lean extra."""
    lean = extras["agentic"]
    for heavy in ("docling", "fastembed"):
        assert heavy not in lean, f"{heavy} must move to agentic-extra"


def test_agentic_extra_holds_only_the_torch_pullers(extras):
    assert "agentic-extra" in extras, "agentic-extra extra must exist"
    xtra = extras["agentic-extra"]
    for q in ("docling", "fastembed"):
        assert q in xtra, f"agentic-extra must provide {q}"
    # semanticscholar is light → lives in core, NOT in the heavy extra
    assert "semanticscholar" not in xtra


def test_lit_is_alias_of_agentic_extra(extras):
    """Legacy `lit` extra folds into agentic-extra (back-compat alias).

    Accepts either a literal subset OR a self-referential alias
    (``f3dasm[agentic-extra]``), which normalises to the project name.
    """
    lit = extras["lit"]
    is_self_ref = lit == {"f3dasm"}
    assert is_self_ref or lit <= extras["agentic-extra"]


def test_literature_corpus_degrades_to_bm25_without_heavy_deps(
    tmp_path, monkeypatch
):
    """The exact lean-image path: with docling AND fastembed unimportable,
    LiteratureCorpus still constructs and retrieval falls back to BM25 (the
    dense embedder resolves to None)."""
    import f3dasm._src.agentic.literature_corpus as lc

    # Force both heavy deps unimportable (a None entry → ImportError on import).
    monkeypatch.setitem(sys.modules, "fastembed", None)
    monkeypatch.setitem(sys.modules, "docling", None)
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)
    # Pin the out-of-process embed probe as already-unavailable so we exercise
    # the pure BM25-only path deterministically (no uv subprocess spin-up).
    monkeypatch.setattr(lc, "_subprocess_embedder_state", False)

    corpus = lc.LiteratureCorpus(tmp_path)  # constructs fine in the lean image
    assert corpus._get_embedding_model() is None  # → BM25-only retrieval
