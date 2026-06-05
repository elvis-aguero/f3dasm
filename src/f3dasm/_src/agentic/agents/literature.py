"""LiteratureReviewAgent — specialist for searching scientific literature."""

from __future__ import annotations

from ..backends.base import Agent

# Module-level constant kept for backward compatibility with agent_prompts.py
# re-export.  The class sets system_prompt directly from this string.
LITERATURE_REVIEW_SYSTEM_PROMPT = """\
<role>
You are the Literature Reviewer. You answer specific research questions
by building a corpus of primary literature and quoting exact passages.

NEVER cite memory — corpus quotes only. Format: > "..." — Author et al., Year, p. X
If the corpus does not contain evidence, write: "Not found in corpus."
</role>

<corpus_tools>
  CorpusAdd(source, title, authors, year, doi, arxiv_id, citation_count=0)
                                  — index a LOCAL file. citation_count boosts
                                    BM25 retrieval weight: log10(c+1) scaling.
                                    Pass citationCount from S2 or OpenAlex.
  CorpusSearch(query, top_k=10)   — passage search across all indexed papers
  CorpusRank(passages, question)  — re-rank CorpusSearch results by BM25
                                    relevance. Use when merging results from
                                    multiple CorpusSearch calls.
  CorpusGetPaper(paper_id)        — full extracted text of one paper
  CorpusList()                    — metadata table of all corpus papers

  Corpus location: debug/lit_reviewer_notes/
    corpus.csv             — metadata index
    papers/{id}/paper.md   — page-annotated extracted text
</corpus_tools>

<discovery_tools>
  mcp__arxiv__search_papers(query, max_results)
                                  — find papers; returns IDs, titles, abstracts
  mcp__arxiv__download_paper(paper_id, dir)
                                  — download PDF to dir; returns local path
  mcp__arxiv__read_paper(paper_id)
                                  — extract text directly (no PDF needed)
  search_semantic_scholar(query, num_results=10)
                                  — search Semantic Scholar (200M+ papers across all disciplines)
  get_semantic_scholar_paper_details(paper_id)
                                  — citation count, influential citations, venue, TLDR
  get_semantic_scholar_citations_and_references(paper_id)
                                  — forward + backward citation traversal (≤20 each)
  search_openalex(query, n_results=10)
                                  — 300M+ works; strong for non-arXiv engineering
                                    journals (JMPS, CMAME, Acta Materialia).
                                    Returns pdf_url for open-access works.
  get_semantic_scholar_recommendations(paper_id, n_results=10)
                                  — semantically similar papers without citation
                                    links. Complements citation traversal.
  get_semantic_scholar_author_details(author_id)
  Read(path)                      — read any local file
  Grep(pattern, path)             — search text in files
</discovery_tools>

<workflow>
1. Expand queries: restate the question as 3-5 domain-specific keywords
   (e.g. "critical buckling stress" → "buckling instability slender rod elastic").
   Search mcp__arxiv__search_papers, search_semantic_scholar, AND search_openalex.
   Use get_semantic_scholar_recommendations(paper_id) on any seed paper found.
2. For each relevant paper, get its text via ONE of:
   a) mcp__arxiv__read_paper(paper_id) → write to {delegation_id}/{paper_id}.md, then
      CorpusAdd(source="{delegation_id}/{paper_id}.md", arxiv_id=paper_id, title=..., ...)
   b) mcp__arxiv__download_paper(paper_id, dir="{delegation_id}/") → then
      CorpusAdd(source="{delegation_id}/{paper_id}.pdf", arxiv_id=paper_id, title=..., ...)
3. CorpusSearch() for passages. Call with multiple phrasings. CorpusRank() to
   merge and reorder results from different queries before quoting.
4. Quote verbatim; don't paraphrase.
5. If no passage answers a question, say "Not found in corpus." and list queries tried.
</workflow>

<operating_principles>
1. VERBATIM QUOTES ONLY: Copy exactly; never paraphrase.
   Cite every quote: Author et al., Year, p. X

2. CITATION REQUIRED
   Every factual claim in ### Key findings and ### Conclusions needs a corpus citation.
   Never cite a paper you have not added to the corpus and read.

3. CORPUS FIRST: Try multiple phrasings before concluding "not found".

4. NO MEMORY SYNTHESIS: Don't fill gaps with background knowledge. "Not found in corpus." is valid.
</operating_principles>

<output_format>
## Report

### Papers reviewed
- {paper_id}: Author et al. (Year). "Title". Venue/Source.

### Key findings
**Q: {question from delegation}**
> "{exact verbatim quote}" — Author et al., Year, p. X
Relevance: {one sentence}.

OR: Not found in corpus. Searched for: {list of queries tried}.

### Conclusions

### Numbers
questions_addressed: N
papers_consulted: M
new_papers_added: K
quotes_used: Q
</output_format>
"""


class LiteratureReviewAgent(Agent):
    """Literature reviewer: answers epistemic questions from a primary-source corpus.

    Owns debug/lit_reviewer_notes/corpus.csv and debug/lit_reviewer_notes/papers/.
    Never answers from memory — all claims must cite exact passages from corpus.
    inject_problem_statement=True ensures the research domain is always visible.
    """

    inject_problem_statement = True
    tools = frozenset({"Read", "Grep", "Glob"})
    reset_on_checkpoint = True
    description = (
        "Searches and synthesises primary scientific literature to answer "
        "epistemic questions: what methods exist, what has been tried, what "
        "the field recommends. Use before committing to a strategy you are "
        "uncertain about, or when you need to know the state of the art. "
        "Never for questions answerable from workspace data."
    )
    report_sections = (
        "### Papers reviewed",
        "### Key findings",
        "### Conclusions",
        "### Numbers",
    )
    mcp_servers = {}
    extra_allowed_tools = frozenset()

    system_prompt = LITERATURE_REVIEW_SYSTEM_PROMPT

    def build_closure_tools(self, study_dir, delegation_id=None, lit_reviewer_notes_dir=None):
        """Inject corpus + Semantic Scholar tools as runtime closures."""
        import json as _json
        from pathlib import Path as _Path
        try:
            from ..literature_corpus import LiteratureCorpus
            from ..literature_corpus import _robust_get, _robust_post
        except ImportError:
            return {}

        corpus_dir = (
            _Path(lit_reviewer_notes_dir)
            if lit_reviewer_notes_dir is not None
            else _Path(study_dir) / "delegations" / "literature"  # fallback for tests
        )
        corpus = LiteratureCorpus(corpus_dir)

        tools = {
            "CorpusAdd": lambda source, title="", authors="", year="", doi="", arxiv_id="", venue="", abstract="", citation_count=0: corpus.add(
                source, title=title, authors=authors, year=year, doi=doi,
                arxiv_id=arxiv_id, venue=venue, abstract=abstract,
                citation_count=int(citation_count or 0),
            ),
            "CorpusSearch": lambda query, top_k=10: corpus.search(query, int(top_k)),
            "CorpusGetPaper": lambda paper_id: corpus.get_paper(paper_id),
            "CorpusList": lambda: corpus.list_papers(),
        }

        # Semantic Scholar tools via the semanticscholar Python library.
        # Mirrors the four tools from JackKuo666/semanticscholar-mcp-server
        # without needing an external process.
        try:
            from semanticscholar import SemanticScholar as _SS
            _sch = _SS()

            def search_semantic_scholar(query: str, num_results: int = 10) -> str:
                """Search for papers on Semantic Scholar."""
                results = _sch.search_paper(query, limit=int(num_results),
                                             fields=["title", "authors", "year",
                                                     "abstract", "externalIds",
                                                     "venue", "citationCount"])
                papers = []
                for p in results:
                    papers.append({
                        "paperId": p.paperId,
                        "title": p.title,
                        "year": p.year,
                        "venue": p.venue,
                        "citationCount": p.citationCount,
                        "authors": [a["name"] for a in (p.authors or [])],
                        "abstract": (p.abstract or "")[:300],
                        "externalIds": p.externalIds or {},
                    })
                return _json.dumps(papers, indent=2)

            def get_semantic_scholar_paper_details(paper_id: str) -> str:
                """Get details for a paper by its S2, DOI, or arxiv ID."""
                paper = _sch.get_paper(paper_id,
                                       fields=["title", "authors", "year", "abstract",
                                               "venue", "citationCount",
                                               "influentialCitationCount",
                                               "tldr", "externalIds"])
                return _json.dumps({
                    "paperId": paper.paperId,
                    "title": paper.title,
                    "year": paper.year,
                    "venue": paper.venue,
                    "citationCount": paper.citationCount,
                    "influentialCitationCount": paper.influentialCitationCount,
                    "tldr": (paper.tldr or {}).get("text"),
                    "authors": [a["name"] for a in (paper.authors or [])],
                    "abstract": paper.abstract,
                    "externalIds": paper.externalIds or {},
                }, indent=2)

            def get_semantic_scholar_author_details(author_id: str) -> str:
                """Get details for an author by their S2 author ID."""
                author = _sch.get_author(author_id,
                                         fields=["name", "affiliations",
                                                 "paperCount", "citationCount",
                                                 "hIndex"])
                return _json.dumps({
                    "authorId": author.authorId,
                    "name": author.name,
                    "affiliations": author.affiliations,
                    "paperCount": author.paperCount,
                    "citationCount": author.citationCount,
                    "hIndex": author.hIndex,
                }, indent=2)

            def get_semantic_scholar_citations_and_references(paper_id: str) -> str:
                """Get citing papers and references for a paper (up to 20 each)."""
                paper = _sch.get_paper(paper_id, fields=["citations", "references"])
                refs = [{"paperId": r.get("paperId"), "title": r.get("title")}
                        for r in (paper.references or [])[:20]]
                cits = [{"paperId": c.get("paperId"), "title": c.get("title")}
                        for c in (paper.citations or [])[:20]]
                return _json.dumps({"references": refs, "citations": cits}, indent=2)

            tools.update({
                "search_semantic_scholar": search_semantic_scholar,
                "get_semantic_scholar_paper_details": get_semantic_scholar_paper_details,
                "get_semantic_scholar_author_details": get_semantic_scholar_author_details,
                "get_semantic_scholar_citations_and_references": get_semantic_scholar_citations_and_references,
            })
        except ImportError:
            pass  # semanticscholar not installed; SS tools unavailable

        def search_openalex(query: str, n_results: int = 10) -> str:
            """Search OpenAlex (300M+ works; strong coverage of closed-venue engineering journals).

            Returns papers with metadata and open-access PDF URLs where available.
            Prefer this for papers NOT on arXiv (JMPS, CMAME, Acta Materialia, etc.).
            """
            import json as _j
            try:
                resp = _robust_get(
                    "https://api.openalex.org/works",
                    params={
                        "search": query,
                        "per-page": min(int(n_results), 25),
                        "select": "id,title,authorships,publication_year,doi,"
                                  "primary_location,abstract_inverted_index",
                    },
                    headers={"User-Agent": "f3dasm-agent/1.0 (mailto:f3dasm@brown.edu)"},
                )
                works = resp.json().get("results", [])
            except Exception as exc:
                return f"ERROR: OpenAlex search failed: {exc}"

            out = []
            for w in works:
                authors = ", ".join(
                    a["author"]["display_name"]
                    for a in (w.get("authorships") or [])[:3]
                )
                doi = (w.get("doi") or "").replace("https://doi.org/", "")
                loc = w.get("primary_location") or {}
                pdf_url = loc.get("pdf_url") or loc.get("landing_page_url") or ""
                # reconstruct abstract from inverted index
                inv = w.get("abstract_inverted_index") or {}
                abstract = ""
                if inv:
                    pairs = [(pos, word) for word, positions in inv.items() for pos in positions]
                    pairs.sort()
                    abstract = " ".join(word for _, word in pairs)[:400]
                out.append({
                    "id": w.get("id", ""),
                    "title": w.get("title", ""),
                    "year": w.get("publication_year", ""),
                    "authors": authors,
                    "doi": doi,
                    "pdf_url": pdf_url,
                    "abstract": abstract,
                })
            return _j.dumps(out, indent=2)

        def get_semantic_scholar_recommendations(paper_id: str, n_results: int = 10) -> str:
            """Find papers semantically similar to paper_id (no citation link needed).

            Complements citation traversal — surfaces related work that doesn't
            cite or get cited by the seed paper.
            paper_id: S2 paperId, DOI, or 'arXiv:XXXX.XXXXX'.
            """
            import json as _j
            try:
                resp = _robust_post(
                    "https://api.semanticscholar.org/recommendations/v1/papers/",
                    json={"positivePaperIds": [paper_id]},
                    params={
                        "fields": "paperId,title,authors,year,abstract,externalIds,openAccessPdf",
                        "limit": min(int(n_results), 50),
                    },
                )
                papers = resp.json().get("recommendedPapers", [])
            except Exception as exc:
                return f"ERROR: S2 recommendations failed: {exc}"

            out = []
            for p in papers:
                oa = p.get("openAccessPdf") or {}
                out.append({
                    "paperId": p.get("paperId", ""),
                    "title": p.get("title", ""),
                    "year": p.get("year", ""),
                    "authors": [a["name"] for a in (p.get("authors") or [])[:3]],
                    "abstract": (p.get("abstract") or "")[:300],
                    "externalIds": p.get("externalIds") or {},
                    "pdf_url": oa.get("url", ""),
                })
            return _j.dumps(out, indent=2)

        def CorpusRank(passages: str, question: str) -> str:
            """Re-rank a newline-separated list of corpus passages by BM25 relevance to question.

            Pass the raw output of CorpusSearch as `passages`.
            Returns the same passages reordered from most to least relevant.
            Useful when combining results from multiple CorpusSearch calls.
            """
            if not passages or passages == "No results found.":
                return passages

            try:
                from rank_bm25 import BM25Okapi
            except ImportError:
                return passages  # no-op if not installed

            # Split on passage separators (--- Title (Year), p.N ---)
            import re as _re
            blocks = _re.split(r"(?=--- .+ \(\d*\), p\.\d+ ---)", passages.strip())
            blocks = [b.strip() for b in blocks if b.strip()]
            if len(blocks) <= 1:
                return passages

            tokenized = [b.lower().split() for b in blocks]
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(question.lower().split())
            ranked = sorted(zip(blocks, scores), key=lambda x: x[1], reverse=True)
            return "\n\n".join(b for b, _ in ranked)

        tools["search_openalex"] = search_openalex
        tools["get_semantic_scholar_recommendations"] = get_semantic_scholar_recommendations
        tools["CorpusRank"] = CorpusRank

        # arxiv tools — Python-native, same implementation for Claude and Ollama.
        from ..backends.ollama import _build_arxiv_closures
        tools.update(_build_arxiv_closures())

        return tools
