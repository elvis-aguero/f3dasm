"""LiteratureReviewAgent — specialist for scientific literature."""

from __future__ import annotations

import logging

from ..backends.base import Agent


def _call_in_fresh_thread(fn, *args, timeout=30.0, **kwargs):
    """Run *fn* in a thread with no running event loop.

    The semanticscholar sync client manages its own asyncio loop and
    breaks when called from within an already-running loop (the
    Claude SDK closure context). A fresh thread has no loop.
    """
    import concurrent.futures as _cf
    with _cf.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn, *args, **kwargs).result(timeout=timeout)

log = logging.getLogger(__name__)

# Module-level constant kept for backward compatibility with
# agent_prompts.py re-export.
LITERATURE_REVIEW_SYSTEM_PROMPT = """\
<role>
You are the Literature Reviewer. You answer specific research questions
by building a corpus of primary literature and quoting exact passages.

NEVER cite memory — corpus quotes only. Format: > "..." — Author et al., Year, p. X
If the corpus does not contain evidence, write: "Not found in corpus."
</role>

<primary_source_rule>
Quote ONLY from full-text papers. Abstract-only corpus entries are
leads, not sources — CorpusSearch will not return their text.

Acquisition chain:
1. Search (mcp__arxiv__search_papers / search_semantic_scholar /
   search_openalex) — note pdf_url fields in results.
2. Download PDF via DownloadPdf(url, filename) or
   mcp__arxiv__download_paper — saves the file locally.
3. CorpusAdd(source=<saved path>, ...) — indexes the full text.
4. CorpusSearch() — now returns real passages to quote.

Until a paper has been CorpusAdded from a PDF or full-text markdown
(>5000 chars), do not quote from it.
</primary_source_rule>

<corpus_tools>
  CorpusAdd(source, title, authors, year, doi, arxiv_id, citation_count=0)
                                  — index a LOCAL file. citation_count boosts
                                    BM25 retrieval weight: log10(c+1) scaling.
                                    Pass citationCount from S2 or OpenAlex.
  CorpusSearch(query, top_k=10)   — passage search across FULL-TEXT papers only.
                                    Returns ERROR if no full-text papers exist.
  CorpusRank(passages, question)  — re-rank CorpusSearch results by BM25
                                    relevance. Use when merging results from
                                    multiple CorpusSearch calls.
  CorpusGetPaper(paper_id)        — full extracted text of one paper
  CorpusList()                    — metadata table; [full-text]/[abstract-only]
  DownloadPdf(url, filename)      — rate-limited fetch of a PDF URL to disk;
                                    returns local path or ERROR.
                                    Validates content-type/magic bytes.

  Corpus location: delegations/literature/  (under the run's debug_dir)
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
                                  — search Semantic Scholar (200M+ papers)
  get_semantic_scholar_paper_details(paper_id)
                                  — citation count, influential citations, venue, TLDR
  get_semantic_scholar_citations_and_references(paper_id)
                                  — forward + backward citation traversal (≤20)
  search_openalex(query, n_results=10)
                                  — 300M+ works; returns pdf_url (best_oa_location)
                                    for open-access works. Strong for non-arXiv
                                    journals (JMPS, CMAME, Acta Materialia).
  get_openalex_citations(work_id, n_results=20)
                                  — papers that cite work_id, sorted by citation
                                    count; citation-graph traversal fallback when
                                    Semantic Scholar is rate-limited.
  get_openalex_references(work_id)
                                  — reference list of work_id (first 40), hydrated
                                    in one batched call; citation-graph traversal
                                    fallback when Semantic Scholar is rate-limited.
  get_semantic_scholar_recommendations(paper_id, n_results=10)
                                  — semantically similar papers without citation
                                    links.
  get_semantic_scholar_author_details(author_id)
  Read(path)                      — read any local file
  Grep(pattern, path)             — search text in files
</discovery_tools>

<workflow>
1. Expand queries: restate the question as 3-5 domain-specific keywords.
   Search mcp__arxiv__search_papers, search_semantic_scholar, AND
   search_openalex. Note pdf_url in results.
2. For each relevant paper, acquire full text via ONE of:
   a) mcp__arxiv__read_paper(paper_id) → write to {delegation_id}/{paper_id}.md
      (only if result is >5000 chars), then CorpusAdd(source=…)
   b) pdf_url from search_openalex / get_semantic_scholar_paper_details →
      DownloadPdf(url, "{delegation_id}/{paper_id}.pdf") →
      CorpusAdd(source=<path>, arxiv_id=…, title=…, …)
   c) mcp__arxiv__download_paper(paper_id, dir="{delegation_id}/") →
      CorpusAdd(source=…)
3. CorpusSearch() for passages. Call with multiple phrasings.
   CorpusRank() to merge and reorder results before quoting.
4. Quote verbatim; don't paraphrase.
5. If no passage answers a question, say "Not found in corpus." and
   list queries tried.
</workflow>

<operating_principles>
1. VERBATIM QUOTES ONLY: Copy exactly; never paraphrase.
   Cite every quote: Author et al., Year, p. X

2. CITATION REQUIRED
   Every factual claim in Key findings and Conclusions needs a citation.
   Never cite a paper you have not added to the corpus and read.

3. CORPUS FIRST: Try multiple phrasings before concluding "not found".

4. NO MEMORY SYNTHESIS: Don't fill gaps with background knowledge.
   "Not found in corpus." is valid.

5. RATE LIMIT HANDLING: If a tool returns ERROR containing
   "rate-limited", switch to a different source (e.g. arXiv instead of
   Semantic Scholar) or wait before retrying.
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
    """Literature reviewer: answers epistemic questions from a corpus.

    Owns debug/lit_reviewer_notes/corpus.csv and papers/.
    Never answers from memory — all claims must cite exact passages.
    inject_problem_statement=True ensures research domain is visible.
    """

    inject_problem_statement = True
    tools = frozenset({"Read", "Grep", "Glob"})
    reset_on_checkpoint = True
    description = (
        "Searches and synthesises primary scientific literature to"
        " answer epistemic questions: what methods exist, what has"
        " been tried, what the field recommends. Use before committing"
        " to a strategy you are uncertain about, or when you need to"
        " know the state of the art. Never for questions answerable"
        " from workspace data."
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

    def build_closure_tools(
        self,
        study_dir,
        delegation_id=None,
        lit_reviewer_notes_dir=None,
    ):
        """Inject corpus + discovery tools as runtime closures."""
        import json as _json
        from pathlib import Path as _Path

        try:
            from ..literature_corpus import LiteratureCorpus
            from ..literature_corpus import (
                SourceCooldownError,
                _robust_get,
                _robust_post,
            )
        except ImportError:
            return {}

        corpus_dir = (
            _Path(lit_reviewer_notes_dir)
            if lit_reviewer_notes_dir is not None
            else _Path(study_dir) / "delegations" / "literature"
        )
        corpus = LiteratureCorpus(corpus_dir)
        cache_dir = corpus._http_cache_dir

        tools = {
            "CorpusAdd": lambda source, title="", authors="",
            year="", doi="", arxiv_id="", venue="", abstract="",
            citation_count=0: corpus.add(
                source, title=title, authors=authors, year=year,
                doi=doi, arxiv_id=arxiv_id, venue=venue,
                abstract=abstract,
                citation_count=int(citation_count or 0),
            ),
            "CorpusSearch": lambda query, top_k=10: corpus.search(
                query, int(top_k)
            ),
            "CorpusGetPaper": lambda paper_id: corpus.get_paper(
                paper_id
            ),
            "CorpusList": lambda: corpus.list_papers(),
        }

        # Semantic Scholar tools via the semanticscholar library.
        try:
            from semanticscholar import SemanticScholar as _SS
            _sch = _SS()

            def search_semantic_scholar(
                query: str, num_results: int = 10
            ) -> str:
                """Search for papers on Semantic Scholar."""
                try:
                    results = _call_in_fresh_thread(
                        _sch.search_paper,
                        query,
                        limit=int(num_results),
                        fields=[
                            "title", "authors", "year", "abstract",
                            "externalIds", "venue", "citationCount",
                        ],
                    )
                except TimeoutError:
                    return (
                        "ERROR: Semantic Scholar request timed out"
                        " after 30s. Try again or use OpenAlex."
                    )
                except Exception as exc:
                    return f"ERROR: Semantic Scholar search failed: {exc}"
                papers = []
                for p in results:
                    papers.append({
                        "paperId": p.paperId,
                        "title": p.title,
                        "year": p.year,
                        "venue": p.venue,
                        "citationCount": p.citationCount,
                        "authors": [
                            a["name"] for a in (p.authors or [])
                        ],
                        "abstract": (p.abstract or "")[:300],
                        "externalIds": p.externalIds or {},
                    })
                return _json.dumps(papers, indent=2)

            def get_semantic_scholar_paper_details(
                paper_id: str,
            ) -> str:
                """Get details for a paper by S2/DOI/arxiv ID."""
                try:
                    paper = _call_in_fresh_thread(
                        _sch.get_paper,
                        paper_id,
                        fields=[
                            "title", "authors", "year", "abstract",
                            "venue", "citationCount",
                            "influentialCitationCount",
                            "tldr", "externalIds",
                        ],
                    )
                except TimeoutError:
                    return (
                        "ERROR: Semantic Scholar request timed out"
                        " after 30s. Try again or use OpenAlex."
                    )
                except Exception as exc:
                    return (
                        f"ERROR: Semantic Scholar paper details failed: {exc}"
                    )
                return _json.dumps({
                    "paperId": paper.paperId,
                    "title": paper.title,
                    "year": paper.year,
                    "venue": paper.venue,
                    "citationCount": paper.citationCount,
                    "influentialCitationCount": (
                        paper.influentialCitationCount
                    ),
                    "tldr": (paper.tldr or {}).get("text"),
                    "authors": [
                        a["name"] for a in (paper.authors or [])
                    ],
                    "abstract": paper.abstract,
                    "externalIds": paper.externalIds or {},
                }, indent=2)

            def get_semantic_scholar_author_details(
                author_id: str,
            ) -> str:
                """Get details for an author by their S2 author ID."""
                try:
                    author = _call_in_fresh_thread(
                        _sch.get_author,
                        author_id,
                        fields=[
                            "name", "affiliations", "paperCount",
                            "citationCount", "hIndex",
                        ],
                    )
                except TimeoutError:
                    return (
                        "ERROR: Semantic Scholar request timed out"
                        " after 30s. Try again or use OpenAlex."
                    )
                except Exception as exc:
                    return (
                        f"ERROR: Semantic Scholar author details failed: {exc}"
                    )
                return _json.dumps({
                    "authorId": author.authorId,
                    "name": author.name,
                    "affiliations": author.affiliations,
                    "paperCount": author.paperCount,
                    "citationCount": author.citationCount,
                    "hIndex": author.hIndex,
                }, indent=2)

            def get_semantic_scholar_citations_and_references(
                paper_id: str,
            ) -> str:
                """Get citing papers and references (≤20 each)."""
                try:
                    paper = _call_in_fresh_thread(
                        _sch.get_paper,
                        paper_id,
                        fields=["citations", "references"],
                    )
                except TimeoutError:
                    return (
                        "ERROR: Semantic Scholar request timed out"
                        " after 30s. Try again or use OpenAlex."
                    )
                except Exception as exc:
                    return (
                        "ERROR: Semantic Scholar citations/references"
                        f" failed: {exc}"
                    )
                refs = [
                    {
                        "paperId": r.get("paperId"),
                        "title": r.get("title"),
                    }
                    for r in (paper.references or [])[:20]
                ]
                cits = [
                    {
                        "paperId": c.get("paperId"),
                        "title": c.get("title"),
                    }
                    for c in (paper.citations or [])[:20]
                ]
                return _json.dumps(
                    {"references": refs, "citations": cits}, indent=2
                )

            tools.update({
                "search_semantic_scholar": search_semantic_scholar,
                "get_semantic_scholar_paper_details": (
                    get_semantic_scholar_paper_details
                ),
                "get_semantic_scholar_author_details": (
                    get_semantic_scholar_author_details
                ),
                "get_semantic_scholar_citations_and_references": (
                    get_semantic_scholar_citations_and_references
                ),
            })
        except ImportError:
            log.warning(
                "semanticscholar not installed — S2 tools not"
                " registered for literature_reviewer"
            )

        def search_openalex(
            query: str, n_results: int = 10
        ) -> str:
            """Search OpenAlex (300M+ works; open-access PDF URLs).

            Returns papers with metadata and open-access PDF URLs
            where available (best_oa_location.pdf_url / oa_url).
            Prefer for papers NOT on arXiv.
            """
            import json as _j
            try:
                resp = _robust_get(
                    "https://api.openalex.org/works",
                    params={
                        "search": query,
                        "per-page": min(int(n_results), 25),
                        "select": (
                            "id,title,authorships,publication_year"
                            ",doi,primary_location,open_access"
                            ",best_oa_location"
                            ",abstract_inverted_index"
                        ),
                    },
                    headers={
                        "User-Agent": (
                            "f3dasm-agent/1.0"
                            " (mailto:f3dasm@brown.edu)"
                        ),
                    },
                    cache_dir=cache_dir,
                )
                works = resp.json().get("results", [])
            except SourceCooldownError as exc:
                return f"ERROR: {exc}"
            except Exception as exc:
                return f"ERROR: OpenAlex search failed: {exc}"

            out = []
            for w in works:
                authors = ", ".join(
                    a["author"]["display_name"]
                    for a in (w.get("authorships") or [])[:3]
                )
                doi = (w.get("doi") or "").replace(
                    "https://doi.org/", ""
                )
                # Best OA location first, then primary_location
                boa = w.get("best_oa_location") or {}
                loc = w.get("primary_location") or {}
                oa = w.get("open_access") or {}
                pdf_url = (
                    boa.get("pdf_url")
                    or oa.get("oa_url")
                    or loc.get("pdf_url")
                    or loc.get("landing_page_url")
                    or ""
                )
                # reconstruct abstract from inverted index
                inv = w.get("abstract_inverted_index") or {}
                abstract = ""
                if inv:
                    pairs = [
                        (pos, word)
                        for word, positions in inv.items()
                        for pos in positions
                    ]
                    pairs.sort()
                    abstract = " ".join(
                        word for _, word in pairs
                    )[:400]
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

        def get_semantic_scholar_recommendations(
            paper_id: str, n_results: int = 10
        ) -> str:
            """Find semantically similar papers (no citation link).

            paper_id: S2 paperId, DOI, or 'arXiv:XXXX.XXXXX'.
            """
            import json as _j
            try:
                resp = _robust_post(
                    "https://api.semanticscholar.org"
                    "/recommendations/v1/papers/",
                    json={"positivePaperIds": [paper_id]},
                    params={
                        "fields": (
                            "paperId,title,authors,year,abstract"
                            ",externalIds,openAccessPdf"
                        ),
                        "limit": min(int(n_results), 50),
                    },
                )
                papers = resp.json().get("recommendedPapers", [])
            except SourceCooldownError as exc:
                return f"ERROR: {exc}"
            except Exception as exc:
                return f"ERROR: S2 recommendations failed: {exc}"

            out = []
            for p in papers:
                oa = p.get("openAccessPdf") or {}
                out.append({
                    "paperId": p.get("paperId", ""),
                    "title": p.get("title", ""),
                    "year": p.get("year", ""),
                    "authors": [
                        a["name"]
                        for a in (p.get("authors") or [])[:3]
                    ],
                    "abstract": (p.get("abstract") or "")[:300],
                    "externalIds": p.get("externalIds") or {},
                    "pdf_url": oa.get("url", ""),
                })
            return _j.dumps(out, indent=2)

        def CorpusRank(passages: str, question: str) -> str:
            """Re-rank corpus passages by BM25 relevance to question.

            Pass the raw output of CorpusSearch as ``passages``.
            Returns passages reordered from most to least relevant.
            """
            if not passages or passages == "No results found.":
                return passages

            try:
                from rank_bm25 import BM25Okapi
            except ImportError:
                return passages  # no-op if not installed

            import re as _re
            blocks = _re.split(
                r"(?=--- .+ \(\d*\), p\.\d+ ---)",
                passages.strip(),
            )
            blocks = [b.strip() for b in blocks if b.strip()]
            if len(blocks) <= 1:
                return passages

            tokenized = [b.lower().split() for b in blocks]
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(question.lower().split())
            ranked = sorted(
                zip(blocks, scores),
                key=lambda x: x[1],
                reverse=True,
            )
            return "\n\n".join(b for b, _ in ranked)

        def DownloadPdf(url: str, filename: str) -> str:
            """Fetch a PDF URL and save to disk.

            Parameters
            ----------
            url:
                Direct URL to the PDF (content-type must contain
                'pdf' or body must start with ``%PDF``).
            filename:
                Destination filename.  If not absolute, saved under
                the corpus papers directory.

            Returns
            -------
            str
                Absolute path to the saved file, or ``"ERROR: …"``.
            """
            from pathlib import Path as _Path
            try:
                resp = _robust_get(
                    url,
                    cache_dir=cache_dir,
                    headers={
                        "User-Agent": (
                            "f3dasm-agent/1.0"
                            " (mailto:f3dasm@brown.edu)"
                        ),
                    },
                )
            except SourceCooldownError as exc:
                return f"ERROR: {exc}"
            except Exception as exc:
                return f"ERROR: DownloadPdf fetch failed: {exc}"

            # Validate content
            ct = ""
            if hasattr(resp, "headers"):
                ct = resp.headers.get("Content-Type", "")
            elif hasattr(resp, "_content_type"):
                ct = resp._content_type
            body = resp.content
            if "pdf" not in ct.lower() and not body.startswith(
                b"%PDF"
            ):
                return (
                    "ERROR: not a PDF — content-type is"
                    f" {ct!r} and body does not start with %PDF."
                    " Check the URL."
                )

            dest = _Path(filename)
            if not dest.is_absolute():
                dest = corpus._papers_dir / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            return str(dest)

        def get_openalex_citations(
            work_id: str, n_results: int = 20
        ) -> str:
            """Fetch papers that cite *work_id* via OpenAlex citation-graph traversal; works during Semantic Scholar cooldowns.

            work_id may be a bare OpenAlex ID (``W123…``) or a full URL
            (``https://openalex.org/W123…``).  Returns a JSON list of
            ``{id, title, year, cited_by_count, doi, pdf_url}`` sorted
            by citation count descending.
            """
            import json as _j
            # Normalise: strip full URL prefix if present
            _wid = work_id.strip()
            if _wid.startswith("https://openalex.org/"):
                _wid = _wid[len("https://openalex.org/"):]
            try:
                resp = _robust_get(
                    "https://api.openalex.org/works",
                    params={
                        "filter": f"cites:{_wid}",
                        "per-page": min(int(n_results), 50),
                        "sort": "cited_by_count:desc",
                        "select": (
                            "id,title,publication_year,doi"
                            ",cited_by_count,best_oa_location"
                            ",open_access,primary_location"
                        ),
                    },
                    headers={
                        "User-Agent": (
                            "f3dasm-agent/1.0"
                            " (mailto:f3dasm@brown.edu)"
                        ),
                    },
                    cache_dir=cache_dir,
                )
                works = resp.json().get("results", [])
            except SourceCooldownError as exc:
                return f"ERROR: {exc}"
            except Exception as exc:
                return f"ERROR: OpenAlex citations failed: {exc}"

            out = []
            for w in works:
                doi = (w.get("doi") or "").replace(
                    "https://doi.org/", ""
                )
                boa = w.get("best_oa_location") or {}
                loc = w.get("primary_location") or {}
                oa = w.get("open_access") or {}
                pdf_url = (
                    boa.get("pdf_url")
                    or oa.get("oa_url")
                    or loc.get("pdf_url")
                    or loc.get("landing_page_url")
                    or ""
                )
                out.append({
                    "id": w.get("id", ""),
                    "title": w.get("title", ""),
                    "year": w.get("publication_year", ""),
                    "cited_by_count": w.get("cited_by_count", 0),
                    "doi": doi,
                    "pdf_url": pdf_url,
                })
            return _j.dumps(out, indent=2)

        def get_openalex_references(work_id: str) -> str:
            """Fetch the reference list of *work_id* hydrated from OpenAlex; citation-graph traversal fallback when S2 is rate-limited.

            Returns a JSON list of ``{id, title, year, cited_by_count,
            doi, pdf_url}`` for the first 40 referenced works, or a
            plain message when no references are listed.
            """
            import json as _j
            _wid = work_id.strip()
            if _wid.startswith("https://openalex.org/"):
                _wid = _wid[len("https://openalex.org/"):]
            try:
                work_resp = _robust_get(
                    f"https://api.openalex.org/works/{_wid}",
                    params={
                        "select": "id,referenced_works",
                    },
                    headers={
                        "User-Agent": (
                            "f3dasm-agent/1.0"
                            " (mailto:f3dasm@brown.edu)"
                        ),
                    },
                    cache_dir=cache_dir,
                )
                ref_ids = work_resp.json().get(
                    "referenced_works", []
                )[:40]
            except SourceCooldownError as exc:
                return f"ERROR: {exc}"
            except Exception as exc:
                return f"ERROR: OpenAlex references failed: {exc}"

            if not ref_ids:
                return f"No references listed for {_wid}."

            # Strip URL prefixes to get bare W-ids for the filter
            bare_ids = [
                r.replace("https://openalex.org/", "")
                for r in ref_ids
            ]
            filter_str = "openalex_id:" + "|".join(bare_ids)
            try:
                hydrate_resp = _robust_get(
                    "https://api.openalex.org/works",
                    params={
                        "filter": filter_str,
                        "per-page": 50,
                        "select": (
                            "id,title,publication_year,doi"
                            ",cited_by_count,best_oa_location"
                            ",open_access,primary_location"
                        ),
                    },
                    headers={
                        "User-Agent": (
                            "f3dasm-agent/1.0"
                            " (mailto:f3dasm@brown.edu)"
                        ),
                    },
                    cache_dir=cache_dir,
                )
                works = hydrate_resp.json().get("results", [])
            except SourceCooldownError as exc:
                return f"ERROR: {exc}"
            except Exception as exc:
                return f"ERROR: OpenAlex reference hydration failed: {exc}"

            out = []
            for w in works:
                doi = (w.get("doi") or "").replace(
                    "https://doi.org/", ""
                )
                boa = w.get("best_oa_location") or {}
                loc = w.get("primary_location") or {}
                oa = w.get("open_access") or {}
                pdf_url = (
                    boa.get("pdf_url")
                    or oa.get("oa_url")
                    or loc.get("pdf_url")
                    or loc.get("landing_page_url")
                    or ""
                )
                out.append({
                    "id": w.get("id", ""),
                    "title": w.get("title", ""),
                    "year": w.get("publication_year", ""),
                    "cited_by_count": w.get("cited_by_count", 0),
                    "doi": doi,
                    "pdf_url": pdf_url,
                })
            return _j.dumps(out, indent=2)

        tools["search_openalex"] = search_openalex
        tools["get_openalex_citations"] = get_openalex_citations
        tools["get_openalex_references"] = get_openalex_references
        tools["get_semantic_scholar_recommendations"] = (
            get_semantic_scholar_recommendations
        )
        tools["CorpusRank"] = CorpusRank
        tools["DownloadPdf"] = DownloadPdf

        # arxiv tools — Python-native, same for Claude and Ollama.
        from ..backends.ollama import _build_arxiv_closures
        tools.update(_build_arxiv_closures())

        return tools
