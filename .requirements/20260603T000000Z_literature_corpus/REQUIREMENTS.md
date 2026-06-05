# REQUIREMENTS: LiteratureCorpus

## As Is

The `LiteratureReviewAgent` (`src/f3dasm/_src/agentic/agents/literature.py`) depends on
Zotero and arxiv MCP servers, which are external processes. There is no local canonical
corpus that the agent owns on disk. The agent writes free-form notes to
`workspace/literature_notes.md` but has no structured, searchable representation of
the papers it has seen.

There is no `literature_corpus.py` module in `src/f3dasm/_src/agentic/`.
There are no tests in `tests/agentic/test_literature_corpus.py`.

## To Be

A new `LiteratureCorpus` class at
`src/f3dasm/_src/agentic/literature_corpus.py` that:

- Owns a corpus directory (`workspace/literature/` by default).
- Maintains a `corpus.csv` metadata index.
- Stores per-paper subdirectories under `papers/`.
- Exposes `add(identifier)`, `search(query, top_k)`, `get_paper(paper_id)`,
  `list_papers()` as the main API.
- Supports arxiv IDs, `arxiv:` prefix, DOI strings (`doi:` prefix or bare
  `10.xxxx/xxx`), and Semantic Scholar paper IDs.
- Downloads PDFs using the `arxiv` Python library (arxiv IDs) or via Unpaywall
  (DOIs). Gracefully degrades when download fails (adds metadata without text).
- Extracts text to page-annotated Markdown using `pymupdf` (`fitz`). Falls back
  to `"(PDF extraction unavailable)"` if `fitz` is not installed.
- Is thread-safe: a `threading.Lock()` protects CSV read/write.

CSV schema:
`paper_id, title, authors, year, doi, arxiv_id, venue, abstract,
local_pdf_path, local_md_path, added_at, source`

## Requirements

1. `__init__(corpus_dir)` creates `corpus_dir` and `corpus_dir/papers/` if
   they do not exist.
2. `add(identifier)` normalises the identifier, assigns a slug `paper_id`,
   checks for duplicates, downloads the PDF (best effort), extracts text,
   updates `corpus.csv`, and returns the `paper_id` on success or
   `"ERROR: …"` on failure.
3. `add()` returns `"Already in corpus: {paper_id}"` when the same paper is
   added a second time.
4. `list_papers()` returns a Markdown table
   (`paper_id | title | authors | year | source`) or `"Corpus is empty."`.
5. `search(query, top_k=10)` scans all `paper.md` files for lines containing
   the query (case-insensitive), returns up to `top_k` formatted passages, or
   `"No results found."`.
6. `get_paper(paper_id)` returns full Markdown of the paper or
   `"ERROR: paper '{paper_id}' not found in corpus."`.
7. `_extract_pdf_to_md(pdf_path)` returns page-annotated Markdown; falls back
   to `"(PDF extraction unavailable)"` when `fitz` is unavailable.
8. `_normalize_identifier(identifier)` maps:
   - `"2312.12345"` or `"arxiv:2312.12345"` → `("arxiv_2312_12345", "arxiv")`
   - `"doi:10.1234/test"` or `"10.1234/test"` → `("doi_10_1234_test", "doi")`
   - Other strings → sanitised slug + `"unknown"`
9. CSV read/write protected by `threading.Lock()` for thread safety.

## Acceptance Criteria

1. After `LiteratureCorpus(tmp_path / "lit")`, `tmp_path / "lit"` and
   `tmp_path / "lit" / "papers"` both exist on disk.
2. After a successful `add()`, `corpus.csv` exists and has at least one data
   row.
3. `list_papers()` on a freshly-created corpus returns exactly
   `"Corpus is empty."`.
4. After `add()` succeeds, `list_papers()` output includes the paper's
   `paper_id`.
5. `search("nonexistent_xyz")` returns `"No results found."`.
6. `get_paper("not_a_real_id")` returns a string starting with `"ERROR"`.
7. `add(same_id_twice)` second call returns `"Already in corpus: {paper_id}"`.
8. `_extract_pdf_to_md(real_pdf)` output contains `<!-- page 1 -->`.
9. `_normalize_identifier("2312.12345")` == `("arxiv_2312_12345", "arxiv")`.
10. `_normalize_identifier("arxiv:2312.12345")` == `("arxiv_2312_12345", "arxiv")`.
11. `_normalize_identifier("doi:10.1234/test")` == `("doi_10_1234_test", "doi")`.
12. `search()` finds text that was manually placed in a paper's `paper.md`.
13. Two threads adding different papers concurrently → `corpus.csv` has exactly
    two rows and is valid CSV.

## Testing Plan

All tests written **before** implementation (TDD red phase). Mocking used to
avoid network calls in every test.

Test file: `tests/agentic/test_literature_corpus.py`

| Test | Strategy |
|------|----------|
| `test_corpus_creates_dir_on_init` | assert both dirs exist after init |
| `test_corpus_csv_created_on_first_add` | mock `_fetch_and_add_arxiv`; patch network; assert CSV exists |
| `test_list_papers_empty_corpus` | no papers added; check return value |
| `test_list_papers_shows_added_paper` | inject a row into CSV via `_save_csv`; call `list_papers` |
| `test_search_no_results` | no papers; assert "No results found." |
| `test_get_paper_not_found` | call `get_paper("bad_id")`; assert starts with "ERROR" |
| `test_add_already_in_corpus` | add same row via CSV, call add() again; assert "Already in corpus" |
| `test_extract_pdf_to_md_page_annotations` | create minimal PDF with reportlab or use pre-existing PDF fixture |
| `test_normalize_arxiv_id` | call `_normalize_identifier("2312.12345")` |
| `test_normalize_arxiv_prefix` | call `_normalize_identifier("arxiv:2312.12345")` |
| `test_normalize_doi` | call `_normalize_identifier("doi:10.1234/test")` |
| `test_search_finds_passage` | write known text to a paper.md; verify search finds it |
| `test_concurrent_add_is_thread_safe` | two threads, patched add, join → check CSV rows |

Network calls (`requests`, `arxiv`) are patched with `unittest.mock.patch`.

## Implementation Plan

### Step 1 — Stub file (import only)
Create `literature_corpus.py` with an empty `LiteratureCorpus` class.
Test: `from f3dasm._src.agentic.literature_corpus import LiteratureCorpus` succeeds.

### Step 2 — `__init__` + directory creation
Implement `__init__`, `_paper_dir`, `_load_csv`, `_save_csv`.
Test: `test_corpus_creates_dir_on_init`.

### Step 3 — `_normalize_identifier`
Implement the regex-based normalisation.
Tests: `test_normalize_arxiv_id`, `test_normalize_arxiv_prefix`, `test_normalize_doi`.

### Step 4 — `list_papers` (empty)
Return `"Corpus is empty."` when CSV missing/empty.
Test: `test_list_papers_empty_corpus`.

### Step 5 — `get_paper` (not found)
Return ERROR string when `paper.md` not found.
Test: `test_get_paper_not_found`.

### Step 6 — `search` (no results)
Implement grep over paper.md files; return "No results found." when no files.
Test: `test_search_no_results`.

### Step 7 — `add` with full mocking
Implement `add()` with patched arxiv/requests. Add metadata to CSV. Return paper_id.
Tests: `test_corpus_csv_created_on_first_add`, `test_add_already_in_corpus`.

### Step 8 — `list_papers` (with entries)
Implement Markdown table rendering.
Test: `test_list_papers_shows_added_paper`.

### Step 9 — `search` (with results)
Test: `test_search_finds_passage`.

### Step 10 — `_extract_pdf_to_md`
Implement with fitz; fallback when unavailable.
Test: `test_extract_pdf_to_md_page_annotations`.

### Step 11 — Thread safety
Test: `test_concurrent_add_is_thread_safe`.

### Step 12 — Add dependencies to pyproject.toml
Add `arxiv` and `pymupdf` under `[project.optional-dependencies] agentic`.
