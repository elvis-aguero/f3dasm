"""LiteratureCorpus — owns a canonical paper corpus on disk.

Manages a directory with::

    corpus_dir/
        corpus.csv          — metadata index
        papers/
            <paper_id>/
                paper.pdf   — source file (optional copy)
                paper.md    — extracted text, page-annotated

**Design principle:** this module does NO network I/O.  Discovery and
download are the agent's responsibility via MCP tools.  ``CorpusAdd``
only accepts paths to files already on disk.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

try:
    import fitz  # type: ignore[import]
except ImportError:
    fitz = None  # type: ignore[assignment]

__all__ = ["LiteratureCorpus", "_robust_get", "_robust_post"]


# ---------------------------------------------------------------------------
# Robust HTTP helpers (retry + timeout) used by closure tools in literature.py
# ---------------------------------------------------------------------------

def _robust_get(url: str, *, params=None, headers=None, retries: int = 3,
                timeout: float = 15.0):
    """GET *url* with exponential back-off retry and connection timeout.

    Returns a :class:`requests.Response` on success; raises on final failure.
    """
    import time
    import requests  # type: ignore[import]

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers,
                                timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise last_exc  # type: ignore[misc]


def _robust_post(url: str, *, json=None, params=None, headers=None,
                 retries: int = 3, timeout: float = 15.0):
    """POST *url* with exponential back-off retry and connection timeout.

    Returns a :class:`requests.Response` on success; raises on final failure.
    """
    import time
    import requests  # type: ignore[import]

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=json, params=params, headers=headers,
                                 timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise last_exc  # type: ignore[misc]

_CSV_FIELDS = [
    "paper_id",
    "title",
    "authors",
    "year",
    "doi",
    "arxiv_id",
    "venue",
    "abstract",
    "local_pdf_path",
    "local_md_path",
    "added_at",
    "source",
    "citation_count",
]

_ARXIV_BARE_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_DOI_BARE_RE = re.compile(r"^10\.\d{4,}/\S+$")


def _slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", s).strip("_")


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


_PUNCT_RE = re.compile(r"[^\w\-]", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase and strip trailing/leading punctuation from each token.

    Preserves hyphens so that ``self-attention`` stays as one token.
    """
    return [
        _PUNCT_RE.sub("", w).lower()
        for w in text.split()
        if _PUNCT_RE.sub("", w)
    ]


class LiteratureCorpus:
    """Local paper corpus: extraction, indexing, search.

    No network I/O — the agent downloads files via MCP tools and hands
    local paths to :meth:`add`.

    Parameters
    ----------
    corpus_dir:
        Root directory.  Created on construction if absent.
    """

    def __init__(self, corpus_dir: Path) -> None:
        self._corpus_dir = Path(corpus_dir)
        self._papers_dir = self._corpus_dir / "papers"
        self._csv_path = self._corpus_dir / "corpus.csv"
        self._chunks_path = self._corpus_dir / "chunks.jsonl"
        self._lock = threading.Lock()
        self._corpus_dir.mkdir(parents=True, exist_ok=True)
        self._papers_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_model = None  # lazy-loaded on first CorpusAdd

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        source: str,
        title: str = "",
        authors: str = "",
        year: str = "",
        doi: str = "",
        arxiv_id: str = "",
        venue: str = "",
        abstract: str = "",
        citation_count: int = 0,
    ) -> str:
        """Add a paper to the corpus from a local file.

        Parameters
        ----------
        source:
            Absolute or relative path to a PDF (``.pdf``) or extracted
            text (``.md`` / ``.txt``) file already on disk.  Download
            the file first with ``mcp__arxiv__download_paper`` or write
            the text returned by ``mcp__arxiv__read_paper`` via the
            Write tool.
        title, authors, year, doi, arxiv_id, venue, abstract:
            Optional metadata.  Pass values obtained from the MCP search
            result that identified this paper.

        Returns
        -------
        str
            ``paper_id`` on success, ``"Already in corpus: {id}"``,
            or ``"ERROR: …"`` on failure.
        """
        src = Path(source)
        if not src.exists():
            return f"ERROR: file not found: {source!r}. Download it first."

        # Derive a stable paper_id from arxiv_id, doi, or filename
        paper_id = self._derive_paper_id(arxiv_id, doi, src)

        with self._lock:
            rows = self._load_csv()
            if any(r["paper_id"] == paper_id for r in rows):
                return f"Already in corpus: {paper_id}"

        paper_dir = self._paper_dir(paper_id)
        paper_dir.mkdir(parents=True, exist_ok=True)

        # Copy the source file into the corpus directory
        suffix = src.suffix.lower()
        local_pdf_path = ""
        local_md_path = ""
        md_content = ""

        if suffix == ".pdf":
            dest_pdf = paper_dir / "paper.pdf"
            if src.resolve() != dest_pdf.resolve():
                shutil.copy2(src, dest_pdf)
            local_pdf_path = str(dest_pdf)
            # Extract text
            md_content = self._extract_pdf_to_md(dest_pdf)
            dest_md = paper_dir / "paper.md"
            dest_md.write_text(md_content, encoding="utf-8")
            local_md_path = str(dest_md)
        elif suffix in {".md", ".txt"}:
            dest_md = paper_dir / "paper.md"
            if src.resolve() != dest_md.resolve():
                shutil.copy2(src, dest_md)
            md_content = dest_md.read_text(encoding="utf-8")
            local_md_path = str(dest_md)
        else:
            return (
                f"ERROR: unsupported file type {suffix!r}. "
                "Provide a .pdf, .md, or .txt file."
            )

        # Chunk and index the text
        chunks = self._chunk_text(md_content)
        self._append_chunks(paper_id, chunks)

        # Embed chunks and save per-paper .npy for dense search
        model = self._get_embedding_model()
        if model is not None and chunks:
            import numpy as _np
            texts = [c["text"] for c in chunks]
            embs = _np.array(list(model.embed(texts)), dtype=_np.float32)
            _np.save(str(paper_dir / "chunks.npy"), embs)

        # Infer source label
        source_label = "arxiv" if arxiv_id else ("doi" if doi else "local")

        row = {
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "year": year,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "venue": venue,
            "abstract": abstract,
            "local_pdf_path": local_pdf_path,
            "local_md_path": local_md_path,
            "added_at": _now_iso(),
            "source": source_label,
            "citation_count": str(int(citation_count)),
        }

        with self._lock:
            rows = self._load_csv()
            if any(r["paper_id"] == paper_id for r in rows):
                return f"Already in corpus: {paper_id}"
            rows.append(row)
            self._save_csv(rows)

        return paper_id

    def _get_embedding_model(self):
        """Lazy-load bge-small-en-v1.5 via fastembed. Returns None if unavailable."""
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            from fastembed import TextEmbedding
            self._embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        except Exception:
            self._embedding_model = None
        return self._embedding_model

    def _load_all_embeddings(self) -> "tuple[list[dict], object]":
        """Load all chunks and their embeddings. Returns (chunks, embeddings_matrix) or (chunks, None)."""
        import numpy as _np
        chunks = self._load_chunks()
        if not chunks:
            return chunks, None

        # Group chunks by paper_id to load matching .npy files
        paper_embs: dict = {}
        for paper_id in dict.fromkeys(c["paper_id"] for c in chunks):
            npy_path = self._paper_dir(paper_id) / "chunks.npy"
            if npy_path.exists():
                paper_embs[paper_id] = _np.load(str(npy_path))

        if not paper_embs:
            return chunks, None

        # Build aligned embedding matrix (rows match chunks order)
        paper_chunk_idx: dict = {}
        rows = []
        for chunk in chunks:
            pid = chunk["paper_id"]
            if pid not in paper_embs:
                return chunks, None  # missing embeddings for at least one paper → fallback
            idx = paper_chunk_idx.get(pid, 0)
            if idx >= len(paper_embs[pid]):
                return chunks, None
            rows.append(paper_embs[pid][idx])
            paper_chunk_idx[pid] = idx + 1

        return chunks, _np.array(rows, dtype=_np.float32)

    def search(self, query: str, top_k: int = 10) -> str:
        """Search all paper.md files for passages relevant to *query*.

        Uses Reciprocal Rank Fusion (BM25 + dense embeddings) when both
        ``rank_bm25`` and ``fastembed`` are installed.  Falls back to
        BM25-only when dense embeddings are unavailable, or to substring
        search when rank_bm25 is also absent.

        Returns up to *top_k* formatted passages with page citations,
        or ``"No results found."``.
        """
        import math as _math
        import numpy as _np

        chunks, emb_matrix = self._load_all_embeddings()
        if not chunks:
            return "No results found."

        rows = self._load_csv()
        citation_counts = {r["paper_id"]: int(r.get("citation_count") or 0) for r in rows}
        meta = {r["paper_id"]: r for r in rows}

        K_RRF = 60  # standard RRF constant

        # --- BM25 ranking ---
        bm25_ranks: dict = {}
        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import]
            tokenized = [_tokenize(c["text"]) for c in chunks]
            bm25 = BM25Okapi(tokenized)
            bm25_scores = bm25.get_scores(_tokenize(query))
            # Apply citation weight
            weighted = [
                bm25_scores[i] * (1 + _math.log10(citation_counts.get(chunks[i]["paper_id"], 0) + 1))
                for i in range(len(chunks))
            ]
            bm25_order = sorted(range(len(chunks)), key=lambda i: weighted[i], reverse=True)
            for rank, idx in enumerate(bm25_order):
                bm25_ranks[idx] = rank
        except ImportError:
            return self._search_substring(query, top_k)

        # --- Dense ranking (if embeddings available) ---
        dense_ranks: dict = {}
        if emb_matrix is not None:
            model = self._get_embedding_model()
            if model is not None:
                q_emb = _np.array(next(model.embed([query])), dtype=_np.float32)
                q_norm = q_emb / (_np.linalg.norm(q_emb) + 1e-9)
                norms = _np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                emb_n = emb_matrix / (norms + 1e-9)
                cos_scores = emb_n @ q_norm
                dense_order = _np.argsort(cos_scores)[::-1].tolist()
                for rank, idx in enumerate(dense_order):
                    dense_ranks[idx] = rank

        # --- RRF fusion ---
        if dense_ranks:
            rrf_scores = {
                i: (2.0 / 3.0) / (K_RRF + dense_ranks.get(i, len(chunks)))
                  + (1.0 / 3.0) / (K_RRF + bm25_ranks.get(i, len(chunks)))
                for i in range(len(chunks))
            }
        else:
            rrf_scores = {
                i: 1.0 / (K_RRF + bm25_ranks.get(i, len(chunks)))
                for i in range(len(chunks))
            }

        top_idx = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:top_k]

        passages = []
        for idx in top_idx:
            if rrf_scores[idx] <= 0:
                break
            chunk = chunks[idx]
            pid = chunk["paper_id"]
            m = meta.get(pid, {})
            title = m.get("title", pid)
            year = m.get("year", "")
            passages.append(f"--- {title} ({year}), p.{chunk['page']} ---\n{chunk['text']}\n")

        return "\n".join(passages) if passages else "No results found."

    def _search_substring(self, query: str, top_k: int) -> str:
        """Fallback substring search (used when rank_bm25 is not installed)."""
        query_lower = query.lower()
        rows = self._load_csv()
        passages: list[str] = []

        for row in rows:
            if len(passages) >= top_k:
                break
            md_path_str = row.get("local_md_path", "")
            if not md_path_str:
                continue
            md_path = Path(md_path_str)
            if not md_path.exists():
                continue

            text = md_path.read_text(encoding="utf-8")
            lines = text.splitlines()
            current_page = 1

            for i, line in enumerate(lines):
                if len(passages) >= top_k:
                    break
                page_match = re.match(r"<!--\s*page\s*(\d+)\s*-->", line)
                if page_match:
                    current_page = int(page_match.group(1))
                    continue
                if query_lower in line.lower():
                    start = max(0, i - 1)
                    end = min(len(lines), i + 2)
                    context = "\n".join(lines[start:end])
                    title = row.get("title", row["paper_id"])
                    year = row.get("year", "")
                    passages.append(
                        f"--- {title} ({year}), p.{current_page} ---\n{context}\n"
                    )

        return "\n".join(passages) if passages else "No results found."

    def get_paper(self, paper_id: str) -> str:
        """Return full extracted Markdown of *paper_id*, or ``"ERROR: …"``."""
        for row in self._load_csv():
            if row["paper_id"] == paper_id:
                md_path_str = row.get("local_md_path", "")
                if not md_path_str:
                    return (
                        f"ERROR: paper '{paper_id}' has no extracted text. "
                        "Add it with CorpusAdd using a local PDF or .md file."
                    )
                md_path = Path(md_path_str)
                if not md_path.exists():
                    return f"ERROR: text file not found at {md_path_str!r}."
                return md_path.read_text(encoding="utf-8")
        return f"ERROR: paper '{paper_id}' not found in corpus."

    def list_papers(self) -> str:
        """Return a Markdown table of all papers, or ``"Corpus is empty."``."""
        rows = self._load_csv()
        if not rows:
            return "Corpus is empty."
        header = "| paper_id | title | authors | year | source |"
        sep = "| --- | --- | --- | --- | --- |"
        lines = [header, sep]
        for r in rows:
            lines.append(
                f"| {r.get('paper_id','')} | {r.get('title','')} | "
                f"{r.get('authors','')} | {r.get('year','')} | {r.get('source','')} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # PDF extraction
    # ------------------------------------------------------------------

    def _extract_pdf_to_md(self, pdf_path: Path) -> str:
        """Extract page-annotated Markdown from *pdf_path*.

        Tries Docling first (layout-aware, table recognition, structured
        output), then falls back to pymupdf, then returns a placeholder.
        """
        # Try Docling first (layout-aware, table recognition, structured output)
        try:
            from docling.document_converter import DocumentConverter  # type: ignore[import]

            converter = DocumentConverter()
            result = converter.convert(str(pdf_path))
            md = result.document.export_to_markdown()
            if md and len(md.strip()) > 100:
                return md
        except Exception:
            pass  # fall through to pymupdf

        # pymupdf fallback
        if fitz is not None:
            try:
                doc = fitz.open(str(pdf_path))
                parts: list[str] = []
                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text()
                    parts.append(f"<!-- page {page_num} -->\n{text}")
                doc.close()
                return "\n\n".join(parts)
            except Exception:
                pass

        return "(PDF extraction unavailable — install docling or pymupdf)"

    # ------------------------------------------------------------------
    # Chunking infrastructure
    # ------------------------------------------------------------------

    def _chunk_text(
        self, md_text: str, chunk_size: int = 150, overlap: int = 50
    ) -> list[dict]:
        """Split page-annotated markdown into overlapping word chunks.

        Respects ``<!-- page N -->`` markers: each chunk records its
        starting page.

        Returns
        -------
        list[dict]
            Each entry has ``{"page": int, "text": str}``.
        """
        lines = md_text.splitlines()
        current_page = 1
        all_words: list[tuple[str, int]] = []  # (word, page)
        for line in lines:
            pm = re.match(r"<!--\s*page\s*(\d+)\s*-->", line)
            if pm:
                current_page = int(pm.group(1))
                continue
            for w in line.split():
                all_words.append((w, current_page))

        chunks: list[dict] = []
        i = 0
        while i < len(all_words):
            window = all_words[i : i + chunk_size]
            if not window:
                break
            page = window[0][1]
            text = " ".join(w for w, _ in window)
            chunks.append({"page": page, "text": text})
            i += max(1, chunk_size - overlap)
        return chunks

    def _append_chunks(self, paper_id: str, chunks: list[dict]) -> None:
        """Append *chunks* for *paper_id* to chunks.jsonl."""
        with self._lock:
            with self._chunks_path.open("a", encoding="utf-8") as f:
                for i, chunk in enumerate(chunks):
                    f.write(
                        json.dumps(
                            {
                                "paper_id": paper_id,
                                "chunk_id": i,
                                "page": chunk["page"],
                                "text": chunk["text"],
                            }
                        )
                        + "\n"
                    )

    def _load_chunks(self) -> list[dict]:
        """Load all chunks from chunks.jsonl."""
        if not self._chunks_path.exists():
            return []
        with self._lock:
            with self._chunks_path.open(encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_paper_id(self, arxiv_id: str, doi: str, src: Path) -> str:
        """Derive a stable paper_id from available metadata."""
        if arxiv_id:
            return "arxiv_" + _slugify(arxiv_id.strip())
        if doi:
            return "doi_" + _slugify(doi.strip())
        # Fall back to the source filename without extension
        return _slugify(src.stem) or "paper_unknown"

    def _paper_dir(self, paper_id: str) -> Path:
        return self._papers_dir / paper_id

    def _load_csv(self) -> list[dict]:
        if not self._csv_path.exists():
            return []
        with self._csv_path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _save_csv(self, rows: list[dict]) -> None:
        with self._csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in _CSV_FIELDS})
