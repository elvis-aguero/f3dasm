"""Ollama-backed adapter for the f3dasm LangGraph agentic runtime.

Uses ChatOpenAI pointed at Ollama's OpenAI-compatible endpoint so that
LangGraph's tool-calling machinery (ToolNode, tools_condition) works
against the well-tested OpenAI code path.  create_react_agent handles
the full tool-execution loop; the adapter exposes the same
closure_tools / invoke() interface as ClaudeAdapter.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

__all__ = ["OllamaAdapter"]


def _to_lc_messages(messages: list[dict]) -> list:
    from langchain_core.messages import AIMessage, HumanMessage

    result = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        if role in ("human", "user"):
            result.append(HumanMessage(content=content))
        elif role in ("ai", "assistant"):
            result.append(AIMessage(content=content))
    return result


def _make_edit_tool(cwd: Path | None) -> Any:
    from langchain_core.tools import StructuredTool

    workspace = (cwd or Path(".")).resolve()

    def edit_file(path: str, old_str: str, new_str: str) -> str:
        """Replace old_str with new_str in file at path (first occurrence).
        Restricted to the workspace directory."""
        try:
            candidate = (workspace / path).resolve()
            candidate.relative_to(workspace)
        except ValueError:
            return f"ERROR: edit rejected — {path!r} resolves outside workspace."
        except Exception as exc:
            return f"ERROR: invalid path {path!r}: {exc}"
        if not candidate.exists():
            return f"ERROR: {path} not found"
        text = candidate.read_text(encoding="utf-8")
        if old_str not in text:
            return f"ERROR: string not found in {path}"
        candidate.write_text(text.replace(old_str, new_str, 1), encoding="utf-8")
        return f"Edited {candidate}"

    return StructuredTool.from_function(edit_file, name="Edit")


def _make_grep_tool() -> Any:
    from langchain_core.tools import StructuredTool

    def grep_files(pattern: str, path: str = ".") -> str:
        """Search for pattern in files under path; return matching lines."""
        import subprocess

        try:
            result = subprocess.run(
                ["grep", "-r", "-n", pattern, path],
                capture_output=True, text=True, timeout=30,
            )
            return result.stdout or "(no matches)"
        except Exception as exc:
            return f"ERROR: {exc}"

    return StructuredTool.from_function(grep_files, name="Grep")


def _make_glob_tool(cwd: Path | None) -> Any:
    from langchain_core.tools import StructuredTool

    base = cwd or Path(".")

    def glob_files(pattern: str) -> str:
        """Find files matching a glob pattern relative to the workspace."""
        matches = sorted(str(p) for p in base.glob(pattern))
        return "\n".join(matches) if matches else "(no matches)"

    return StructuredTool.from_function(glob_files, name="Glob")


def _make_bash_tool(cwd: Path | None, nudge: Any = None) -> Any:
    import subprocess

    from langchain_core.tools import StructuredTool

    work_dir = str(cwd) if cwd else None

    def bash(command: str) -> str:
        """Run a shell command and return stdout + stderr."""
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=120, cwd=work_dir,
        )
        out = result.stdout + result.stderr
        out = out if out else "(no output)"
        if nudge is not None:
            msg = nudge("Bash", {"command": command})
            if msg:
                out = f"{out}\n\n{msg}"
        return out

    return StructuredTool.from_function(bash, name="Bash")


def _make_read_tool(cwd: Path | None) -> Any:
    from langchain_core.tools import StructuredTool

    def read_file(path: str) -> str:
        """Read a file and return its contents."""
        p = (
            Path(path) if Path(path).is_absolute()
            else (cwd or Path(".")) / path
        )
        if not p.exists():
            return f"ERROR: {p} not found"
        return p.read_text(encoding="utf-8")

    return StructuredTool.from_function(read_file, name="Read")


def _make_write_tool(cwd: Path | None, nudge: Any = None) -> Any:
    from langchain_core.tools import StructuredTool

    def write_file(path: str, content: str) -> str:
        """Write content to a file, creating it if it doesn't exist."""
        p = (
            Path(path) if Path(path).is_absolute()
            else (cwd or Path(".")) / path
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        out = f"Written: {p}"
        if nudge is not None:
            msg = nudge("Write", {"file_path": path, "content": content})
            if msg:
                out = f"{out}\n\n{msg}"
        return out

    return StructuredTool.from_function(write_file, name="Write")


def _native_tool_map(cwd: Path | None, nudge: Any = None) -> dict[str, Any]:
    return {
        "Bash":  _make_bash_tool(cwd, nudge),
        "Read":  _make_read_tool(cwd),
        "Write": _make_write_tool(cwd, nudge),
        "Edit":  _make_edit_tool(cwd),
        "Glob":  _make_glob_tool(cwd),
        "Grep":  _make_grep_tool(),
    }


def _build_arxiv_closures() -> dict:
    """Return arxiv tool callables keyed by their MCP-compatible tool name.

    Returns an empty dict if the ``arxiv`` package is not installed.
    Used by both OllamaAdapter (via _make_literature_tools) and
    LiteratureReviewAgent.build_closure_tools() so both backends share
    identical arxiv tool implementations.
    """
    try:
        import arxiv as _arxiv
    except ImportError:
        return {}

    def search_papers(query: str, max_results: int = 10) -> str:
        """Search arxiv for papers matching query."""
        client = _arxiv.Client()
        results = list(client.results(_arxiv.Search(query=query, max_results=max_results)))
        lines = []
        for r in results:
            lines.append(f"[{r.entry_id}] {r.title} ({r.published.year})\n  {r.summary[:300]}")
        return "\n\n".join(lines) or "(no results)"

    def list_papers(category: str, max_results: int = 10) -> str:
        """List recent arxiv papers in a category (e.g. 'cs.LG')."""
        client = _arxiv.Client()
        results = list(client.results(_arxiv.Search(query=f"cat:{category}", max_results=max_results)))
        return "\n".join(f"[{r.entry_id}] {r.title}" for r in results) or "(no results)"

    def download_paper(paper_id: str, output_dir: str = ".") -> str:
        """Download a paper PDF from arxiv by its ID."""
        client = _arxiv.Client()
        paper = next(client.results(_arxiv.Search(id_list=[paper_id])))
        path = paper.download_pdf(dirpath=output_dir)
        return f"Downloaded: {path}"

    def read_paper(paper_id: str) -> str:
        """Download and extract text from an arxiv paper."""
        import tempfile
        client = _arxiv.Client()
        paper = next(client.results(_arxiv.Search(id_list=[paper_id])))
        with tempfile.TemporaryDirectory() as tmp:
            path = paper.download_pdf(dirpath=tmp)
            try:
                import fitz
                doc = fitz.open(path)
                return "\n".join(page.get_text() for page in doc)
            except ImportError:
                import subprocess
                result = subprocess.run(
                    ["pdftotext", path, "-"], capture_output=True, text=True
                )
                return result.stdout or "(could not extract text — install pymupdf or pdftotext)"

    return {
        "mcp__arxiv__search_papers": search_papers,
        "mcp__arxiv__list_papers": list_papers,
        "mcp__arxiv__download_paper": download_paper,
        "mcp__arxiv__read_paper": read_paper,
    }


def _make_literature_tools() -> list:
    """Build LangChain StructuredTools equivalent to the literature MCP stack."""
    from langchain_core.tools import StructuredTool

    tools = []

    # --- arxiv (via shared _build_arxiv_closures) ---
    for name, fn in _build_arxiv_closures().items():
        tools.append(StructuredTool.from_function(fn, name=name))

    # --- semantic scholar ---
    try:
        from semanticscholar import SemanticScholar as _SS

        _ss = _SS()

        def get_paper_details(paper_id: str) -> str:
            """Get details for a paper by its S2, DOI, or arxiv ID."""
            import json
            paper = _ss.get_paper(paper_id, fields=[
                "title", "year", "venue", "citationCount",
                "influentialCitationCount", "tldr", "authors",
            ])
            return json.dumps({
                "title": paper.title,
                "year": paper.year,
                "venue": paper.venue,
                "citations": paper.citationCount,
                "influential_citations": paper.influentialCitationCount,
                "tldr": paper.tldr.get("text") if paper.tldr else None,
                "authors": [a["name"] for a in (paper.authors or [])],
            }, indent=2)

        def get_citations_and_references(paper_id: str) -> str:
            """Get citing papers and references for a paper."""
            import json
            paper = _ss.get_paper(paper_id, fields=["citations", "references"])
            refs = [{"title": r.get("title"), "paperId": r.get("paperId")} for r in (paper.references or [])]
            cits = [{"title": c.get("title"), "paperId": c.get("paperId")} for c in (paper.citations or [])]
            return json.dumps({"references": refs[:20], "citations": cits[:20]}, indent=2)

        tools += [
            StructuredTool.from_function(
                get_paper_details,
                name="mcp__semanticscholar__get_semantic_scholar_paper_details",
            ),
            StructuredTool.from_function(
                get_citations_and_references,
                name="mcp__semanticscholar__get_semantic_scholar_citations_and_references",
            ),
        ]
    except ImportError:
        pass  # semanticscholar package not installed

    # --- zotero (read-only) ---
    try:
        import os
        from pyzotero import zotero as _pyzotero

        _lib_id = os.environ.get("ZOTERO_LIBRARY_ID")
        _api_key = os.environ.get("ZOTERO_API_KEY")

        if _lib_id and _api_key:
            _zot = _pyzotero.Zotero(_lib_id, "user", _api_key)

            def zotero_search_items(query: str, limit: int = 10) -> str:
                """Search Zotero library for items matching query."""
                import json
                items = _zot.items(q=query, limit=limit)
                return json.dumps([
                    {"key": i["key"], "title": i["data"].get("title"), "year": i["data"].get("date", "")[:4]}
                    for i in items
                ], indent=2)

            def zotero_get_item_metadata(item_key: str) -> str:
                """Get metadata for a Zotero item by key."""
                import json
                return json.dumps(_zot.item(item_key)["data"], indent=2)

            def zotero_get_item_fulltext(item_key: str) -> str:
                """Get full text content of a Zotero item."""
                try:
                    return _zot.fulltext_item(item_key).get("content", "(no fulltext)")
                except Exception as e:
                    return f"(fulltext unavailable: {e})"

            def zotero_semantic_search(query: str, limit: int = 10) -> str:
                """Semantic search over Zotero library."""
                return zotero_search_items(query, limit)  # fallback to keyword

            tools += [
                StructuredTool.from_function(zotero_search_items, name="mcp__zotero-mcp__zotero_search_items"),
                StructuredTool.from_function(zotero_get_item_metadata, name="mcp__zotero-mcp__zotero_get_item_metadata"),
                StructuredTool.from_function(zotero_get_item_fulltext, name="mcp__zotero-mcp__zotero_get_item_fulltext"),
                StructuredTool.from_function(zotero_semantic_search, name="mcp__zotero-mcp__zotero_semantic_search"),
            ]
    except ImportError:
        pass  # pyzotero not installed

    return tools


class OllamaAdapter:
    """Adapter for Ollama-served open-weight models.

    Uses ChatOpenAI pointed at Ollama's OpenAI-compatible endpoint and
    create_react_agent for the tool-execution loop.  Exposes the same
    interface as ClaudeAdapter: a mutable closure_tools dict and invoke().

    Parameters
    ----------
    model : str
        Ollama model name (e.g. ``"llama3.2"``).
    system_prompt : str
        System prompt prepended to each invocation.
    study_dir : path-like or None
        Working directory; used as root for file-management tools.
    native_tools : list[str] or None
        Tool names to enable (subset of Bash/Read/Write/Edit/Glob/Grep).
    closure_tools : dict or None
        Initial closure tools.  Nodes append to this after __init__.
    base_url : str
        Ollama OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str,
        study_dir: Any = None,
        native_tools: list[str] | None = None,
        closure_tools: dict[str, Any] | None = None,
        base_url: str = "http://localhost:11434/v1",
        extra_mcp_servers: dict | None = None,
        extra_allowed_tools: list[str] | None = None,
        persistent: bool = False,
        max_history_pairs: int = 5,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.study_dir = Path(study_dir) if study_dir else None
        # Use 'native_tools' (not '_native_tool_names') so ImplementerNode's
        # sandboxed-Write setup can find it by the same attribute name as
        # ClaudeAdapter.
        self.native_tools: list[str] = list(native_tools or [])
        self.closure_tools: dict[str, Any] = dict(closure_tools or {})
        self._base_url = base_url
        self.extra_mcp_servers: dict = dict(extra_mcp_servers or {})
        self.extra_allowed_tools: list[str] = list(extra_allowed_tools or [])
        # persistent and max_history_pairs kept for backward compatibility with
        # Agent subclasses and tests that read these attributes; not used
        # in the core invocation path (history is demand-driven via DelegationLog).
        self.persistent: bool = persistent
        self.max_history_pairs: int = max_history_pairs
        # Lock serializes concurrent delegations to the same shared adapter.
        self._lock: threading.Lock = threading.Lock()
        # Built lazily so that closure_tools are fully populated before first
        # invoke().  Reset to None whenever native_tools or closure_tools change
        # so the next invoke() picks up the updated tool set.
        self._agent: Any = None
        # route_watcher is set by StrategizerNode; unused by OllamaAdapter
        # (create_react_agent runs the full tool loop to completion) but must
        # be present so StrategizerNode.__init__ doesn't raise AttributeError.
        self.route_watcher: Any = None
        # Non-blocking raw-oracle nudge, capped per delegation (= per invoke).
        from .base import OracleNudgeBudget
        self._oracle_nudge = OracleNudgeBudget()
        # Populated after each invoke() with token counts for run-level accounting.
        self.last_usage: dict = {}

    def copy(self) -> "OllamaAdapter":
        """Always return self.

        Concurrent delegations share this adapter instance and are serialized
        via _lock in invoke(). Episodic memory is demand-driven via RecallHistory
        (backed by DelegationLog) rather than per-adapter history injection.
        """
        return self

    def _build_tools(self) -> list[Any]:
        from langchain_core.tools import StructuredTool

        native_map = _native_tool_map(self.study_dir, self._oracle_nudge.check)
        tools: list[Any] = [
            native_map[name]
            for name in self.native_tools
            if name in native_map
        ]
        for name, fn in self.closure_tools.items():
            tools.append(StructuredTool.from_function(fn, name=name))
        # Inject MCP-equivalent tools for declared extra_allowed_tools.
        if self.extra_allowed_tools:
            lit_tools = _make_literature_tools()
            allowed = set(self.extra_allowed_tools)
            tools += [t for t in lit_tools if t.name in allowed]
        return tools

    def _build_agent(self) -> Any:
        from langchain_core.messages import SystemMessage
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent

        llm = ChatOpenAI(
            model=self.model, base_url=self._base_url, api_key="local"
        )
        return create_react_agent(
            llm,
            self._build_tools(),
            prompt=SystemMessage(content=self.system_prompt),
        )

    def invoke(self, messages: list[dict]) -> str:
        """Run one full agent turn; return final assistant text.

        Acquires _lock to serialize concurrent callers (e.g. parallel
        delegations to the same shared worker adapter). Transient API/network
        failures are retried with exponential backoff (see retry_on_transient).
        """
        from .base import retry_on_transient
        with self._lock:
            return retry_on_transient(lambda: self._invoke_once(messages))

    def _invoke_once(self, messages: list[dict]) -> str:
        """Core invoke logic — build agent if needed, run, return text."""
        # One invoke == one delegation's worker run; reset the per-delegation
        # nudge cap. The cached agent's tool closures read this live.
        self._oracle_nudge.reset()
        if self._agent is None:
            self._agent = self._build_agent()

        lc_msgs = _to_lc_messages(messages)
        result = self._agent.invoke(
            {"messages": lc_msgs},
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
        last = result["messages"][-1]

        # Extract token usage from LangChain response metadata.
        meta = getattr(last, "usage_metadata", None) or {}
        self.last_usage = {
            "input_tokens": meta.get("input_tokens", 0),
            "output_tokens": meta.get("output_tokens", 0),
            "cache_read_input_tokens": meta.get("input_token_details", {}).get("cache_read", 0),
            "cache_creation_input_tokens": meta.get("input_token_details", {}).get("cache_creation", 0),
            "total_cost_usd": None,  # not available from Ollama
        }

        return str(last.content)
