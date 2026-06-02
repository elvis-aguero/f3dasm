"""Ollama-backed adapter for the f3dasm LangGraph agentic runtime.

Uses ChatOpenAI pointed at Ollama's OpenAI-compatible endpoint so that
LangGraph's tool-calling machinery (ToolNode, tools_condition) works
against the well-tested OpenAI code path.  create_react_agent handles
the full tool-execution loop; the adapter exposes the same
closure_tools / invoke() interface as ClaudeAdapter.
"""

from __future__ import annotations

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


def _make_edit_tool() -> Any:
    from langchain_core.tools import StructuredTool

    def edit_file(path: str, old_str: str, new_str: str) -> str:
        """Replace old_str with new_str in file at path (first occurrence)."""
        p = Path(path)
        if not p.exists():
            return f"ERROR: {path} not found"
        text = p.read_text(encoding="utf-8")
        if old_str not in text:
            return f"ERROR: string not found in {path}"
        p.write_text(text.replace(old_str, new_str, 1), encoding="utf-8")
        return f"Edited {path}"

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


def _make_bash_tool(cwd: Path | None) -> Any:
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
        return out if out else "(no output)"

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


def _make_write_tool(cwd: Path | None) -> Any:
    from langchain_core.tools import StructuredTool

    def write_file(path: str, content: str) -> str:
        """Write content to a file, creating it if it doesn't exist."""
        p = (
            Path(path) if Path(path).is_absolute()
            else (cwd or Path(".")) / path
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written: {p}"

    return StructuredTool.from_function(write_file, name="Write")


def _native_tool_map(cwd: Path | None) -> dict[str, Any]:
    return {
        "Bash":  _make_bash_tool(cwd),
        "Read":  _make_read_tool(cwd),
        "Write": _make_write_tool(cwd),
        "Edit":  _make_edit_tool(),
        "Glob":  _make_glob_tool(cwd),
        "Grep":  _make_grep_tool(),
    }


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
        # Built lazily so that closure_tools are fully populated before first
        # invoke().  Reset to None whenever native_tools or closure_tools change
        # so the next invoke() picks up the updated tool set.
        self._agent: Any = None
        # route_watcher is set by StrategizerNode; unused by OllamaAdapter
        # (create_react_agent runs the full tool loop to completion) but must
        # be present so StrategizerNode.__init__ doesn't raise AttributeError.
        self.route_watcher: Any = None

    def copy(self) -> "OllamaAdapter":
        """Return a fresh adapter with the same config but independent state.

        Used by StrategizerNode.Delegate() to give each concurrent delegation
        its own adapter instance so they never race on closure_tools or the
        cached _agent.
        """
        return OllamaAdapter(
            model=self.model,
            system_prompt=self.system_prompt,
            study_dir=self.study_dir,
            native_tools=list(self.native_tools),
            closure_tools=dict(self.closure_tools),
            base_url=self._base_url,
        )
        # _agent is intentionally left as None in the copy so it is built fresh.

    def _build_tools(self) -> list[Any]:
        from langchain_core.tools import StructuredTool

        native_map = _native_tool_map(self.study_dir)
        tools: list[Any] = [
            native_map[name]
            for name in self.native_tools
            if name in native_map
        ]
        for name, fn in self.closure_tools.items():
            tools.append(StructuredTool.from_function(fn, name=name))
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
        """Run one full agent turn; return final assistant text."""
        if self._agent is None:
            self._agent = self._build_agent()

        result = self._agent.invoke(
            {"messages": _to_lc_messages(messages)},
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
        last = result["messages"][-1]
        return str(last.content)
