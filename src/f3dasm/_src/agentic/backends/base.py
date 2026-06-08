"""Agent, Edge, and Graph primitives for agentic-f3dasm backends."""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Agent base class
# ---------------------------------------------------------------------------


class Agent:
    """Base class for all agentic nodes in a Graph.

    Subclasses override class-level attributes (``system_prompt``,
    ``tools``, etc.) to configure behaviour.  Behavioural differences
    belong in class attributes, not constructor arguments.

    **Tool system — three categories:**

    ``tools: frozenset[str]`` declares tools from two categories:

    1. **Native backend tools** — names from :data:`NATIVE_TOOL_NAMES`
       (``"Bash"``, ``"Read"``, ``"Write"``, etc.).  The runtime passes
       these to the backend session's native tool executor.

    2. **Protocol closure tools** — names from
       :data:`PROTOCOL_CLOSURE_NAMES` (``"Done"``, ``"WriteNote"``,
       ``"ReadNote"``).  The runtime builds Python callables for these and
       passes them as ``closure_tools`` to the session factory.

    3. **Topology-injected tools** — ``"Delegate"``, ``"Parallel"``,
       ``"Debate"``, ``"Retry"`` (outgoing edges), ``"Ask"`` (entry node),
       and ``"FollowUp"`` (incoming edges).  **Never declare these in**
       ``Agent.tools``.  The runtime injects them automatically from the
       graph topology; any declaration here is ignored.

    4. **External MCP server tools** — names declared in
       ``extra_allowed_tools`` (e.g.
       ``"mcp__arxiv__search_papers"``).  The runtime passes these to the
       backend together with ``mcp_servers``, a dict of
       ``{server_name: McpStdioServerConfig}`` that declares which external
       MCP servers to start.

    Default is ``frozenset()`` — no tools (opt-in, conservative).

    Parameters
    ----------
    model : str or None
        Model identifier.  ``None`` delegates to the backend default.
    """

    system_prompt: str = ""
    tools: frozenset[str] = frozenset()
    reset_on_checkpoint: bool = True
    description: str = ""
    role: str = "implementer"
    backend: str | None = None
    mcp_servers: dict = {}
    extra_allowed_tools: frozenset[str] = frozenset()
    inject_problem_statement: bool = False
    max_history_pairs: int = 5
    report_sections: tuple[str, ...] = (
        "### Actions taken",
        "### Files touched",
        "### Conclusions",
        "### Numbers",
    )

    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def forward(self) -> None:
        """ADAS hook — override for inspectable Python orchestration."""

    def build_closure_tools(
        self,
        study_dir: "Path",
        delegation_id: str | None = None,
        lit_reviewer_notes_dir: "Path | None" = None,
    ) -> dict:
        """Return runtime closure tools for this agent. Override in subclasses.

        Called by the runtime when constructing the worker adapter so agents can
        inject Python callables (e.g. corpus management tools) without declaring
        them in Agent.tools.
        """
        return {}


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """A directed delegation edge between two named agents.

    Parameters
    ----------
    source : str
        Name of the agent that is allowed to call ``Delegate``.
    target : str
        Name of the agent that receives the delegated task.
    preamble : str
        Text prepended to the task message when this edge is traversed.
    """

    source: str
    target: str
    preamble: str = ""


@dataclass
class Graph:
    """Agent graph passed to :class:`~agent_runtime.AgenticRun`.

    Declares which agents exist, which directed delegation edges connect
    them, and which agent starts the run.  Loops are permitted.

    Parameters
    ----------
    nodes : dict[str, Agent]
        Maps unique agent names to :class:`Agent` instances.
    edges : sequence of Edge
        Directed delegation edges.  An agent with no outgoing edges
        receives no ``Delegate`` tool.
    entry : str
        Name of the agent that receives the initial briefing.

    Raises
    ------
    ValueError
        If any edge endpoint names an undeclared node, or if *entry* is
        not declared.
    """

    nodes: dict  # dict[str, Agent]
    edges: tuple = ()
    entry: str = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.edges = tuple(self.edges)
        if self.entry is None:
            raise ValueError("Graph.entry is required and must not be None.")
        bad = [k for k, v in self.nodes.items() if not isinstance(v, Agent)]
        if bad:
            raise TypeError(
                f"Graph nodes values must be Agent instances; "
                f"got invalid values for keys: {sorted(bad)}"
            )
        names = set(self.nodes)
        for e in self.edges:
            if e.source not in names or e.target not in names:
                raise ValueError(
                    f"Edge {e!r} references undeclared node. "
                    f"Declared names: {sorted(names)}"
                )
        if self.entry not in names:
            raise ValueError(
                f"entry={self.entry!r} not in nodes (entry node undeclared). "
                f"Declared names: {sorted(names)}"
            )
        missing_desc = [n for n, a in self.nodes.items() if not a.description]
        if missing_desc:
            raise ValueError(
                f"All agents must define a non-empty description. "
                f"Missing in: {sorted(missing_desc)}"
            )

    def outgoing(self, name: str) -> list[str]:
        """Return target names for all edges out of *name*."""
        return [e.target for e in self.edges if e.source == name]

    def incoming(self, name: str) -> list[str]:
        """Return source names for all edges into *name*."""
        return [e.source for e in self.edges if e.target == name]

    def edge(self, source: str, target: str) -> Edge | None:
        """Return the Edge from *source* to *target*, or None if absent."""
        for e in self.edges:
            if e.source == source and e.target == target:
                return e
        return None

    # Curated semantic colours (fill, stroke) by agent class name;
    # unknown classes cycle a fallback palette so any graph stays legible.
    _MERMAID_COLOURS = {
        "StrategizerAgent": ("#1d4ed8", "#1e3a8a"),
        "LiteratureReviewAgent": ("#6d28d9", "#4c1d95"),
        "DataGeneratorAgent": ("#b45309", "#7c2d12"),
        "F3dasmImplementerAgent": ("#15803d", "#14532d"),
        "AdversarialCritiqueAgent": ("#b91c1c", "#7f1d1d"),
        "DebuggerAgent": ("#475569", "#1e293b"),
    }
    _MERMAID_FALLBACK = [
        ("#0f766e", "#134e4a"), ("#a16207", "#713f12"),
        ("#be185d", "#831843"), ("#4338ca", "#312e81"),
    ]

    def to_mermaid(self) -> str:
        """Return a styled Mermaid flowchart for this graph.

        Generated from the live spec: node labels carry each agent's
        class, role, and a short description; nodes are coloured by agent
        class; edges from the entry node render as solid delegation
        arrows and edges from worker nodes as dotted consultation arrows.
        Paste at https://mermaid.live or any Mermaid-aware renderer
        (GitHub markdown, Jupyter, VS Code).
        """
        def _clean(text: str, n: int = 46) -> str:
            t = " ".join(str(text).split()).replace('"', "'")
            return (t[: n - 1] + "…") if len(t) > n else t

        lines = ["flowchart TD"]
        colours: dict = {}
        members: dict = {}
        fb = 0
        for name in self.nodes:
            agent = self.nodes[name]
            cls = type(agent).__name__
            if cls not in colours:
                if cls in self._MERMAID_COLOURS:
                    colours[cls] = self._MERMAID_COLOURS[cls]
                else:
                    colours[cls] = self._MERMAID_FALLBACK[
                        fb % len(self._MERMAID_FALLBACK)]
                    fb += 1
            members.setdefault(cls, []).append(name)
            desc = _clean(
                getattr(agent, "description", "") or agent.role)
            label = f"<b>{name}</b><br/>{cls}<br/><i>{desc}</i>"
            if name == self.entry:
                lines.append(f'    {name}(["{label}"])')
            else:
                lines.append(f'    {name}["{label}"]')
        for e in self.edges:
            arrow = "-->" if e.source == self.entry else "-.->"
            if e.preamble:
                snippet = _clean(e.preamble, 35)
                lines.append(
                    f'    {e.source} {arrow}|"{snippet}"| {e.target}')
            else:
                lines.append(f'    {e.source} {arrow} {e.target}')
        for cls, (fill, stroke) in colours.items():
            lines.append(
                f"    classDef {cls} fill:{fill},stroke:{stroke},"
                "color:#fff,stroke-width:1px;")
        for cls, names in members.items():
            lines.append(f"    class {','.join(names)} {cls}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        out_map: dict[str, list] = {}
        for e in self.edges:
            out_map.setdefault(e.source, []).append(e)
        lines = [f"Graph(entry={self.entry!r})"]
        for name in self.nodes:
            tag = " [entry]" if name == self.entry else ""
            edges = out_map.get(name, [])
            if edges:
                for e in edges:
                    preamble = f"  # {e.preamble!r}" if e.preamble else ""
                    lines.append(f"  {name}{tag}  ──▶  {e.target}{preamble}")
                    tag = ""  # only label first edge row
            else:
                lines.append(f"  {name}{tag}  (leaf)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Transient-error retry (shared by all backend adapters)
# ---------------------------------------------------------------------------

# Substrings (case-insensitive) marking a retryable transient failure. The
# Claude path runs through claude-agent-sdk (not the anthropic SDK) and the
# Ollama path through langchain/httpx, so there is no shared exception class to
# catch — we classify heuristically on the message plus a few stdlib types.
_TRANSIENT_SUBSTRINGS = (
    "overloaded", "rate limit", "ratelimit", "429", "502", "503",
    "timeout", "timed out", "temporarily unavailable", "service unavailable",
    "connection error", "connection reset", "connection aborted",
    "remote disconnected", "econnreset",
)
_TRANSIENT_TYPES = (TimeoutError, ConnectionError)


def is_transient_error(exc: BaseException) -> bool:
    """True if *exc* looks like a retryable transient API/network failure.

    Conservative: anything not recognised as transient (auth, 400, tool/logic
    errors) returns False so we never silently retry a real bug.
    """
    if isinstance(exc, _TRANSIENT_TYPES):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(s in text for s in _TRANSIENT_SUBSTRINGS)


def retry_on_transient(
    fn,
    *,
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float = 60.0,
):
    """Call ``fn()`` retrying transient failures with backoff + jitter.

    Non-transient exceptions propagate immediately. Defaults come from
    ``F3DASM_LLM_RETRY_MAX`` (default 5) and ``F3DASM_LLM_RETRY_BASE`` (2.0s).
    """
    if max_attempts is None:
        max_attempts = int(os.environ.get("F3DASM_LLM_RETRY_MAX", "5"))
    if base_delay is None:
        base_delay = float(os.environ.get("F3DASM_LLM_RETRY_BASE", "2.0"))
    attempt = 0
    while True:
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised unless transient
            attempt += 1
            if attempt >= max_attempts or not is_transient_error(exc):
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay += random.uniform(0, delay * 0.5)
            time.sleep(delay)

