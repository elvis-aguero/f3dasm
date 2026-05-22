"""Agent, Edge, and Graph primitives for agentic-f3dasm backends."""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

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
       :data:`PROTOCOL_CLOSURE_NAMES` (``"Done"``, ``"WriteMarkdown"``,
       ``"ReadNote"``).  The runtime builds Python callables for these and
       passes them as ``closure_tools`` to the session factory.

    3. **Topology-injected tools** — ``"Delegate"``, ``"Parallel"``,
       ``"Debate"``, ``"Retry"`` (outgoing edges), ``"Ask"`` (entry node),
       and ``"FollowUp"`` (incoming edges).  **Never declare these in**
       ``Agent.tools``.  The runtime injects them automatically from the
       graph topology; any declaration here is ignored.

    Default is ``frozenset()`` — no tools (opt-in, conservative).

    Parameters
    ----------
    model : str or None
        Model identifier.  ``None`` delegates to the backend default.
    """

    system_prompt: str = ""
    tools: frozenset[str] = frozenset()
    reset_on_checkpoint: bool = True
    description: str | None = None

    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def forward(self) -> None:
        """ADAS hook — override for inspectable Python orchestration."""


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
    """

    source: str
    target: str


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
        Defaults to ``"strategizer"``.

    Raises
    ------
    ValueError
        If any edge endpoint names an undeclared node, or if *entry* is
        not declared.
    """

    nodes: dict  # dict[str, Agent]
    edges: tuple = ()
    entry: str = "strategizer"

    def __post_init__(self) -> None:
        self.edges = tuple(self.edges)
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

    def outgoing(self, name: str) -> list[str]:
        """Return target names for all edges out of *name*."""
        return [e.target for e in self.edges if e.source == name]

    def incoming(self, name: str) -> list[str]:
        """Return source names for all edges into *name*."""
        return [e.source for e in self.edges if e.target == name]

