"""LangGraph StateGraph builder for f3dasm agentic runs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from .backends.base import Agent, Graph
from .graph_state import AgenticState
from .nodes import ImplementerNode, StrategizerNode

__all__ = ["build_graph"]


def build_graph(
    spec: Graph,
    make_adapter: Callable[[str, Agent], Any],
    checkpointer: Any = None,
    study_dir: Any = None,
    interactive: bool = False,
) -> Any:
    """Build and compile a LangGraph StateGraph from a Graph spec.

    Parameters
    ----------
    spec : Graph
        Agent graph specification (nodes, edges, entry).
    make_adapter : callable
        ``(name: str, agent: Agent) -> adapter`` — factory that produces a
        ``ClaudeAdapter`` or ``OllamaAdapter`` for the given node.
    checkpointer : any, optional
        LangGraph checkpointer.  Defaults to an in-memory :class:`MemorySaver`.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph ready to invoke.
    """
    builder = StateGraph(AgenticState)

    for name, agent in spec.nodes.items():
        adapter = make_adapter(name, agent)
        outgoing = spec.outgoing(name)

        if agent.role == "strategizer":
            node = StrategizerNode(
                adapter,
                name=name,
                outgoing=outgoing,
                spec=spec,
                study_dir=study_dir,
                interactive=interactive,
            )
        else:
            node = ImplementerNode(adapter)

        builder.add_node(name, node)

    builder.set_entry_point(spec.entry)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
