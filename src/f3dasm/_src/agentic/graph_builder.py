"""LangGraph StateGraph builder for f3dasm agentic runs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from .backends.base import Agent, Graph
from .graph_state import AgenticState
from .nodes import ImplementerNode, StrategizerNode, WorkerNode  # ImplementerNode re-exported for backward compat

__all__ = ["build_graph"]


def build_graph(
    spec: Graph,
    make_adapter: Callable[[str, Agent], Any],
    checkpointer: Any = None,
    study_dir: Any = None,
    interactive: bool = False,
    max_ask: int = 1,
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
            # Pre-build worker adapters for each outgoing edge target so the
            # Strategizer can spawn them in background threads.
            worker_adapters = {
                target_name: make_adapter(target_name, spec.nodes[target_name])
                for target_name in outgoing
                if target_name in spec.nodes
            }
            node = StrategizerNode(
                adapter,
                name=name,
                outgoing=outgoing,
                spec=spec,
                study_dir=study_dir,
                interactive=interactive,
                max_ask=max_ask,
                worker_adapters=worker_adapters,
            )
        else:
            node = WorkerNode(adapter, study_dir=study_dir)

        builder.add_node(name, node)

    builder.set_entry_point(spec.entry)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
