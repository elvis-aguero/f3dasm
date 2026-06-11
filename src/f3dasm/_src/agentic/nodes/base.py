"""Base node class for ADAS-inspectable LangGraph nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..graph_state import AgenticState


class AgentNode:
    """Base class for ADAS-inspectable LangGraph nodes.

    Subclasses override __call__ to define routing topology.
    inspect.getsource(MyNode.__call__) reads the full routing logic.
    """

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def __call__(self, state: AgenticState) -> Any:
        raise NotImplementedError
