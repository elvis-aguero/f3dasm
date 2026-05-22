"""LangGraph node classes for the f3dasm agentic runtime.

Each node class implements ``__call__(state: AgenticState) -> Command``,
which is the ADAS-inspectable topology entry point.
``inspect.getsource(StrategizerNode.__call__)`` reads the full routing logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .graph_state import AgenticState

__all__ = ["AgentNode", "StrategizerNode", "ImplementerNode", "_to_adapter_messages"]


def _to_adapter_messages(lc_messages: list) -> list[dict]:
    """Convert LangChain message objects to adapter-format dicts."""
    from langchain_core.messages import AIMessage, HumanMessage

    result: list[dict] = []
    for msg in lc_messages:
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            result.append({"role": "user", "content": str(content)})
        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            result.append({"role": "ai", "content": str(content)})
    return result


class AgentNode:
    """Base class for ADAS-inspectable LangGraph nodes.

    Subclasses override __call__ to define routing topology.
    inspect.getsource(MyNode.__call__) reads the full routing logic.
    """

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def __call__(self, state: "AgenticState") -> Any:
        raise NotImplementedError


class StrategizerNode(AgentNode):
    """Orchestrator node: reads Reports, decides next Delegation, signals Done/Ask."""

    def __init__(
        self,
        adapter: Any,
        outgoing: list[str],
        entry: str = "strategizer",
    ) -> None:
        super().__init__(adapter)
        self._outgoing = list(outgoing)
        self._entry = entry
        self._route: dict = {}
        self.adapter.closure_tools.update(self._build_routing_closures())

    def _build_routing_closures(self) -> dict:
        """Return routing closure tools that write to self._route."""
        route = self._route
        outgoing = self._outgoing

        def Delegate(intent: str, expected_report: str) -> str:
            """Delegate a task to the implementer agent."""
            route["kind"] = "delegate"
            route["target"] = outgoing[0] if outgoing else None
            route["task"] = intent
            return "Task delegated. Waiting for Report."

        def Done(summary: str) -> str:
            """Signal end of run with a summary of findings."""
            route["kind"] = "done"
            route["summary"] = summary
            return "Run complete."

        def Ask(question: str) -> str:
            """Ask the human operator a question and wait for input."""
            route["kind"] = "ask"
            route["question"] = question
            return "Awaiting user response."

        return {"Delegate": Delegate, "Done": Done, "Ask": Ask}

    def __call__(self, state: "AgenticState") -> Any:
        from langchain_core.messages import AIMessage, HumanMessage
        from langgraph.graph import END
        from langgraph.types import Command, interrupt

        self._route.clear()

        text = self.adapter.invoke(_to_adapter_messages(state["messages"]))
        ai_msg = AIMessage(content=text)

        route = self._route
        if route.get("kind") == "delegate" and route.get("target"):
            return Command(
                goto=route["target"],
                update={
                    "messages": [ai_msg, HumanMessage(content=route.get("task", ""))],
                    "total_delegations": state["total_delegations"] + 1,
                },
            )
        if route.get("kind") == "ask" and route.get("question"):
            answer = interrupt(route["question"])
            return Command(
                goto=self._entry,
                update={"messages": [ai_msg, HumanMessage(content=str(answer))]},
            )
        # "done" or no routing tool called → end run
        summary = route.get("summary") or text
        return Command(
            goto=END,
            update={"messages": [ai_msg], "done": True, "last_report": summary},
        )


class ImplementerNode(AgentNode):
    """Worker node: executes tasks, writes Reports, returns to caller."""

    def __init__(self, adapter: Any, return_to: str) -> None:
        super().__init__(adapter)
        self._return_to = return_to

    def __call__(self, state: "AgenticState") -> Any:
        from langchain_core.messages import AIMessage
        from langgraph.types import Command

        text = self.adapter.invoke(_to_adapter_messages(state["messages"]))
        ai_msg = AIMessage(content=text)
        return Command(
            goto=self._return_to,
            update={"messages": [ai_msg], "last_report": text},
        )
