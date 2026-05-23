"""LangGraph node classes for the f3dasm agentic runtime.

Each node class implements ``__call__(state: AgenticState) -> Command``,
which is the ADAS-inspectable topology entry point.
``inspect.getsource(StrategizerNode.__call__)`` reads the full routing logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .graph_state import AgenticState

__all__ = ["AgentNode", "StrategizerNode", "ImplementerNode", "_to_adapter_messages"]

_REQUIRED_SUBSECTIONS = [
    "### Actions taken",
    "### Files touched",
    "### Conclusions",
    "### Numbers",
]
_CAPABILITY_PHRASES = [
    "i cannot", "i can't", "i don't have access",
    "unable to", "not able to", "i am unable",
]


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


def _classify_response(text: str) -> str | None:
    """Return a REFLECT diagnosis string if text is malformed, else None."""
    from .agent_prompts import (
        REFLECT_DIAGNOSIS_CAPABILITY_LIMIT,
        REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE,
        REFLECT_DIAGNOSIS_NO_REPORT_HEADING,
        REFLECT_DIAGNOSIS_SHORT,
    )

    if len(text.strip()) < 100:
        return REFLECT_DIAGNOSIS_SHORT
    low = text.lower()
    if any(p in low for p in _CAPABILITY_PHRASES):
        return REFLECT_DIAGNOSIS_CAPABILITY_LIMIT
    if "## report" not in low:
        return REFLECT_DIAGNOSIS_NO_REPORT_HEADING
    missing = [s for s in _REQUIRED_SUBSECTIONS if s.lower() not in low]
    if missing:
        return REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE.format(
            missing_subsections=", ".join(f"'{s}'" for s in missing)
        )
    return None


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
        study_dir: Any = None,
        interactive: bool = False,
    ) -> None:
        super().__init__(adapter)
        self._outgoing = list(outgoing)
        self._entry = entry
        self._route: dict = {}
        self._study_dir = study_dir
        self._interactive = interactive
        self._current_notes_dir: Path | None = None
        self.adapter.closure_tools.update(self._build_routing_closures())

    def _build_routing_closures(self) -> dict:
        """Return routing closure tools that write to self._route."""
        node = self
        route = self._route
        outgoing = self._outgoing
        study_dir = self._study_dir
        interactive = self._interactive

        def Delegate(intent: str, expected_report: str) -> str:
            """Delegate a task to the implementer agent."""
            route["kind"] = "delegate"
            route["target"] = outgoing[0] if outgoing else None
            route["task"] = intent
            route["expected_report"] = expected_report
            return "Task delegated. Waiting for Report."

        def Done(summary: str) -> str:
            """Signal end of run with a summary of findings."""
            route["kind"] = "done"
            route["summary"] = summary
            return "Run complete."

        def Ask(question: str) -> str:
            """Ask the human operator a question and wait for input."""
            if interactive:
                route["kind"] = "ask"
                route["question"] = question
                return "Awaiting user response."
            # Non-interactive: auto-respond so run proceeds autonomously.
            return (
                "No human operator is present. Proceed autonomously "
                "using only information available in the problem statement "
                "and files in the study directory."
            )

        def WriteMarkdown(path: str, body: str) -> str:
            """Write a Markdown (.md) file to strategizer_notes/."""
            notes_dir = node._current_notes_dir
            if notes_dir is None:
                return "ERROR: notes_dir not set (run_dir missing from state)."
            bare = Path(path).name
            if not bare.endswith(".md"):
                bare = bare + ".md"
            target = Path(notes_dir) / bare
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body)
            return f"Written: {target}"

        def ReadNote(path: str) -> str:
            """Read a file from the study directory."""
            if study_dir is None:
                return "ERROR: study_dir not set."
            target = Path(study_dir) / path
            if not target.exists():
                return f"NOT FOUND: {target}"
            return target.read_text()

        return {
            "Delegate": Delegate,
            "Done": Done,
            "Ask": Ask,
            "WriteMarkdown": WriteMarkdown,
            "ReadNote": ReadNote,
        }

    def __call__(self, state: "AgenticState") -> Any:
        import time

        from langchain_core.messages import AIMessage, HumanMessage
        from langgraph.graph import END
        from langgraph.types import Command, interrupt

        # Update notes_dir from current state run_dir
        run_dir = state.get("run_dir")
        if run_dir:
            self._current_notes_dir = Path(run_dir) / "strategizer_notes"

        # Wall-clock budget check
        budget = state.get("budget_seconds")
        start = state.get("start_time")
        if budget is not None and start is not None:
            elapsed = time.time() - start
            if elapsed >= budget:
                return Command(
                    goto=END,
                    update={
                        "messages": [],
                        "done": True,
                        "last_report": state.get("last_report") or "Budget exhausted.",
                    },
                )

        # Eval budget check
        eval_budget = state.get("eval_budget")
        evals_used = state.get("evals_used", 0)
        if eval_budget is not None and evals_used >= eval_budget:
            return Command(
                goto=END,
                update={
                    "messages": [],
                    "done": True,
                    "last_report": state.get("last_report") or "Eval budget exhausted.",
                },
            )

        self._route.clear()

        text = self.adapter.invoke(_to_adapter_messages(state["messages"]))
        ai_msg = AIMessage(content=text)

        route = self._route
        if route.get("kind") == "delegate" and route.get("target"):
            task_msg = route.get("task", "")
            expected = route.get("expected_report", "")
            if expected:
                task_msg = (
                    f"{task_msg}\n\n**Required deliverables / acceptance criteria:**\n{expected}"
                )
            return Command(
                goto=route["target"],
                update={
                    "messages": [ai_msg, HumanMessage(content=task_msg)],
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
        self._evals_reported: dict = {}
        self.adapter.closure_tools.update(self._build_eval_closures())

    def _build_eval_closures(self) -> dict:
        evals = self._evals_reported

        def ReportEvals(count: int) -> str:
            """Report the number of function evaluations used in this task."""
            evals["count"] = int(count)
            return f"Recorded {count} evaluations."

        return {"ReportEvals": ReportEvals}

    def __call__(self, state: "AgenticState") -> Any:
        from langchain_core.messages import AIMessage
        from langgraph.types import Command

        from .agent_prompts import IMPLEMENTER_REPORT_RETRY_PROMPT

        self._evals_reported.clear()
        messages = _to_adapter_messages(state["messages"])
        text = self.adapter.invoke(messages)

        diagnosis = _classify_response(text)
        if diagnosis is not None:
            # One retry with correction prompt
            retry_messages = messages + [
                {"role": "ai", "content": text},
                {
                    "role": "user",
                    "content": (
                        f"{IMPLEMENTER_REPORT_RETRY_PROMPT}\n\nDiagnosis: {diagnosis}"
                    ),
                },
            ]
            text = self.adapter.invoke(retry_messages)

        ai_msg = AIMessage(content=text)
        evals_delta = self._evals_reported.get("count", 0)
        return Command(
            goto=self._return_to,
            update={
                "messages": [ai_msg],
                "last_report": text,
                "evals_used": state.get("evals_used", 0) + evals_delta,
            },
        )
