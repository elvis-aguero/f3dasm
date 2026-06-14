"""Forward-compatible tool catalog (Spec B).

The <tools> prompt section is GENERATED from the live closure set, so it can't
drift from the actual tools and a new tool self-registers. The tool owns its
guidance (docstring + optional @tool_examples).
"""
from __future__ import annotations

from f3dasm._src.agentic.tool_catalog import (
    render_tool_catalog,
    system_prompt_with_catalog,
    tool_examples,
)


def test_catalog_includes_every_live_tool_by_exact_name():
    def Alpha():
        """Do the alpha thing."""
    def Beta():
        """Do the beta thing."""
    cat = render_tool_catalog({"Alpha": Alpha, "Beta": Beta})
    assert "### Alpha" in cat and "Do the alpha thing." in cat
    assert "### Beta" in cat and "Do the beta thing." in cat


def test_new_tool_self_registers_forward_compatible():
    """Adding a closure makes it appear with no other change — the whole point."""
    def Existing():
        """existing"""
    base = render_tool_catalog({"Existing": Existing})
    assert "### NewlyAdded" not in base

    def NewlyAdded():
        """brand new"""
    grown = render_tool_catalog({"Existing": Existing, "NewlyAdded": NewlyAdded})
    assert "### NewlyAdded" in grown and "brand new" in grown


def test_examples_render_when_present():
    @tool_examples("Foo(1, 2)", "Foo('x')")
    def Foo():
        """foo desc"""
    cat = render_tool_catalog({"Foo": Foo})
    assert "Examples:" in cat
    assert "Foo(1, 2)" in cat and "Foo('x')" in cat


def test_missing_docstring_marked_not_crashed():
    def NoDoc():
        pass
    cat = render_tool_catalog({"NoDoc": NoDoc})
    assert "### NoDoc" in cat and "(no description)" in cat


def test_deterministic_sorted_order():
    def B():
        """b"""
    def A():
        """a"""
    cat = render_tool_catalog({"B": B, "A": A})
    assert cat.index("### A") < cat.index("### B")


def test_empty_closures_render_nothing():
    assert render_tool_catalog({}) == ""


def test_system_prompt_with_catalog_appends():
    def T():
        """t"""
    out = system_prompt_with_catalog("BASE PROMPT", {"T": T})
    assert out.startswith("BASE PROMPT")
    assert "<tools>" in out and "### T" in out


def test_catalog_fixes_observed_drift_on_a_real_strategizer():
    """Regression for the drift the audit found: CancelDelegation and ReadNote
    are live closures that were MISSING from the hand-written prompt prose, and
    the prose said 'Read' not 'ReadNote'. The generated catalog uses the live
    closure keys, so they're present and correctly named."""
    from f3dasm._src.agentic.backends.base import Agent, Edge, Graph
    from f3dasm._src.agentic.nodes import StrategizerNode

    class _Stub:
        def __init__(self):
            self.closure_tools: dict = {}
            self.last_usage: dict = {}
            self.model = "m"

        def invoke(self, messages):
            return ""

    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done"})
        description = "strategizer"

    class B(Agent):
        role = "implementer"
        description = "implementer"

    spec = Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),), entry="strategizer")
    n = StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()})

    cat = render_tool_catalog(n.adapter.closure_tools)
    assert "### CancelDelegation" in cat   # was missing from the old prose
    # the catalog name-set is EXACTLY the live closure key-set — so a tool can
    # never be missing (CancelDelegation) or misnamed ('Read' vs 'ReadNote')
    # in the prompt again, by construction.
    catalog_names = {
        line[4:].strip() for line in cat.splitlines() if line.startswith("### ")}
    assert catalog_names == set(n.adapter.closure_tools)
