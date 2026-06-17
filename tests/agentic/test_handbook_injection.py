"""ConsultHandbook is injected UNIVERSALLY at adapter construction.

Every node's adapter must get the read-only handbook lookup equally, from the
single construction point (agent_runtime._make_adapter) — not duplicated per
worker/strategizer path, and not special-cased per agent. This pins that the
critic (invoked off the Delegate path) gets it just like a worker, and that the
tool is a runtime closure (its description owned by _consult_handbook), never
hardcoded into a prompt.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from f3dasm._src.agentic.agent_runtime import AgenticRun
from f3dasm._src.agentic.agents.critic import AdversarialCritiqueAgent
from f3dasm._src.agentic.agents.implementer import ImplementerAgent


class _NoOutgoing:
    def outgoing(self, name):
        return []


@pytest.mark.parametrize(
    "name,agent",
    [("critic", AdversarialCritiqueAgent()), ("implementer", ImplementerAgent())],
)
def test_make_adapter_injects_consult_handbook(tmp_path, name, agent):
    run = AgenticRun(study_dir=tmp_path)        # no config.yaml → defaults
    run._run_dir = None
    run._graph_spec = _NoOutgoing()

    adapter = run._make_adapter(name, agent)

    # Universal, runtime-injected closure (NOT declared in the agent prompt).
    assert "ConsultHandbook" in adapter.closure_tools
    # It is the shared handbook function (one source), not a per-agent copy.
    from f3dasm._src.agentic.nodes.parsing import _consult_handbook
    assert adapter.closure_tools["ConsultHandbook"] is _consult_handbook
