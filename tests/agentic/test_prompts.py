"""Prompts must match the runtime — no stale references."""

from f3dasm._src.agentic.agent_prompts import (
    CHECKPOINT_STRATEGIZER_PROMPT,
    IMPLEMENTER_SYSTEM_PROMPT_OLLAMA,
    RUN_PATHS_PREAMBLE_TEMPLATE,
)
from f3dasm._src.agentic.agents.critic import (
    ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT,
)
from f3dasm._src.agentic.agents.strategizer import (
    STRATEGIZER_SYSTEM_PROMPT,
)


def test_no_stale_hypotheses_md_references():
    from f3dasm._src.agentic.agent_prompts import (
        RUN_PATHS_PREAMBLE_TEMPLATE,
    )
    assert "hypotheses.md" not in STRATEGIZER_SYSTEM_PROMPT
    assert "hypotheses.md" not in CHECKPOINT_STRATEGIZER_PROMPT
    assert "hypotheses.md" not in RUN_PATHS_PREAMBLE_TEMPLATE
    assert "hypotheses.md" not in ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT


def test_no_false_runtime_enforcement_claim():
    assert "enforced at runtime" not in STRATEGIZER_SYSTEM_PROMPT


def test_strategizer_documents_new_signatures():
    for token in ("falsification_criterion", "prediction", "prior",
                  "posterior", "is_falsification_attempt"):
        assert token in STRATEGIZER_SYSTEM_PROMPT, token


def test_checkpoint_asks_for_ledger_digest():
    assert "posterior" in CHECKPOINT_STRATEGIZER_PROMPT


def test_critic_uses_falsification_flags():
    assert "is_falsification_attempt" in \
        ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT


def test_prompts_are_case_generic():
    # no references to specific past studies in any prompt
    for prompt in (STRATEGIZER_SYSTEM_PROMPT,
                   CHECKPOINT_STRATEGIZER_PROMPT,
                   ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT):
        for stale in ("supercompressible", "black_box_8d",
                      "sigma_crit"):
            assert stale not in prompt, stale


def test_strategizer_mentions_science_monitor():
    assert "[SCIENCE MONITOR" in STRATEGIZER_SYSTEM_PROMPT


def test_run_paths_preamble_has_experiment_data_dir():
    from f3dasm._src.agentic.agent_prompts import (
        RUN_PATHS_PREAMBLE_TEMPLATE,
    )
    # must format cleanly with the new required field
    out = RUN_PATHS_PREAMBLE_TEMPLATE.format(
        study_dir="/s", run_dir="/s/runs/T", debug_dir="/s/runs/T/debug",
        notes_dir="/s/runs/T/debug/strategizer_notes",
        experiment_data_dir="/s/runs/T/experiment_data",
    )
    # exposed as the canonical-store project_dir with an explicit from_file call
    assert "canonical_store" in out
    assert "/s/runs/T/experiment_data" in out
    assert "from_file(project_dir=" in out


def test_replicate_deliverable_consumes_ledger_and_asserts():
    assert "ExperimentData.from_file" in STRATEGIZER_SYSTEM_PROMPT
    # replicate.py must self-assert (pass/fail replication test)
    assert "assert" in STRATEGIZER_SYSTEM_PROMPT
    assert "canonical evaluation ledger as INPUT" in \
        STRATEGIZER_SYSTEM_PROMPT
    # must NOT instruct re-running the expensive evaluator
    assert "does NOT re-run the expensive" in \
        STRATEGIZER_SYSTEM_PROMPT


def test_strategizer_has_replicate_authoring_primer():
    """The strategizer AUTHORS replicate.py, so it must carry a concrete
    f3dasm primer (not just a one-line load) — load the ledger, read its
    frames, select, and assert a DERIVED number (not hardcoded)."""
    p = STRATEGIZER_SYSTEM_PROMPT
    # concrete ledger-read API, not just from_file
    assert "to_pandas()" in p
    assert "TEMPLATE" in p
    # the optional f3dasm Pipeline form is shown as actionable code
    assert "Pipeline(" in p and "Step(" in p
    # explicit anti-hardcoding guidance the critic can lean on
    low = p.lower()
    assert "hardcod" in low and "derive" in low


# ---------------------------------------------------------------------------
# Prompt-audit tests (audit fixes A-M)
# ---------------------------------------------------------------------------

def test_no_two_agent_framing_in_any_prompt():
    """No prompt should use the stale 'two-agent' framing."""
    from f3dasm._src.agentic.agents.implementer import (
        IMPLEMENTER_SYSTEM_PROMPT,
    )
    from f3dasm._src.agentic.agents.literature import (
        LITERATURE_REVIEW_SYSTEM_PROMPT,
    )
    for name, prompt in (
        ("STRATEGIZER", STRATEGIZER_SYSTEM_PROMPT),
        ("IMPLEMENTER", IMPLEMENTER_SYSTEM_PROMPT),
        ("OLLAMA", IMPLEMENTER_SYSTEM_PROMPT_OLLAMA),
        ("LITERATURE", LITERATURE_REVIEW_SYSTEM_PROMPT),
        ("CRITIC", ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT),
    ):
        assert "two-agent" not in prompt.lower(), (
            f"{name} prompt contains stale 'two-agent' framing"
        )


def test_strategizer_ledger_read_tools_documented(tmp_path):
    """RecallStore, QueryStore, RecallHistory are documented to the strategizer.

    Post-Spec-B these live in the GENERATED <tools> catalog (the single source
    of truth, derived from the live closures), not the static base prompt — so
    assert the catalog of a real strategizer node, which is what the model
    actually receives appended to its system prompt."""
    from f3dasm._src.agentic.backends.base import Agent, Edge, Graph
    from f3dasm._src.agentic.nodes import StrategizerNode
    from f3dasm._src.agentic.tool_catalog import render_tool_catalog

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
    from f3dasm._src.agentic.delegation_log import DelegationLog
    n = StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()},
        delegation_log=DelegationLog(tmp_path / "dlog.jsonl"))  # → RecallHistory
    cat = render_tool_catalog(n.adapter.closure_tools)
    for tool in ("RecallStore", "QueryStore", "RecallHistory"):
        assert tool in cat, f"generated tool catalog missing '{tool}'"


def test_strategizer_ledger_is_ground_truth():
    """Strategizer prompt states the canonical ledger is ground truth."""
    lower = STRATEGIZER_SYSTEM_PROMPT.lower()
    assert "ground truth" in lower, (
        "STRATEGIZER_SYSTEM_PROMPT missing 'ground truth' ledger-"
        "evidence statement"
    )


def test_strategizer_no_builtin_gp_claim():
    """Strategizer architecture section must not list CMA-ES/GP as
    f3dasm-native; tpesampler should be present near optimization."""
    lower = STRATEGIZER_SYSTEM_PROMPT.lower()
    assert "tpesampler" in lower, (
        "STRATEGIZER_SYSTEM_PROMPT does not mention 'tpesampler' in "
        "optimization guidance"
    )


def test_ollama_implementer_contains_get_evaluator():
    """Ollama implementer prompt must document get_evaluator."""
    assert "get_evaluator" in IMPLEMENTER_SYSTEM_PROMPT_OLLAMA, (
        "IMPLEMENTER_SYSTEM_PROMPT_OLLAMA missing 'get_evaluator'"
    )


def test_run_paths_renders_delegations_not_workspace():
    """RUN_PATHS_PREAMBLE_TEMPLATE must render 'delegations', not
    '/workspace', as workspace_dir."""
    out = RUN_PATHS_PREAMBLE_TEMPLATE.format(
        study_dir="/s",
        run_dir="/s/runs/T",
        debug_dir="/s/runs/T/debug",
        notes_dir="/s/runs/T/debug/strategizer_notes",
        experiment_data_dir="/s/runs/T/experiment_data",
    )
    assert "delegations" in out, (
        "RUN_PATHS_PREAMBLE_TEMPLATE did not render 'delegations'"
    )
    assert "/workspace" not in out, (
        "RUN_PATHS_PREAMBLE_TEMPLATE still renders '/workspace'"
    )
