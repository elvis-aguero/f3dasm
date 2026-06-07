"""Prompts must match the runtime — no stale references."""

from f3dasm._src.agentic.agent_prompts import (
    CHECKPOINT_STRATEGIZER_PROMPT,
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
    assert "experiment_data_dir" in out
    assert "/s/runs/T/experiment_data" in out


def test_replicate_deliverable_consumes_ledger_and_asserts():
    assert "ExperimentData.from_file" in STRATEGIZER_SYSTEM_PROMPT
    # replicate.py must self-assert (pass/fail replication test)
    assert "assert" in STRATEGIZER_SYSTEM_PROMPT
    assert "canonical evaluation ledger as INPUT" in \
        STRATEGIZER_SYSTEM_PROMPT
    # must NOT instruct re-running the expensive evaluator
    assert "does NOT re-run the expensive" in \
        STRATEGIZER_SYSTEM_PROMPT
