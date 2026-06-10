"""Pin the scientific-method charter: single source, injected verbatim into
both adjudicating nodes, with the substantive Popperian wording intact."""

from f3dasm._src.agentic.knowledge.charter import FALSIFICATION_CHARTER
from f3dasm._src.agentic.agent_prompts import (
    STRATEGIZER_SYSTEM_PROMPT,
    ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT,
)


def test_charter_injected_verbatim_into_both_nodes():
    """DRY: the one charter string is the literal substring of both prompts."""
    assert FALSIFICATION_CHARTER in STRATEGIZER_SYSTEM_PROMPT
    assert FALSIFICATION_CHARTER in ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT


def test_charter_is_the_single_source():
    """The verdict logic must not be hand-restated outside the charter.

    The strategizer/critic may *reference* clauses ("Charter §3"); they must
    not re-define the FALSIFIED truth-condition in their own prose (which is
    exactly the drift that caused the 4-round argument). We approximate this
    by requiring the only 'if and only if' falsification statement to live in
    the charter, and the prompts to cite section numbers."""
    assert "Charter §" in STRATEGIZER_SYSTEM_PROMPT
    assert "Charter §" in ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT


def test_charter_keeps_the_strong_biconditional_with_adequacy():
    """The substantive decision: biconditional, NOT the weaker 'only when',
    qualified by 'adequate' so Duhem–Quine is respected."""
    c = FALSIFICATION_CHARTER.lower()
    assert "if and only if" in c
    assert "adequate" in c
    assert "duhem" in c  # the qualifier's named justification


def test_charter_encodes_attempt_vs_verdict_and_no_goalposts():
    c = FALSIFICATION_CHARTER.lower()
    # attempt vs verdict are explicitly distinguished
    assert "attempt" in c and "verdict" in c
    # the goalpost-move is named and forbidden
    assert "post-hoc" in c
    assert "texas" in c  # the fallacy is named


def test_charter_only_uses_the_four_canonical_statuses():
    for status in ("OPEN", "SUPPORTED", "FALSIFIED", "INCONCLUSIVE"):
        assert status in FALSIFICATION_CHARTER
    # no invented states leaked in
    assert "EVIDENCE_AGAINST" not in FALSIFICATION_CHARTER


def test_charter_survives_existing_required_tokens():
    """Sanity: the edits did not drop tokens other tests rely on."""
    assert "is_falsification_attempt" in STRATEGIZER_SYSTEM_PROMPT
    assert "is_falsification_attempt" in ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT
    assert "falsified" in STRATEGIZER_SYSTEM_PROMPT.lower()
