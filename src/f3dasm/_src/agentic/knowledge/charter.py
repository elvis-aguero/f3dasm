"""The scientific-method charter — the single, citable source of truth for how
hypotheses are tested and labelled across the agentic system.

``FALSIFICATION_CHARTER`` is the ONE place the Popperian rules live (DRY). It is
injected verbatim into the strategizer and critic system prompts at compile time
(a stable prefix → cached by the SDK, so ~no per-turn cost), and is reachable by
workers on demand through ConsultHandbook. Because both adjudicating nodes are
given the *identical* numbered text, either may cite a clause ("Charter §3") in a
disagreement and the other defers to the same words — no paraphrase drift, no
multi-round negotiation over what falsification means.

Do not restate these rules elsewhere; reference the clause number instead.
``tests/agentic/test_charter.py`` pins the injection and the wording.
"""

FALSIFICATION_CHARTER = """\
SCIENTIFIC-METHOD CHARTER — the shared, binding contract for testing and
labelling hypotheses. It is identical in the strategizer's and the critic's
instructions: cite a clause by number ("Charter §3") and the other party
defers to it.

§1  A hypothesis is ONE falsifiable claim carrying a registered prediction —
    the observable whose occurrence would refute the claim — plus a prior.

§2  ATTEMPT and VERDICT are distinct. Before a hypothesis may be closed, an
    adequate falsification ATTEMPT must have been made: a delegation flagged
    is_falsification_attempt that genuinely tests the registered prediction (a
    token probe is not adequate). The VERDICT is a separate judgement that
    follows the test's OUTCOME — never the mere fact that an attempt was run.

§3  A hypothesis is FALSIFIED if and only if an ADEQUATE test of its registered
    prediction yields a contradiction. Concretely:
      - adequate test, prediction contradicted  -> FALSIFIED (you may not
        decline the verdict to protect a favoured claim);
      - adequate test, prediction NOT contradicted -> the hypothesis SURVIVED;
        it stays OPEN or closes INCONCLUSIVE — it is NOT falsified, even if
        other evidence makes it look wrong;
      - inadequate or confounded test -> INCONCLUSIVE. A contradiction from a
        flawed test indicts the test, not the hypothesis (Duhem–Quine).

§4  No moving the goalposts. A FALSIFIED verdict must rest on the contradiction
    of the SAME prediction that was registered — not a different, post-hoc
    observation chosen after seeing the data (the Texas-sharpshooter fallacy).
    If the registered prediction was the wrong test, revise it explicitly and
    run a new adequate test; do not reinterpret the old result.

§5  SUPPORTED is corroboration, not proof. It means the hypothesis survived at
    least one adequate attempt to refute it (§2). You never "confirm" a
    hypothesis; you only fail to falsify it.

§6  The four statuses are the only ones: OPEN, SUPPORTED, FALSIFIED,
    INCONCLUSIVE. A closing status (the latter three) cites a real delegation
    ID and at least one number drawn from that delegation's report.
"""
