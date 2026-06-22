# Agentic system — backlog

Deferred ideas with enough context to resume cold. Not committed work; pick up
when prioritized. Ordered by priority (highest first); item numbers are stable
references, not a queue position — see the order above.

> **Detailed, evidence-grounded specs** for every item live in
> [`specs/`](specs/README.md) (primary-evidence citations, TDD test names, DRY
> reuse, KPI done-when). The entries below are the short version.

---

## 1. Reconcile cancelled-but-completed delegations
**Status:** partially mitigated 2026-06-15 (cancel hardened + eval-count now
from ledger); the core inconsistency remains. **Highest priority** — it is the
recurring root cause of UNGATED / hypotheses-left-OPEN outcomes across runs.

`CancelDelegation` detaches a worker and tells the strategizer "result will be
ignored," but the **detached worker keeps running**, finishes, stamps real evals
into the canonical ledger, and writes its report to disk. The run then holds two
contradictory truths: the official record (delegation_log / registry / what the
strategizer is told) says *ignored / not executed*, while the ledger + on-disk
report say *done*. Primary evidence (run 20260615T192313): D005/D006 were
cancelled-detached, absent from delegation_log, yet have 145/32 ledgered evals +
`D006/REPORT_SUMMARY.txt` ("H1 SUPPORTED"). The strategizer then claimed "the
falsification was not executed" (per the runtime's "ignore it") while the critic
read the disk report — a **non-converging gate loop** rooted in inconsistent
state, NOT agent hallucination or prompt friction. Re-confirmed 2026-06-16: the
8d e2e left H1/H2 OPEN citing "D004 cancelled post-completion."

**Done so far:** (a) `evals_used` now counts from the ledger so those evals
aren't dropped from the run total (`agent_runtime.py`); (b) cancel is hardened
against impatience — two-shot for delegations already producing ledgered evals;
docstring + poll/premature nudges reframed (`routing.py`).

**Still to design:** when a detached worker completes with stamped ledger rows,
**reconcile** it — record its completion in the delegation_log (so the
strategizer sees it finished, not "ignored"), or stop the worker BEFORE it
stamps. Pick one source of truth so strategizer and critic never see
contradictory delegation state. Related: the stuck-delegation detection in #6.

---

## 2. Richer delegator↔worker comms — typed blocker/escalation
**Status:** deferred (prefer benchmarking the current system first). Design explored 2026-06-15.

**Today:** the protocol is near single-shot — `Delegate(task, expected_report, …)`
down, the worker's structured report up, and exactly **one blocking
`FollowUp(question)` clarification** mid-task (≤1 per delegation; routes to the
delegating agent — or the human for the entry node). When no operator/TTY is
present FollowUp now returns an autonomous-proceed notice instead of blocking
(headless `input()` EOFError fixed 2026-06-16). The delegator can only
`CancelDelegation` a running worker — it **cannot steer** one.

**Principle:** a *blocker* differs from a *clarification* on **who must act** —
a clarification needs information back; a blocker needs the delegator to take an
**action** the worker can't take itself. So the fix is not "let the worker ask
more," it's "give the delegator the right levers." The real levers a blocked
worker needs: **provide** a missing input/capability, **grant** more eval
budget, **revise** scope, **reroute** to another specialist, **abort** cleanly.

**Chosen direction (Approach A + C's logging):** a worker tool `Escalate(blocker,
kind)` distinct from `FollowUp` — `kind ∈ {missing_input, over_budget,
capability_gap, scope_conflict, unrecoverable}` — that *blocks* (reuse the
FollowUp wait/Reply plumbing) and the delegator answers with a **typed
decision** (`provide / grant_budget(n) / revise_scope / reroute(target) /
abort`) that the runtime *applies* (budget bump, clean abort, re-delegation).
Log the escalation + its resolution to the delegation log (auditable, fits the
science-integrity ethos). These same `abort`/`grant_budget`/`reroute` levers are
what the strategizer needs to act on the stuck-delegation signal in #6.

**Open question to settle first:** which of the 5 delegator actions earn their
place vs YAGNI?

---

## 3. ProblemDefinerAgent — a pre-strategizer intake stage
**Status:** deferred. Raised 2026-06-16.

A new agent that sits **between the human and the strategizer**, running once at
the very start of a run, before the strategizer takes over. Its job is to turn a
raw human problem statement into a high-signal, airtight brief so the strategizer
spends its budget on science, not on plumbing/ambiguity.

**Responsibilities:**
- **(a) Airtight problem statement** — resolve ambiguity, pin the objective
  (minimise/maximise), constraints, success criterion, and what "the result"
  is. Today this is partly covered by the advisory `_review_problem_statement`
  pre-run pass (`agent_runtime.py:680`); the ProblemDefiner would *own* and
  extend it (interactive with the human, not just advisory).
- **(b) Tech stack + ExperimentData schema** — decide/confirm the f3dasm Domain
  (input variables + bounds + types) and the **output columns** of the canonical
  ExperimentData (objective col name, feasibility cols, units), so the ledger
  schema is fixed before any delegation runs.
- **(c) Hard-to-automate plumbing** — the evaluator entrypoint, eval/wall
  budgets, output_names, any study-specific config that today lives in
  `config.yaml` / `PROBLEM_STATEMENT.md` and is easy to get subtly wrong.

**Why:** it **offloads the strategizer** (which currently has to infer schema,
reconcile config vs problem statement, and self-review well-posedness) and hands
it a higher-quality signal. Net effect: fewer SCIENCE_DRIFT / MILESTONE_BLOCK
diagnostics traceable to an under-specified brief, and a fixed ledger schema from
turn one.

**To design when picked up:** is it a graph node (entry before strategizer) or a
runtime pre-pass like the current problem-statement review? How interactive with
the human (blocking Q&A vs one-shot)? Does it *write* `config.yaml` + an enriched
`PROBLEM_STATEMENT.md` as its output artifacts (so the brief is itself a
reproducible deliverable)? Relationship to the existing
`_review_problem_statement` advisory pass (replace vs wrap).

---

## 4. Single Jupyter-notebook deliverable
**Status:** deferred. Raised 2026-06-15.

Merge the two deliverables (`solution.md` prose + `pipeline.py` executable) into
**one `.ipynb`** — markdown cells for the writeup, code cells for the lazy
create→run→analyze pipeline. One human-readable, runnable artifact.

**Implications to design when picked up:**
- The reproduction gate (`StrategizerNode._reproduction_gate`) would execute the
  *notebook* lazily (e.g. `jupyter nbconvert --execute` / `nbclient`) instead of
  `python pipeline.py`, keeping the same asserts: exit clean + **zero new oracle
  evals** + headline self-assert (now a pre-critic gate, timeout = max(10% of
  the time budget, 180s)).
- `WriteDeliverable` accepts `.ipynb`; `_missing_deliverables` requires it.
- The runtime currently auto-writes `solution.md` from the `Done()` summary —
  that prose would instead become the notebook's leading markdown cells (decide
  who authors it: agent vs runtime injection).
- Keep the lazy + cache-or-load contract intact (notebook re-run = reproduction,
  no sims, no heavy refit).

---

## 5. KB entry: running a study on SLURM
**Status:** deferred. Raised 2026-06-15.

Add a knowledge-base entry (and/or doc) on running an agentic study on a SLURM
cluster. f3dasm's `Pipeline.run(mode="slurm", cluster=...)` already exists
(`pipeline/executors/`); the KB should cover how the agentic layer maps onto it
— the canonical store / FileLock semantics across nodes, how `get_evaluator()`
and the lazy job-status resume behave under cluster parallelism, resource
declaration (`Step.resources`, `orchestrator_resources`), and what changes vs
the current local mode.

---

## 6. Detect a delegation that is running but making zero ledger progress
**Status:** raised 2026-06-16. Cross-links #1 (reconcile) and #2 (steering levers).

A delegation can be **alive but unproductive**: the worker is still `Working`,
wall-time is climbing, but it is stamping **zero new rows** into the canonical
ledger. Today the strategizer cannot tell "slow but progressing" from "stuck /
spinning," so it polls, then either waits until the time budget dies or cancels
on impatience (feeding #1). Primary evidence (run 20260616T004655): the
implementer (D004) was alive ~228s+ yet produced **0 ledgered evals**; the run
ended UNGATED with an empty ledger.

**The signal:** status == Working AND wall-time since last ledger row > threshold
AND ledger delta == 0. That is detectable from the same `RunStateSummary` /
delegation timing the runtime already tracks. Surface it to the strategizer as a
distinct notice ("D004 has run 200s with no ledger progress") rather than letting
it guess from poll counts.

**Then it needs a lever, not just a notice** — which is exactly the typed
delegator decisions in #2 (`grant_budget` if it's genuinely close, `reroute` or
`abort` if it's stuck). Without #2 the only response is still the blunt
`CancelDelegation`. So #6 is the *detector*; #2 supplies the *actuators*; #1
ensures whatever the worker already stamped is reconciled rather than orphaned.

**Open question:** the threshold — fixed seconds, a fraction of the time budget,
or adaptive to the worker's own first-row latency? A cheap robust default: warn
once past max(120s, 15% of budget) with no row, escalate past 2× that.

---

## 7. Remove the literature hand-listed tool docs (BF-13a)
**Status:** raised 2026-06-21. Follow-through on BF-13(b) (commit 6b9c8dd8).

`literature.py` still hand-lists every tool in `<corpus_tools>`/`<discovery_tools>`
with bare signatures; the implementer's `<role>` "Available tools" list does the
same. BF-13(b) made the auto-generated `<tools>` catalog the authoritative,
MCP-qualified source, so these hand-lists are now a second source that can drift
from it.

**Why not done yet:** the corpus closures (`CorpusAdd`/`CorpusSearch`/
`CorpusGetPaper`/`CorpusList`) are docstring-less lambdas in
`build_closure_tools`, and `render_tool_catalog` falls back to "(no description)"
for them. Deleting the hand-list before the closures carry real docstrings would
DEGRADE the catalog (lose the BM25-weighting note, the "ERROR if no full-text"
semantics, the acquisition workflow). The implementer list also mixes native
tools (Read/Bash — bare-correct) with closures (qualified) — a blanket delete
would lose the native-tool descriptions too.

**The principled fix:** give each corpus/discovery closure a real docstring (one
line is enough — the SDK already reads `fn.__doc__` for its tool schema), then
delete the hand-lists and let the qualified auto-catalog be the single source.
Test: assert the literature catalog (qualified) carries a non-"(no description)"
entry for every corpus tool, and that no bare hand-list signature survives.

## 8. Localize literature str/int error (lit-bug #3)
**Status:** raised 2026-06-21. Observed in a wet literature run (prior session),
exact site not captured.

A wet literature delegation hit a `str`/`int` type error twice ("str-int error
×2"), separate from the dense-ranking crash (fixed 33d8d3f4) and the name
mismatch (fixed 6b9c8dd8). No traceback was captured this session — the wet test
was killed to save resources before it streamed output.

**Likely neighbourhood:** the citation-count / BM25 weighting path
(`log10(c+1)`), where a `citationCount` arriving as a string from S2/OpenAlex
JSON would break arithmetic; `build_closure_tools` already guards
`int(citation_count or 0)` on `CorpusAdd`, so the unguarded site is probably in
`CorpusRank` or the corpus's internal ranking, or in an OpenAlex/S2 field read.
**Needs a real wet run with the traceback** (`uv run pytest
tests/agentic/test_literature_wet.py -s --no-cov`) to localize before fixing —
do not guess-patch without the stack.

## 9. Orchestrator-owned live validator for HypothesisUpdate
**Status:** raised 2026-06-22 (user's idea). **§4 — epistemic contract, user owns it.**

Today the strategizer carries a triple burden for every hypothesis: it (1) states
the hypothesis + falsification criterion, (2) frames the falsification attempt,
and (3) judges the attempt's results — proposer, experiment-designer, and judge in
one agent. The only live guard on `HypothesisUpdate` is RULE-based (the Popperian
charter's `ERROR_RETURN`: cannot mark SUPPORTED without a falsification attempt on
record; cite only completed delegations). It checks *form*, not *substance*.

**Idea:** an **orchestrator-owned LLM** that validates each `HypothesisUpdate`
call LIVE at the tool boundary (where `ERROR_RETURN` fires now), independent of the
strategizer's own reasoning — offloading the judging role. It would check the
substance a rule cannot:
- the falsification attempt actually PROBES the registered prediction (severity —
  could it have refuted?);
- the verdict (SUPPORTED / FALSIFIED / INCONCLUSIVE) FOLLOWS from the cited
  evidence/ledger numbers;
- prediction and falsification_criterion test the SAME claim (catches goalpost-moving).

**Evidence motivating it (n=5 audit, `_audit_preserved/n5_20260621_222224`):**
- run03 strategizer moved its own goalposts — registered prediction ("reach f ≤ −1.0")
  ≠ falsification_criterion (relative budget test); the critic flagged it Charter §4.
  The strategizer self-reported the CONSISTENCY contradiction in its retrospective.
- `LinkFalsificationAttempt` was used only 2× across 6 runs — falsification linkage
  is effectively advisory/bypassed (a run02 critic: "the flag is advisory, not binding").
- verdicts are non-reproducible: the same H2 question (multi-start vs BO) came back
  FALSIFIED (r1, r2) and SUPPORTED (r4, r5) depending on the run's self-chosen budget.

**Open design questions (resume-cold):**
- BLOCK vs ADVISE: does it reject the update (hard, like `ERROR_RETURN`) or annotate
  it for the critic (soft)? Budgets are soft elsewhere — lean advise-then-escalate.
- Relation to the critic: the critic judges the DELIVERABLE at gate-time; this judges
  HYPOTHESIS verdicts LIVE as asserted — complementary, not a replacement.
- Cost/latency: one extra LLM call per HypothesisUpdate; pick a model tier; cache.
- Where it lives: `science_monitor.py` / the HypothesisUpdate tool wrapper — the
  Popperian-rules surface the user owns. Do NOT implement without the user's design call.
- Relates to the deferred "separate the falsification-judge from the hypothesis-author"
  options (freeze criterion at proposal time; dedicated verdict-adjudicator node;
  binding LinkFalsificationAttempt + INCONCLUSIVE as valid closure).

**Done-when (KPI):** verdict reproducibility across same-problem runs improves;
goalpost-moves caught LIVE (not only post-hoc by the critic); the falsification-attempt
linkage becomes load-bearing rather than advisory.
