# Agentic system — backlog

Deferred ideas with enough context to resume cold. Not committed work; pick up
when prioritized. Ordered by priority (highest first); item numbers are stable
references, not a queue position — see the order above.

> **Detailed, evidence-grounded specs** for every item live in
> [`specs/`](specs/README.md) (primary-evidence citations, TDD test names, DRY
> reuse, KPI done-when). The entries below are the short version.

## Status checklist
Resolved items keep their write-up below for the record; `(commit)` is what fixed them.

- [ ] **#1** Reconcile cancelled-but-completed delegations — *open, highest priority* (recurring UNGATED root cause; partially mitigated 2026-06-15)
- [ ] **#2** Richer delegator↔worker comms (typed blocker/escalation) — *deferred*
- [ ] **#3** ProblemDefinerAgent pre-strategizer intake stage — *deferred*
- [x] **#4** Single Jupyter-notebook deliverable — *DONE* (the live system; gate runs the notebook via nbclient; toolset completed `db288d5b`)
- [x] **#5** KB entry: running a study on SLURM — *DONE* (`entries/0010-running-a-study-on-slurm.md`; the deliverable pipeline runs on SLURM, the agent graph stays local)
- [ ] **#6** Detect a delegation running but making zero ledger progress — *open*
- [ ] **#7** Remove the literature hand-listed tool docs (BF-13a) — *open* (needs corpus closures to carry docstrings first)
- [x] **#8** Literature str/int error (lit-bug #3) — `64a8230a` (coerce arxiv `max_results` to int)
- [x] **#9** Orchestrator-owned live validator for HypothesisUpdate — *DONE* (advise-with-teeth, charter-grounded, kill-switchable via `F3DASM_VERDICT_VALIDATOR`; merged into dev/confer; validated live run `20260623T002417` — fired, judged H1 correctly vs the gate critic, non-blocking)
- [ ] **#10** Strategizer delegates the optimization as one monolithic un-budgeted campaign — *open, §4 user-owned* (**current binding constraint** — watchdog-kills runs)
- [x] **#11** Orphaned background process survives watchdog kill — `5199b593` (process-group reap)
- [x] **#12** Watchdog kill loses the strategizer's retrospective — `5199b593` (synthetic post-mortem entry)
- [x] **#13** Per-cell notebook debugger — *DONE* (`RunPipelineCell` closure + `diagnose_notebook`; per-cell trace localizes a repro failure by cell name + traceback; runs against a ledger copy; kill-switch-free read-only diagnostic)
- [ ] **#14** Watchdog reap (#11) MISSES detached campaign processes — *open, HIGH (real CPU leak)* — they escape the process-group kill (new session) and outlive the run

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
**Status:** RESOLVED — this IS the live system: `pipeline.ipynb` is the sole deliverable,
the gate executes it via nbclient (`notebook_exec.run_deliverable` / `_reproduction_gate`)
with the zero-new-evals + `REPRODUCED:` asserts, no `solution.md`/`pipeline.py`, and the
agent authors the cells through the structured tools (the CRUD set completed in `db288d5b`).
Every implication below was implemented. Original write-up kept for the record.
**Status (orig):** deferred. Raised 2026-06-15.

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
**Status:** RESOLVED 2026-06-23 — `entries/0010-running-a-study-on-slurm.md`.
Raised 2026-06-15.

KB entry added: `Pipeline.run(mode="slurm", cluster=SlurmCluster(...))`, per-step
`SlurmResources`, `parallel=True` → `cluster_array` striping, canonical-store
FileLock making concurrent array writes + lazy FINISHED-skip resume safe, and the
key boundary — the deliverable pipeline runs on SLURM, the agent graph stays
LOCAL (no slurm path in agent_runtime/run.py). Lightweight by request; deeper
how-to docs deferred (none requested).

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
**Status:** RESOLVED 2026-06-22 (commit 64a8230a) — localized via the
string-`max_results` repro in run 20260622T165943; `arxiv_search_papers` /
`arxiv_list_papers` now coerce `max_results` to int. Original write-up below.
**Status (orig):** raised 2026-06-21. Observed in a wet literature run (prior session),
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

## 10. Strategizer delegates the optimization as ONE monolithic, un-budgeted campaign
**Status:** PARTIALLY MITIGATED 2026-06-22 (commit pending) — added a "scope each
delegation to one hypothesis" principle to the strategizer planning prompt
(`<scientific_process>`), grounded in Charter §3 (bundling confounds the test) with
a fail-fast/self-correct rationale. This NUDGES toward targeted campaigns but is not
enforced — the strategizer already ignored a sharper signal (H1's explicit 300-eval
criterion), so it likely needs the #9-family check to truly bind, and the wall-clock/
algorithm half (uncapped GP × 700 iters, non-resumable re-run) is untouched. Stays open.
**Status (orig):** raised 2026-06-22 (run 20260622T165943 watchdog post-mortem). **§4 — budget/decomposition, user-owned.** The recurring binding constraint once upstream phases are clean.

The strategizer delegates the WHOLE optimization to one implementer call. Verbatim
intent (D004): "Execute a comprehensive 1000-eval black-box minimization campaign …
PHASE 1 LHS 250 … PHASE 2 BO loop ~700 evals … PHASE 3 multi-start 50". Its thinking:
"Now I'll delegate to the implementer to execute the full campaign … fairly
open-ended." No wall-clock budgeting, no chunking (it does NOT delegate Phase 1,
check the budget, then Phase 2). The implementer runs it as a monolithic background
script (optimize_1000eval.py) whose GP-surrogate BO loop is O(n³) per refit over 700
iterations — inherently slow. Result: ~41 min, 673/1000 evals, never finished, watchdog.

**KPI contrast (the tell):** run 20260622T043904 GATED with ~996 evals in **22 min**;
run 20260622T050137 GATED with 2570 evals in 49 min. So a ~1000-eval campaign CAN fit
and gate — the failure is per-eval SLOWNESS (heavy GP-BO refits) + a no-chunk, no-time-
budget delegation, not the eval count itself.

**Classification:** JUDGMENT CALL, not a bug — a reasonable agent following the spec
(pipeline = LHS→GP→BO→optimization; implementer is "the only agent that evaluates")
would delegate "the campaign". But the system gives the strategizer no wall-clock
signal and no incentive to chunk. The implementer's own retrospective even claims an
"~18-min allocation" while the campaign needed 41 — a budget-estimate mismatch.

**Resume-cold options (user's call):** (a) strategizer chunks the campaign (delegate a
bounded slice, check budget/ledger, delegate the next) so it can gate mid-way; (b) give
the implementer a wall-clock-aware budget that caps the campaign and returns partial
results; (c) cheaper default surrogate/acquisition (the slowness is GP-refit cost); (d)
raise the 3600s watchdog. Do NOT pick without the user — budget is soft by charter.

## 11. Orphaned background process survives the watchdog kill (resource leak)
**Status:** RESOLVED 2026-06-22 (commit 5199b593) — run.py is now a process-group
leader (`os.setpgrp`) and the watchdog reaps the group (`reap_process_group` in
`watchdog_cleanup.py`) before `os._exit`. Residual: a child that `setsid`'s escapes
the group (documented). Original write-up below.

The implementer backgrounds its BO campaign as a detached process
(`uv run python …/optimize_1000eval.py &`-style). When the watchdog force-exits
run.py via os._exit(2), that detached child is NOT reaped — it kept running at ~230%
CPU after the run died (had to be killed manually). Backgrounded children outlive the
run. **Fix direction:** the watchdog should kill its whole process group (or the
implementer's Bash backgrounding should be tracked and reaped on close); alternatively
discourage detaching the campaign. No reasonable reading justifies a live orphan after
a kill.

## 12. Watchdog kill loses the strategizer's retrospective (blinds §1 Step 1)
**Status:** RESOLVED 2026-06-22 (commit 5199b593) — the watchdog now appends a synthetic
`role="watchdog"` post-mortem (`write_watchdog_retrospective`) with the last delegation
state + diagnostics + a transcript pointer, so §1 Step 1 gets a breadcrumb. (A real
first-person strategizer retro on watchdog remains impossible — the agent is killed
mid-flight.) Original write-up below.

Retrospectives are written at clean run close; the watchdog's os._exit kills the
strategizer BEFORE it writes one. So EVERY watchdog-killed run has ZERO first-person
signal from the orchestrator — the very runs we most need to diagnose (its DECISION/
FRICTION on delegation, budgeting, gating). Observed: run 20260622T165943 has
retrospectives only for D001/D004, none for the strategizer, so its campaign-
decomposition reasoning had to be reconstructed from the transcript (§1 Step 4) instead
of read directly (Step 1). **Fix direction:** have the watchdog handler flush a
best-effort strategizer retrospective (or a partial "interrupted" one) before os._exit.

---

## 13. Per-cell notebook debugger (agent can't see WHICH cell failed reproduction)
**Status:** RESOLVED 2026-06-23 — `RunPipelineCell` closure (`nodes/tools/routing.py`)
+ `diagnose_notebook` (`notebook_exec.py`); 5 headless tests + live closure smoke.
Original spec below. Surfaced by run 20260623T002417 (FAILED).

**Problem.** `CheckDeliverable` runs `pipeline.ipynb` top-to-bottom via nbclient and
returns a BINARY pass/fail with a high-level message. When reproduction fails the
strategizer cannot see *which* cell raised, its traceback, or its stdout/state — so it
iterates blindly. Primary evidence: run 20260623T002417 burned ~10 gate attempts
(`REPRO_GATE_BOUNCE`×6 → `REPRO_GATE_FAILED`) and FAILED; the strategizer's DONE
retrospective BLOCKED field reads verbatim: *"No tool to execute and debug pipeline.ipynb
cell-by-cell in isolation… the CheckDeliverable gate was binary fail/pass with high-level
error messages only. I needed a cell-level executor (run cell 'analysis' and return
stdout/stderr/state) to pinpoint whether the failure was in data loading, ledger path
resolution, output formatting, or the gate's expectations."* It never diagnosed that its
own pillar cells were stubs (`print('doe')`, …), so the REJECT was unavoidable.

**Why this is parsimonious (not overfit).** "An agent must be able to observe the failure
it is asked to fix" is a general observability principle — a philosopher nods. The binary
gate throws away per-cell information nbclient ALREADY produces. This is not a workaround
for one run; it is the diagnostic counterpart the repro gate has always lacked.

**Proposed tool (DRY — reuse the gate's runner).** Add a strategizer closure, e.g.
`RunPipelineCell(name: str | None = None)`, registered alongside `ShowNotebook` /
`RunScratch` / `CheckDeliverable` (`nodes/tools/routing.py`), backed by the SAME nbclient
execution `notebook_exec.py` already uses for the gate:
- `name` given → execute `pipeline.ipynb` top-to-bottom up to AND INCLUDING the cell with
  that `metadata.name` (fresh kernel, the gate's environment + canonical store), return that
  cell's stdout / stderr / traceback + an `errored` flag. (Top-to-bottom because cells share
  state — imports/vars from earlier cells; "run cell N in isolation" would spuriously fail.)
- `name` omitted → execute the whole notebook and return a PER-CELL trace (each cell:
  ok | errored, the first failing cell's name + traceback, stdout tail) — the granular
  version of `CheckDeliverable`.

**Boundary / safety.** Read-only diagnostic: run in the same sandboxed, zero-new-eval mode
the gate uses (must NOT mutate the canonical ledger — a notebook that calls
`get_evaluator()` against a full store is lazy/no-op; guard the same way the gate does).
Pure diagnostic — does not change the gate's accept/reject criteria (not §4 critic
substance). Likely the minimal change is to expose what nbclient already captures, so much
of the work is surfacing, not new execution.

**Done-when (KPI).** A subsequent run that hits a repro failure resolves it WITHOUT
exhausting gate attempts — i.e. `REPRO_GATE_BOUNCE` count on a recovered run drops to ≤2
(vs 6 in 20260623T002417), or the strategizer's retrospective no longer lists the
cell-level executor as a BLOCKED gap. Headless: a test that a deliberately-broken cell is
pinpointed by name + traceback (not a binary fail).

**Reuse.** `notebook_exec.py` (nbclient runner behind `CheckDeliverable`), the notebook
closures in `nodes/tools/routing.py` (`ShowNotebook`/`RunScratch` registration pattern),
`metadata.name` cell addressing (already the notebook CRUD convention).

---

## 14. Watchdog reap (#11) misses detached campaign processes (real CPU leak)
**Status:** raised 2026-06-23. Discovered live during audit run 20260623T015907.
HIGH — leaks a full CPU core per watchdog-killed run; orphans accumulate for days.

**Problem.** #11's reap (`watchdog_cleanup.reap_process_group` -> `os.killpg(pgid)`)
kills only run.py's process GROUP. The implementer launches its optimization
campaign as a DETACHED background process (`/tmp/campaign_v2.py`) with poll-loop
bashes waiting on `campaign_report.json` -- these run in a NEW session, so they
escape the group kill (the residual risk the code comments already flagged) and
outlive the watchdog `os._exit(2)`.

**Primary evidence (audit run 20260623T015907, watchdog_killed at 60min).** Minutes
after the watchdog fired, `ps` showed PID 47561 `python3 /tmp/campaign_v2.py` at
125% CPU / 42min CPU still running, plus two `until [ -f .../D003/
campaign_report.json ]` poll bashes (63962, 59331), AND an ancient zombie from a
2-day-old run (26085, "Waiting for 975 evaluations", polling run 20260621T155717).
The leak accumulates across runs/days. Had to `kill` them by hand to free the CPU
before the next run.

**Fix direction (touches the watchdog -- needs approval).** At reap time the
campaign's parent chain is still intact (it reparents to launchd only AFTER run.py
exits), so a recursive process-TREE walk by PPID at reap time catches it where the
process-group kill does not. Options: (a) dependency-free recursive `pgrep -P`
BFS from os.getpid(), SIGTERM each (recommended); (b) `psutil.children(recursive=True)`
(adds dep); (c) a spawned-PID registry. Keep the walk BEFORE `os._exit` (chain intact).

**Done-when.** After a watchdog kill, `ps aux | grep -E 'campaign|campaign_report'`
returns zero leftovers. Headless: extend `test_reap_kills_a_detached_background_process`
to a GRANDCHILD in a new session that the group kill misses (the current test's
child is a session leader pid==pgid, so the group kill happens to catch it).

**Reuse.** `watchdog_cleanup.reap_process_group`, `studies/agentic_black_box_3d/run.py`
`_watchdog`, `tests/agentic/test_watchdog_cleanup.py`.
