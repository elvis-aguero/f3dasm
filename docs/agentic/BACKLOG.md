# Agentic system — backlog

Deferred ideas with enough context to resume cold. Not committed work; pick up
when prioritized.

---

## 1. Richer delegator↔worker comms — typed blocker/escalation
**Status:** deferred (prefer benchmarking the current system first). Design explored 2026-06-15.

**Today:** the protocol is near single-shot — `Delegate(task, expected_report, …)`
down, the worker's structured report up, and exactly **one blocking
`FollowUp(question)` clarification** mid-task (`nodes/tools/routing.py:471`,
≤1 per delegation, waits ≤5 min then proceeds). The delegator can only
`CancelDelegation` a running worker — it cannot steer one.

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
science-integrity ethos).

**Open question to settle first:** which of the 5 delegator actions earn their
place vs YAGNI?

---

## 2. Single Jupyter-notebook deliverable
**Status:** deferred. Raised 2026-06-15.

Merge the two deliverables (`solution.md` prose + `pipeline.py` executable) into
**one `.ipynb`** — markdown cells for the writeup, code cells for the lazy
create→run→analyze pipeline. One human-readable, runnable artifact.

**Implications to design when picked up:**
- The reproduction gate (`StrategizerNode._reproduction_gate`) would execute the
  *notebook* lazily (e.g. `jupyter nbconvert --execute` / `nbclient`) instead of
  `python pipeline.py`, keeping the same asserts: exit clean + **zero new oracle
  evals** + headline self-assert.
- `WriteDeliverable` accepts `.ipynb`; `_missing_deliverables` requires it.
- The runtime currently auto-writes `solution.md` from the `Done()` summary —
  that prose would instead become the notebook's leading markdown cells (decide
  who authors it: agent vs runtime injection).
- Keep the lazy + cache-or-load contract intact (notebook re-run = reproduction,
  no sims, no heavy refit).

---

## 3. KB entry: running a study on SLURM
**Status:** deferred. Raised 2026-06-15.

Add a knowledge-base entry (and/or doc) on running an agentic study on a SLURM
cluster. f3dasm's `Pipeline.run(mode="slurm", cluster=...)` already exists
(`pipeline/executors/`); the KB should cover how the agentic layer maps onto it
— the canonical store / FileLock semantics across nodes, how `get_evaluator()`
and the lazy job-status resume behave under cluster parallelism, resource
declaration (`Step.resources`, `orchestrator_resources`), and what changes vs
the current local mode.

---

## 4. Reconcile cancelled-but-completed delegations
**Status:** partially mitigated 2026-06-15 (cancel hardened + eval-count now
from ledger); the core inconsistency remains.

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
state, NOT agent hallucination or prompt friction.

**Done so far:** (a) `evals_used` now counts from the ledger so those evals
aren't dropped from the run total (`agent_runtime.py`); (b) cancel is hardened
against impatience — two-shot for delegations already producing ledgered evals;
docstring + poll/premature nudges reframed (`routing.py`).

**Still to design:** when a detached worker completes with stamped ledger rows,
**reconcile** it — record its completion in the delegation_log (so the
strategizer sees it finished, not "ignored"), or stop the worker BEFORE it
stamps. Pick one source of truth so strategizer and critic never see
contradictory delegation state.
