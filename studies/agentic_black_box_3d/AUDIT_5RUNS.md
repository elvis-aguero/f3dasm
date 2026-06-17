# 5-Run Overnight Audit — agentic_black_box_3d

Autonomous loop: run the e2e 5×, audit each, close system back doors (not
agent-dumbness), report KPIs + changes on wake.

Starting commit baseline (ROOT 1/2/4 fixed):
- `e8662f3c` ROOT 1 — un-clobberable ledger (sentinel + shrink guard)
- `74aa505e` ROOT 2 — load-or-create deliverable, proven satisfiable
- `a5c26297` ROOT 4 — HypothesisList absorbs stray kwarg
- (+ ROOT 1 prompt rule)

Discipline: ground-truth before any claim; lead with friction; fix ONLY
beyond-reasonable-doubt system defects; deterministic regression test per fix.

NOTE: `run.py` wipes `runs/` each launch — audit + persist BEFORE relaunching.

---

## KPI table (append-only, mirrors run_ledger.csv)

| Run | run_id | outcome | critic | deleg | ledger_rows (budget 1000) | cost | wall | provenance | friction codes |
|-----|--------|---------|--------|-------|---------------------------|------|------|------------|----------------|
| 1 | 20260617T004224 | UNGATED | REJECT (4 crit) | 4 | **1523 (+52%)** | $1.77 | 22:09 | **D003: 915 orphan rows** | MILESTONE_BLOCK:1, ERROR_RETURN:7, SCIENCE_DRIFT:6, CONSISTENCY_FLAG:1 |
| 2 | 20260617T012012 | UNGATED | MAJOR (criticals fixed) | 9 | 543 (in budget ✅) | $1.45 | 22:03 | **0 orphans ✅** (D003 traceable=RUNNING) | ERROR_RETURN:5, MILESTONE_BLOCK:1, SCIENCE_DRIFT:5, CONSISTENCY_FLAG:1 |
| 3 | 20260617T015040 | **GATED ✅** | PASS | 6 | 384 ✅ | $1.15 | 32:26 | 0 orphans ✅ | ERROR_RETURN:**1**, MILESTONE_BLOCK:1, SCIENCE_DRIFT:**2** (no consistency flag) |

---

## Run 1 — audit

**Outcome:** UNGATED; critic REJECT with 4 CRITICAL findings; strategizer
ended BLOCKED. ERROR_RETURN:7 + SCIENCE_DRIFT:6 = high friction.

**Retrospectives (the goldmine) converged on ONE system back door**, flagged by
critic ×2 AND the strategizer's own BLOCKER — explicitly NOT agent fault:

- **Orphaned ledger rows.** Ground truth: the canonical store has rows stamped
  `D003`=915 and `D005`=608 (total 1523), but `delegation_log.jsonl` logs only
  D002/D004/D005/D007. **D003's 915 evals have NO delegation record.** Strategizer:
  *"delegations … start evaluating … then error or get cancelled. The ledger
  accumulated evaluations from D003 (200 rows) that have no corresponding
  delegation record … breaking the provenance audit trail."* Critic FRICTION:
  *"the delegation_log and the canonical ledger are meant to be coupled … they
  are decoupled here."*

- **Code trace (beyond reasonable doubt = system, not agent):** `record()` is
  written only at COMPLETION; the instrumented evaluator flushes rows DURING
  execution. A delegation cancelled/killed mid-flight (wall/eval budget) flushes
  rows but never reaches `record()` → orphan. AND both eval-budget counters
  (`routing.py` `_spent` = `sum(query_all().evals)`; strategizer `evals_used`
  accumulator) sum the LOG, so orphan rows escape the counter → the 1523-vs-1000
  overrun. Same root for both symptoms, even though the system's own comments
  declare "the canonical store is the single source of truth for the eval count."

**Critic CONSISTENCY check on the charter/checklist: "ok — self-consistent, no
contradictions"** → corroborates that ROOT 2 (the prior self-contradictory
deliverable spec) is now resolved; the recurring "lazily resumable is ambiguous"
complaint did NOT reappear.

**NOT fixed (agent-judgment / ambiguous, deferred to watch):**
- Headline f=-0.701 vs ledger true min ≈-1.0 (finding 1) — downstream of the
  orphan rows (agent derived over the traceable subset); revisit if it persists
  after the provenance fix.
- INCONCLUSIVE vs OPEN (finding 3) — the critic ITSELF called this a Charter
  ambiguity, not a clear violation. Candidate spec-clarification if it recurs.

---

## Changes made (chronological, with evidence)

### After Run 1 — close the orphaned-delegation back door (one root, two symptoms)
Evidence: D003 = 915 ledger rows, 0 delegation_log entries; 1523 > 1000 budget.
- `delegation_log.py`: add `record_started()` (RUNNING entry at dispatch) +
  `_load_all()` collapses last-wins → killed/cancelled delegations stay
  traceable; terminal record supersedes. (provenance)
- `routing.py`: log RUNNING at dispatch; `_spent` now counts canonical-ledger
  rows (RunStateSummary.n_rows), not the log-sum. (budget truthfulness)
- `strategizer.py`: budget warning `evals_used` = max(accumulator, ledger rows).
- Tests: `test_delegation_log_provenance.py` (3, pin orphan-traceable + last-wins
  + completion order); updated 4 tests that read the raw JSONL to the collapsed
  view. Full deterministic suite: 923 passed.

### After Run 2 — allow a verdict RETRACTION (carry-forward), fix the close deadlock
Run-2 audit confirms the run-1 fix WORKED: critic — *"the prior review's three
CRITICAL findings (wrong headline, underreported budget, non-reproducible
pipeline) have all been fixed"*; provenance check: **0 orphan stamps**, budget
543 < 1000.

New back door (strategizer's explicit BLOCKER — beyond reasonable doubt a system
deadlock, not agent judgment): the agent CORRECTLY realised it marked H1/H3
SUPPORTED without a falsification ATTEMPT (Charter §2 → should be OPEN), tried to
fix it, but `HypothesisUpdate` refused the downgrade ("reopening requires new
evidence with a 'delegation' key") — and at close time there is no new
delegation to cite. Forced contradiction (narrative OPEN vs ledger SUPPORTED) →
critic re-raised across two gate calls → CONSISTENCY_FLAG → UNGATED.

Fix (`hypothesis_ledger.py`): retracting SUPPORTED/INCONCLUSIVE → OPEN carries
forward the evidence the verdict was based on (no new delegation needed —
withdrawing a claim isn't asserting one). Un-falsifying a FALSIFIED hypothesis
STILL requires new evidence (anti-dodge preserved). Docstring updated so the
agent knows it can self-correct. Tests: `test_hypothesis_retraction.py`; existing
anti-dodge test green. 925 passed.

**Watch:** headline-subset discrepancy reappeared in a new guise (critic: pipeline
idxmin = −0.79 but conclusion claimed −0.569 from a 250-row subset) — still
downstream of agent analysis, not a clear system defect yet.

### After Run 3 — first GATED run; close two friction back doors
Run 3 **GATED** (critic PASS), 0 orphans, retraction fix held (H1/H2 honestly
OPEN, no forced contradiction). Friction fell sharply: ERROR_RETURN 7→5→**1**,
SCIENCE_DRIFT 6→5→**2**, no CONSISTENCY_FLAG. Two systemic frictions remained
(flagged in retrospectives, ground-truth confirmed — both beyond reasonable
doubt system, not agent fault):

1. **`Domain` not exported from `f3dasm` top-level** (confirmed: `f3dasm.Domain`
   was False; only `f3dasm.design.Domain` worked). The strategizer's natural
   `from f3dasm import Domain` failed → ~4 wasted CheckDeliverable calls. Fix:
   export `Domain` at top level (peers ExperimentData/Pipeline already are).

2. **Write sandbox double-nested `D###/D###/`** (flagged by 3 agents across
   runs). The sandbox is already rooted at `{delegation_id}/`, but the prompt
   calls it "your D### subfolder" so agents prefix paths with it → `_ws/D###/…`
   nested. Fix: `_sandboxed_write` absorbs a redundant leading `{delegation_id}/`
   (absolute-path rejection preserved). The prior allow-test silently passed
   while double-nesting; now pinned.

Tests: `test_worker_write_strips_redundant_delegation_prefix`; Domain export
checked. 926 passed.
