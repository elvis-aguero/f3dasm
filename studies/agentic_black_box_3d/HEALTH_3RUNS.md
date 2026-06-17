# 3-Run e2e Health Check — agentic_black_box_3d

Commit under test: `40236b41` (handbook injected universally + critic optional
pointer + ReadNote containment fix).
NO code changes between runs — clean read of current state.

**Pre-run-1 note:** the first launch HUNG permanently — the strategizer's first
tool call was `ReadNote("/")`, which (via `Path(study)/"/"` == `/`) recursively
walked the entire filesystem and never returned; the synchronous hang froze the
turn so the time backstop never fired. Killed, root-caused, fixed (`40236b41`),
and restarted. That hung launch does NOT count as a run.

Per-run deep audit (self-directed):
- **solution.md**: headline consistency vs ledger, completeness, honesty.
- **pipeline.py**: valid (gate) AND interpretable (composable, reads as the method,
  captures the real phases vs LHS-only/stub) — primary-read, not gate-trust.
- **retrospectives.jsonl**: every agent — friction, BLOCKERs, CONSISTENCY flags.
- **critic_reviews**: verdict + did it use the new OPTIONAL handbook pointer?
- **transcripts**: actual tool calls — quirks (premature Done, poll loops,
  D###/D### nesting, stale RecallStore), did strategizer ConsultHandbook?
- **provenance**: orphan ledger rows (should be 0).

| Run | run_id | outcome | evals | cost | wall | provenance | solution.md | pipeline.py (valid/faithful) | handbook pointer? | key friction |
|-----|--------|---------|-------|------|------|------------|-------------|------------------------------|-------------------|--------------|
| 1 | 20260617T122559 | GATED | 355 | $0.91 | 33:56 | 0 orphans ✅ | healthy (headline = ledger min exactly) | reproduces ✅ but LHS-only + **broken create branch** | critic *used* handbook, pointer NOT emitted | MILESTONE_BLOCK:2, ERROR_RETURN:1 |
| 2 | 20260617T130405 | GATED (4 critic calls, REJECT→PASS) | 250 | $1.54 | 37:42 | 0 orphans ✅ (D004=RUNNING, traceable) | states coords ✅ but thin; metadata bug (0 delegations) | **pure analysis script — NO get_evaluator** (Criterion-6 anti-pattern, passed as "honest") | **pointer FIRED** ✅ (on H3 criterion, Charter §4) | SCIENCE_DRIFT:12, CONSISTENCY_FLAG:1, REPRO_GATE_BOUNCE:1, ERROR_RETURN:5 |

---

## Run 1 — deep audit

**Outcome:** GATED, 1 critic call (passed first gate), 355 evals, 0 orphans. The
ReadNote containment fix held — no hang.

**solution.md — HEALTHY.** Headline `f = -0.6997451525205849` matches the ledger
true min EXACTLY; 355 evals correct; both hypotheses honestly OPEN (no
falsification attempt made — Charter §2); future work stated. No stale-RecallStore
mismatch this time (unlike the earlier run-5).

**pipeline.py — reproduces, but two real gaps:**
1. **LHS-only again.** Composable create→evaluate→analyze, load-or-create,
   real get_evaluator(), idxmin headline — so it REPRODUCES. But no
   surrogate/optimizer; "Phase 2 deferred." Same thinness as before.
2. **Broken `create` branch (latent).** It calls `domain.add_continuous_input(...)`
   — which does NOT exist (Domain has `add_float`). On a FRESH/empty store
   `python pipeline.py` would crash at create. The repro gate runs the LOAD
   branch (ledger present), so it NEVER executes create → the bug is invisible
   to the gate, and the critic READ it without catching the fake API. So the
   pipeline can reproduce but CANNOT regenerate — uncaught.

**Critic + handbook.** The universal ConsultHandbook injection WORKS — the critic
consulted the handbook (reproducibility + charter) and lists it as available
(FRICTION line). BUT it did NOT emit the optional "Handbook pointer" despite the
LHS-only thinness — it passed it as "transparent, defensible incompleteness."
→ The advisory nudge is too weak/skippable to actually surface the thinness.

**Retrospectives (reflection/interview):** cleanest yet — datagenerator, critic,
strategizer all CONSISTENCY: ok, FRICTION: none, no BLOCKERs. (Implementer wrote
no retrospective text.) Note: the datagenerator correctly used `add_float`; the
STRATEGIZER introduced the fake `add_continuous_input` when assembling pipeline.py.

---

## Run 2 — deep audit

**Outcome:** GATED but only after **4 critic calls** (REJECT → … → PASS). High
friction: SCIENCE_DRIFT:12, CONSISTENCY_FLAG:1, REPRO_GATE_BOUNCE:1, ERROR_RETURN:5.
Worse science: ledger min **−0.36** (250 LHS evals; much of the run spent fighting
the gate, not optimizing). 0 orphans (D004 logged RUNNING → traceable).

**Wins (the new machinery works):**
- The **optional Handbook pointer FIRED** (call_004): a 3-line advisory citing
  "hypothesis-registration-criteria-validity" to fix H3's falsification criterion
  before Phase 3 — advisory, didn't change the PASS verdict. Exactly as designed.
- The critic (call_002) **caught the thin/false pipeline**: 4-phase docstring but
  dead, never-called `fit_surrogate`/`optimize_with_bo`; the `add_continuous_*`
  vs `add_float` API mismatch (same fake-API class as h1); H3 criterion ≠
  prediction (Charter §4); H2/H3 confounding; and a headline-vs-ledger x-coord
  mismatch (the stale-draft-before-verify issue recurs).

**The collapse, though:** the agent "resolved" the dead-code MAJOR not by
implementing the phases but by **stripping pipeline.py to a pure read-only
analysis script** — `load ledger → idxmin → print`, **zero get_evaluator()**, no
Pipeline/Step. That is the Criterion-6 anti-pattern ("read-only analysis script…
is a CRITICAL finding"), yet the critic PASSED it: *"fails the regeneration
criterion … no longer an overclaim problem … resolved by honest restatement."*
**The deliverable bar collapsed to "reproduces + honest about being a stub."**

**Root tension RESURFACED (strategizer's own CONSISTENCY flag):** *"pipeline.py
must be a faithful, COMPOSABLE pipeline of the WHOLE process (create → sample →
run → fit surrogate → optimize → analyze)" — but the campaign is incomplete (only
Phase 1).* When budget runs out before the method completes, the agent cannot
ship a whole-process pipeline, so it ships either a false 4-phase shell (REJECTED)
or an honest analysis script (PASSED). The regenerate-vs-lazy tension I thought
ROOT 2 closed re-appears whenever the SCIENCE is incomplete.

**solution.md quality:** better than h1 — it now **states the coordinates**
(`x = [+2.516, +1.273, −2.885]`, forced by the critic's x-coord CRITICAL). But
still no landscape characterization (where to look), generic "Phases 2–4 PENDING"
next steps, and a metadata bug (`total_delegations: 0` despite 3+).
