# Agentic system — feature catalog

The single place that says **what the agentic system can do, why, and where it
lives.** Read this to get your bearings without reading code.

> **Contract (enforced):** every agent tool listed in an agent's `tools` set MUST
> appear in the "Tools" table below — `tests/agentic/test_features_documented.py`
> fails the build otherwise. Every new *capability* (tool OR infrastructure)
> MUST get an entry here in the same commit that adds it. The test can only
> enumerate tools; infrastructure features rely on this written contract.

Format per feature: **what** (plain language) · **why** · **where** (files) ·
**config** (if any) · **status**.

---

## A. Science & orchestration

### Hypothesis ledger
- **What:** the run's record of falsifiable hypotheses and their verdicts (OPEN /
  SUPPORTED / FALSIFIED / INCONCLUSIVE), append-only.
- **Where:** `hypothesis_ledger.py`; per-run file `debug/strategizer_notes/hypotheses.json`.
- **Status:** core.

### Falsification charter (the Popperian rules)
- **What:** the single binding text defining how a hypothesis may be tested and
  labelled (severity of the attempt, verdict follows the result, no goalpost-moving).
- **Why:** one shared standard both the strategizer and the critic cite.
- **Where:** `knowledge/charter.py`. **Status:** core (§4 user-owned).

### Live verdict validator (#9)
- **What:** when a hypothesis is closed, an independent referee checks — *live* —
  that the verdict obeys the charter, and nudges the strategizer if not.
- **Why:** the gate critic only checks at the end; this catches charter violations
  at the moment of assertion.
- **Where:** `verdict_validator.py` (judge logic); invoked by `nodes/strategizer.py`
  HypothesisUpdate via `node._run_verdict_validator`, which is defined in
  `nodes/critic_gate.py`. Runs on the **critic's** model (reuses the critic adapter),
  not the strategizer's — one refereeing standard, decoupled from the agent it judges.
- **Memory (anti-oscillation):** the judge is fed its own prior rulings on the SAME
  hypothesis (from the ledger `status_log`, via `_prior_rulings_digest`) with a
  justify-any-reversal guard, so a borderline verdict can't silently flip between
  calls. Mirrors the gate critic's prior-reviews digest.
- **Config:** kill switch `F3DASM_VERDICT_VALIDATOR=0`. **Status:** advisory, non-blocking.

### Science monitor
- **What:** background rules that flag scientific drift (unledgered evals, premature
  closes, …) and escalate repeated drift to the critic.
- **Where:** `science_monitor.py`. **Status:** core (§4 user-owned).

### Process milestones
- **What:** a small backlog (assess-literature, oracle-ready, …) that gates the
  implementer until the strategizer resolves each (complete or skip).
- **Where:** `milestones.py`. **Status:** core.

### Delegation + inter-agent messaging
- **What:** the strategizer delegates work to specialist agents and they report back;
  agents can ask one clarifying question, send async messages, and report progress.
- **Where:** `nodes/tools/routing.py`, `nodes/strategizer.py`.
- **Tools:** `Delegate`*, `GetStatus`, `Wait`, `FollowUp`, `Confer`, `ReportEvals`.
  (*Delegate is injected dynamically, not in a static `tools` set.)
- **Status:** core.

## B. The deliverable (pipeline.ipynb)

### Notebook authoring + reproduction gate
- **What:** the single deliverable is a Jupyter notebook; the runtime re-executes it
  lazily and accepts it only if it runs cleanly, adds zero new oracle evals, and
  leaves the ledger unchanged. The printed `REPRODUCED:` headline is informational —
  the critic checks its provenance (it must trace to a real ledger row); the runtime
  no longer machine-matches it to an objective extremum (that wrongly rejected
  constrained optima — audit 20260624T021359).
- **Where:** `notebook_exec.py`, `nodes/tools/routing.py`, `nodes/strategizer.py`
  (`_reproduction_gate`).
- **Tools:** `AddPipelineCell`, `AddPipelineMarkdownCell`, `EditPipelineCell`,
  `DeletePipelineCell`, `ShowNotebook`, `WriteDeliverable`, `CheckDeliverable`.
- **Status:** core (the live deliverable).

### Per-cell notebook debugger (#13)
- **What:** run pipeline.ipynb against a *copy* of the ledger and get a per-cell
  pass/error trace, so a failing cell can be pinpointed instead of guessing.
- **Where:** `notebook_exec.py` `diagnose_notebook`, `RunPipelineCell` closure.
- **Tools:** `RunPipelineCell`. **Status:** done.

### Output-column guidance fix
- **What:** notebook guidance requires naming the objective column EXPLICITLY. The
  earlier "first non-provenance output" auto-detect was unsafe: `output_names` is
  sorted, so with multiple outputs a constraint flag (e.g. `coilable`) sorts before
  the objective and gets silently picked (audit 20260624T021359). If derived, read
  `run_config['evaluator_output_names'][0]`, not column order.
- **Where:** `notebook_exec.py`. **Status:** done.

## C. Workers & ground truth

### Metered oracle (get_evaluator) + canonical ledger
- **What:** the one door to the registered ground-truth oracle; every evaluation is
  written to the canonical store with provenance, under a file lock.
- **Where:** `instrumented.py`. **Tools (worker scratch):** `RunScratch`, `ReportEvals`.
- **Status:** core.

### Literature reviewer
- **What:** a specialist agent that searches papers (arXiv / Semantic Scholar) and
  returns findings; degrades to lexical search without the heavy extras.
- **Where:** `agents/literature.py`, backends. **Status:** core.

### Delegation-ID allocation fix (D002)
- **What:** delegation IDs are allocated *after* the milestone gate, so a blocked
  attempt no longer burns an ID (IDs stay contiguous).
- **Where:** `nodes/tools/routing.py`. **Status:** done.

## D. Resource governance (this is the big recent addition)

### Soft eval-budget nudge
- **What:** when the shared ledger crosses 80/100/150% of the eval budget, the
  *offender* (the running campaign) is nudged via its own output — capped at one per
  band. **Soft: never stops the campaign** (the eval budget is the agent's call).
- **Where:** `instrumented.py` `_flush` governor; budget plumbed via `agent_runtime.py`.
- **Config:** `eval_budget` (config.yaml / `F3DASM_EVAL_BUDGET`). **Status:** done.

### Hard memory cap (the one hard boundary)
- **What:** a 5-second watchman sums each delegation's process-tree **resident (RSS)**
  memory and kills the tree if it exceeds the cap. Real-usage based; verifies the
  process is still ours (start-time match) before killing, so a recycled PID is never
  hit. Per-delegation, absolute (not a share of system RAM), does not sum across
  delegations.
- **Where:** `studies/.../run.py` `_memory_watcher`; `watchdog_cleanup.py`
  (`check_memory_and_kill`, `_owned_pids`); `resource_backend.py`.
- **Config:** `mem_cap` (config.yaml / `F3DASM_MEM_CAP`); default 4 GiB. On SLURM set
  below the job's `--mem`. **Status:** done (cgroup-native HPC backend = future seam).

### Resource backend (OS abstraction)
- **What:** one interface (`set_self_limit` / `read_rss` / `kill` / `proc_start_time`)
  so memory/kill OS-specifics live in one place; psutil impl + stdlib fallback.
- **Where:** `resource_backend.py`. **Status:** done (Linux cgroup backend = future).

### Per-delegation resource telemetry
- **What:** `GetStatus` shows a delegation's eval count and RSS, so the strategizer
  can see a fat campaign (and `Confer` the implementer).
- **Where:** `nodes/tools/routing.py`, `watchdog_cleanup.py` `delegation_rss`.
- **Status:** done.

### Per-delegation ledger KPIs auto-appended to the report
- **What:** when a delegation completes, a KPI footer is appended to the result
  the strategizer auto-receives (GetStatus/Confer/Done) — per-eval wall-time
  (median, max), this delegation's total eval wall-time, and the ledger total.
  Measured from the rows the delegation actually wrote, so budget planning runs
  on observed sim cost instead of an a priori per-sim estimate. Auto-delivered,
  not on-demand. Plain measurements only — interpretation is the strategizer's.
- **Where:** `instrumented.py` `RunStateSummary.{wall_per_delegation,
  delegation_footer}`; appended in `nodes/tools/routing.py`.
- **Status:** done.

## E. Runtime safety

### Wall-clock watchdog + recursive reap (#11/#14)
- **What:** a hard wall-clock timer force-exits a stalled run; on exit it recursively
  kills every campaign process tree — including detached/new-session ones that a
  process-group kill misses.
- **Where:** `studies/.../run.py` `_watchdog`; `watchdog_cleanup.py` `reap_governor_pids`.
- **Config:** watchdog = 2× the run's time budget. Operational kill-switch
  `F3DASM_DISABLE_WATCHDOG=1` turns the wall-clock force-exit OFF (the memory-cap
  watcher stays on) — for long supervised runs. **Status:** done.

### Synthetic watchdog retrospective (#12)
- **What:** a watchdog kill leaves a labelled post-mortem so the analysis protocol
  isn't blind.
- **Where:** `watchdog_cleanup.py` `write_watchdog_retrospective`. **Status:** done.

### KB (handbook) entries
- **What:** curated knowledge the agents consult (incl. running on SLURM, pipeline
  patterns). **Where:** `knowledge/entries/`. **Status:** core.

---

## Tools (every one must be documented above; the test enforces it)

| Tool | Feature |
|---|---|
| `Delegate` | Delegation (dynamically injected) |
| `GetStatus` · `Wait` · `FollowUp` · `Confer` · `ReportEvals` | Delegation + messaging + telemetry |
| `AddPipelineCell` · `AddPipelineMarkdownCell` · `EditPipelineCell` · `DeletePipelineCell` · `ShowNotebook` · `WriteDeliverable` · `CheckDeliverable` | Notebook authoring + reproduction gate |
| `RunPipelineCell` | Per-cell notebook debugger (#13) |
| `RunScratch` | Worker scratch execution against a ledger copy |
| `WriteNote` · `ReadNote` | Agent scratch notes |
| `Read` · `Write` · `Edit` · `Bash` · `Glob` · `Grep` | Workspace file/shell primitives |
| `Done` | Close the run for the gate |
