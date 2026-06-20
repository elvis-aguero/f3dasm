# Authoring a study folder — the agentic-f3dasm contract

A **study folder** is the single input to a run: `AgenticRun(study_dir=...)`.
The agent reads the folder, designs and runs experiments, and writes its results
back into the same folder. This page is the **contract** — the exact files the
runtime reads, the evaluator interface it expects, and the artifacts it
produces — so you can prepare a folder that runs correctly the first time.

For framework internals (architecture, tools, backends, HPC) see
[`../../README-agentic.md`](../../README-agentic.md); this page is the narrow
"what must my folder contain" reference.

---

## Minimal study folder

```
my_study/
  PROBLEM_STATEMENT.md      # REQUIRED — the brief the agent works from
  config.yaml               # recommended — declares model / budget / oracle
  workspace/                # optional — your evaluator + the agent's scratch space
    evaluator.py            #   (only if you ship the oracle as a callable/class)
```

Launch:

```python
from f3dasm.agentic import AgenticRun
AgenticRun(study_dir="my_study").execute()
```

Everything else under `my_study/` (`solution.md`, `pipeline.py`, `runs/…`) is
**produced by the run** — you do not author it.

---

## The input files

### `PROBLEM_STATEMENT.md` — REQUIRED

The runtime reads this file (`agent_runtime.py`) and injects it into the agents
that request it; it is the entire task definition, and the critic checks the
final deliverables against it. Write it precisely — vague briefs produce vague,
ungatable claims. Include:

- **Objective + explicit success criteria** — the headline number or claim the
  run must deliver (e.g. "maximise normalised buckling load; report the design
  vector and the ledgered value").
- **Design space** — every input variable with bounds, type (continuous /
  integer / categorical) and **units**.
- **Deliverables** — what the run must hand back beyond `pipeline.py`
  (e.g. `solution.md`, a mechanism explanation, a plot).
- **Resources** — datasets, solver paths, reference values, prior work to beat.
- **Validity / constraint gates** — feasibility flags, regimes of validity,
  noise gates. These are where reward-hacking happens; state them.

### `config.yaml` — optional (defaults apply), recommended

Every key has a default; an absent `config.yaml` is valid (claude backend,
no budget, honor-system evaluator). Keys the runtime actually **reads**:

| key | type | default | meaning |
|---|---|---|---|
| `model` | str | backend default | LLM id. `AgenticRun(model=...)` overrides. |
| `backend` | `claude` \| `ollama` | `claude` | which LLM backend |
| `budget` | `"HH:MM:SS"` or seconds | none (unlimited) | **soft** wall-clock (warn at 95/100%); a separate `RUN_BACKSTOP_MULTIPLE`× backstop guards runaway cost |
| `eval_budget` | int | none | **soft** cap on ground-truth evaluations |
| `required_deliverables` | list[str] | `[]` | EXTRA files that must exist before `Done()` is accepted (`pipeline.py` is **always** required regardless; `solution.md` is auto-written by the runtime) |
| `evaluator` | block | none → honor-system | declares the oracle — see below |

> ⚠️ **`checkpoint_every` is a no-op.** It appears in `README-agentic.md`
> examples and several shipped configs, but the runtime does **not** read it
> today. Setting it does nothing. (Tracked as a doc/contract gap — see bottom.)

### `workspace/` — optional

Where your evaluator file lives (if you ship one) and where the agents do scratch
work. Not needed if you use a `lookup` pool or let the DataGenerator author the
oracle.

### `run.py` — optional

If present, it defines a custom graph topology / launch and is executed *instead*
of the default entrypoint (this is how the container runner launches a study that
ships its own topology). Most studies don't need it — the default graph is used.

---

## The evaluator contract (the oracle)

This is the most important contract. **`get_evaluator()` (no arguments) is the
only metered path to ground truth**: only rows produced through it are counted
against `eval_budget` and admitted to the canonical ledger. Surrogates,
acquisition functions and optimisers the agent builds are its *own* free
`DataGenerator`s — not routed through `get_evaluator()`, not metered. You
declare the oracle in one of these ways (decision order):

1. **Shipped callable** — a `**kwargs`-style function, one kwarg per input:
   ```yaml
   evaluator:
     entrypoint: "workspace/evaluator.py:evaluate_kw"   # path relative to study_dir
     output_names: [f]                                   # required for a bare callable
   ```
2. **`DataGenerator` subclass:**
   ```yaml
   evaluator:
     entrypoint: "workspace/data_generator.py:MyDataGenerator"
   ```
3. **Precomputed lookup pool** (probes resolve to the nearest pool row):
   ```yaml
   evaluator:
     lookup:
       pool: "experiment_data"            # path under study_dir
       input_columns: [x1, x2, x3]
       output_columns: [y]                # optional
   ```
4. **Agent-authored** — omit the `evaluator` block entirely *and* describe the
   oracle (a binary, dataset, or physics) in `PROBLEM_STATEMENT.md`; the
   DataGenerator agent writes a normalised `DataGenerator` and the runtime
   registers it mid-run. `get_evaluator()` then resolves it.
5. **Nothing** — no `evaluator` and nothing authored → an **honor-system**
   fallback where the agent self-reports eval counts (no instrumented ledger).
   Fine for exploration; **not** recommended for benchmarks you want gated.

`fidelity_column` (optional) names a column for multi-fidelity oracles.

---

## What the run PRODUCES (do not author these)

```
my_study/
  solution.md                              ← headline + critic verdict (study root, written post-gate)
  pipeline.py                              ← lazy f3dasm Pipeline; re-running reproduces the headline from the ledger (0 new evals)
  runs/<timestamp>/
    experiment_data/experiment_data/        ← canonical ledger: output.csv / input.csv / jobs.csv / domain.json
    debug/
      strategizer_notes/hypotheses.json     ← Popperian hypothesis trail
      delegation_log.jsonl                  ← every Delegate() + status
      critic_reviews/call_NNN.md            ← each critic audit + verdict
      diagnostics.jsonl                     ← ScienceMonitor rule firings
      retrospectives.jsonl                  ← per-node consistency notes
      run.log                               ← human-readable log
```

A `solution.md` prefixed `## ⚠ UNGATED RUN` (or `BUDGET EXCEEDED`) did **not**
pass the adversarial critic — treat it as unaudited.

---

## Authentication

Runs use Claude via a **subscription OAuth token** (preferred) or an API key
(fallback). Export `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`) before
launching; see the benchmarks tutorial's Authentication section for the
generate-on-laptop / use-on-cluster recipe.

---

## Pre-flight checklist

Before a long run, confirm:

- [ ] `PROBLEM_STATEMENT.md` exists and states **explicit** success criteria,
      the design space (bounds/types/units), and deliverables.
- [ ] `config.yaml` parses, and its `evaluator` block points at a real
      `file.py:attr` (path relative to study root) or a real `lookup.pool`.
- [ ] The oracle imports and runs on **one** sample without error.
- [ ] Auth: `CLAUDE_CODE_OAUTH_TOKEN` is exported and `api.anthropic.com` is
      reachable from where you'll run.
- [ ] You expect a `pipeline.py` deliverable — it's always required, the
      `Done()` gate refuses to close without it, and the runtime executes it
      lazily to verify the headline reproduces from the ledger.

---

## Minimal worked example

A runnable, **tested** copy of this folder lives at
[`example_study/`](example_study/) and is executed end-to-end by
`tests/agentic/test_study_contract.py` (config loader + `get_evaluator()`), so
this example cannot silently drift from the runtime contract — if it does, the
test fails.

`my_study/config.yaml`:

```yaml
model: claude-haiku-4-5-20251001
backend: claude
eval_budget: 200
evaluator:
  entrypoint: "workspace/evaluator.py:evaluate"
  output_names: [y]
# pipeline.py is the single deliverable (auto-required); solution.md is
# auto-written. Add required_deliverables only for EXTRA files.
```

`my_study/workspace/evaluator.py`:

```python
def evaluate(x1: float, x2: float) -> float:
    """One kwarg per input; returns the single output named in output_names."""
    return (x1 - 1.0) ** 2 + (x2 + 2.0) ** 2
```

`my_study/PROBLEM_STATEMENT.md` (sketch):

```markdown
# Minimise a 2-D quadratic
Objective: minimise y = (x1-1)^2 + (x2+2)^2 over x1,x2 ∈ [-5, 5].
Success: report argmin (x1*, x2*) and the ledgered y*, with pipeline.py
reproducing y* from the canonical store (lazy, zero new evals).
Design space: x1, x2 — continuous, [-5, 5], dimensionless.
Deliverables: solution.md, pipeline.py.
```

Then `AgenticRun(study_dir="my_study").execute()`.

---

## Run analysis protocol

> **Written in stone.** Analyzing a run is a defined procedure, not an
> improvised reading of whatever output happens to be convenient. Deviation
> from this order is a methodology failure — not a style preference.

When a run closes (GATED, UNGATED, FAILED, or watchdog_killed), analysis
proceeds in this exact order. Do not skip steps, do not reorder them.

### Step 1 — Read `debug/retrospectives.jsonl` FIRST

Each node that executes writes a retrospective entry after the run closes.
Fields: `ts`, `source_id`, `role`, `flagged`, `text`. The `text` is
structured as four sections:

- **CONSISTENCY** — rule contradictions the agent detected in its own work
- **DECISION** — the most uncertain strategic choice the agent made, with its
  reasoning
- **FRICTION** — rules, APIs, or contracts that were counterintuitive or
  required guessing
- **BLOCKED** — capability gaps that prevented the agent from doing its job

This is the highest-signal method. The agents report what was confusing,
contradictory, or blocked from *inside* the run — not just what visibly
failed. A FRICTION entry often names the root cause of a bug that never
appears in any traceback. A DECISION entry often explains an apparent budget
overrun or unexpected behavior.

Do not skip this step. If `retrospectives.jsonl` does not exist (the run was
killed before any node closed), note this explicitly — it means you have zero
first-person signal and must rely entirely on external evidence.

> ⚠️ **Retrospectives are ephemeral.** `run.py` wipes the `runs/` directory
> before each new run. Once overwritten, they are permanently lost — the run
> ledger does not archive them. Read retrospectives *before* launching the
> next run.

### Step 2 — Read `debug/diagnostics.jsonl`

ScienceMonitor rule firings: `ERROR_RETURN`, `RAW_ORACLE_NUDGE`,
`BUDGET_WARN`, `HYPOTHESIS_*`, etc. Cross-reference against the
retrospective's FRICTION and BLOCKED fields. A ScienceMonitor event with no
corresponding retrospective FRICTION entry means the agent did not notice it
— that gap is itself a finding.

### Step 3 — Read `debug/critic_reviews/call_NNN.md` (gate audit)

If there were multiple gate attempts, read each one in order. The sequence
reveals whether (a) the agent was fixing real issues, (b) the critic
miscategorised something (false-positive REJECT), or (c) the same underlying
bug was patched superficially and re-surfaced.

### Step 4 — Read delegation transcripts, targeted

Only read `debug/delegations/<ID>/` for delegations flagged by steps 1–3.
Reading all transcripts without a prior hypothesis is expensive and produces
false diagnoses: surface-level error matching without causal understanding.
The transcript is evidence for a hypothesis you already have, not a place to
go fishing.

### Step 5 — Classify every finding before touching any code

For each finding from steps 1–4, assign exactly one category:

| Category | Criterion | Action |
|---|---|---|
| **Bug** | No reasonable reading of the spec justifies this behavior | Fix it; one commit per root cause |
| **Judgment call** | A design tradeoff — science methodology, budget policy, behavioral optimization | Document it; defer to the user; do NOT fix without explicit approval |

The distinguishing test: if a reasonable agent following the spec would make
the same choice, it is a judgment call. If it would not, it is a bug.

### Constraint on prompt and spec fixes — do not overfit

When a bug is fixed by adding a rule to an agent's prompt or spec (as opposed
to fixing a code path), that rule must pass the **parsimony test** before it
is written:

> Would a philosopher of science, reading this rule in isolation, nod at it
> as a general methodological principle — or frown at it as a specific
> workaround for one observed case?

If they would frown, the rule is overfit. Do not add it.

A rule is **parsimonious** if:
- It is general enough to prevent an entire *class* of failures, not just the
  one you observed.
- It is grounded in epistemology or scientific method, not in the accident of
  what happened in run N.
- Removing the observed failure from history, the rule would still belong in
  the spec on its own merits.

A rule is **overfit** if:
- It names a specific tool call, a specific argument format, or a specific
  sequence that failed once.
- It reads as a post-hoc patch: "remember to do X before Y" where the only
  reason to say so is that the agent didn't do it last time.
- A philosopher of science would ask "why is this a rule and not just correct
  behavior implied by the existing principles?"

When a failure is real but the obvious rule would be overfit, look for the
**underlying principle** the failure violated — and add that instead. If no
general principle can be stated, the fix belongs in code (a validation, a
guard, an assertion), not in the prompt.

---

## Commit discipline — git log is the canonical progress diary

> **Written in stone.** Git history is the only record that survives across
> sessions, context resets, and tool failures. It is the diary the user reads
> on wakeup to understand what happened while they were away. Write it
> accordingly.

### One commit per root cause

Every atomic fix gets exactly one commit. "Atomic" means: one root cause,
one change, one commit. Two bugs found in the same run get two commits, even
if both touch the same file.

The test: can you revert this commit cleanly, without undoing anything else?
If not, split it.

### Required commit message structure

Subject line (≤72 chars), conventional commits format:

```
type(scope): short imperative summary
```

The long description is **mandatory** for every fix commit. It must contain
these four sections, in this order, with no omissions:

```
ROOT CAUSE
The actual cause — not just "the bug was X" but "X because Y, which happened
because Z." Cite the specific file and function (e.g.
src/f3dasm/_src/agentic/instrumented.py:_build_batch_domain). One to three
sentences.

EVIDENCE
What revealed this — which retrospective entry (source_id + field + quoted
line), which diagnostics.jsonl event, which critic review call, which
traceback. If the evidence came from a retrospective that no longer exists,
say so explicitly and name the run it came from.

RUN
The run timestamp that triggered this fix (e.g. 20260620T102928).

DEFERRED
Related issues NOT fixed in this commit, and why: behavioral (agent
decision-making that is not a spec violation), requires user judgment
(name the specific question), or out of scope for this change.
If nothing was deferred, write "DEFERRED: none."
```

All four sections are required. A commit missing any section is incomplete.

### Why this matters

The run ledger records *what* happened. The commit message records *why* the
code changed in response. A future reader — including a context-reset Claude
Code instance with no session memory — must be able to reconstruct the full
audit trail from `git log` alone, without access to run artifacts (which are
ephemeral) or session notes (which are lost on context reset).

### Dirty-tree guard

`run.py` refuses to start if there are uncommitted changes (enforced by a
`git status --porcelain` check at launch). The run ledger stamps the current
git SHA as provenance; a dirty tree means that SHA cannot reproduce the run.
Commit or stash all changes before launching.

---

## Known doc/contract gaps (2026-06-10)

- **`checkpoint_every`** is documented and shipped in configs but **not read**
  by the runtime — a no-op key. Either wire it or drop it from docs/examples.
- **Lookup projection distance is invisible** — a probe that resolves to a far
  pool row is not flagged; projection distance is not stamped into provenance.
  Treat lookup results as approximate.
- **Fidelity is not stamped into provenance** today even when `fidelity_column`
  is set on the oracle — a headline can rest on low-fidelity rows without the
  ledger recording the tier.
