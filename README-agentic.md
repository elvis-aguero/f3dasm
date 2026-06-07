# agentic-f3dasm

An LLM-orchestrated, **scientifically-disciplined** design-of-experiments loop
on top of f3dasm. You drop a `PROBLEM_STATEMENT.md` into a study directory; a
graph of LLM agents reads it, forms falsifiable hypotheses, delegates concrete
experiments, runs f3dasm pipelines, accumulates a provenance-tracked evaluation
ledger, and returns a structured, replication-tested result.

```bash
uv run python -m f3dasm.agentic <study-dir>
```

The only required input is a natural-language problem description. Everything
else — sampling strategy, surrogate choice, optimiser, when to stop — is decided
by the agents and recorded as ground-truth data, not prose.

---

## Table of contents

- [Why this is different](#why-this-is-different)
- [Part I — Production guide (running workflows)](#part-i--production-guide)
- [Part II — Architecture (for interns extending the codebase)](#part-ii--architecture)
- [Tool catalog](#tool-catalog)
- [Agent roster](#agent-roster)
- [Run outputs: what lands on disk](#run-outputs-what-lands-on-disk)
- [Backends](#backends)
- [Containerized runs](#containerized-runs)
- [Literature tooling & external-API guardrails](#literature-tooling--external-api-guardrails)
- [Extending the system](#extending-the-system)
- [Testing](#testing)
- [Repository layout](#repository-layout)

---

## Why this is different

Most "agentic optimisation" loops let an LLM type numbers into a report and
trust them. This one treats the run as a **self-healing dynamical system** that
resists drift toward vague, unsupported claims. Three pillars:

1. **Popperian hypothesis ledger** — every hypothesis is one falsifiable claim
   with an explicit falsification criterion, a prediction, and a prior; every
   status change cites evidence (a delegation + numbers) and an updated
   posterior. Managed only through tools, never hand-edited.

2. **Canonical evaluation ledger** — a single run-level f3dasm `ExperimentData`
   store is the ground truth. Every real evaluation is appended (with
   provenance) as a *side effect of evaluating*, concurrency-safe under a file
   lock. The agent cannot fake an evaluation or its budget — the count is
   mechanical, not self-reported.

3. **ScienceMonitor + adversarial critic** — deterministic drift rules run after
   every hypothesis update and delegation, inject bounded corrective feedback
   when the run drifts, and escalate to an adversarial critic. The run closes
   *only* through a critic-gated `Done()`; ending a turn after a refused `Done()`
   re-prompts, and after repeated refusals the result is stamped **UNGATED**.

The competitive advantage is **transparency**: every number in the final report
is traceable to a row in the ledger that the runtime itself counted.

---

# Part I — Production guide

For running agentic workflows reliably.

## Install

```bash
uv venv
uv pip install -e ".[agentic]"          # core agentic stack
uv pip install -e ".[agentic,lit]"       # + literature tooling (S2, embeddings)
```

The `agentic` extra installs `claude-agent-sdk`, `langchain-*`, `langgraph`, and
`pymupdf`. The `lit` extra adds `semanticscholar` and `fastembed` (dense
retrieval; on Intel macOS dense retrieval runs out-of-process — see the
literature section).

For the **Claude** backend you need the `claude` CLI on `PATH` with an active
session (run `claude` once interactively to authenticate). For **Ollama** you
need a local Ollama server.

## Run a study

```bash
# existing study
uv run python -m f3dasm.agentic studies/agentic_black_box_8d

# your own
mkdir studies/my_problem
cat > studies/my_problem/PROBLEM_STATEMENT.md << 'EOF'
# Maximise f(x) over an 8-D box [-5,5]^8.
# A DataGenerator wrapping the evaluator is provided in workspace/.
# Budget: 1000 evaluations.
EOF
# ship an evaluator (see config below) and run:
uv run python -m f3dasm.agentic studies/my_problem
```

## config.yaml reference

`config.yaml` is optional; every key has a default.

```yaml
model: claude-haiku-4-5-20251001    # LLM model id
backend: claude                      # "claude" (default) or "ollama"
budget: "00:30:00"                   # HH:MM:SS wall-clock (soft warn at 95/100%,
                                     #   HARD stop past the 5% cleanup window)
eval_budget: 1000                    # max function evaluations (soft warn)
checkpoint_every: 30                 # delegations between strategizer checkpoints

# Declares the run's oracle so the runtime can instrument it (see "canonical
# ledger"). OPTIONAL — absent => honor-system ReportEvals fallback.
evaluator:
  # (a) a bare callable, path:attr relative to study_dir, **kwargs-style:
  entrypoint: "workspace/evaluator.py:evaluate_kw"
  output_names: [f]
  # (b) OR a DataGenerator subclass:
  # entrypoint: "workspace/data_generator.py:BioreactorDataGenerator"
  # (c) OR a precomputed lookup pool:
  # lookup:
  #   pool: "experiment_data"               # path under study_dir
  #   input_columns: [ratio_d, ratio_pitch, ratio_top_diameter]
  #   output_columns: [coilable, sigma_crit, energy]
  fidelity_column: null                      # e.g. "fidelity" for multi-fidelity

required_deliverables:                        # hard gate before Done() (replicate.py
  - replicate.py                              #   is always required regardless)
```

**Budget semantics.** Wall-clock budget gives a soft warning at 95% and 100%,
then a **hard stop** once the 5% cleanup window is exhausted (the run terminates
with a `BUDGET EXCEEDED` banner rather than drifting). Eval budget is a soft
warning. `replicate.py` is always a required deliverable; `Done()` is refused
until it exists.

## What "good" looks like

A healthy run ends with `solution.md` (the critic-passed conclusion + run
metadata) and a `replicate.py` that, when run, loads the canonical ledger and
**asserts** the headline number. If you see `## ⚠ UNGATED RUN` or
`## ⚠ BUDGET EXCEEDED` at the top of `solution.md`, the conclusion did **not**
pass the gate — treat it as unaudited.

See [Run outputs](#run-outputs-what-lands-on-disk) for the full artifact map.

## Cost & model guidance

The benchmark studies run end-to-end on **Haiku 4.5** at roughly $1–3 and
15–35 minutes per run. Use a stronger model (Sonnet/Opus) for harder reasoning;
set per-node models to mix (cheap workers, stronger strategizer). Both Claude
and Ollama are first-class — every feature works on both.

---

# Part II — Architecture

For interns extending the codebase.

## LangGraph under the hood

The runtime compiles an `Agent`/`Graph` spec into a LangGraph `StateGraph`. Each
`Agent` becomes a node; routing is `Command(goto=...)` driven by the closure
tools an agent calls. `AgenticState` (a `MessagesState` subclass) carries
messages, budget counters, `run_dir`, `experiment_data_dir`, and routing
metadata across turns. **Users never touch LangGraph** — the `Agent`/`Graph`/
`Edge` API is the only surface.

State is for control-flow scalars and the message channel. The *scientific*
state lives in dedicated, file-backed objects discovered via `run_dir`:
`HypothesisLedger` (`hypotheses.json`), `DelegationLog` (`delegation_log.jsonl`),
the canonical `ExperimentData` store, and the `ScienceMonitor`'s diagnostics.
Heavy data never travels through the LangGraph checkpointer (it would be
re-serialized every super-step) — only a path handle does.

## The scientific loop (the differentiator)

```
        ┌──────────────────────── StrategizerNode ────────────────────────┐
        │ Hypothesis ledger  ─┐                                            │
        │  Propose/Update    │ provenance        ScienceMonitor (rules)   │
        │  (Popperian schema)─┘  ┌── reads ledger + delegation log + store │
        │                        ▼   → drift violations → bounded inject   │
        │ Delegate(target, intent, hypothesis_ids, is_falsification_attempt)│
        │      │ fires worker (own thread)         │ escalation            │
        │      ▼                                    ▼                       │
        │ DelegationLog ◄── worker Report      AskForFeedback / Done gate   │
        │      │                                    │ critic verdict        │
        │      ▼                                    ▼                       │
        │ canonical ExperimentData store ◄── InstrumentedDataGenerator      │
        │  (one append path, FileLock, provenance, mechanical eval count)   │
        └───────────────────────────────────────────────────────────────────┘
```

### 1. Hypothesis ledger (`hypothesis_ledger.py`)

`HypothesisLedger` manages `hypotheses.json`. Schema (enforced at propose/update
time, all failures return `ERROR:` strings — never raise to the agent):

- `propose(statement, falsification_criterion, prediction, prior, proposed_by)`
  — `prior` strictly in (0,1); fields ≤500 chars; a compound-claim heuristic
  rejects statements bundling multiple quantified sub-claims; max 3 OPEN.
- `update(h_id, status, comment, evidence, posterior, triggered_by)` — closing
  statuses (SUPPORTED/FALSIFIED/INCONCLUSIVE) require `evidence` citing a real
  delegation; `posterior` always required; no-op updates rejected.

`from_dict` is strict (no backward compat — old ledgers fail loudly at load).

### 2. Canonical evaluation ledger (`instrumented.py`)

`InstrumentedDataGenerator` wraps the study's evaluator. On each `execute()` it
delegates to the inner generator, stamps provenance (`_delegation_id`, `source`,
`_ts`), buffers, and flushes to the run-level store **inside a `FileLock`**
(reusing the proven `datagenerator.py:_store_experiment_sample` read-merge-write
pattern — bare `store()` is not atomic). It also writes a **mechanical** eval
counter per delegation.

Workers obtain it with zero arguments:

```python
from f3dasm.agentic import get_evaluator
gen = get_evaluator()          # binds delegation id from cwd, reads run_config.json
data = gen.call(data, mode="sequential")
gen.flush()
```

`get_evaluator(inner=None)` resolves the inner evaluator from the study's
`evaluator:` config (`load_inner_evaluator`): bare callable, `DataGenerator`
subclass, or `LookupDataGenerator` over a pool. Pass `inner=` explicitly when an
agent authors its own evaluator.

**The contract (lean, append-only):** all artifacts are append-only — nothing is
deleted. Evaluations are facts (the store); surrogates/samplers/EDA are
interpretations (delegation workspace, superseded not deleted); decisions live
in the ledger. There is ONE append path and ONE provenance column set; whether a
row is a real-oracle fact or a tagged surrogate guess is read via `source`, not
gated in code. Provenance enables *filtered views and attribution, never
deletion* — you never roll back an evaluation, and a regretted strategy's evals
still enrich the next one.

### 3. ScienceMonitor (`science_monitor.py`)

A pure-Python observer over the ledger + delegation log + store. Stateless rule
evaluation: a violation that resolves simply stops appearing. Rules (each
`error` or `warn`):

| Rule | Fires when |
|---|---|
| `EVIDENCE_DELEGATION_EXISTS` | a hypothesis cites a delegation that doesn't exist |
| `EVIDENCE_NUMBERS_MATCH` | cited numbers don't appear in that delegation's report |
| `SUPPORTED_WITHOUT_ATTACK` | SUPPORTED with no `is_falsification_attempt` delegation |
| `STALE_OPEN` | an OPEN hypothesis untouched by the last K delegations |
| `UNANCHORED_DELEGATION` | a completed delegation produced no `### Numbers` |
| `POSTERIOR_INERTIA` | status changed but belief barely moved |
| `UNLEDGERED_EVALS` | a DONE delegation reported evals but wrote 0 store rows (bypassed `get_evaluator`) |

Violations are re-validated at injection time, deduped, capped (≤2/turn with a
digest), and logged to `diagnostics.jsonl` as `SCIENCE_DRIFT`. Repeated drift
escalates to a synchronous adversarial-critic audit (≤2/run).

### 4. Termination (route-aware, `nodes.py` `StrategizerNode.__call__`)

A run closes **only** through an accepted `Done()` (two-shot, critic-gated when a
critic is connected). Ending a turn without one → bounded re-prompt (×3) → forced
finish with an `UNGATED` banner. Past the budget's 5% cleanup window → hard stop
with a `BUDGET EXCEEDED` banner. Live delegations survive loop-backs.

## Defining a graph

```python
from f3dasm.agentic import Edge, Graph, AgenticRun
from f3dasm.agentic.agents import (
    StrategizerAgent, ImplementerAgent,
    AdversarialCritiqueAgent, LiteratureReviewAgent,
)

graph = Graph(
    nodes={
        "strategizer": StrategizerAgent(),
        "implementer": ImplementerAgent(),
        "critic": AdversarialCritiqueAgent(),
        "literature_reviewer": LiteratureReviewAgent(),
    },
    edges=(
        Edge("strategizer", "implementer"),
        Edge("strategizer", "critic"),
        Edge("strategizer", "literature_reviewer"),
    ),
    entry="strategizer",
)
AgenticRun(study_dir="studies/my_problem", graph=graph).execute()
```

When no `graph=` is passed, a sensible default topology is built. A connected
`critic` activates the `Done()` gate and `AskForFeedback`; without it, `Done()`
closes directly.

### Agent class attributes

| Attribute | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str` | `""` | system prompt for this node |
| `tools` | `frozenset[str]` | `frozenset()` | declared tool names (opt-in) |
| `role` | `str` | `"implementer"` | `"strategizer"`, `"implementer"`, `"critic"` — controls injected closures |
| `backend` | `str \| None` | `None` | per-node `"claude"`/`"ollama"` override |
| `model` | `str \| None` | `None` | per-node model override |
| `reset_on_checkpoint` | `bool` | `True` | clear history between delegations |
| `description` | `str` | `""` | shown to the strategizer as a delegation target hint |

---

## Tool catalog

`Agent.tools` is opt-in (default: no tools). Three categories.

### Native backend tools (declare in `Agent.tools`)

`Bash`, `Read`, `Write`, `Edit`, `Glob`, `Grep`. SDK-native on Claude; custom
`StructuredTool` wrappers on Ollama. Worker `Write` is sandboxed to the
delegation's `debug/delegations/D###/` folder.

### Protocol closure tools (declare in `Agent.tools`)

| Name | Role | Purpose |
|---|---|---|
| `Done` | strategizer | end the run (two-shot, critic-gated) |
| `FollowUp` | any | one clarifying question to the delegating party |
| `WriteNote` / `ReadNote` | strategizer | `.md` lab-notebook notes / read study files |
| `WriteDeliverable` | strategizer | write `replicate.py` (or other top-level deliverable) |
| `ReportEvals` | implementer | honor-system eval count (fallback when not using `get_evaluator`) |

### Topology-injected tools (never declare — added by the runtime)

| Tool | Injected when |
|---|---|
| `Delegate(target, intent, expected_report, hypothesis_ids, wait, is_falsification_attempt)` | node has outgoing edges |
| `GetStatus` / `Reply` | orchestrating node (poll / answer a worker FollowUp) |
| `HypothesisPropose` / `Update` / `List` / `Get` | entry node with a ledger |
| `RecallHistory(n)` | any node (demand-driven episodic memory from the delegation log) |
| `RecallStore()` | strategizer (derived summary of the canonical ledger) |
| `QueryStore(delegation_ids, source, n_best, output_name)` | strategizer (read-only filtered view) |
| `AskForFeedback(hypothesis_ids)` | entry node when a critic is connected |

---

## Agent roster

| Agent | role | Tools | Purpose |
|---|---|---|---|
| `StrategizerAgent` | strategizer | Done, FollowUp, WriteNote, ReadNote, WriteDeliverable (+ injected) | orchestrates: hypotheses, delegations, synthesis. Entry node. |
| `F3dasmImplementer` (`ImplementerAgent`) | implementer | Bash, Edit, Read, Write, Glob, Grep, ReportEvals | runs f3dasm pipelines, evaluates designs, produces data |
| `AdversarialCritiqueAgent` | critic | Read, Glob | adversarial quality gate; verdict PASS/REVISE/REJECT |
| `LiteratureReviewAgent` | implementer | Read, Grep, Glob (+ corpus/MCP tools) | primary-source literature review |
| `DebuggerAgent` | implementer | Bash, Read, Grep, Edit, Write | debugging delegations |
| `DataGeneratorImplementerAgent` | implementer | Bash, Edit, Read, Write, Glob, Grep, ReportEvals | authors `DataGenerator` subclasses |

---

## Run outputs: what lands on disk

```text
studies/<study>/
    PROBLEM_STATEMENT.md          # the only required input
    config.yaml                   # optional
    solution.md                   # ← final report + metadata (study root)
    replicate.py                  # ← agent-written: loads the ledger, asserts the headline
    runs/<timestamp>/
        experiment_data/          # ← CANONICAL LEDGER (input/output/jobs/domain csv+json)
        debug/
            run_config.json       # store/counter/lock paths + evaluator entrypoint
            delegation_log.jsonl  # every delegation (full task + deliverable + provenance)
            diagnostics.jsonl     # tool errors + SCIENCE_DRIFT records
            eval_counter/D###.count  # mechanical per-delegation eval counts
            strategizer_notes/
                hypotheses.json   # the Popperian ledger
                *.md              # planner notes
            delegations/D###/     # per-delegation worker scratch space
```

**Reading a run:** start at `solution.md` (banner check first). Cross-check
claims against `experiment_data/` (the ledger), audit the reasoning trail in
`hypotheses.json` + `delegation_log.jsonl`, and inspect `diagnostics.jsonl` for
`SCIENCE_DRIFT` to see where the run was nudged.

---

## Backends

Both backends are first-class ("gold standard") — every feature is implemented
for both.

- **Claude (default)** — `ClaudeAdapter` wraps the claude-agent-sdk `query()`
  with an MCP server for closure tools; native tools are SDK tool names.
- **Ollama** — `OllamaAdapter` uses `ChatOpenAI(base_url=…/v1)` + LangGraph
  `create_react_agent`; native tools are `StructuredTool` wrappers. Closure
  tools behave identically.

```yaml
backend: ollama
model: qwen2.5:7b      # any tool-calling Ollama model
```

Per-node override: set `backend = "ollama"` on one `Agent` subclass to mix
backends within a graph.

---

## Containerized runs

`AgenticRun(container=True)` runs the whole study in an isolated container via
`ContainerRunner` (`container_runner.py`) — `docker run` / `docker compose`
(Colima on macOS). Isolation is whole-run (filesystem, network, process tree),
useful for untrusted implementer code and pinned reproducibility.

```python
AgenticRun(study_dir="studies/x", container=True,
           container_image="f3dasm-agentic:latest").execute()
```

---

## Literature tooling & external-API guardrails

The literature reviewer is the only tool surface that reaches the public
internet, so it is heavily guarded (`literature_corpus.py`, `agents/literature.py`):

- **Primary-source enforcement:** corpus entries carry a `full_text` flag;
  `CorpusSearch` returns quotable text only from full-text papers. No full text ⇒
  not quotable. `DownloadPdf` + open-access URLs (from OpenAlex) complete the
  fetch chain.
- **Rate limiting + circuit breaker:** per-domain pacing (Semantic Scholar 1
  req/s, arXiv 1 req/3s + jitter for shared-IP friendliness, OpenAlex 5 req/s),
  `Retry-After` honored, no-retry on 4xx, 60s per-source cooldown after repeated
  429s.
- **24h on-disk GET cache** keyed by URL+params (cache hits skip rate limiting).
- **Two citation-graph backends:** Semantic Scholar + OpenAlex `cites:`/
  `references` (the resilient fallback during S2 cooldowns).
- **Dense retrieval:** bge-small via `fastembed`; on Intel macOS (numpy-2 vs
  onnxruntime wheels) it runs out-of-process in an ephemeral env
  (`_embed_worker.py`); BM25 is the always-available fallback.

---

## Extending the system

**Add an agent:** subclass `Agent` (or a canonical agent), set `role`,
`system_prompt`, `tools`; add it to a `Graph` with `Edge`s. Closures are injected
by role automatically.

**Add a closure tool:** add a builder in the relevant `_build_*_closures` method
in `nodes.py`, register it in the returned dict, and gate its injection (by role
or by graph topology). Wrap it via `_wrap_closure` so `ERROR:`/exceptions are
recorded.

**Add a ScienceMonitor rule:** add a `_check_*` method returning `Violation`s and
call it from `evaluate()`. Keep it stateless (recompute from ledger + log +
store); choose `warn` vs `error` severity.

**Add a study:** create `studies/<name>/PROBLEM_STATEMENT.md`, ship an evaluator,
declare it under `evaluator:` in `config.yaml`. Lookup studies point at a pool;
live studies point at a `DataGenerator` subclass or a bare callable.

**Backend parity is non-negotiable:** any new feature must work on both Claude
and Ollama. Closures are pure Python (backend-neutral by construction); native
tools need a wrapper in both adapters.

---

## Testing

```bash
# fast unit/integration suite (no network, no LLM)
uv run pytest tests/agentic/ -q --no-cov --ignore=tests/agentic/test_literature_wet.py

# wet integration tests (real LLM + APIs; marked `integration`, opt-in)
uv run pytest tests/agentic/test_literature_wet.py -v -s --no-cov
```

The suite covers the ledger, ScienceMonitor rules, the instrumented store
(including a real-threads concurrency no-loss proof), evaluator resolution, the
route-aware termination, and prompt hygiene. Wet runs against the benchmark
studies are the end-to-end truth and catch prompt-adherence failures unit tests
can't.

---

## Repository layout

```text
src/f3dasm/agentic/__init__.py         # public API
src/f3dasm/_src/agentic/
    agent_runtime.py        # AgenticRun: config, execute(), run-dir + store init
    agent_prompts.py        # prompt constants + run/workspace preambles
    graph_builder.py        # compile Agent/Graph spec → LangGraph StateGraph
    graph_state.py          # AgenticState
    nodes.py                # StrategizerNode / WorkerNode (closures, routing, termination)
    hypothesis_ledger.py    # Popperian HypothesisLedger
    science_monitor.py      # drift rules + escalation
    instrumented.py         # InstrumentedDataGenerator, get_evaluator, RunStateSummary
    delegation_log.py       # graph-wide append-only delegation log
    lookup.py               # LookupDataGenerator (dataset/pool studies)
    literature_corpus.py    # corpus + rate-limited HTTP + dense retrieval
    container_runner.py     # ContainerRunner (Colima/Docker whole-run isolation)
    optimizer.py            # AgenticOptimizer (f3dasm Optimizer interface)
    _embed_worker.py        # out-of-process bge-small embedder
    agents/                 # strategizer, implementer, critic, literature, debugger, datagenerator
    backends/               # base (Agent/Edge/Graph), claude, ollama

studies/<study>/            # PROBLEM_STATEMENT.md (+ config.yaml, evaluator, runs/)
tests/agentic/              # unit + integration + wet tests
```

---

## Studies

| Study | Description |
|---|---|
| `agentic_black_box_8d` | 8-D black-box optimisation; compiled evaluator |
| `agentic_supercompressible_3d` / `_7d` | metamaterial design (Bessa benchmark); lookup pool |
| `agentic_modular_resonance` | integer optimisation; agent-authored `DataGenerator` |
| `agentic_project_euler_078` | smoke test (no f3dasm DOE) |

Flagship benchmark problem statements (bioreactor CFD, supercompressible FEM,
interfacial-locomotion FSI) live in the sibling `f3dasm-agentic-benchmarks` repo.

---

## Authorship & license

Elvis Aguero (`elvis_alexander_aguero_vera@brown.edu`), Bessa Research Group,
Brown University. Inherits the host project's **BSD-3-Clause** license.

## Status

**Experimental.** Class-based `Agent` hierarchy, declarative `Graph`/`Edge`
topology, LangGraph orchestration, unified Claude/Ollama backends, Popperian
hypothesis enforcement, canonical evaluation ledger with mechanical eval
counting, and adversarial-critic-gated termination. Exercised end-to-end on the
benchmark studies with Haiku 4.5.
