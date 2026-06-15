---
id: pipeline-building-patterns
title: Building an f3dasm Pipeline (patterns from real studies)
tags: [pipeline, step, block, loop, datagenerator, sampler, experimentdata, domain, deliverable]
audience: [strategizer, implementer]
---
How to compose a data-driven solution as an f3dasm `Pipeline` — distilled from
the bessagroup `l2co_experiments` study suite. (Those run on SLURM via Hydra;
our agentic runs are **local** — ignore the cluster/Hydra machinery and keep the
composition idioms below. Build the pipeline in code, not YAML.)

## Canonical skeleton: create → run → collect
A pipeline is an ordered list of `Step`s; the recurring shape is *build the DoE,
evaluate every row through the oracle, collect the results*. The **run step is
`get_evaluator()`** — the one door to the registered ground-truth oracle. You do
NOT write your own `Oracle(DataGenerator)`; the datagenerator agent already built
and registered it, and `get_evaluator()` returns it as a ready Step block that
meters every call into the run's canonical store with provenance.

```python
from f3dasm import Pipeline, Step, CollectArrayResults, ExperimentData, create_sampler
from f3dasm.agentic import get_evaluator
from f3dasm.design import Domain

def create_experimentdata(project_dir):                 # Step 1: the DoE
    domain = Domain()
    domain.add_float("x0", -5.0, 5.0); domain.add_float("x1", -5.0, 5.0)
    domain.add_output("f")                              # declare outputs up front
    data = ExperimentData(domain=domain)
    sampler = create_sampler("random", seed=0)
    sampler.arm(data=data)                              # ARM before CALL (two-phase API)
    data = sampler.call(data=data, n_samples=50)
    data.store(project_dir)

Pipeline(name="solve", steps=[
    Step(name="create", block=create_experimentdata),
    Step(name="run",    block=get_evaluator(), parallel=True),  # ← the ONE oracle door
    Step(name="post",   block=CollectArrayResults()),
]).run(mode="local", project_job="run_001")
```

`get_evaluator()` must be called from inside your delegation workspace (it reads
the run config to resolve the registered source and the canonical store). Every
row it evaluates is ledgered automatically — derive your headline from that
canonical store (what the runtime reproduces from), never from hardcoded
numbers or a step's local scratch copy.

## The composition API (the actual idioms)
- **`Step(block=, name=, parallel=, kwargs=)`** wraps a callable, a `Block`, or a
  `DataGenerator`. A plain callable receives `project_dir` auto-injected.
- **`block_a >> block_b`** chains blocks into ONE step (run sequentially, shared
  state) — use for post-processing that builds on the previous block.
- **`Loop(n_iterations=N, steps=[...])`** repeats inner steps; carry state across
  iterations on the on-disk `ExperimentData` (`data.mark_all("open")` at the end
  of a round so the next iteration recomputes) — the active-learning / BO-rounds
  shape.
- **`Block` subclass** for ED→ED post-processing: `def call(self, data, **kw) -> ExperimentData`.
- **`DataGenerator` subclass** ONLY for your OWN off-ledger predictors (a fitted
  surrogate, a stub): heavy state in `__init__`, per-row work in `execute`. The
  ground-truth oracle is NEVER hand-written here — it is always `get_evaluator()`.
- **Stub a block** with a trivial body or fake return to draft a *working*
  skeleton first (the baseline-to-beat), then swap real blocks in. Swapping a
  block (a different sampler, surrogate, or optimizer) is how you test a
  hypothesis; the oracle run step stays `get_evaluator()` throughout.

## Domain / oracle / sampler
- `Domain()` + `add_float/add_int/add_category(name, ...)` for inputs;
  `add_output(name)` for every output column (custom objects:
  `add_output(name, to_disk=True, store_function=, load_function=)`).
- Wrap an existing fn as a generator: `datagenerator(output_names="y")(fn)`.
- Samplers (`"random"`, `"latin"`, `"sobol"`, `"grid"`): always `arm(data=)`
  then `call(data=, n_samples=)`.

## ExperimentData (the ledger)
The **one canonical ledger** is written by `get_evaluator()` — every true-oracle
row lands there, stamped with a `_delegation_id`, across all delegations in the
run. That store is the single source of truth for the eval count and the
headline. Read it with `ExperimentData.from_file(project_dir)` and query via
`.to_pandas()` / `.select(...)` / `.select_with_status("finished")`; the headline
is *derived* from those rows, never hardcoded. You may `.store(...)` your own
scratch ExperimentData (a surrogate's predictions, intermediate DoEs) wherever
you like — just don't confuse it with the canonical ledger.

## Pipeline archetypes ("use when")
1. **Sampling sweep** (create→evaluate→collect): a labelled dataset over a space.
2. **Surrogate-fit sweep** (grid over model configs → fit → score): compare surrogates.
3. **Iterative `Loop`** (create once, then loop run+update): the design depends on
   prior rounds — BO, active learning, on-policy RL.
4. **Independent-per-row** (one fit per row, parallel) + a joined evaluation.
5. **Baseline + method** spliced together: ship a method and its apples-to-apples
   baseline in one pipeline.
6. **Resume tail-only**: re-derive results from an existing run without recomputing.

## Anti-patterns the examples warn against
- Don't put heavy loads in `execute` — they run per-row; load once in `__init__`.
- Don't forget to `arm()` before `call()` — it's a required two-phase API.
- Don't carry `Loop` state in memory — persist to the ED, `mark_all("open")`.
- Don't hardcode the headline — derive it from the canonical store.
- The oracle run step is ALWAYS `get_evaluator()` — never a hand-written
  `DataGenerator`, never the raw evaluator, never a redirected store. That one
  door is what makes the result ledgered and reproducible; see
  [[evaluate-through-get-evaluator]] and [[surrogates-are-off-ledger]].
