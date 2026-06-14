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
evaluate every row, collect the results*:

```python
from f3dasm import Pipeline, Step, CollectArrayResults, ExperimentData, DataGenerator, create_sampler
from f3dasm.design import Domain

def create_experimentdata(project_dir):                 # Step 1: the DoE / ledger
    domain = Domain()
    domain.add_float("x0", -5.0, 5.0); domain.add_float("x1", -5.0, 5.0)
    domain.add_output("f")                              # declare outputs up front
    data = ExperimentData(domain=domain)
    sampler = create_sampler("random", seed=0)
    sampler.arm(data=data)                              # ARM before CALL (two-phase API)
    data = sampler.call(data=data, n_samples=50)
    data.store(project_dir)

class Oracle(DataGenerator):                            # Step 2: the evaluator
    def execute(self, experiment_sample, **kw):
        x = list(experiment_sample.to_numpy()[0])
        experiment_sample.store("f", objective(x))
        return experiment_sample

Pipeline(name="solve", steps=[
    Step(name="create", block=create_experimentdata),
    Step(name="run",    block=Oracle(), parallel=True),
    Step(name="post",   block=CollectArrayResults()),
]).run(mode="local", project_job="run_001")
```

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
- **`DataGenerator` subclass** for evaluation: heavy state (loaded models,
  datasets) in `__init__`/`__post_init__`; per-row work in `execute(self, sample)`.
- **Stub a block** with a trivial body or fake return to draft a *working*
  skeleton first (the baseline-to-beat), then swap real blocks in.

## Domain / oracle / sampler
- `Domain()` + `add_float/add_int/add_category(name, ...)` for inputs;
  `add_output(name)` for every output column (custom objects:
  `add_output(name, to_disk=True, store_function=, load_function=)`).
- Wrap an existing fn as a generator: `datagenerator(output_names="y")(fn)`.
- Samplers (`"random"`, `"latin"`, `"sobol"`, `"grid"`): always `arm(data=)`
  then `call(data=, n_samples=)`.

## ExperimentData (the ledger)
`ExperimentData(domain=)`, `.from_file(project_dir)`, `.store(project_dir)`;
inside a generator `sample.get("name")` / `sample.store("name", obj)`; query via
`.to_pandas()` / `.select(...)` / `.select_with_status("finished")`. The headline
result is *derived* in a post block from the collected ED, not hardcoded.

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
- Don't hardcode the headline — derive it from the collected `ExperimentData`.
- Don't evaluate the true oracle outside `get_evaluator()` — see
  [[evaluate-through-get-evaluator]]; off-ledger numbers can't anchor a finding.
