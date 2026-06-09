---
id: evaluate-through-get-evaluator
title: Run ground-truth evaluations through get_evaluator()
tags: [evaluation, get_evaluator, canonical-store, provenance, ledger]
audience: [implementer, datagenerator]
---
Every ground-truth (true-oracle) evaluation must go through
`get_evaluator()` so it is written to the canonical ExperimentData store with
provenance — a `_delegation_id` stamp — and counts toward the run's evaluation
ledger. Numbers produced any other way cannot anchor a headline and are not
reproducible from the store.

```python
from f3dasm.agentic import get_evaluator
gen = get_evaluator()
data = data.run(data_generator=gen)
gen.flush()
```

Do NOT compute the objective in a private loop and write your own CSV/.h5, and
do NOT call `ExperimentData.store()` directly against the run store — both
produce rows without provenance (or no canonical rows at all). If you reported
evaluations but none landed in the canonical store, you bypassed
`get_evaluator()`; re-run them through it.
