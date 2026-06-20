# Black-Box Optimisation: 3-Dimensional Continuous Function

Find the point **x ∈ [−5, 5]³** that minimises f(x), and report its coordinates and objective value.

The function has multiple local minima. Their locations, values, and count are unknown.

---

## Domain

Three continuous inputs, each in [−5, 5]:

| Variable | Lower bound | Upper bound |
|----------|-------------|-------------|
| x1       | −5          | 5           |
| x2       | −5          | 5           |
| x3       | −5          | 5           |

---

## Evaluator

The evaluator is registered with f3dasm. Access it exclusively through `get_evaluator()`:

```python
from f3dasm.agentic import get_evaluator

gen = get_evaluator()          # resolves the registered oracle
gen.call(data, mode="sequential")  # evaluate and log to canonical store
gen.flush()
```

**Never** import from `workspace/evaluator.py` directly. Direct imports bypass the
canonical ledger — evaluations go unlogged and the reproduction gate fails (zero-new-evals
contract broken). `get_evaluator()` is the only oracle door.

The underlying function takes three continuous inputs (x1, x2, x3) each in [−5, 5]
and returns a scalar. All inputs must lie within the bounds above; behaviour outside
[−5, 5]³ is undefined. The function is **deterministic**.

---

## What to report

- The best point (x1, x2, x3) found and its f(x) value
- Total number of evaluations used
- The search strategy used and why
- Evidence the reported point is not merely a local minimum (e.g. multiple restarts, deliberate exploration of distant regions)

---

## Research context

This is a standard problem in numerical optimisation and experimental design.
You have a limited evaluation budget (1000 calls) and a multimodal landscape you
cannot see directly. The landscape is **deceptive**: several local minima sit
close in value to one another, and the best one is appreciably deeper than the
rest but occupies a small region. Uniform random sampling alone will not find it
— think about how to spend your budget wisely (coverage first, then refinement).
