# Black-Box Optimisation: 8-Dimensional Continuous Function

Find the point **x ∈ [−5, 5]⁸** that minimises f(x), and report its coordinates and objective value.

The function has multiple local minima. Their locations, values, and count are unknown.

---

## Domain

Eight continuous inputs, each in [−5, 5]:

| Variable | Lower bound | Upper bound |
|----------|-------------|-------------|
| x1       | −5          | 5           |
| x2       | −5          | 5           |
| x3       | −5          | 5           |
| x4       | −5          | 5           |
| x5       | −5          | 5           |
| x6       | −5          | 5           |
| x7       | −5          | 5           |
| x8       | −5          | 5           |

---

## Evaluator

A pre-compiled evaluator lives in `workspace/`:

```python
import sys
sys.path.insert(0, "workspace")
from evaluator import evaluate   # evaluate(x: list[float]) -> float
```

`evaluate(x)` takes a list or array of exactly 8 floats and returns a scalar. All inputs must lie within the bounds above; behaviour outside [−5, 5]⁸ is undefined. The function is **deterministic**.

---

## What to report

- The best point (x1, …, x8) found and its f(x) value
- Total number of evaluations used
- The search strategy used and why
- Evidence the reported point is not a local minimum (e.g. multiple restarts, deliberate exploration of distant regions)
