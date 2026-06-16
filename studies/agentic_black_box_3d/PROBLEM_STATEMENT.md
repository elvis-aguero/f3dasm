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

A pre-compiled evaluator lives in `workspace/`:

```python
import sys
from pathlib import Path
# study_dir is shown in your <workspace> preamble — use it to locate the evaluator
sys.path.insert(0, str(Path(study_dir) / "workspace"))
from evaluator import evaluate        # evaluate(x: list[float]) -> float
```

`evaluate(x)` takes a list or array of exactly 3 floats and returns a scalar. All inputs must lie within the bounds above; behaviour outside [−5, 5]³ is undefined. The function is **deterministic**.

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
