# Find the high-scoring region of a concentrated 2-D landscape

Maximise a fixed objective `score(x, y)` over the closed unit disk
(`x² + y² ≤ 1`). Higher is better; the maximum value is `1.0`. The landscape is
**concentrated**: almost the whole disk scores near zero, and the high-scoring
region is a thin set that uniform `(x, y)` sampling rarely lands on.

This problem is deliberately small and fast — it is a test of *how you frame the
search*, not of compute.

---

## The baseline design space — `cartesian`

The default (registered) design space parametrises a point by its coordinates
directly:

| Variable | Lower | Upper |
|----------|-------|-------|
| `x`      | −1    | 1     |
| `y`      | −1    | 1     |

Reach the oracle through `get_evaluator()` (never import the oracle directly):

```python
from f3dasm.agentic import get_evaluator
gen = get_evaluator()              # the 'cartesian' baseline oracle
gen.call(data, mode="sequential")  # evaluate + log to the canonical ledger
```

A uniform/LHS sweep of this box is the **baseline to beat**. You will find it
spends most of its budget in the near-zero region.

---

## You may invent a better *parametrization* — the key idea

The deeper lever here is not the optimiser — it is **how the design space is
represented**. The objective `score(x, y)` is FIXED (it is the one ruler), but
*the variables you search over are yours to choose*. A representation that
concentrates its samples where the score lives will beat the Cartesian box with
the same budget.

When a different representation is the question, open it as a **new design
namespace**: `Delegate(target="datagenerator", namespace="your_name", …)` to
build + register its oracle, then `Delegate(target="implementer",
namespace="your_name", …)` to search it. Each namespace keeps its own ledger;
the baseline is untouched.

**Worked example you are encouraged to try — `polar`.** Parametrise the same
point by a radius and an angle instead of coordinates:

```python
# the 'polar' oracle the datagenerator authors — it calls the SAME objective,
# so its scores are directly comparable to the cartesian baseline:
from objective import score
import math
def evaluate_kw(**kw):
    r, theta = float(kw["r"]), float(kw["theta"])     # r in [0,1], theta in [0, 2*pi]
    return score(r * math.cos(theta), r * math.sin(theta))
```

This is not the only possible reparametrization — propose your own if you see a
better one. The discipline: a new namespace's oracle must call `objective.score`
so a "this design wins" claim is measured on the same ruler. If you ever change
what is measured, say so and justify why the comparison still holds.

---

## What to report

- The best point found and its `score`, **per namespace** you explored.
- Which parametrization searched the landscape most efficiently for the budget,
  and **why** (the mechanism — not just the number).
- Total evaluations used (per namespace) and the search strategy.
- Honest epistemic status: if a parametrization did *not* beat the baseline, say
  so — the idea can still be worth recording.
