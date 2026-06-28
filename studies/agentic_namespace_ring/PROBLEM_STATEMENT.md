# Find the high-scoring region of a concentrated 2-D landscape

We have a fixed objective `score(x, y)` defined over the closed unit disk
(`x² + y² ≤ 1`). Higher is better; the best attainable value is `1.0`. The catch:
the landscape is **concentrated** — almost the entire disk scores near zero, and
the high-scoring region is a thin sliver. Sample `(x, y)` uniformly and you burn
your whole budget in the dead zone.

The budget is small and the problem is fast on purpose. This is not a
compute problem — it is a question of how cleverly you go after that sliver.

## The baseline

The default design space searches the coordinates directly — `x ∈ [−1, 1]`,
`y ∈ [−1, 1]` — and you reach the (one, fixed) objective through the metered
oracle:

```python
from f3dasm.agentic import get_evaluator
gen = get_evaluator()              # the baseline oracle
gen.call(data, mode="sequential")  # evaluate + log to the canonical ledger
```

A uniform / space-filling sweep of that box is the number to beat under the same
budget. You will see most of it land where the score is ~0.

## What I'm asking

Beat the baseline. Spend the same budget, find a better-scoring design, and —
more importantly — tell me **why** your approach worked. Here the way you set up
the search will buy you more than the choice of optimizer, so think about that
first.

If it helps, the framework lets you stand up alternative design spaces of your
own and search those instead of the default box (the handbook has the mechanics).
Use that freedom if you see a reason to; don't if you don't.

Two non-negotiables, the way they'd be in any honest study:
- **One ruler.** Whatever you search, it must be scored by the same
  `objective.score`, so "this beats that" is a real comparison. If you ever
  change what is measured, say so and justify why the comparison still stands.
- **Honest result.** Report the best design and score for each thing you tried,
  the budget each used, and the mechanism behind the winner. If an idea did not
  beat the baseline, say so plainly — a negative result that's understood is
  worth recording.
