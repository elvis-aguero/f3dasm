# Minimise a 2-D quadratic (contract example)

This is the canonical minimal study folder for the agentic-f3dasm **study-folder
contract** (see `../authoring-a-study.md`). It is intentionally trivial — its job
is to be a *runnable, tested* reference, not a hard problem.

## Objective
Minimise `y = (x1 - 1)^2 + (x2 + 2)^2` over the design space below. The global
minimum is `y = 0` at `(x1, x2) = (1, -2)`.

## Success criteria
Report the argmin `(x1*, x2*)` and the **ledgered** `y*`, and ship a
`replicate.py` that loads the canonical store and asserts `y*` from ledgered
rows (not a hardcoded number).

## Design space
| variable | type | bounds | units |
|---|---|---|---|
| `x1` | continuous | [-5, 5] | dimensionless |
| `x2` | continuous | [-5, 5] | dimensionless |

## Oracle
A shipped callable: `workspace/evaluator.py:evaluate` (one kwarg per input,
returns `y`). Declared in `config.yaml` under `evaluator.entrypoint`.

## Deliverables
- `solution.md` — the argmin, the ledgered `y*`, and a one-line conclusion.
- `replicate.py` — re-derives `y*` from the canonical store and asserts it.
