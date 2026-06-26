"""The BASELINE oracle (the 'cartesian' design space).

Wraps the fixed objective for the default Cartesian parametrization: the design
variables ARE the coordinates (x, y), each in [-1, 1]. This is the registered
oracle for the default namespace; the agent reaches it via get_evaluator().

A new design namespace (e.g. 'polar') is a DIFFERENT parametrization that the
datagenerator authors at run time — its variables map to (x, y) and it calls the
SAME objective.score, so its results compare to this baseline on one ruler.
"""
from __future__ import annotations

from objective import score


def evaluate_kw(**kwargs) -> float:
    """Cartesian baseline: design variables x, y in [-1, 1] -> objective."""
    x = float(kwargs["x"])
    y = float(kwargs["y"])
    return score(x, y)
