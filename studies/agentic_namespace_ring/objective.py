"""The FIXED objective — the shared ruler for this study.

This is the ONE thing every design parametrization is measured against. Do NOT
modify it: a "better design" only means something if it is scored by the same
objective as the baseline (comparable-by-construction). A new parametrization
must ultimately call ``score(x, y)`` — it may change HOW (x, y) are produced
(the design variables), never WHAT is measured.

The objective is defined on the closed unit disk x**2 + y**2 <= 1. Higher is
better; the maximum is 1.0. The landscape is concentrated: almost all of the
disk scores near zero, and the high-scoring region is a thin set that uniform
(x, y) sampling rarely lands on. Finding it efficiently is the point.
"""
from __future__ import annotations

import math


def score(x: float, y: float) -> float:
    """Score a point on the unit disk. Higher is better; max 1.0.

    Concentrated on a thin circular ridge at a preferred radius. Angle does not
    matter — only the distance from the origin. Points outside the unit disk are
    clamped to score 0.0 (infeasible).
    """
    if x * x + y * y > 1.0 + 1e-9:
        return 0.0
    r = math.hypot(x, y)
    return math.exp(-((r - 0.75) ** 2) / 0.0002)
