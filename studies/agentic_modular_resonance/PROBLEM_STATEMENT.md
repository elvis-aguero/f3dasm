# Modular Resonance Search

I am studying a two-parameter integer optimisation problem. The question I want answered is: **what (k, m) pair in the domain below maximises `resonance(k, m)`, and what is the maximum value?**

## The function

For positive integers `k` and `m`, define the multiplicative order

```
ord(k, m)  =  the smallest positive integer r such that k^r ≡ 1  (mod m)
```

when `gcd(k, m) = 1`. When `gcd(k, m) > 1` the multiplicative order is undefined; treat such pairs as **infeasible** (assign a large negative sentinel).

The objective to maximise is

```
resonance(k, m)  =  ord(k, m) / ln(m)
```

over

```
k  ∈  {2, 3, …, 50}            (integer)
m  ∈  {1000, 1001, …, 100000}  (integer)
```

The feasible domain has approximately 4.85 million points. A brute-force evaluation of every point is not required.

## Implementation notes

`ord(k, m)` for `m` up to 10⁵ requires some care — a naive loop `r = 1; while pow(k, r, m) != 1: r += 1` is correct but can be slow when `r` is large. A faster approach: factor the Carmichael function `λ(m)` and test divisors of `λ(m)` in ascending order.

**You must use `f3dasm` for the search.** Specifically:
- Build a `Domain` with two `add_int` parameters (`k` and `m`) and one `add_output("resonance")`.
- Wrap the objective in a `DataGenerator` subclass.
- Sample candidate points using f3dasm samplers (`Latin`, `Sobol`, `Grid`, etc.) and evaluate them through the `DataGenerator`. Iterate and refine as needed.
- Use `ExperimentData` to store every evaluation and track the running best.

Everything you write — scripts, intermediate data — must live under `workspace/`. Files written elsewhere will not be included in the deliverable.

## Constraints

- No Python packages beyond `f3dasm`, `numpy`, and the standard library.
- Infeasible points (gcd > 1) get a sentinel; do not raise exceptions.

## What to report

- The best `(k, m, resonance)` triple observed during the search
- The search strategy used and why
- Whether the reported value is likely the global optimum or a local one, and the evidence for that assessment
