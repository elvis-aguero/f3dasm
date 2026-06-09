---
id: replicate-reproduces-from-store
title: replicate.py must re-derive the headline from the canonical store
tags: [replicate, reproducibility, deliverable, critic, headline]
audience: [strategizer, implementer]
---
`replicate.py` must load the canonical ExperimentData store and **re-derive**
the headline number from ledgered rows — never hardcode it. The critic's
reproducibility gate checks exactly this: would a clean run reproduce the
headline from the store alone?

Practical notes that have bitten runs before:
- Use an absolute, self-locating path to the store, not a bare relative one:

  ```python
  import os
  here = os.path.dirname(os.path.abspath(__file__))
  store = os.path.join(here, "experiment_data")  # adjust to your layout
  ```

- The headline must be reachable from provenance-stamped rows (see
  [run ground-truth evaluations through get_evaluator()]). If the headline
  came from an off-ledger computation, the gate will reject it.
