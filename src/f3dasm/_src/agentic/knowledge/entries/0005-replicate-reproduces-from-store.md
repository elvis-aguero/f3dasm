---
id: pipeline-reproduces-from-store
title: pipeline.py is the deliverable — lazy, and it reproduces the headline from the store
tags: [pipeline, reproducibility, deliverable, critic, headline, lazy]
audience: [strategizer, implementer]
---
The single deliverable is `pipeline.py`: a human-readable f3dasm Pipeline of the
whole data-driven process that ALSO reproduces the headline. The runtime
EXECUTES it lazily after the critic gate, asserting the headline re-derives from
the canonical ledger with ZERO new oracle evaluations. Hand-authored idxmin
scripts are gone — running the pipeline IS the reproduction.

It must be LAZY:
- **Oracle:** load the canonical store (`ExperimentData.from_file`) and reach the
  oracle only via `get_evaluator()`. f3dasm skips already-`FINISHED` rows, so a
  fresh run does the full campaign while a re-run evaluates nothing (see
  [[evaluate-through-get-evaluator]]).
- **Heavy non-oracle blocks** (a fitted GP/NN/RF surrogate, costly analysis):
  f3dasm's row-laziness does NOT cover these — cache-or-load them yourself.
  Persist the artifact (`Domain.add_output(name, to_disk=True, store_function=,
  load_function=)`, or a plain "load if the file exists else fit+save" guard) so
  a re-run does not refit. See [[surrogates-are-off-ledger]].

And SELF-REPRODUCING:
- Derive the headline from ledgered rows in a final analysis step and `assert`
  it against the reported number — never hardcode the answer; deriving it IS the
  reproduction.
- Read the store from the `F3DASM_CANONICAL_STORE` env var when set (the runtime
  gate sets it), else a self-locating path — never a brittle cwd-relative guess
  (that is what broke past runs).
