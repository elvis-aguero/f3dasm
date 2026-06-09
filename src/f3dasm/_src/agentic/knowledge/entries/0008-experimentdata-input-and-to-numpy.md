---
id: experimentdata-gotchas
title: ExperimentData gotchas — input shaping and to_numpy()
tags: [experimentdata, to_numpy, domain, gotcha, idiom]
audience: [implementer]
---
The exact, CI-verified call forms (sampling, `to_numpy`, candidate
construction, best-N) live in the core f3dasm idioms injected into your
prompt — use those verbatim. This entry only records the *why* behind the
gotchas they encode, so you trust them:

- **Domain variable names are your contract.** Build the `Domain` with the
  parameter names the study uses (e.g. `x1..x8`, or `k`/`m`) and keep input
  dicts consistent with them. Mixing `x0..x7` with a domain declared `x1..x8`
  silently misaligns columns.

- **`to_numpy()` returns a tuple `(X, y)` — it takes no argument.** Not
  `to_numpy("input")`. The output array can carry metadata columns; if in
  doubt about the objective column, go through pandas (`_, df_out =
  data.to_pandas(); y = df_out["<output_name>"].to_numpy()`).

- **There is no `ExperimentData.from_numpy` and no `data.sample(...)`.** To
  wrap proposed points you build `ExperimentSample(_input_data={...})` (the
  kwarg is `_input_data`, not `input`) and pass them to
  `ExperimentData.from_data`; to sample you use `create_sampler(...).call(...)`.
  Both are shown in the injected core idioms.
