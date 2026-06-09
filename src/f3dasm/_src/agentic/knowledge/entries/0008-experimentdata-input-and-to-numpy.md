---
id: experimentdata-gotchas
title: ExperimentData gotchas — input shaping and to_numpy()
tags: [experimentdata, to_numpy, domain, gotcha, idiom]
audience: [implementer]
---
A couple of f3dasm idioms that have tripped workers up:

- **Domain variable names are your contract.** Build the `Domain` with the
  parameter names the study uses (e.g. `x1..x8`, or `k`/`m`) and keep input
  dicts consistent with them. Mixing `x0..x7` with a domain declared `x1..x8`
  silently misaligns columns.

- **`to_numpy()` returns a tuple, with metadata in the output array.**
  `ExperimentData.to_numpy()` returns `(input_array, output_array)`, and the
  output array can carry provenance/metadata columns alongside the objective.
  Extract the objective column explicitly rather than assuming a bare 1-D
  array. When in doubt, go through pandas:

  ```python
  _, df_out = data.to_pandas()
  y = df_out["f"].to_numpy()   # use the study's actual output column name
  ```
