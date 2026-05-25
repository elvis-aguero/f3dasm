# Supercompressible Metamaterial Design (3D)

## Background

A mechanical metamaterial building block is being designed to achieve
*supercompressibility*: compressive strains exceeding 90% that are fully
recoverable — the structure springs back to its original shape once the load is
removed. This sounds paradoxical when the base material is polylactic acid
(PLA), a brittle polymer that fractures at roughly 4% local strain. The key is
geometry: for the right proportions, axial compression drives a global elastic
buckling mode — the longerons coil helically around the central axis — that
produces enormous macroscopic deformation while keeping local material strains
below the fracture limit.

The concept combines two structural principles: the deployable mast (a
helically-wound slender structure developed for space applications) and the
thin-walled conical frustum (common in impact-absorption applications).
Together they form a unit cell consisting of a bottom ring (the base, larger),
a top ring (the apex, smaller), and a set of three slender vertical elements
called *longerons* connecting the two rings and arranged helically. The number
of longerons is not a design variable: by the principle of superposition,
adding more longerons scales strength proportionally without changing the
buckling mode, so only the geometry of the unit cell with three longerons is
studied.

When compressed axially, a given geometry will respond in one of three ways:

- **Non-coilable** — the structure deforms in a global bending mode or by
  local buckling of individual longerons. Supercompressibility is not
  achieved; the longerons do not wind around the axis.
- **Coilable and reversible** — the longerons wind helically around the
  central axis in a smooth, symmetric global buckling pattern. Maximum local
  strain throughout the full compression cycle stays below the elastic limit;
  the structure recovers completely on unloading. This is the only acceptable
  outcome.
- **Coilable but yielding** — the coiling mode activates, but local strains
  in the longerons exceed the elastic limit before full compression is reached.
  The material undergoes permanent plastic deformation and the structure does
  not recover.

Predicting which regime a given design falls into requires full nonlinear
finite-element analysis. No closed-form criterion separates the three modes.

---

## From seven parameters to three

The general unit-cell geometry is described by seven parameters: bottom ring
diameter *D*₁, top ring diameter *D*₂, mast height *P*, and four cross-section
parameters of the longerons — area *A*, moments of inertia *Iₓ* and *Iᵧ*, and
torsional constant *Jₜ*. The material adds two elastic constants: Young's
modulus *E* and shear modulus *G*. By dimensional analysis (scaling geometry by
*D*₁ and elastic moduli by *E*), these nine parameters reduce to seven
independent non-dimensional ratios.

For this 3D study, two simplifications are applied:

1. **Circular cross-sections.** Constraining the longeron cross-section to be
   circular with diameter *d* collapses *A*, *Iₓ*, *Iᵧ*, and *Jₜ* into a
   single variable. For a circle: *Iₓ* = *Iᵧ* = π*d*⁴/64, *Jₜ* = π*d*⁴/32 =
   2*Iₓ*, and *A* = π*d*²/4 — all uniquely determined by *d*. A global
   sensitivity analysis on a companion 50 000-point dataset (7D version, not
   available here) further confirms that *A* and *G*/*E* contribute negligibly
   to both outputs relative to the moment-of-inertia terms.

2. **Fixed material.** PLA is chosen as the base material, fixing *E* = 3 500 MPa
   and *G*/*E* = 0.3677 and removing them from the design space.

After these simplifications, three non-dimensional parameters fully define the
geometry:

| Parameter            | Definition                          | Physical meaning                                |
|----------------------|-------------------------------------|-------------------------------------------------|
| `ratio_d`            | *d* / *D*₁                          | Slenderness of the longerons                    |
| `ratio_pitch`        | *P* / *D*₁                          | Aspect ratio of the mast (tall vs. squat)       |
| `ratio_top_diameter` | (*D*₁ − *D*₂) / *D*₁               | Degree of taper (0 = cylindrical, 1 = pointed) |

---

## Design space

The bottom ring diameter *D*₁ = 100 mm is fixed as the physical reference for
all non-dimensional ratios. This is a simulation convention, not a physical
constraint — the underlying mechanics is scale-invariant, so any physical
design can be obtained by uniform scaling. The top ring diameter follows from
*D*₂ = *D*₁ (1 − `ratio_top_diameter`), ranging from a cylinder (`ratio_top_diameter`
= 0, *D*₂ = *D*₁) to a strongly tapered cone (`ratio_top_diameter` = 0.8,
*D*₂ = 0.2 *D*₁).

| Parameter            | Lower bound | Upper bound |
|----------------------|-------------|-------------|
| `ratio_d`            | 0.004       | 0.073       |
| `ratio_pitch`        | 0.25        | 1.50        |
| `ratio_top_diameter` | 0.0         | 0.80        |

Fixed simulation constants (not design variables):

| Constant              | Value     | Meaning                                      |
|-----------------------|-----------|----------------------------------------------|
| `young_modulus`       | 3 500 MPa | PLA Young's modulus                          |
| `ratio_shear_modulus` | 0.3677    | *G*/*E* for PLA                              |
| `n_longerons`         | 3         | Number of longerons in the unit cell         |
| `bottom_diameter`     | 100 mm    | Reference length *D*₁ for non-dimensionalisation |

---

## Outputs

Each design was evaluated by two sequential finite-element analyses: a linear
eigenvalue buckling analysis to classify the coiling mode, followed by a
nonlinear Riks (arc-length) analysis that traces the complete post-buckling
response. Three quantities are recorded:

**`coilable`** — integer classification:
- `0` — not coilable (the critical buckling mode is global bending or local longeron buckling, not the helical coiling mode)
- `1` — coilable and reversible (the coiling mode is critical and maximum local strain stays below the elastic limit throughout full compression)
- `2` — coilable but yielding (the coiling mode activates but local strains exceed the elastic limit; the structure does not recover)

Of the 1 000 designs in the dataset: 318 are class 0, 214 are class 1, and
468 are class 2. The target class (`coilable == 1`) therefore represents
roughly 21% of the dataset.

**`sigma_crit`** — critical buckling stress (kPa). The effective compressive
stress at which the coiling instability first initiates. Deterministic,
extracted directly from the linear buckling analysis. Set to `NaN` for
non-coilable designs (`coilable == 0`), because the coiling mode either does
not appear or is not the critical mode.

**`energy`** — elastic energy absorption (kJ/m³). Area under the complete
compressive stress–strain curve from zero to full compression, computed from
the Riks output by piecewise cubic Hermite interpolation and Simpson's rule
integration. This quantity is stochastic: geometric imperfections are sampled
from a lognormal distribution in each Riks simulation, introducing variability.
Each design is evaluated once, so the recorded value is a single noisy sample.
Set to `NaN` for non-coilable designs and for simulations that failed to
converge. Of the 214 class-1 designs, only 99 have a valid `energy` value.

---

## Objective

Find the design with the **highest `sigma_crit`** subject to **`coilable == 1`**.
Designs with `coilable ≠ 1` are mechanically unusable regardless of their
other output values.

`energy` is a secondary objective: among designs satisfying `coilable == 1`,
higher energy absorption is preferable, but only after the primary criterion
has been met.

---

## Available data

No new finite-element simulations can be run. The dataset of 1 000 pre-computed
designs is the only source of ground-truth information about the design space.

The dataset was sampled using a Sobol sequence — a low-discrepancy quasi-random
method that fills the parameter space more uniformly than pseudo-random
sampling. Coverage is roughly uniform across the domain; no subregion has been
preferentially sampled.

```
experiment_data/experiment_data/
  input.csv    — 1 000 rows × 3 columns (ratio_d, ratio_pitch, ratio_top_diameter)
  output.csv   — 1 000 rows × 3 columns (coilable, sigma_crit, energy)
  domain.json  — f3dasm Domain definition (parameter bounds)
  jobs.csv     — job status metadata
```

Note: the nested `experiment_data/experiment_data/` path is an artefact of how
f3dasm stores data relative to the study directory.

---

## Success criterion

Report the best design found — its parameter values, its `sigma_crit`, its
`coilable` classification, and its `energy`. If multiple high-performing
designs are identified, include them for trade-off analysis.
