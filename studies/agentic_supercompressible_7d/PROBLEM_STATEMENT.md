# Supercompressible Metamaterial Design (7D)

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

When compressed axially, a given geometry will respond in one of two ways:

- **Non-coilable** — the structure deforms in a global bending mode or by
  local buckling of individual longerons. Supercompressibility is not
  achieved; the longerons do not wind around the axis.
- **Coilable** — the longerons wind helically around the central axis in a
  smooth, symmetric global buckling pattern. This is the only acceptable
  outcome.

Predicting which regime a given design falls into requires full nonlinear
finite-element analysis. No closed-form criterion separates the two modes.

---

## The seven parameters

The unit-cell geometry is described by seven non-dimensional parameters obtained
by scaling all lengths by the bottom ring diameter *D*₁ and all moduli by the
Young's modulus *E*. The longeron cross-section is unrestricted: area *A*, two
independent area moments of inertia *Iₓ* and *Iᵧ*, and polar moment *Jₜ* are
all free parameters. The cross-section need not be symmetric (*Iₓ* ≠ *Iᵧ* is
allowed) and is not constrained to any particular shape. The material shear
modulus ratio *G*/*E* is also free, spanning the plausible range for
polymer-like materials.

| Parameter             | Definition                          | Physical meaning                                           |
|-----------------------|-------------------------------------|------------------------------------------------------------|
| `ratio_area`          | *A* / *D*₁²                         | Cross-sectional area of a longeron relative to *D*₁²       |
| `ratio_Ixx`           | *Iₓ* / *D*₁⁴                        | Second moment of area about the x-axis, relative to *D*₁⁴ |
| `ratio_Iyy`           | *Iᵧ* / *D*₁⁴                        | Second moment of area about the y-axis, relative to *D*₁⁴ |
| `ratio_J`             | *Jₜ* / *D*₁⁴                        | Polar moment of area, relative to *D*₁⁴                   |
| `ratio_pitch`         | *P* / *D*₁                          | Aspect ratio of the mast (tall vs. squat)                  |
| `ratio_top_diameter`  | (*D*₁ − *D*₂) / *D*₁               | Degree of taper (0 = cylindrical, 1 = pointed)             |
| `ratio_shear_modulus` | *G* / *E*                           | Shear modulus relative to Young's modulus                  |

---

## Design space

The bottom ring diameter *D*₁ = 100 mm is fixed as the physical reference for
all non-dimensional ratios. This is a simulation convention, not a physical
constraint — the underlying mechanics is scale-invariant, so any physical
design can be obtained by uniform scaling. The top ring diameter follows from
*D*₂ = *D*₁ (1 − `ratio_top_diameter`).

| Parameter             | Lower bound      | Upper bound      |
|-----------------------|------------------|------------------|
| `ratio_area`          | 1.17 × 10⁻⁵      | 4.10 × 10⁻³      |
| `ratio_Ixx`           | 1.128 × 10⁻¹¹    | 1.40 × 10⁻⁶      |
| `ratio_Iyy`           | 1.128 × 10⁻¹¹    | 1.40 × 10⁻⁶      |
| `ratio_J`             | 1.353 × 10⁻¹¹    | 7.77 × 10⁻⁶      |
| `ratio_pitch`         | 0.25             | 1.50             |
| `ratio_top_diameter`  | 0.0              | 0.80             |
| `ratio_shear_modulus` | 0.035            | 0.45             |

Fixed simulation constants (not design variables):

| Constant           | Value     | Meaning                                           |
|--------------------|-----------|---------------------------------------------------|
| `young_modulus`    | 3 500 MPa | PLA Young's modulus                               |
| `n_longerons`      | 3         | Number of longerons in the unit cell              |
| `bottom_diameter`  | 100 mm    | Reference length *D*₁ for non-dimensionalisation  |

---

## Outputs

Each design was evaluated by two sequential finite-element analyses: a linear
eigenvalue buckling analysis to classify the coiling mode, followed by a
nonlinear Riks (arc-length) analysis that traces the complete post-buckling
response. Three quantities are recorded:

**`coilable`** — integer classification:
- `0` — not coilable (the critical buckling mode is global bending or local longeron buckling, not the helical coiling mode)
- `1` — coilable (the helical coiling mode is the critical buckling mode)

The classification is based purely on the eigenvalue buckling analysis: a design
is coilable if and only if the coiling instability is the lowest buckling mode.
No additional yield or fracture criterion is applied. Of the 50 000 designs in
the dataset: 17 669 are class 0 and 32 331 are class 1. The target class
(`coilable == 1`) therefore represents roughly 65% of the dataset.

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
converge. Of the 32 331 class-1 designs, roughly 62% have a valid `energy`
value.

---

## Objective

Identify the design in the **continuous** seven-dimensional parameter space
that achieves the **highest `sigma_crit`** while remaining **`coilable == 1`**.
The 50 000-point dataset is a training resource — your deliverable is a specific
parameter vector anywhere in the bounded domain, not necessarily a row that was
already simulated. The question is whether we can design a metamaterial that
exceeds the best compressibility strength observed so far.

`energy` is a secondary objective: among feasible designs, higher energy
absorption is preferable, but only after maximising `sigma_crit`.

---

## Available data

No new finite-element simulations can be run. The 50 000 pre-computed designs
are the only ground truth available. They were sampled via Sobol sequence —
coverage is roughly uniform across the domain with no intentionally dense
regions.

```
experiment_data/experiment_data/
  input.csv    — 50 000 rows × 7 columns (ratio_area, ratio_Ixx, ratio_Iyy,
                 ratio_J, ratio_pitch, ratio_top_diameter, ratio_shear_modulus)
  output.csv   — 50 000 rows × 3 columns (coilable, sigma_crit, energy)
  domain.json  — f3dasm Domain definition (parameter bounds)
  jobs.csv     — job status metadata
```

Note: the nested `experiment_data/experiment_data/` path is an artefact of how
f3dasm stores data relative to the study directory.

---

## More context

**Is the coilable==1 region a simple shape?**
No. The boundary is set by nonlinear buckling mechanics, is not analytically
known. The relatively high class-1 fraction (65%) means much of the domain is
coilable, but the high-sigma_crit frontier is a narrow subset of that region.

**What physically drives sigma_crit?**
Euler buckling theory links critical load to the bending stiffness of the
longerons and the effective length of the mast. The cross-section parameters
and the pitch both enter this relationship, but the interactions in the
post-buckling regime are nonlinear and not captured by simple beam theory.
Different parameters influence sigma_crit and energy absorption differently —
understanding which ones matter most, and how, is part of the challenge.

**Can I propose a design not in the dataset?**
Yes — that is the goal. Use the data to learn about the space; the answer
should be the best design you can justify in the continuous domain.

**How do I know if a proposed design is coilable==1 if I can't simulate it?**
You cannot run new simulations, so any prediction carries uncertainty. Be
honest about what you don't know.

---

## Success criterion

Report a specific design (parameter values) predicted to be `coilable == 1`
with the highest achievable `sigma_crit`. Include the predicted `sigma_crit`,
why you believe the design is feasible.

You have freedom to experiment innovative solutions within your time constraints. We strive for state-of-the-art supercompressible designs.
