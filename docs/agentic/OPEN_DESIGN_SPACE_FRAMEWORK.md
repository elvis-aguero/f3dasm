# Open Design-Space Discovery — Scientific Framework Spec

> Branch: `exp/open-design-space` (epistemic change, isolated from `dev/confer`).
> This is a **framework spec**, not an implementation plan. The implementation
> plan (TDD tasks) comes *after* the decisive 2D experiment validates the
> direction. Nothing here is proven — it is a hypothesis about how to reframe
> the task, to be tested cheaply before any code is written.
>
> Status as of 2026-06-26: spec approved; awaiting the 2D experiment (step 2).
> Grounded in three codebase audits run 2026-06-26 (epistemic-contract,
> block-vs-nudge, abstraction-mutability).

## Context

**Why.** PI feedback: prescribing a fixed 14D box and asking the agent to beat a
baseline by 15% wastes the one thing an LLM uniquely brings. BO already solves a
14D box. The frontier is *design representation* — inventing new low-dimensional
parametrizations (e.g. top/bottom rings as ellipses with a parametrized phase
offset), each motivated by literature / physics / memory, each studied for
breakthrough potential. The reframe: **"here is the 7D baseline; propose new,
low-dimensional-ish designs and study each for signs of a breakthrough."**

**Why a spec first.** This changes the epistemic contract (CLAUDE.md §4, user-owned),
so it earns its own branch and a written framework before code. The three axes the
user named — epistemics, ruleset, abstraction levels — were audited against the
current codebase.

**Intended outcome.** A scientific framework where a single self-directed agent can
invent and study *new* design spaces over a run, while (a) single-design problems
stay byte-for-byte as they are today, and (b) the provenance floor that makes results
checkable stays intact.

## Governing principle (the user's steer)

**Additive with a default that equals current behavior.** A one-design study must use
zero new concepts and zero new tools — it *is* today's system. Every mechanism below
activates only when the agent opens more than one design space. The simplest, most
general framing wins; no apparatus is imposed on problems that don't need it.

---

## Axis 1 — Epistemics: a claim is scoped to its design namespace

This is a *clarification* of the existing Charter, not new machinery. (User's call:
the per-namespace idea is the one closest to blessing; the "new claim primitive" and
"portfolio layer" options were rejected as overfit / expressivity-hindering.)

- **The Charter, the 4 statuses (OPEN/SUPPORTED/FALSIFIED/INCONCLUSIVE), and the
  verdict validator are untouched.** §2/§3 severity already means "the search had power
  over *the space the claim ranges over*" (`charter.py:62-68`). That space is now
  explicitly **the namespace**, not "the problem." With one namespace this reading is
  identical to today.
- **A new parametrization = a new namespace.** Proposing one is exploration; the
  results *within* it are ordinary hypotheses, tested by the existing machinery, scoped
  to that namespace. No new status, no new lifecycle.
- **Cross-design claims are ordinary hypotheses too** — and the *existing* critic
  criterion 4 (over-generalization, `critic.py:64-74`) already governs them: you may not
  assert across namespaces what you tested in one. The synthesis is **emergent only when
  >1 namespace exists**; it is not a mandatory artifact.
- **The one genuinely new obligation:** a cross-namespace comparison must rest on a
  comparable metric. Axis 3 makes that structural, not policed.

Net epistemic change: near-zero new contract. The space the Charter already references
becomes a first-class, multipliable thing.

## Axis 2 — Ruleset: per-block decision, keep-or-nudge

The block-vs-nudge audit established the floor that **stays HARD** (provenance +
host-safety): schema validation, graph connectivity, nbformat validity, the
per-delegation memory cap, reproduction-must-run, and closing-a-hypothesis-requires-
evidence (Popperian provenance). All unchanged — these have a strong reason.

For each remaining creativity-constraining block, decide **per block**: keep it
blocking *only if there is a strong reason*, otherwise convert to a **two-shot confirm
nudge** (the existing pattern in `CancelDelegation` `routing.py:1307` and `Done()`
`routing.py:1479`): first call emits a message and refuses; an identical second call
proceeds. **The message regenerates per feature and per time-sensitivity** — a
time-sensitive feature (e.g. near a budget/watchdog limit) gets a terse, urgent
second-shot message and a short reconsider window; a non-time-sensitive one (e.g.
deliverable shape) gets a fuller explanatory message.

| Block | Site | Strong reason to keep blocking? | Decision |
|---|---|---|---|
| max-3-OPEN-hypotheses | `hypothesis_ledger.py:205` | No — it was closure *discipline*, not safety; exploring several designs needs >3 open | **Two-shot nudge.** Low time-sensitivity → full message ("you have N open; usual ceiling is 3 — re-call to confirm you want to track more") |
| process milestones (`oracle-ready`, …) | `routing.py:383`, `1449` | Partial — they ensure literature/oracle gates happen, but a nudge keeps the prompt while allowing skip | **Two-shot nudge, recurring per namespace.** Medium time-sensitivity |
| 5-pillar deliverable structure | `AddPipelineCell` phase enum, `routing.py:2409` | No — structure was convenience; a new-design notebook may not fit doe/data_generation/ml/optimization/analysis | **Two-shot nudge.** Low time-sensitivity → full message |

Each conversion is re-checked against this test at implementation time; any block found
to have a strong (provenance/host-safety) reason stays hard. Parsimony check
(CLAUDE.md §2): "let the cooperating agent proceed after confirming intent" is a general
principle, not a patch for one observed failure — a philosopher nods.

## Axis 3 — Abstraction: the namespace primitive (comparable-by-construction)

The mutability audit's headline: **"one space, one oracle" is a config choice, not an
abstraction limit.** Minimal, additive touch-points (default namespace = today; old
runs unaffected):

- `Delegation` gains optional `namespace: str | None = None` (`graph_state.py:64`).
  `None` → default → current path.
- `run_config`: single `evaluator_entrypoint` → `oracles: {default: {...}, <ns>: {...}}`,
  with a back-compat read of the old field (`agent_runtime.py:119`, `137-203`).
- `get_evaluator(namespace=None)` resolves the oracle for the namespace
  (`instrumented.py:532`).
- Per-namespace store dir + its own `.f3dasm_protected` sentinel (`instrumented.py:98`,
  `experimentdata.py:911`). **Provenance preserved**: rows stay `_delegation_id`-stamped;
  the delegation log stays global; each namespace ledger is independently auditable.
- The DataGenerator agent becomes **re-invokable per namespace**; its registration writes
  `oracles[<ns>]` instead of overwriting `default` (`routing.py:939`).

**Comparable-by-construction (user's choice).** A new namespace's oracle =
*(the agent's design→geometry mapping)* composed with the **shared, fixed
objective + feasibility evaluator**. A "15% beat" is then automatically on the same
ruler — comparability falls out of architecture, with no enforcement rule. The agent
*may* swap the objective evaluator (the PI's "touch the datagenerator" freedom), but
then it must **narrate why the new number is still comparable** — checked by the critic
(criterion 4 / CLAUDE.md §4.5 faithfulness), never blocked.

## What does NOT change (expressivity guarantee)

- One-namespace studies: identical to today — no new tools, no new concepts.
- Charter, 4 statuses, verdict validator, the provenance + host-safety floor: untouched.
- The reproduction gate still binds the deliverable (per namespace's ledger).

## Critical files (when implementation begins, post-experiment)

- `src/f3dasm/_src/agentic/graph_state.py` — `Delegation.namespace`
- `src/f3dasm/_src/agentic/agent_runtime.py` — `oracles` registry + back-compat
- `src/f3dasm/_src/agentic/instrumented.py` — `get_evaluator(namespace)`, per-ns store
- `src/f3dasm/_src/experimentdata.py` — per-namespace sentinel (additive guard)
- `src/f3dasm/_src/agentic/hypothesis_ledger.py` — max-open → confirm nudge
- `src/f3dasm/_src/agentic/nodes/tools/routing.py` — milestone + pillar nudges, ns plumbing
- `src/f3dasm/_src/agentic/agents/datagenerator.py` — re-invokable per namespace
- `docs/agentic/FEATURES.md` — new "design namespace" capability entry (same-commit rule)

## Verification — the decisive test BEFORE any code

The framework is an untested hypothesis. The cheap decisive test (no code change — uses
today's single-study machinery): stand up the PI's **ellipse-with-phase 2D design as ONE
namespace** in the current system and run the agent. Watch two things:

1. **Capability** — can it explore the low-D space competently (sample → surrogate →
   optimize) and find good feasible designs?
2. **Honesty** — does it report truthfully whether the design beats the baseline on the
   *comparable* metric, including "no, it didn't, but the idea was worth trying"?

If it cannot get past one hand-built design, iterate there — the multi-namespace
primitive is wasted until the single-design loop is honest and competent. Only after this
holds do we implement Axis 3 (and the Axis 2 nudges). Axis 1 needs no code beyond Axis 3.

## Sequencing

1. **This spec** — conceptual framework + touch-point map (now).
2. **2D single-namespace experiment** — no code; the decisive cheap test. Problem
   statement lives in the benchmark repo (`f3dasm-agentic-benchmarks`), not here.
3. **If it holds** — implementation plan (TDD): Axis 3 namespace primitive + Axis 2
   nudge conversions. Each block→nudge and the namespace plumbing get headless
   regression tests first, e2e last (per the headless-smoke-before-e2e rule).
4. **§4 record** — Axes 1 and 2 are epistemic-contract decisions the user made in this
   session; logged in this doc and the BACKLOG.
