# Phase 3: Canonical Store — RunStateSummary, RecallStore/QueryStore, UNLEDGERED_EVALS

## As Is

- `InstrumentedDataGenerator` in `instrumented.py` stamps provenance and flushes to a shared `ExperimentData` store at `store_dir/experiment_data/`.
- `StrategizerNode._build_routing_closures()` provides `Delegate`, `GetStatus`, `Reply`, `Done`, `RecallHistory` etc.
- `ScienceMonitor` has rules: `EVIDENCE_DELEGATION_EXISTS`, `EVIDENCE_NUMBERS_MATCH`, `SUPPORTED_WITHOUT_ATTACK`, `POSTERIOR_INERTIA`, `STALE_OPEN`, `UNANCHORED_DELEGATION`. No store access.
- `DelegationLog.record()` fields: id, from_node, to_node, task, deliverable, hypothesis_ids, started_at, completed_at, status, tokens_in, tokens_out, cost_usd, is_falsification_attempt. No `evals` field.
- Wet run showed D001 claimed ReportEvals but wrote 0 store rows (bypassed get_evaluator). This gap cannot be caught by ScienceMonitor.

## To Be

### Part A: `RunStateSummary` (in `instrumented.py`)
- Class that computes a summary from the canonical ExperimentData store.
- `from_store(store_dir, *, fidelity_column=None) -> RunStateSummary | None`: returns None if store absent/empty.
- Module-level mtime cache keyed by `store_dir`; recomputes only when `output.csv` mtime changes.
- Computes: `n_rows`, `n_per_delegation`, `n_per_source`, `n_per_fidelity` (if fidelity_column given and present in input columns), per-output numeric column stats (min/max/mean, skip provenance cols).
- `format() -> str`: compact human/LLM-readable block, ≤25 lines typical.

### Part B: `RecallStore` and `QueryStore` closures in `nodes.py`
- `RecallStore() -> str`: ALWAYS injected (not gated on delegation_log). Returns `RunStateSummary.from_store(...).format()` or empty-message when no store. Store dir derived: `_current_notes_dir.parent.parent / "experiment_data"`. Docstring is LLM-facing.
- `QueryStore(delegation_ids=None, source=None, n_best=None, output_name=None) -> str`: READ-ONLY filtered view. Loads store, filters by delegation_ids (JSON/comma/bare string decoding) and/or source. If n_best+output_name given, returns n best rows sorted by that output. Never mutates store. Docstring is LLM-facing.
- Both go through `_wrap_closure`.

### Part C: `UNLEDGERED_EVALS` drift rule in `ScienceMonitor`
- New rule, warn severity, fires when a DONE delegation has `evals > 0` in the log AND zero rows in the canonical store with that delegation_id.
- `DelegationLog.record()` gets new `evals: int = 0` field (additive, like `is_falsification_attempt`).
- `_run` epilogue in `nodes.py` passes resolved `_evals` into `delegation_log.record()`.
- `ScienceMonitor` gets `store_dir=None` constructor param; the UNLEDGERED_EVALS rule no-ops when None.
- `store_dir` is wired in `StrategizerNode.__call__` when `_current_notes_dir` is set: `node._science_monitor.store_dir = _current_notes_dir.parent.parent / "experiment_data"`. (Lazy read, no eager load.)
- Monitor uses `RunStateSummary.from_store()` (mtime-cached) to get store row counts per delegation.

## Requirements

1. `RunStateSummary.from_store(store_dir)` returns `None` when store_dir has no `experiment_data/output.csv`.
2. `RunStateSummary.from_store(store_dir)` returns a valid `RunStateSummary` instance with correct counts when populated.
3. `RunStateSummary` caches by store_dir+mtime; same object returned on repeated calls when file unchanged; new object returned when mtime changes.
4. `RunStateSummary.n_per_fidelity` populated when fidelity_column provided and present in INPUT columns; None otherwise.
5. `RunStateSummary.format()` returns a string ≤25 lines for typical stores.
6. `RecallStore` closure returns empty-message when no store exists.
7. `RecallStore` closure returns formatted summary when store populated.
8. `RecallStore` is injected ALWAYS (not gated on delegation_log).
9. `QueryStore` filters rows by delegation_ids (JSON/comma/bare string forms).
10. `QueryStore` with n_best+output_name returns n smallest rows by that output as compact text.
11. `QueryStore` never mutates the store (mtime preserved, content unchanged).
12. `DelegationLog.record()` accepts and stores `evals: int = 0` field.
13. `_run` epilogue in `nodes.py` passes `evals=_evals` to `delegation_log.record()`.
14. `ScienceMonitor` accepts `store_dir=None` constructor param.
15. `UNLEDGERED_EVALS` rule fires for DONE delegation with evals>0 and 0 store rows for that id.
16. `UNLEDGERED_EVALS` rule is silent when store has rows for the delegation.
17. `UNLEDGERED_EVALS` rule is silent when `store_dir=None`.
18. `store_dir` is set on `ScienceMonitor` in `StrategizerNode.__call__` when `_current_notes_dir` is known.

## Acceptance Criteria

1. `RunStateSummary.from_store(tmp_path_without_store)` returns `None`.
2. After writing 5 rows with delegation_ids D001×3 and D002×2: `summary.n_rows == 5`, `summary.n_per_delegation == {"D001": 3, "D002": 2}`.
3. Calling `from_store` twice on same path returns `is` same object; after touching file returns new object.
4. `summary.n_per_fidelity` is None when fidelity_column not in input columns; populated when present.
5. `len(summary.format().splitlines()) <= 25` for a store with 10 rows.
6. `RecallStore()` on a missing store returns the exact empty-message string.
7. `RecallStore()` on populated store returns a non-empty string containing delegation counts.
8. `RecallStore` key present in `node.adapter.closure_tools` regardless of `delegation_log` presence.
9. `QueryStore(delegation_ids="D001,D002")`, `QueryStore(delegation_ids='["D001"]')`, `QueryStore(delegation_ids="D001")` all filter correctly.
10. `QueryStore(n_best=2, output_name="f")` returns the 2 rows with smallest f values.
11. `output.csv` mtime before == mtime after `QueryStore(...)` call.
12. `log.query_all()[0]["evals"] == 5` after `log.record(..., evals=5)`.
13. The delegation_log record written in `_run` epilogue has `evals` matching `_evals`.
14. `ScienceMonitor(ledger, dlog, store_dir="/some/path")` does not raise.
15. `UNLEDGERED_EVALS` in violation rules when DONE record has evals=5 and store has 0 rows for D001.
16. No `UNLEDGERED_EVALS` when store has rows for that delegation.
17. No violations (not even UNLEDGERED_EVALS) when `store_dir=None`.
18. After `__call__` with run_dir set, `node._science_monitor.store_dir` is a Path.

## Testing Plan

TDD order:
1. Write tests for `DelegationLog.record(evals=...)` → implement the field.
2. Write tests for `RunStateSummary` (empty, populated, cache, fidelity, format) → implement.
3. Write tests for `RecallStore` and `QueryStore` closures → implement.
4. Write tests for `UNLEDGERED_EVALS` rule and `store_dir` constructor param → implement.
5. Write test that `_run` epilogue passes evals to delegation_log → implement wiring.
6. Write test that `store_dir` is set on science_monitor in `__call__` → implement.

## Implementation Plan

1. **`DelegationLog.record()` evals field** — add `evals: int = 0` param; write to record dict. Test: round-trip test in `test_delegation_log_extra.py`.

2. **`RunStateSummary` class** — add to `instrumented.py`. Module-level cache dict `_RSS_CACHE: dict[str, tuple[float, RunStateSummary]]`. `from_store` checks mtime of `output.csv`, reuses cache or recomputes. Test: `test_instrumented.py` additions.

3. **`RecallStore` closure** — in `_build_routing_closures`, unconditionally add `RecallStore` (alongside `Delegate`). Derive store_dir from `_current_notes_dir` (or None → return empty message). Test: stub-node fixture in `test_nodes_coverage.py` or new file.

4. **`QueryStore` closure** — also in `_build_routing_closures`. Decode delegation_ids like Delegate does. Test: same file.

5. **`ScienceMonitor` store_dir** — add `store_dir=None` to constructor. Add `UNLEDGERED_EVALS` rule method. Wire in `StrategizerNode.__call__`. Test: `test_science_monitor.py` additions.

6. **`_run` epilogue evals wiring** — pass `evals=_evals` to `delegation_log.record()`. Test: extend test_nodes_coverage.
