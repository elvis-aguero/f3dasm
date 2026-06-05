# Requirements: Hypothesis Ledger & Delegation Log

## As Is

- Strategizer writes free-form `hypotheses.md` under `strategizer_notes/`
- Structure enforced only by prompt instructions — model can drift
- `Delegate(target, intent, expected_report)` has no hypothesis linkage
- Delegation IDs are `TASK-{8-hex-chars}` (UUID fragment)
- Workers sandbox writes to `workspace/` root (any path inside workspace)
- No machine-readable record of delegation lifecycle or hypothesis status changes

## To Be

Two structured files written under `strategizer_notes/` during every run:

**`hypotheses.json`** — keyed by `H1`, `H2`..., each entry:
```json
{
  "id": "H1",
  "statement": "...",
  "proposed_by": "strategizer",   // auto-injected
  "proposed_at": "ISO8601",       // auto-injected
  "status_log": [
    {"status": "OPEN", "comment": "...", "triggered_by": null, "ts": "ISO8601"},
    {"status": "FALSIFIED", "comment": "...", "triggered_by": "D003", "ts": "ISO8601"}
  ]
}
```

**`delegations.jsonl`** — append-only, one complete record per delegation at end:
```json
{"id": "D001", "from_node": "strategizer", "to_node": "implementer",
 "hypothesis_ids": ["H1"], "task": "...", "outcome_summary": "...",
 "started_at": "ISO8601", "completed_at": "ISO8601", "status": "DONE"}
```

Four new strategizer closure tools: `HypothesisPropose`, `HypothesisUpdate`,
`HypothesisList`, `HypothesisGet`.

`Delegate()` gains a required `hypothesis_ids: list` parameter.

Workers write only to `workspace/{delegation_id}/`; reads across `workspace/` unrestricted.

---

## Requirements

1. **HypothesisLedger module** — new `hypothesis_ledger.py` with `HypothesisLedger` class
2. **HypothesisPropose** — proposes a hypothesis; runtime injects `id`, `proposed_by`, `proposed_at`
3. **Max-3-OPEN guard** — `HypothesisPropose` returns error if 3 OPEN hypotheses already exist
4. **HypothesisUpdate** — appends to `status_log`; runtime injects `triggered_by` and `ts`
5. **Valid status set** — only `OPEN`, `SUPPORTED`, `FALSIFIED`, `INCONCLUSIVE` accepted
6. **HypothesisList** — returns `{id, statement, current_status}` only (no logs)
7. **HypothesisGet** — returns full entry including `status_log`
8. **Disk persistence** — `hypotheses.json` written after every mutating operation
9. **Sequential delegation IDs** — `D{n:03d}` replacing `TASK-{hex}`
10. **Delegate requires hypothesis_ids** — empty list returns error
11. **Workspace subfolder injection** — `<workspace_subfolder>workspace/D001/</workspace_subfolder>` prepended to task message
12. **Write isolation per delegation** — `Write` closure rejects paths outside `workspace/{delegation_id}/`
13. **Delegation JSONL record** — written on delegation completion (Done or Errored)
14. **notes_dir threading** — `notes_dir` passed from `agent_runtime` → `build_graph` → `StrategizerNode`
15. **Prompt updates** — strategizer system prompt and preambles document new tools

---

## Acceptance Criteria

1. `HypothesisLedger(notes_dir).propose(statement, proposed_by)` creates entry with `status="OPEN"`, auto-assigned `id`, `proposed_at` timestamp
2. Proposing when 3 OPEN exist returns an error string beginning with `"ERROR:"`
3. `update(h_id, "FALSIFIED", comment, "D003")` appends `{status: "FALSIFIED", triggered_by: "D003"}` to `status_log`
4. `update` with invalid status returns error string
5. `list_all()` returns list of dicts with only `id`, `statement`, `current_status` keys
6. `get(h_id)` returns full dict including `status_log`
7. `hypotheses.json` can be deleted and reloaded: `HypothesisLedger(same_dir).get("H1")` returns same data
8. `record_delegation(...)` appends one line to `delegations.jsonl`
9. `Delegate(..., hypothesis_ids=[])` returns `"ERROR: hypothesis_ids must not be empty."`
10. Task message for a delegation contains `<workspace_subfolder>workspace/D001/</workspace_subfolder>`
11. Worker `Write("outside/path.txt", ...)` returns error; `Write("D001/result.csv", ...)` succeeds
12. First delegation in a run gets ID `D001`; next gets `D002`
13. `delegations.jsonl` contains one JSON line per completed delegation with all required fields
14. `AgenticRun` with default graph creates `hypotheses.json` + `delegations.jsonl` in `strategizer_notes/`

---

## Testing Plan

### Unit tests — `tests/agentic/test_hypothesis_ledger.py` (NEW, written first)
- `test_propose_creates_open_entry`
- `test_propose_assigns_sequential_ids` (H1, H2, H3)
- `test_propose_rejects_fourth_when_three_open`
- `test_propose_allows_fourth_after_close`
- `test_update_appends_to_status_log`
- `test_update_injects_triggered_by`
- `test_update_rejects_invalid_status`
- `test_update_rejects_unknown_hypothesis_id`
- `test_list_returns_only_summary_fields`
- `test_get_returns_full_entry`
- `test_persists_and_reloads_from_disk`
- `test_record_delegation_appends_to_jsonl`
- `test_concurrent_propose_is_thread_safe`

### Integration tests — additions to `tests/agentic/test_nodes.py`
- `test_delegate_requires_hypothesis_ids`
- `test_delegate_injects_workspace_subfolder_in_task`
- `test_delegate_id_is_sequential`
- `test_delegate_writes_delegation_jsonl_on_done`
- `test_hypothesis_propose_via_strategizer_closure`
- `test_hypothesis_update_injects_triggered_by`
- `test_max_three_open_hypothesis_guard`
- `test_worker_write_rejected_outside_delegation_subfolder`
- `test_worker_write_allowed_inside_delegation_subfolder`

---

## Implementation Plan

1. **Write `test_hypothesis_ledger.py`** (all tests, all failing) → confirm failures
2. **Implement `hypothesis_ledger.py`** → run ledger tests until green
3. **Write new `test_nodes.py` tests** (failing)
4. **Modify `nodes.py`**: sequential IDs, `hypothesis_ids` param, workspace injection, write isolation, hypothesis closures, JSONL record
5. **Modify `graph_builder.py`**: add `notes_dir` param, pass to `StrategizerNode`
6. **Modify `agent_runtime.py`**: pass `notes_dir` to `build_graph`
7. **Modify `agent_prompts.py`**: update preambles and strategizer prompt
8. **Run full suite** — confirm 90+ passing
