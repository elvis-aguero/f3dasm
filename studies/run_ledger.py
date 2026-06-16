"""Append one row of run telemetry to studies/run_ledger.csv.

Longitudinal record so we can measure whether changes make agentic runs better
or worse over time. Each row pins the commit SHA + run id and the headline
process/outcome metrics.

Usage:
    python studies/run_ledger.py <run_dir> [--commit <sha>]

<run_dir> is a study's runs/<timestamp>/ directory. --commit overrides the
detected HEAD (use it to backfill a row for a run that executed on an older
commit).
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

LEDGER = Path(__file__).resolve().parent / "run_ledger.csv"
COLUMNS = [
    "commit", "study", "run_id", "outcome", "critic_consults", "delegations",
    "ledger_rows", "mean_wall_ms", "input_tokens", "output_tokens",
    "cost_usd", "time_used", "milestones_done", "milestones_skipped",
    "milestones_pending", "diagnostics",
]


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _solution_field(text: str, key: str) -> str:
    # matches "- key: value" or "| key | value |" style metadata lines
    m = re.search(rf"{re.escape(key)}\s*[:|]\s*([^\n|]+)", text)
    return m.group(1).strip() if m else ""


def extract(run_dir: Path) -> dict:
    debug = run_dir / "debug"
    study = run_dir.parent.parent.name  # studies/<study>/runs/<id>
    row = {c: "" for c in COLUMNS}
    row["study"] = study
    row["run_id"] = run_dir.name

    # outcome
    status_f = debug / "run_status.json"
    sol = run_dir.parent.parent / "solution.md"
    if status_f.exists():
        try:
            row["outcome"] = json.loads(status_f.read_text()).get(
                "status", "halted")
        except Exception:
            row["outcome"] = "halted?"
    elif sol.exists() and (
        "FAILED RUN" in (_s := sol.read_text()) or "⛔" in _s
    ):
        # Distinct, loud terminal state: the deliverable never reproduced.
        row["outcome"] = "FAILED"
    elif sol.exists() and "UNGATED" in sol.read_text():
        row["outcome"] = "UNGATED"
    elif sol.exists():
        row["outcome"] = "GATED"
    else:
        row["outcome"] = "no_solution"

    # critic consults
    cr = debug / "critic_reviews"
    row["critic_consults"] = len(list(cr.glob("*.md"))) if cr.exists() else 0

    # delegations + phases
    dlog = debug / "delegation_log.jsonl"
    if dlog.exists():
        recs = [json.loads(line) for line in dlog.read_text().splitlines()
                if line.strip()]
        row["delegations"] = len(recs)

    # ledger rows + mean wall_ms (store is nested: <run>/experiment_data/experiment_data)
    out_csv = run_dir / "experiment_data" / "experiment_data" / "output.csv"
    if out_csv.exists():
        rows = list(csv.DictReader(out_csv.open()))
        row["ledger_rows"] = len(rows)
        wm = [float(r["_wall_ms"]) for r in rows
              if r.get("_wall_ms") not in (None, "", "nan")
              and r.get("_source") != "precomputed_pool"]
        row["mean_wall_ms"] = round(sum(wm) / len(wm), 3) if wm else ""

    # tokens / cost / wall from solution.md metadata
    if sol.exists():
        t = sol.read_text()
        row["input_tokens"] = _solution_field(t, "input_tokens").replace(",", "")
        row["output_tokens"] = _solution_field(
            t, "output_tokens").replace(",", "")
        row["cost_usd"] = _solution_field(t, "estimated_cost").lstrip("$")
        row["time_used"] = _solution_field(t, "time_used")

    # milestones
    ms = debug / "strategizer_notes" / "milestones.json"
    if ms.exists():
        try:
            mvals = list(json.loads(ms.read_text()).values())
            for st in ("DONE", "SKIPPED", "PENDING"):
                row[f"milestones_{st.lower()}"] = sum(
                    1 for m in mvals if m.get("status") == st)
        except Exception:
            pass

    # diagnostics histogram
    diag = debug / "diagnostics.jsonl"
    if diag.exists():
        from collections import Counter
        rows = [json.loads(line) for line in diag.read_text().splitlines()
                if line.strip()]
        c = Counter(r.get("error_type") or r.get("type")
                    or r.get("intervention") or "other" for r in rows)
        row["diagnostics"] = json.dumps(dict(c))
    return row


def main() -> None:
    args = sys.argv[1:]
    commit = None
    if "--commit" in args:
        i = args.index("--commit")
        commit = args[i + 1]
        args = args[:i] + args[i + 2:]
    run_dir = Path(args[0]).resolve()
    row = extract(run_dir)
    row["commit"] = commit or _git_short_sha()

    new = not LEDGER.exists()
    with LEDGER.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"appended row for {row['study']}/{row['run_id']} "
          f"(commit {row['commit']}, outcome {row['outcome']}) -> {LEDGER}")


if __name__ == "__main__":
    main()
