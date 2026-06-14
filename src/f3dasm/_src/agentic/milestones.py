"""Milestone ledger (Spec C2): the strategizer's process-policy tracker.

DISTINCT from the hypothesis ledger (epistemics — what's true, closed by
evidence). Milestones are PROCESS steps — prescribed defaults or agent-authored
— closed by COMPLETION, and SOFT-enforced: a decision-point nudge + a recorded
diagnostic when you enter a phase with a pending gate, never a refusal. The
proven pattern (mirrors the falsification checkpoint), not a buried prompt
caveat.

Config-seeded default GATES carry an auto-satisfy PREDICATE (code, keyed by the
milestone's ``key``) so they self-resolve the moment their structural condition
holds — no busywork, no advisory drift. Agent-authored milestones have no
predicate and are completed/skipped explicitly.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "MilestoneLedger",
    "DEFAULT_MILESTONES",
    "milestone_gate_nudge",
    "VALID_STATUSES",
]

VALID_STATUSES = frozenset({"PENDING", "DONE", "SKIPPED"})


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Default milestones: built-in gates with auto-satisfy predicates (node->bool).
# Predicates live in code (they can't serialise); the ledger tracks only status.
# --------------------------------------------------------------------------

def _lit_review_done(node) -> bool:
    """A completed literature_reviewer delegation exists."""
    log = getattr(node, "_delegation_log", None)
    if log is None:
        return False
    spec = getattr(node, "_spec", None)
    outgoing = getattr(node, "_outgoing", []) or []
    lit_targets = set()
    for t in outgoing:
        role = ""
        if spec is not None and hasattr(spec, "nodes"):
            role = getattr(spec.nodes.get(t), "role", "") or ""
        if role == "literature_reviewer" or "literature" in t.lower():
            lit_targets.add(t)
    return any(
        r.get("status") == "DONE" and r.get("to_node") in lit_targets
        for r in log.query_all()
    )


def _canonical_source_ready(node) -> bool:
    """A canonical ground-truth oracle is registered/validated."""
    fn = getattr(node, "_canonical_source_registered", None)
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class DefaultMilestone:
    key: str
    description: str
    phase: str | None        # the Phase value this gate guards
    predicate: Callable      # (node) -> bool, auto-satisfy condition


DEFAULT_MILESTONES: list[DefaultMilestone] = [
    DefaultMilestone(
        "lit_review_before_doe",
        "Run a literature review before designing the experiment (DoE).",
        "doe", _lit_review_done,
    ),
    DefaultMilestone(
        "datagenerator_gold_state",
        "Get the datagenerator/oracle registered and validated before "
        "generating data.",
        "data_generation", _canonical_source_ready,
    ),
]

_PREDICATES: dict[str, Callable] = {d.key: d.predicate for d in DEFAULT_MILESTONES}


class MilestoneLedger:
    """Append-only, status-tracked process ledger persisted as JSON."""

    def __init__(self, notes_dir: Path | str) -> None:
        self._path = Path(notes_dir) / "milestones.json"
        self._lock = threading.Lock()

    # -- persistence -------------------------------------------------------
    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _next_id(data: dict) -> str:
        n = 0
        for k in data:
            try:
                n = max(n, int(k[1:]))
            except (ValueError, IndexError):
                pass
        return f"M{n + 1:03d}"

    # -- seeding + authoring ----------------------------------------------
    def seed_defaults(self, disabled: frozenset[str] = frozenset()) -> None:
        with self._lock:
            data = self._load()
            existing_keys = {m.get("key") for m in data.values()}
            for d in DEFAULT_MILESTONES:
                if d.key in disabled or d.key in existing_keys:
                    continue
                mid = self._next_id(data)
                data[mid] = {
                    "id": mid, "key": d.key, "description": d.description,
                    "phase": d.phase, "source": "default", "gate": True,
                    "status": "PENDING", "note": "", "opened_at": _now(),
                    "closed_at": None,
                }
            self._save(data)

    def propose(self, description: str, phase: str | None = None,
                gate: bool = False) -> str:
        if not description or not description.strip():
            return "ERROR: milestone description must be non-empty."
        with self._lock:
            data = self._load()
            mid = self._next_id(data)
            data[mid] = {
                "id": mid, "key": None, "description": description.strip(),
                "phase": phase, "source": "agent", "gate": bool(gate),
                "status": "PENDING", "note": "", "opened_at": _now(),
                "closed_at": None,
            }
            self._save(data)
        return mid

    def _close(self, mid: str, status: str, note: str) -> str:
        with self._lock:
            data = self._load()
            m = data.get(mid)
            if m is None:
                return f"ERROR: unknown milestone {mid!r}."
            m["status"] = status
            m["note"] = note
            m["closed_at"] = _now()
            self._save(data)
        return f"{mid} marked {status}."

    def complete(self, mid: str, note: str = "") -> str:
        return self._close(mid, "DONE", note)

    def skip(self, mid: str, reason: str) -> str:
        return self._close(mid, "SKIPPED", reason)

    # -- queries -----------------------------------------------------------
    def list_all(self) -> list[dict]:
        return list(self._load().values())

    def get(self, mid: str) -> dict | None:
        return self._load().get(mid)

    def pending_gates(self, phase: str | None = None) -> list[dict]:
        out = []
        for m in self._load().values():
            if m.get("gate") and m.get("status") == "PENDING":
                if phase is None or m.get("phase") == phase:
                    out.append(m)
        return out

    def auto_satisfy(self, node) -> None:
        """Flip default gates to DONE whose structural predicate now holds."""
        with self._lock:
            data = self._load()
            changed = False
            for m in data.values():
                if (m.get("source") == "default"
                        and m.get("status") == "PENDING"):
                    pred = _PREDICATES.get(m.get("key"))
                    if pred is not None and pred(node):
                        m["status"] = "DONE"
                        m["note"] = "auto-satisfied (condition met)"
                        m["closed_at"] = _now()
                        changed = True
            if changed:
                self._save(data)

    def format(self) -> str:
        items = self.list_all()
        if not items:
            return "No milestones."
        return "\n".join(
            f"- {m['id']} [{m['status']}]"
            f"{' (gate, phase=' + str(m['phase']) + ')' if m['gate'] else ''}"
            f": {m['description']}"
            for m in items
        )


def milestone_gate_nudge(ledger: MilestoneLedger, node, phase: str) -> str:
    """Decision-point nudge text when entering ``phase`` with pending gates.

    Auto-satisfies first (so a met condition never nags), then reports the
    still-pending gates for this phase. Returns "" when nothing is pending.
    Soft: the caller appends this to the tool return; it never refuses.
    """
    ledger.auto_satisfy(node)
    pend = ledger.pending_gates(phase)
    if not pend:
        return ""
    items = "; ".join(f"{m['id']} ({m['description']})" for m in pend)
    return (
        f"\n\n⚖ MILESTONE CHECKPOINT — you're entering phase '{phase}' but "
        f"these gate milestones are still pending: {items}. Do them first, or "
        "MilestoneSkip('<id>', reason) if this study legitimately doesn't need "
        "one. (Advisory — the delegation still ran; this is recorded for the "
        "gate.)"
    )
