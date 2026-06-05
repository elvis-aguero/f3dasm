"""Structured hypothesis ledger for agentic runs.

Manages one file under ``debug/strategizer_notes/``:

- ``hypotheses.json``   — keyed by hypothesis ID; append-only status_log

Delegation logging has moved to :class:`~delegation_log.DelegationLog`
(``debug/delegation_log.jsonl``).
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["HypothesisLedger", "HypothesisEntry", "StatusLogEntry"]

VALID_STATUSES = frozenset({"OPEN", "SUPPORTED", "FALSIFIED", "INCONCLUSIVE"})
MAX_OPEN = 3


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


@dataclass
class StatusLogEntry:
    status: str
    comment: str
    triggered_by: str | None
    ts: str = field(default_factory=_now_iso)


@dataclass
class HypothesisEntry:
    id: str
    statement: str
    proposed_by: str
    proposed_at: str
    status_log: list[StatusLogEntry] = field(default_factory=list)

    @property
    def current_status(self) -> str:
        return self.status_log[-1].status if self.status_log else "OPEN"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "proposed_by": self.proposed_by,
            "proposed_at": self.proposed_at,
            "status_log": [asdict(e) for e in self.status_log],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HypothesisEntry":
        log = [StatusLogEntry(**e) for e in data.get("status_log", [])]
        return cls(
            id=data["id"],
            statement=data["statement"],
            proposed_by=data["proposed_by"],
            proposed_at=data["proposed_at"],
            status_log=log,
        )


class HypothesisLedger:
    """Manages ``hypotheses.json`` on disk.

    All mutating operations are atomic under a threading lock so that
    concurrent delegation threads (background workers) cannot corrupt state.
    """

    def __init__(self, notes_dir: Path) -> None:
        self._notes_dir = Path(notes_dir)
        self._hypotheses_path = self._notes_dir / "hypotheses.json"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Hypothesis operations
    # ------------------------------------------------------------------

    def propose(self, statement: str, proposed_by: str) -> str:
        """Propose a new hypothesis; returns the assigned ID (e.g. 'H1').

        Returns an error string if 3 OPEN hypotheses already exist.
        """
        with self._lock:
            data = self._load()
            open_count = sum(
                1 for h in data.values()
                if h["status_log"] and h["status_log"][-1]["status"] == "OPEN"
            )
            if open_count >= MAX_OPEN:
                return (
                    f"ERROR: {MAX_OPEN} OPEN hypotheses already exist. "
                    "Close one (SUPPORTED, FALSIFIED, or INCONCLUSIVE) before proposing a new one."
                )
            h_id = f"H{len(data) + 1}"
            ts = _now_iso()
            entry = HypothesisEntry(
                id=h_id,
                statement=statement,
                proposed_by=proposed_by,
                proposed_at=ts,
                status_log=[StatusLogEntry(status="OPEN", comment="initial proposal", triggered_by=None, ts=ts)],
            )
            data[h_id] = entry.to_dict()
            self._save(data)
            return h_id

    def update(
        self,
        h_id: str,
        status: str,
        comment: str,
        triggered_by: str | None,
    ) -> str:
        """Append a status-change entry to the hypothesis's status_log.

        Returns an error string on invalid input.
        """
        if status not in VALID_STATUSES:
            return (
                f"ERROR: invalid status {status!r}. "
                f"Must be one of: {sorted(VALID_STATUSES)}"
            )
        with self._lock:
            data = self._load()
            if h_id not in data:
                return f"ERROR: hypothesis {h_id!r} not found."
            entry = StatusLogEntry(
                status=status,
                comment=comment,
                triggered_by=triggered_by,
                ts=_now_iso(),
            )
            data[h_id]["status_log"].append(asdict(entry))
            self._save(data)
            return f"Updated {h_id}: status → {status}."

    def list_all(self) -> list[dict]:
        """Return summary dicts ``{id, statement, current_status}`` for all hypotheses."""
        with self._lock:
            data = self._load()
        result = []
        for h in data.values():
            log = h.get("status_log", [])
            current = log[-1]["status"] if log else "OPEN"
            result.append({"id": h["id"], "statement": h["statement"], "current_status": current})
        return result

    def get(self, h_id: str) -> dict | None:
        """Return the full hypothesis dict, or None if not found."""
        with self._lock:
            data = self._load()
        return data.get(h_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        """Load hypotheses from disk; returns {} if file absent."""
        if not self._hypotheses_path.exists():
            return {}
        return json.loads(self._hypotheses_path.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        """Atomically write hypotheses to disk."""
        self._hypotheses_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
