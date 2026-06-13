"""Graph-wide append-only delegation log for agentic runs.

Stores all inter-node delegations in ``debug/delegation_log.jsonl``.
Thread-safe via a single lock.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["DelegationLog"]


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


class DelegationLog:
    """Graph-wide append-only log at debug/delegation_log.jsonl.

    Every inter-node delegation — regardless of which node fired it — is
    recorded here. Stores FULL task text and FULL deliverable text (not truncated).
    Thread-safe via a single lock.
    """

    def __init__(self, log_path: Path) -> None:
        self._path = Path(log_path)
        self._lock = threading.Lock()
        # Monotonic sequence counter for globally-unique delegation IDs.
        # Shared across all orchestrating nodes that hold a reference to
        # this log — guarantees D### uniqueness even when datagenerator
        # and implementer both delegate to literature_reviewer.
        self._seq: int = 0
        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def next_id(self) -> str:
        """Return the next globally-unique delegation ID (e.g. ``"D001"``).

        Thread-safe: increments the shared monotonic counter under
        ``self._lock``.
        """
        with self._lock:
            self._seq += 1
            return f"D{self._seq:03d}"

    def record(
        self,
        *,
        id: str,
        from_node: str,
        to_node: str,
        task: str,
        deliverable: str,
        hypothesis_ids: list[str],
        started_at: str,
        completed_at: str,
        status: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float | None = None,
        is_falsification_attempt: bool = False,
        evals: int = 0,
    ) -> None:
        """Append one delegation record.

        task and deliverable are stored in full.
        """
        record: dict[str, Any] = {
            "id": id,
            "from_node": from_node,
            "to_node": to_node,
            "task": task,
            "deliverable": deliverable,
            "hypothesis_ids": hypothesis_ids,
            "started_at": started_at,
            "completed_at": completed_at,
            "status": status,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
            "is_falsification_attempt": is_falsification_attempt,
            "evals": evals,
        }
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def query_received(
        self, node_name: str, n: int | None = None
    ) -> list[dict]:
        """Return last n records where to_node == node_name, oldest-first.

        Returns all matching records if n is None.
        """
        with self._lock:
            records = self._load_all()

        matching = [r for r in records if r.get("to_node") == node_name]
        if n is not None:
            # Return the last n, oldest-first
            matching = matching[-n:]
        return matching

    def last_completed_id(self, from_node: str) -> str | None:
        """Return the ID of the most recently completed (DONE) delegation
        sent BY from_node. Used for triggered_by injection in HypothesisUpdate."""
        with self._lock:
            records = self._load_all()

        # Filter for DONE delegations from this node, return last one's ID
        done = [
            r for r in records
            if r.get("from_node") == from_node and r.get("status") == "DONE"
        ]
        if not done:
            return None
        return done[-1]["id"]

    def mark_attempt(self, delegation_id: str, hypothesis_id: str) -> bool:
        """Retroactively flag an existing record as a falsification ATTEMPT of
        hypothesis_id (the read-time post-hoc link).

        Sets is_falsification_attempt=True, adds hypothesis_id to its
        hypothesis_ids, and stamps attempt_linked_post_hoc=True so the critic
        scrutinises adequacy harder than a flag declared up front at delegate
        time. Rewrites the (small) append-only log in place under the lock.
        Returns True iff a matching record was updated.
        """
        with self._lock:
            records = self._load_all()
            updated = False
            for r in records:
                if r.get("id") == delegation_id:
                    r["is_falsification_attempt"] = True
                    hids = r.get("hypothesis_ids") or []
                    if hypothesis_id not in hids:
                        hids = [*hids, hypothesis_id]
                    r["hypothesis_ids"] = hids
                    r["attempt_linked_post_hoc"] = True
                    updated = True
            if updated:
                with self._path.open("w", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")
            return updated

    def query_all(self) -> list[dict]:
        """Return every record, oldest-first."""
        with self._lock:
            return self._load_all()

    def _load_all(self) -> list[dict]:
        """Load all records from disk. Must be called under lock or read-only context."""
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").strip().splitlines()
        records = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records
