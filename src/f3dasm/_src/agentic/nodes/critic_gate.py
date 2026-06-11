"""Critic consultation: find the connected critic, run a synchronous FEEDBACK/GATE
audit, persist the verdict, build the feedback task message. A mixin on the
strategizer (uses its instance attrs + RecordingMixin methods via MRO)."""
from __future__ import annotations

import traceback
from pathlib import Path


class CriticGateMixin:
    """Mixin carrying the four critic-consultation helpers for StrategizerNode.

    Relies on instance attributes and RecordingMixin methods resolved via MRO:
      self._spec, self._outgoing, self._worker_adapters, self._current_notes_dir,
      self._critic_calls, self._study_dir, self._name,
      self._record_usage, self._role_of, self._record_retrospective.
    Defines no __init__.
    """

    def _find_critic_name(self) -> str | None:
        """Name of the first connected critic worker, or None."""
        spec = self._spec
        if spec is None or not hasattr(spec, "nodes"):
            return None
        for target in self._outgoing:
            agent = spec.nodes.get(target)
            if (
                agent is not None
                and getattr(agent, "role", None) == "critic"
                and target in self._worker_adapters
            ):
                return target
        return None

    def _invoke_critic(self, task_msg: str) -> str:
        """Synchronously invoke the connected critic; returns its
        text or an ERROR string.

        The critic is a worker too: under F3DASM_DEBUG its full transcript is
        streamed to disk, its verdict/review is ALWAYS persisted (the PASS
        branch doesn't echo it to the strategizer, so this is the only place
        the deciding verdict is auditable), and its ### Retrospective is
        recorded like every other node's (#7).
        """
        critic_name = self._find_critic_name()
        if critic_name is None:
            return "ERROR: no critic connected."
        adapter = self._worker_adapters[critic_name]
        worker = (
            adapter.copy() if hasattr(adapter, "copy") else adapter
        )
        self._critic_calls = getattr(self, "_critic_calls", 0) + 1
        _n = self._critic_calls
        from ..backends.base import (
            debug_enabled as _dbg,
        )
        from ..backends.base import (
            get_transcript_sink as _get_sink,
        )
        from ..backends.base import (
            set_transcript_sink as _set_sink,
        )
        _notes = self._current_notes_dir
        _prev_sink = _get_sink()
        if _dbg() and _notes is not None:
            _set_sink(str(
                _notes.parent / "transcripts" / "critic"
                / f"call_{_n:03d}.jsonl"))
        try:
            critique = worker.invoke(
                [{"role": "user", "content": task_msg}]
            )
        except Exception:  # noqa: BLE001
            critique = (
                "ERROR: critic invocation failed:\n"
                f"{traceback.format_exc()}"
            )
        finally:
            # Restore the strategizer's own sink (same thread-local).
            if _dbg() and _notes is not None:
                _set_sink(_prev_sink)
        # Account the critic's tokens/cost — critic consults are real LLM calls
        # and must land in token_totals AND telemetry (they were previously
        # uncounted, undercounting run cost and omitting the 'critic' role).
        self._record_usage(
            getattr(worker, "last_usage", {}) or {},
            role=self._role_of(critic_name),
            model=getattr(worker, "model", None),
            phase="critic_review",
            delegation_id=f"critic-{_n}",
        )
        # Always-on: persist the verdict/review to disk + record retrospective.
        self._persist_critic_review(_n, critique)
        self._record_retrospective("critic", f"critic-{_n}", critique)
        return critique

    def _persist_critic_review(self, n: int, critique_text: str) -> None:
        """Write the critic's full review to debug/critic_reviews/ so the
        deciding verdict is auditable regardless of PASS/REVISE. Best-effort."""
        try:
            notes = self._current_notes_dir
            if notes is None:
                return
            d = Path(notes).parent / "critic_reviews"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"call_{n:03d}.md").write_text(
                critique_text or "", encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    def _build_feedback_task_msg(self, h_ids: list) -> str:
        """<mode>FEEDBACK</mode> task message with paths block."""
        notes_path = str(self._current_notes_dir or "")
        _notes_dir = self._current_notes_dir
        _study_dir = self._study_dir
        _debug_dir = (
            _notes_dir.parent if _notes_dir is not None else None
        )
        return (
            "<mode>FEEDBACK</mode>\n\n"
            "Perform a synchronous find-only adversarial audit.  "
            "PASS is not an available verdict — return REVISE or "
            "REJECT with your findings.\n\n"
            "<paths>\n"
            f"study_dir             = {_study_dir}\n"
            f"debug_dir             = {_debug_dir}\n"
            f"delegation_log        = {_debug_dir}/delegation_log.jsonl\n"
            f"diagnostics           = {_debug_dir}/diagnostics.jsonl\n"
            f"strategizer_notes     = {notes_path}\n"
            f"delegations_workspace = {_debug_dir}/delegations/\n"
            f"deliverable            = {_study_dir}/replicate.py "
            "(solution.md is written by the runtime AFTER this gate, from the "
            "accepted summary — do NOT flag it as missing)\n"
            "</paths>\n\n"
            f"Focus hypotheses: {h_ids if h_ids else 'all'}"
        )
