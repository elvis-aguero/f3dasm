"""Critic consultation: find the connected critic, run a synchronous FEEDBACK/GATE
audit, persist the verdict, build the feedback task message. A mixin on the
strategizer (uses its instance attrs + RecordingMixin methods via MRO)."""
from __future__ import annotations

import traceback
from pathlib import Path

# Bound on how many earlier reviews are echoed back to the critic, and the
# total char budget for the digest — injected context must be current and
# bounded, not an ever-growing transcript.
_MAX_PRIOR_REVIEWS = 6
_PRIOR_REVIEWS_CHAR_BUDGET = 6000


def _extract_md_section(text: str, header: str) -> str:
    """Return the body under a `### Header` up to the next `### ` heading.

    Used to pull just the Verdict and Findings out of a persisted review,
    dropping Actions/Numbers/Retrospective noise. Empty string if absent.
    """
    lines = text.splitlines()
    out: list[str] = []
    capturing = False
    for line in lines:
        if line.strip() == header:
            capturing = True
            continue
        if capturing and line.lstrip().startswith("### "):
            break
        if capturing:
            out.append(line)
    return "\n".join(out).strip()


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

    def _prior_reviews_digest(self) -> str:
        """A bounded `<prior_reviews_this_run>` block of THIS run's earlier
        critic reviews (verdict + findings only), or "" if none.

        The critic is invoked one-shot per gate with no live session and no
        RecallHistory, so without this it cannot see what it already ruled and
        can silently contradict an earlier verdict (the H1 SUPPORTED->FALSIFIED
        ->back whipsaw that drove the REVISE-spin). Echoing its standing
        objections back forces consistency: it may still reverse, but only by
        saying so and citing new evidence — never silently.
        """
        notes = self._current_notes_dir
        if notes is None:
            return ""
        review_dir = Path(notes).parent / "critic_reviews"
        if not review_dir.is_dir():
            return ""
        files = sorted(review_dir.glob("call_*.md"))
        if not files:
            return ""
        elided = max(0, len(files) - _MAX_PRIOR_REVIEWS)
        kept = files[-_MAX_PRIOR_REVIEWS:]
        blocks: list[str] = []
        for f in kept:
            try:
                text = f.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001
                continue
            n = f.stem.replace("call_", "")
            verdict = _extract_md_section(text, "### Verdict") or "(unparsed)"
            findings = _extract_md_section(text, "### Findings") or "(none)"
            blocks.append(
                f"--- call_{n} ---\n"
                f"Verdict: {verdict}\n"
                f"Findings:\n{findings}"
            )
        # Char budget: drop oldest kept blocks until under budget.
        while blocks and sum(len(b) for b in blocks) > _PRIOR_REVIEWS_CHAR_BUDGET:
            blocks.pop(0)
            elided += 1
        if not blocks:
            return ""
        elided_note = (
            f"\n({elided} earlier review(s) elided for brevity.)"
            if elided else ""
        )
        return (
            "<prior_reviews_this_run>\n"
            "You have already reviewed this run. Your standing verdicts and "
            "objections are below. Be CONSISTENT with them: do not silently "
            "contradict a verdict you reached earlier, and do not re-raise an "
            "objection the strategizer has since resolved. You MAY reverse a "
            "prior position, but only by stating which call you are reversing "
            "and citing the Charter clause and the NEW evidence that justifies "
            "it.\n"
            + "\n\n".join(blocks)
            + elided_note
            + "\n</prior_reviews_this_run>\n\n"
        )

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
        # Cross-round memory: prepend this run's earlier reviews so the critic
        # stays consistent instead of whipsawing its own verdicts (Fix A).
        task_msg = self._prior_reviews_digest() + task_msg
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
            f"deliverable            = {_study_dir}/pipeline.ipynb "
            "(the runtime EXECUTES the notebook lazily after this gate to verify "
            "the headline re-derives from the ledger with zero new evals; the "
            "notebook's own markdown cells ARE the writeup — there is no "
            "solution.md, do NOT flag it as missing)\n"
            "</paths>\n\n"
            f"Focus hypotheses: {h_ids if h_ids else 'all'}"
        )
