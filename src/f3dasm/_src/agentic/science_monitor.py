"""Rule-based scientific drift monitor for agentic runs.

Pure-Python observer over HypothesisLedger + DelegationLog. Recomputes
all rules from current state on every evaluation — a violation that has
resolved simply stops appearing (drain-time re-validation). No LLM
calls; backend-neutral.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["ScienceMonitor", "Violation"]

# EVIDENCE_NUMBERS_MATCH (the anchored-evidence rule) is ON by default. It was
# briefly disabled because a tight 1e-6 match rejected legitimately-grounded-
# but-rounded citations (e.g. -0.040279 vs the report's -0.040278683…); that was
# a TOLERANCE bug, not a concept bug. Re-enabled with a 1e-3 RELATIVE tolerance
# (see _numbers_match), which anchors the headline against the report while
# tolerating sensible rounding. It is the only in-flight grounding check (the
# critic gate is end-of-run). Disable with F3DASM_EVIDENCE_NUMBERS_MATCH=0.
def _evidence_numbers_match_enabled() -> bool:
    """Knob: evidence_numbers_match (config.yaml runtime block; default ON;
    F3DASM_EVIDENCE_NUMBERS_MATCH overrides). Read at call time."""
    from .settings import get_bool
    return get_bool("evidence_numbers_match", True)


STALE_K = 3
POSTERIOR_EPSILON = 0.05
# POSTERIOR_INERTIA is OFF by default: it nags when a status change moves belief
# < POSTERIOR_EPSILON, but that mis-fires on already-extreme posteriors (e.g.
# 0.95→0.97 is a legitimately small move near the ceiling). Trust the agent to
# calibrate its own beliefs; the critic judges over/under-claiming. Code kept;
# re-enable with F3DASM_POSTERIOR_INERTIA=1.
def _posterior_inertia_enabled() -> bool:
    """Knob: posterior_inertia (config.yaml runtime block; default OFF;
    F3DASM_POSTERIOR_INERTIA overrides). Read at call time."""
    from .settings import get_bool
    return get_bool("posterior_inertia", False)


MAX_INJECT_PER_TURN = 2
ESCALATION_CAP = 2
ESCALATE_AFTER_VIOLATIONS = 3
ESCALATE_AFTER_ERROR_REPEATS = 2

_NUMBERS_SECTION_RE = re.compile(
    r"###\s*Numbers\s*\n(.*?)(?=\n###|\n##|\Z)", re.DOTALL)
# A substantive Conclusions section also counts as a concrete result (Charter
# §6 allows qualitative evidence — a measurement is not the only anchor).
_CONCLUSIONS_SECTION_RE = re.compile(
    r"###\s*Conclusions?\s*\n(.*?)(?=\n###|\n##|\Z)", re.DOTALL)
# Minimum non-whitespace chars for a Conclusions section to count as a concrete
# qualitative finding rather than an empty/placeholder heading.
_MIN_CONCLUSION_CHARS = 20
# Key charclass tolerates markdown emphasis/code (**key**, `key`) —
# workers often format report keys in bold (observed in wet runs).
_NUM_LINE_RE = re.compile(
    r"^\s*-?\s*[*`_]*([\w.\-/ ]+?)[*`_]*\s*:\s*(.+?)\s*$")
_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


@dataclass(frozen=True)
class Violation:
    rule: str
    severity: str          # "error" | "warn"
    h_id: str | None
    message: str

    @property
    def key(self) -> tuple:
        return (self.rule, self.h_id)


class ScienceMonitor:
    """Evaluates drift rules; formats bounded corrective injections."""

    def __init__(
        self,
        ledger,
        delegation_log,
        diagnostics_writer: Callable[[dict], None] | None = None,
        stale_k: int = STALE_K,
        max_inject: int = MAX_INJECT_PER_TURN,
        escalation_cap: int = ESCALATION_CAP,
        store_dir: str | None = None,
    ) -> None:
        self._ledger = ledger
        self._dlog = delegation_log
        self._diag = diagnostics_writer
        self._stale_k = stale_k
        self._max_inject = max_inject
        self._escalation_cap = escalation_cap
        self._lock = threading.Lock()
        self._logged_keys: set[tuple] = set()      # diagnostics dedupe
        self._h_rule_seen: dict[str, set[str]] = {}  # h_id → rules seen
        self._error_streak: dict[tuple, int] = {}  # key → consecutive
        self._escalations = 0
        self._pending_escalation: list[str] | None = None
        # Optional canonical store path for UNLEDGERED_EVALS rule.
        # Set lazily by StrategizerNode.__call__ when run_dir is known.
        self.store_dir: str | None = store_dir

    def evaluate(self) -> list[Violation]:
        """Run all rules against current state; return live violations."""
        try:
            hypotheses = {
                h["id"]: self._ledger.get(h["id"])
                for h in self._ledger.list_all()
            }
        except Exception:  # noqa: BLE001
            return []
        records = self._dlog.query_all()
        # DONE only: an Errored delegation tested nothing, so it
        # deliberately does NOT count as a falsification attempt.
        completed = [r for r in records if r.get("status") == "DONE"]
        by_id = {r["id"]: r for r in completed}
        out: list[Violation] = []
        for h_id, h in hypotheses.items():
            out += self._check_hypothesis(h_id, h, by_id, completed)
        out += self._check_stale(hypotheses, completed)
        out += self._check_unanchored(hypotheses, completed)
        out += self._check_unledgered(records)
        return out

    def _check_hypothesis(self, h_id, h, by_id, completed):
        out = []
        log = h.get("status_log", [])
        current = log[-1]["status"] if log else "OPEN"
        # Judge ONLY the latest evidence-bearing entry — a superseded early
        # citation must not nag forever (self-healing, matching
        # _check_unanchored). "D000" is the precomputed ground-truth pool: a
        # legitimate evidence anchor in lookup studies, not a delegation, so
        # its numbers come from the canonical store (authoritative), not a
        # delegation report.
        latest_ev = None
        for entry in reversed(log):
            ev = entry.get("evidence") or {}
            if ev.get("delegation") is not None:
                latest_ev = ev
                break
        if latest_ev is not None:
            d_id = latest_ev.get("delegation")
            if d_id != "D000" and d_id not in by_id:
                out.append(Violation(
                    "EVIDENCE_DELEGATION_EXISTS", "error", h_id,
                    f"{h_id} cites evidence from {d_id!r}, which is "
                    "not a completed delegation. Cite a real D-id "
                    "from the delegation log (or D000 for the "
                    "ground-truth pool), or correct the update."))
            elif d_id != "D000" and _evidence_numbers_match_enabled():
                numbers = latest_ev.get("numbers") or {}
                if numbers and not self._numbers_match(
                        numbers, by_id[d_id].get("deliverable", "")):
                    out.append(Violation(
                        "EVIDENCE_NUMBERS_MATCH", "error", h_id,
                        f"{h_id} cites evidence from {d_id} but NONE of "
                        f"its numbers appear in that delegation's report — "
                        f"the claim is ungrounded. Cite at least one "
                        f"measured value from the report (derived counts or "
                        f"classifications you computed yourself can stay "
                        f"alongside it). Numbers cited: {numbers}."))
        if current == "SUPPORTED":
            attacked = any(
                r.get("is_falsification_attempt")
                and h_id in (r.get("hypothesis_ids") or [])
                for r in completed)
            if not attacked:
                crit = h.get("falsification_criterion", "")
                out.append(Violation(
                    "SUPPORTED_WITHOUT_ATTACK", "warn", h_id,
                    f"{h_id} is SUPPORTED but no delegation flagged "
                    "is_falsification_attempt targets it. Criterion: "
                    f"\"{crit}\". Delegate a falsification attempt "
                    "or downgrade to OPEN "
                    "(reopening requires evidence citing the delegation"
                    " that produced the current status)."))
        # posterior inertia on the latest closing entry
        if len(log) >= 2 and current in (
                "SUPPORTED", "FALSIFIED", "INCONCLUSIVE"):
            prev_belief = None
            for entry in log[:-1][::-1]:
                if entry.get("posterior") is not None:
                    prev_belief = entry["posterior"]
                    break
            post = log[-1].get("posterior")
            if (_posterior_inertia_enabled()
                    and prev_belief is not None and post is not None
                    and abs(post - prev_belief) < POSTERIOR_EPSILON):
                out.append(Violation(
                    "POSTERIOR_INERTIA", "warn", h_id,
                    f"{h_id} changed status to {current} but belief "
                    f"barely moved ({prev_belief} → {post}). Either "
                    "the evidence is weak (keep it OPEN) or the "
                    "posterior is wrong — reconcile them. "
                    "To resolve: gather new evidence (new delegation)"
                    " and update with a posterior that reflects"
                    " it."))
        return out

    def _check_stale(self, hypotheses, completed):
        out = []
        recent = completed[-self._stale_k:]
        if len(completed) < self._stale_k:
            return out
        for h_id, h in hypotheses.items():
            log = h.get("status_log", [])
            current = log[-1]["status"] if log else "OPEN"
            if current != "OPEN":
                continue
            linked = any(
                h_id in (r.get("hypothesis_ids") or [])
                for r in recent)
            if not linked:
                out.append(Violation(
                    "STALE_OPEN", "warn", h_id,
                    f"{h_id} is OPEN but none of the last "
                    f"{self._stale_k} delegations addressed it. "
                    "Delegate a test of it or close it as "
                    "INCONCLUSIVE."))
        return out

    def _check_unanchored(self, hypotheses, completed):
        """Per OPEN hypothesis: find its most-recent linked completed
        delegation; fire if that delegation lacks ### Numbers content.
        Older bare delegations stop nagging once a newer linked one
        exists (self-healing).
        """
        out = []
        for h_id, h in hypotheses.items():
            log = h.get("status_log", [])
            if not log or log[-1]["status"] != "OPEN":
                continue
            # Find the most-recent completed delegation linked to h_id
            most_recent = None
            for r in reversed(completed):
                if h_id in (r.get("hypothesis_ids") or []):
                    most_recent = r
                    break
            if most_recent is None:
                continue
            deliverable = most_recent.get("deliverable", "")
            section = _NUMBERS_SECTION_RE.search(deliverable)
            has_numbers = bool(
                section
                and any(
                    _NUM_LINE_RE.match(ln)
                    for ln in section.group(1).splitlines()))
            # Per Charter §6, evidence need not be numeric — a qualitative
            # finding anchors a verdict too. Accept a substantive ### Conclusions
            # section as a concrete result. (The critic remains the semantic
            # judge of adequacy; this is only an early "did it produce anything"
            # nudge, so a generous accept here is fine — the gate backstops it.)
            _concl = _CONCLUSIONS_SECTION_RE.search(deliverable)
            has_conclusion = bool(
                _concl and len(_concl.group(1).strip()) >= _MIN_CONCLUSION_CHARS)
            if not (has_numbers or has_conclusion):
                out.append(Violation(
                    "UNANCHORED_DELEGATION", "warn", h_id,
                    f"Delegation {most_recent['id']} (most recent "
                    f"linked to {h_id}) completed with no concrete result "
                    "(no ### Numbers and no substantive ### Conclusions). A "
                    "verdict needs a concrete result to cite — a measurement, a "
                    "comparison, or a qualitative finding. Re-delegate for one."))
        return out

    def _check_unledgered(self, all_records: list[dict]) -> list[Violation]:
        """UNLEDGERED_EVALS: DONE delegations that reported evals but wrote
        no rows to the canonical store. No-ops when store_dir is None."""
        if self.store_dir is None:
            return []
        from pathlib import Path as _Path
        sd = _Path(self.store_dir)
        # Lazy summary via mtime cache — cheap on repeated calls.
        try:
            from .instrumented import RunStateSummary
            summary = RunStateSummary.from_store(sd)
        except Exception:  # noqa: BLE001
            summary = None
        rows_per_delegation: dict = {}
        if summary is not None:
            rows_per_delegation = summary.n_per_delegation

        out: list[Violation] = []
        for r in all_records:
            if r.get("status") != "DONE":
                continue
            evals = r.get("evals", 0) or 0
            if evals <= 0:
                continue
            d_id = r.get("id", "")
            if rows_per_delegation.get(d_id, 0) > 0:
                continue
            # Key on the delegation id (not None) so multiple
            # simultaneously-unledgered delegations each get their own
            # message + dedupe bookkeeping instead of collapsing to one.
            out.append(Violation(
                "UNLEDGERED_EVALS", "warn", d_id,
                f"Delegation {d_id} reported {evals} evaluation(s) but "
                "wrote none to the canonical ledger — it bypassed "
                "get_evaluator(). Fine for throwaway exploration, but any "
                "number that feeds the HEADLINE must come from ledgered "
                "rows or pipeline.py cannot reproduce it. If this "
                "delegation's results back a conclusion, re-run them via "
                "get_evaluator().",
            ))
        return out

    def _numbers_match(self, numbers: dict, deliverable: str) -> bool:
        """True iff the evidence is ANCHORED to the report: at least one
        cited value appears in the delegation's report.

        A hypothesis update legitimately MIXES raw measurements (which must
        trace to the report) with the strategizer's own DERIVED quantities
        — counts, classifications, reformatted coordinates — which by
        definition are absent from the worker's report. Demanding that every
        value match punishes interpretation; requiring at least one anchors
        the claim while still catching wholesale fabrication or a citation
        of the wrong delegation. The critic judges semantics.

        Prefers the ### Numbers section (key: value lines); falls back to
        the full text. Numeric match: rel tol 1e-3 (tolerates sensible
        rounding while still catching fabrication). Value-only (keys are
        not bound).
        """
        section = _NUMBERS_SECTION_RE.search(deliverable)
        haystack = section.group(1) if section else deliverable
        hay_floats = [float(m) for m in _FLOAT_RE.findall(haystack)]
        numeric_vals: list[float] = []
        string_vals: list[str] = []
        for value in numbers.values():
            try:
                numeric_vals.append(float(value))
            except (TypeError, ValueError):
                string_vals.append(str(value))
        # Anchored if any cited numeric value appears in the report.
        for v in numeric_vals:
            if any(
                abs(v - hv) <= 1e-3 * max(abs(v), abs(hv), 1e-12)
                for hv in hay_floats
            ):
                return True
        # No numeric values cited: anchor on any string token instead;
        # if there is nothing to anchor on at all, do not flag.
        if not numeric_vals:
            return (
                any(s in haystack for s in string_vals)
                if string_vals else True
            )
        return False

    def on_hypothesis_update(self, h_id: str) -> list[str]:
        """Hook after a successful ledger update. Returns error-severity
        messages for inline return to the agent."""
        live = self.evaluate()
        self._bookkeep(live, count_streaks=False)
        return [
            f"[SCIENCE MONITOR — {v.rule}] {v.message}"
            for v in live
            if v.severity == "error" and v.h_id == h_id
        ]

    def on_delegation_complete(self, delegation_id: str) -> None:
        """Hook after a delegation finishes; updates bookkeeping."""
        self._bookkeep(self.evaluate(), count_streaks=False)

    def drain(self) -> str:
        """Re-validate, dedupe, cap, digest. Returns injection text
        (possibly empty). Call from the node's _drain_notifications."""
        live = self.evaluate()
        self._bookkeep(live, count_streaks=True)
        if not live:
            return ""
        seen: dict[tuple, Violation] = {}
        for v in sorted(live, key=lambda v: v.severity != "error"):
            seen.setdefault(v.key, v)
        ordered = list(seen.values())
        head = ordered[: self._max_inject]
        rest = ordered[self._max_inject:]
        lines = [
            f"[SCIENCE MONITOR — {v.rule}] {v.message}" for v in head
        ]
        if rest:
            digest = ", ".join(
                f"{v.rule}({v.h_id})" for v in rest)
            lines.append(
                f"[SCIENCE MONITOR] +{len(rest)} more: {digest}")
        return "\n".join(lines) + "\n"

    def _bookkeep(
        self,
        live: list[Violation],
        count_streaks: bool = False,
    ) -> None:
        """Update diagnostics log, h_rule_seen, and (optionally) streaks.

        Streak counting is intentionally restricted to drain() calls so
        that a persistent error is not double-counted within a single
        agent turn (on_hypothesis_update + drain would both increment).
        Semantics: "persisting across N drain calls".
        """
        with self._lock:
            live_keys = {v.key for v in live}
            for v in live:
                if v.key not in self._logged_keys:
                    self._logged_keys.add(v.key)
                    if self._diag is not None:
                        self._diag({
                            "rule": v.rule, "severity": v.severity,
                            "hypothesis_id": v.h_id,
                            "message": v.message,
                        })
                if v.h_id is not None:
                    self._h_rule_seen.setdefault(
                        v.h_id, set()).add(v.rule)
                if count_streaks and v.severity == "error":
                    self._error_streak[v.key] = \
                        self._error_streak.get(v.key, 0) + 1
            # resolved keys may re-fire later → re-log + reset streaks
            for key in list(self._logged_keys):
                if key not in live_keys:
                    self._logged_keys.discard(key)
                    self._error_streak.pop(key, None)
            if self._escalations >= self._escalation_cap:
                self._pending_escalation = None
                return
            offenders = sorted(
                h for h, rules in self._h_rule_seen.items()
                if len(rules) >= ESCALATE_AFTER_VIOLATIONS)
            offenders += sorted(
                {key[1] for key, n in self._error_streak.items()
                 if n >= ESCALATE_AFTER_ERROR_REPEATS
                 and key[1] is not None
                 and key[1] not in offenders})
            self._pending_escalation = offenders or None

    def escalation_due(self) -> list[str] | None:
        with self._lock:
            return self._pending_escalation

    def note_escalated(self) -> None:
        with self._lock:
            self._escalations += 1
            self._pending_escalation = None
            self._h_rule_seen.clear()
            self._error_streak.clear()
