"""Non-blocking raw-oracle-access nudge.

A worker that reaches the ground-truth oracle directly (e.g. `from evaluator
import evaluate`, or loading evaluator.dylib) instead of via get_evaluator()
gets a just-in-time reminder appended to its tool output — capped twice per
delegation. The nudge must NEVER fire on the correct get_evaluator() path.
"""
from __future__ import annotations

from f3dasm._src.agentic.backends.base import (
    ORACLE_NUDGE_CAP,
    OracleNudgeBudget,
    detect_raw_oracle_access,
)


class TestDetector:
    def test_fires_on_from_evaluator_import(self):
        msg = detect_raw_oracle_access(
            "Bash", {"command": "python3 -c 'from evaluator import evaluate'"})
        assert msg and "ORACLE ACCESS" in msg

    def test_fires_on_import_evaluator(self):
        assert detect_raw_oracle_access("Bash", {"command": "import evaluator"})

    def test_fires_on_dylib_load(self):
        assert detect_raw_oracle_access(
            "Write", {"content": "ctypes.CDLL('workspace/evaluator.dylib')"})

    def test_fires_on_so_load(self):
        assert detect_raw_oracle_access(
            "Bash", {"command": "ldd evaluator.so"})

    def test_fires_on_write_content(self):
        script = "import numpy\nfrom evaluator import evaluate\nevaluate(x)\n"
        assert detect_raw_oracle_access("Write", {"content": script})

    # ---- must NOT fire on the correct path or innocent text ----------------
    def test_silent_on_get_evaluator(self):
        cmd = ("from f3dasm.agentic import get_evaluator\n"
               "gen = get_evaluator()\ndata = data.run(data_generator=gen)\n"
               "gen.flush()")
        assert detect_raw_oracle_access("Bash", {"command": cmd}) is None

    def test_silent_on_import_get_evaluator_only(self):
        assert detect_raw_oracle_access(
            "Write", {"content": "import get_evaluator  # not the module"}
        ) is None

    def test_silent_on_plain_code(self):
        assert detect_raw_oracle_access(
            "Bash", {"command": "ls -la && python train_surrogate.py"}) is None

    def test_silent_on_non_bash_write_tool(self):
        assert detect_raw_oracle_access(
            "Read", {"command": "from evaluator import evaluate"}) is None

    def test_silent_on_empty(self):
        assert detect_raw_oracle_access("Bash", {}) is None
        assert detect_raw_oracle_access("Bash", {"command": ""}) is None

    def test_robust_to_non_dict(self):
        assert detect_raw_oracle_access("Bash", None) is None


class TestBudgetCap:
    def test_cap_is_two(self):
        assert ORACLE_NUDGE_CAP == 2

    def test_fires_up_to_cap_then_silent(self):
        budget = OracleNudgeBudget()
        ti = {"command": "from evaluator import evaluate"}
        assert budget.check("Bash", ti) is not None   # 1
        assert budget.check("Bash", ti) is not None    # 2
        assert budget.check("Bash", ti) is None        # capped

    def test_non_matching_calls_do_not_consume_budget(self):
        budget = OracleNudgeBudget()
        assert budget.check("Bash", {"command": "ls"}) is None
        assert budget.check("Bash", {"command": "echo hi"}) is None
        # still has full budget for real hits
        ti = {"command": "from evaluator import evaluate"}
        assert budget.check("Bash", ti) is not None
        assert budget.check("Bash", ti) is not None
        assert budget.check("Bash", ti) is None

    def test_reset_restores_budget(self):
        budget = OracleNudgeBudget()
        ti = {"command": "from evaluator import evaluate"}
        budget.check("Bash", ti)
        budget.check("Bash", ti)
        assert budget.check("Bash", ti) is None
        budget.reset()
        assert budget.check("Bash", ti) is not None


class TestOllamaWiring:
    """The Ollama bash/write closures append the nudge to their output."""

    def test_bash_tool_appends_nudge(self):
        from f3dasm._src.agentic.backends.ollama import _make_bash_tool
        budget = OracleNudgeBudget()
        tool = _make_bash_tool(None, budget.check)
        # `:` is a no-op shell builtin; the pattern lives in a comment so the
        # detector fires while the command does nothing.
        out = tool.func(command=": # from evaluator import evaluate")
        assert "ORACLE ACCESS" in out

    def test_bash_tool_no_nudge_on_clean_command(self):
        from f3dasm._src.agentic.backends.ollama import _make_bash_tool
        budget = OracleNudgeBudget()
        tool = _make_bash_tool(None, budget.check)
        out = tool.func(command="echo hello")
        assert "ORACLE ACCESS" not in out
        assert "hello" in out

    def test_write_tool_appends_nudge_and_still_writes(self, tmp_path):
        from f3dasm._src.agentic.backends.ollama import _make_write_tool
        budget = OracleNudgeBudget()
        tool = _make_write_tool(tmp_path, budget.check)
        out = tool.func(
            path="phase1.py",
            content="from evaluator import evaluate\nevaluate([1, 2])\n",
        )
        assert "ORACLE ACCESS" in out
        assert (tmp_path / "phase1.py").exists()

    def test_no_nudge_when_callable_absent(self, tmp_path):
        from f3dasm._src.agentic.backends.ollama import _make_bash_tool
        tool = _make_bash_tool(None)  # nudge defaults to None
        out = tool.func(command=": # from evaluator import evaluate")
        assert "ORACLE ACCESS" not in out
