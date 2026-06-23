"""Commit 1: eval_budget + mem_cap are plumbed into run_config.json + env so the
in-process governor (at the eval boundary) can read them — they previously lived
only in graph state, invisible to a campaign process."""
from __future__ import annotations

import json
import os

from f3dasm._src.agentic.agent_runtime import (
    DEFAULT_MEM_CAP_BYTES,
    _init_canonical_store,
)


def test_run_config_carries_eval_budget_and_mem_cap(tmp_path, monkeypatch):
    monkeypatch.delenv("F3DASM_EVAL_BUDGET", raising=False)
    monkeypatch.delenv("F3DASM_MEM_CAP", raising=False)
    run_dir = tmp_path / "run"
    (run_dir / "debug").mkdir(parents=True)
    study_dir = tmp_path / "study"
    study_dir.mkdir()

    cfg = _init_canonical_store(
        run_dir, study_dir, evaluator_config=None,
        eval_budget=1000, mem_cap_bytes=DEFAULT_MEM_CAP_BYTES,
    )
    # returned dict + the on-disk sidecar both carry the knobs
    assert cfg["eval_budget"] == 1000
    assert cfg["mem_cap_bytes"] == DEFAULT_MEM_CAP_BYTES
    on_disk = json.loads((run_dir / "debug" / "run_config.json").read_text())
    assert on_disk["eval_budget"] == 1000
    assert on_disk["mem_cap_bytes"] == DEFAULT_MEM_CAP_BYTES
    # exported to env so child campaign processes inherit them
    assert os.environ["F3DASM_EVAL_BUDGET"] == "1000"
    assert os.environ["F3DASM_MEM_CAP"] == str(DEFAULT_MEM_CAP_BYTES)


def test_none_budget_does_not_set_env(tmp_path, monkeypatch):
    monkeypatch.delenv("F3DASM_EVAL_BUDGET", raising=False)
    run_dir = tmp_path / "run"
    (run_dir / "debug").mkdir(parents=True)
    cfg = _init_canonical_store(run_dir, tmp_path / "s", eval_budget=None)
    assert cfg["eval_budget"] is None
    assert "F3DASM_EVAL_BUDGET" not in os.environ


def test_default_mem_cap_is_sane():
    # 4 GiB: above a healthy campaign, below the runaway blowup.
    assert DEFAULT_MEM_CAP_BYTES == 4 * 1024 ** 3
