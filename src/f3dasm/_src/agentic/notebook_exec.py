"""Execute the run deliverable — a `pipeline.py` OR a `pipeline.ipynb` — under the
reproduction gate, with ONE executor-agnostic entry point.

The reproduction gate's contract (exit-clean + ZERO new oracle evals + the
`REPRODUCED:` headline grounded in the ledger) is unchanged; only the *execution
mechanism* differs by deliverable suffix. `run_deliverable()` returns a
`subprocess.CompletedProcess`-shaped result (`returncode`, `stdout`, `stderr`)
for BOTH, and raises `subprocess.TimeoutExpired` on timeout for BOTH, so the gate
branches on nothing.

Notebook mode is opt-in: enabled only when the `notebook_deliverable` runtime
flag is set AND the `agentic-notebook` extra (nbclient/nbformat/ipykernel) is
importable. A non-notebook install keeps shipping/gating `pipeline.py`.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path

from . import settings

__all__ = [
    "notebook_available",
    "notebook_mode_enabled",
    "required_deliverable_name",
    "run_deliverable",
    "build_notebook",
    "notebook_deliverable_spec",
]


# The canonical notebook structure — ONE source, referenced by the agent prompt
# (notebook_deliverable_spec), KB 0009, and any fallback builder. The spine is the
# Popperian loop (hypotheses → falsification attempt → verdict); the body mirrors
# f3dasm's four pillars (Phase enum: doe / data_generation / ml / optimization),
# each a name-tagged code cell preceded by a WHY explainer markdown cell.
def notebook_deliverable_spec() -> str:
    """The deliverable contract for notebook mode — injected into the relevant
    agent prompts so the .ipynb is a reproducible scientific narrative, not a
    ported .py. Mode-specific: only shown when notebook mode is active."""
    return (
        "\n<deliverable_format>\n"
        "DELIVERABLE = pipeline.ipynb. This SUPERSEDES every 'pipeline.py' /\n"
        "WriteDeliverable('pipeline.py') / solution.md instruction above — for\n"
        "THIS run there is exactly one deliverable, pipeline.ipynb, and no\n"
        "pipeline.py and no solution.md. It is the single merged\n"
        "artifact — the writeup AND the runnable, lazily-reproducible recipe in\n"
        "one. It is a SCIENTIFIC NARRATIVE, not a dumped script. Author it with\n"
        "WriteDeliverable('pipeline.ipynb', <nbformat-v4 JSON>) (or the Jupyter\n"
        "tools when available); never hand-write fragile JSON — build valid\n"
        "nbformat v4.\n\n"
        "STRUCTURE (the Popperian spine + f3dasm's four pillars). Each code cell\n"
        "carries name metadata = its pillar so the structure is machine-checkable:\n"
        "  1. md  '# Problem & objective'  — question, min/max, success criterion.\n"
        "  2. md  '## Hypotheses'          — registered hypotheses + falsifiable\n"
        "         predictions (the Popperian setup; mirror the hypothesis ledger).\n"
        "  3. md WHY-explainer + code name='doe'             — Domain + sampler\n"
        "         (LOAD-OR-CREATE: load the ledger if present, else build the DoE).\n"
        "  4. md WHY-explainer + code name='data_generation' — evaluate via\n"
        "         get_evaluator() ONLY (lazy: skips FINISHED rows → 0 new on re-run).\n"
        "  5. md WHY-explainer + code name='ml'              — fit the surrogate.\n"
        "  6. md WHY-explainer + code name='optimization'    — acquisition / BO loop.\n"
        "  7. md '## Verdict & result' + code name='analysis' — for each hypothesis\n"
        "         state SUPPORTED/FALSIFIED + WHY from the evidence; derive the\n"
        "         headline FROM the ledger and print exactly 'REPRODUCED: <value>'.\n\n"
        "RULES:\n"
        "- The four pillar cells (doe/data_generation/ml/optimization) are ALWAYS\n"
        "  present. A pillar you did NOT run stays present but its explainer says\n"
        "  plainly 'NOT executed (budget)'. Never silently drop a pillar.\n"
        "- Every WHY-explainer justifies the methodological choice (cite the\n"
        "  literature you gathered) — this is the rationale, not just description.\n"
        "- LAZY + reproducible: the runtime executes the notebook against the\n"
        "  shipped ledger and requires ZERO new oracle evals + the REPRODUCED line\n"
        "  grounded in the ledger. Reach the oracle ONLY via get_evaluator().\n"
        "</deliverable_format>\n"
    )


def notebook_available() -> bool:
    """True iff the agentic-notebook extra (nbclient + nbformat) is importable."""
    try:
        import nbclient  # noqa: F401
        import nbformat  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def notebook_mode_enabled() -> bool:
    """Notebook deliverable is active iff the runtime flag is set AND the extra is
    importable. Default off — a plain install/run stays on pipeline.py."""
    return settings.get_bool("notebook_deliverable", False) and notebook_available()


def required_deliverable_name() -> str:
    """The single required deliverable filename for the active mode."""
    return "pipeline.ipynb" if notebook_mode_enabled() else "pipeline.py"


@contextlib.contextmanager
def _patched_environ(env: dict | None):
    """Temporarily replace os.environ with `env` so a freshly-spawned Jupyter
    kernel (which inherits os.environ at launch) sees the gate's injected vars
    (F3DASM_CANONICAL_STORE / F3DASM_RUN_CONFIG / F3DASM_DELEGATION_ID)."""
    if env is None:
        yield
        return
    saved = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(env)
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _execute_notebook(path: Path, cwd: Path, env: dict, timeout: float):
    """Execute a notebook in THIS interpreter's env (so f3dasm/get_evaluator
    import) and return a CompletedProcess-shaped result. Raises
    subprocess.TimeoutExpired on timeout (mirrors the .py path)."""
    import nbformat
    from jupyter_client.manager import KernelManager
    from nbclient import NotebookClient
    from nbclient.exceptions import CellTimeoutError, DeadKernelError

    nb = nbformat.read(str(path), as_version=4)
    # Pin the kernel to THIS interpreter so the notebook runs in the env that
    # has f3dasm installed (the global "python3" kernelspec may point elsewhere).
    km = KernelManager(kernel_name="python3")
    try:
        km.kernel_spec.argv[0] = sys.executable
    except Exception:  # noqa: BLE001 — fall back to the spec's python
        pass
    client = NotebookClient(
        nb, km=km, timeout=int(timeout),
        allow_errors=True,                      # capture errors, don't raise
        resources={"metadata": {"path": str(cwd)}},
    )
    with _patched_environ(env):
        try:
            client.execute()
        except (CellTimeoutError, DeadKernelError) as exc:
            raise subprocess.TimeoutExpired(cmd=str(path), timeout=timeout) from exc

    out_parts, err_parts, errored = [], [], False
    for cell in nb.cells:
        for o in cell.get("outputs", []):
            ot = o.get("output_type")
            if ot == "stream":
                (out_parts if o.get("name") == "stdout" else err_parts).append(
                    o.get("text", ""))
            elif ot == "error":
                errored = True
                err_parts.append(
                    f"{o.get('ename', '')}: {o.get('evalue', '')}\n"
                    + "\n".join(o.get("traceback", [])))
    return subprocess.CompletedProcess(
        args=[str(path)],
        returncode=1 if errored else 0,
        stdout="".join(out_parts),
        stderr="".join(err_parts),
    )


def run_deliverable(path: Path, *, cwd: Path, env: dict, timeout: float):
    """Run the deliverable; return a subprocess.CompletedProcess. `.ipynb` →
    nbclient (in-env kernel); anything else → `python <file>` subprocess.
    Raises subprocess.TimeoutExpired on timeout in BOTH paths."""
    path = Path(path)
    if path.suffix == ".ipynb" and notebook_available():
        return _execute_notebook(path, Path(cwd), env, timeout)
    return subprocess.run(
        [sys.executable, str(path)],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout,
    )


def build_notebook(cells: list[dict]):
    """Assemble a notebook from a simple cell list (avoids hand-written JSON).

    Each cell: {"type": "markdown"|"code", "source": str, "name": str|None}.
    The optional ``name`` is written to cell.metadata.name AND cell.metadata.tags
    (mirrors the f3dasm Phase: doe/data_generation/ml/optimization/analysis), so
    phase-presence is machine-checkable. Returns an nbformat NotebookNode.
    """
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    built = []
    for c in cells:
        src = c.get("source", "")
        if c.get("type") == "markdown":
            cell = nbformat.v4.new_markdown_cell(src)
        else:
            cell = nbformat.v4.new_code_cell(src)
        name = c.get("name")
        if name:
            cell.metadata["name"] = name
            cell.metadata["tags"] = [name]
        built.append(cell)
    nb.cells = built
    return nb
