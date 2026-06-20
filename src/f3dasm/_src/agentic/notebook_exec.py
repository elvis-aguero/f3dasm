"""Execute the run deliverable — `pipeline.ipynb` — under the reproduction gate.

The system is committed to the notebook: `pipeline.ipynb` is THE deliverable.
The gate's contract (exit-clean + ZERO new oracle evals + the `REPRODUCED:`
headline grounded in the ledger) executes it lazily via nbclient. The nbclient/
nbformat/ipykernel deps ship with the `agentic` extra; `run_deliverable()`
returns a `subprocess.CompletedProcess`-shaped result and raises
`subprocess.TimeoutExpired` on timeout, so the gate branches on nothing.

Agents author the notebook cell-by-cell via the structured AddPipelineCell /
SetNotebookIntro closures (pure nbformat) or via the live Jupyter MCP server
(mcp__jupyter__* tools) when a JupyterServer is running for the session.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path

__all__ = [
    "notebook_available",
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
def notebook_deliverable_spec(role: str = "strategizer") -> str:
    """The deliverable contract — injected into the relevant agent prompts so
    the .ipynb is a reproducible scientific narrative, not a ported .py.

    Role-aware: ONLY the strategizer authors the notebook (it alone is granted
    SetNotebookIntro / AddPipelineCell), so only it gets the "author with these
    tools" imperative. The implementer (writes phase code to its workspace) and
    the critic (judges the notebook) get the same STRUCTURE + RULES so their
    work fits / is judged against it — but no instruction to call tools they do
    not have."""
    if role == "strategizer":
        intro = (
            "DELIVERABLE = pipeline.ipynb. This SUPERSEDES every 'pipeline.py' /\n"
            "WriteDeliverable('pipeline.py') / solution.md instruction above —\n"
            "for THIS run there is exactly one deliverable, pipeline.ipynb, and\n"
            "no pipeline.py and no solution.md. It is the single merged artifact\n"
            "— the writeup AND the runnable, lazily-reproducible recipe in one. A\n"
            "SCIENTIFIC NARRATIVE, not a dumped script. AUTHOR IT WITH THE\n"
            "STRUCTURED TOOLS — they make the structure unforgeable plumbing:\n"
            "  - SetNotebookIntro(problem, hypotheses): the two leading narrative\n"
            "    cells (call once, early).\n"
            "  - AddPipelineCell(phase, why, code): one pillar cell + its REQUIRED\n"
            "    WHY-explainer; phase in {doe, data_generation, ml, optimization,\n"
            "    analysis}. Cells stay in canonical order; re-calling a phase\n"
            "    replaces it. Because the pillar name and the rationale are\n"
            "    required arguments, you cannot ship a structureless notebook or\n"
            "    omit the WHY. Do NOT hand-write notebook JSON.\n"
            "    (WriteDeliverable('pipeline.ipynb', …) is a raw fallback only.)\n\n"
        )
    elif role == "implementer":
        intro = (
            "DELIVERABLE CONTEXT: the run's single deliverable is pipeline.ipynb,\n"
            "which the STRATEGIZER assembles from your work (you do NOT author it\n"
            "— you have no notebook tools). Write your phase code so it drops\n"
            "cleanly into one of the pillar cells below; return runnable,\n"
            "non-stub code that reaches the oracle ONLY via get_evaluator().\n\n"
        )
    else:  # critic / reviewer
        intro = (
            "DELIVERABLE CONTEXT: the run's single deliverable is pipeline.ipynb\n"
            "(no pipeline.py, no solution.md — the notebook's markdown IS the\n"
            "writeup). You JUDGE it against the structure below; you do not author\n"
            "it.\n\n"
        )
    return (
        "\n<deliverable_format>\n"
        + intro +
        "STRUCTURE (the Popperian spine + f3dasm's four pillars). Each code cell\n"
        "carries name metadata = its pillar so the structure is machine-checkable:\n"
        "  1. md  '# Problem & objective'  — question, min/max, success criterion.\n"
        "  2. md  '## Hypotheses'          — registered hypotheses + falsifiable\n"
        "         predictions (the Popperian setup; mirror the hypothesis ledger).\n"
        "  3. md WHY-explainer + code name='doe'             — Domain + sampler\n"
        "         (LOAD-OR-CREATE: load the ledger if present, else build the DoE).\n"
        "  4. md WHY-explainer + code name='data_generation' — LAZY eval pattern:\n"
        "         canonical_store = os.environ['F3DASM_CANONICAL_STORE']\n"
        "         try:\n"
        "             from f3dasm.agentic import get_evaluator\n"
        "             gen = get_evaluator()   # raises ValueError if oracle not registered\n"
        "             gen.call(data, mode='sequential')  # skips FINISHED rows → 0 new\n"
        "         except ValueError:\n"
        "             data = ExperimentData.from_file(project_dir=canonical_store)\n"
        "             assert len(data) > 0, 'Canonical store empty and no oracle registered'\n"
        "         (Never use f3dasm._src.* paths — internal, unversioned, not public.)\n"
        "         @datagenerator SCALAR RULE: if you use @datagenerator(output_names=['y'])\n"
        "         directly, return the scalar value itself — NOT a dict. 'return val' not\n"
        "         'return {\"y\": val}'. The output_names mapping already handles the name.\n"
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
        "- NEVER call data.store() after evaluator.call(). The InstrumentedDataGenerator\n"
        "  behind get_evaluator() already writes FINISHED rows to the canonical store.\n"
        "  Calling data.store() afterwards overwrites those FINISHED rows with\n"
        "  IN_PROGRESS — silently corrupting the ledger. Reload if you need the\n"
        "  updated outputs: ExperimentData.from_file(project_dir=canonical_store).\n"
        "- GUARD OPTIONAL IMPORTS. Heavy packages (torch, botorch, gpytorch, jax)\n"
        "  are NOT guaranteed to be installed. An unconditional 'import torch' at the\n"
        "  top of a cell causes returncode=1 and forces a gate-bounce correction\n"
        "  delegation. Always guard: 'import importlib.util; if\n"
        "  importlib.util.find_spec(\"torch\") is None: # use sklearn/scipy fallback'.\n"
        "</deliverable_format>\n"
    )


def notebook_available() -> bool:
    """True iff nbclient + nbformat are importable (ship with the agentic extra)."""
    try:
        import nbclient  # noqa: F401
        import nbformat  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def required_deliverable_name() -> str:
    """The single run deliverable. The system is committed to the notebook —
    pipeline.ipynb is THE deliverable (there is no pipeline.py / solution.md)."""
    return "pipeline.ipynb"


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
