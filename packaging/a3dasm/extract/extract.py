"""Reproducible extraction of a3dasm from the f3dasm agentic fork.

Builds a standalone a3dasm tree that depends on stock f3dasm (no vendored core).
Validated: the full suite passes against f3dasm 2.2.3 from PyPI (1227 passed).

Usage:
    python packaging/a3dasm/extract/extract.py [DEST]

DEST defaults to a sibling `a3dasm-build/` next to the f3dasm repo. This is the
lightweight copy+rename path (fast, no history). For a history-preserving move,
use the `git filter-repo` recipe in packaging/a3dasm/PORTING.md instead; the
rename rules below are the same either way.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[3]                       # the f3dasm repo root
OVERRIDES = HERE.parent / "overrides"       # a3dasm-specific test rewrites
DEST = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC.parent / "a3dasm-build"


def build() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    (DEST / "src" / "a3dasm").mkdir(parents=True)
    (DEST / "tests").mkdir(parents=True)

    # source: agentic package -> src/a3dasm/_src ; public shim -> package root
    shutil.copytree(SRC/"src/f3dasm/_src/agentic", DEST/"src/a3dasm/_src")
    shutil.copy(SRC/"src/f3dasm/agentic/__init__.py", DEST/"src/a3dasm/__init__.py")
    mm = SRC/"src/f3dasm/agentic/__main__.py"
    if mm.exists():
        shutil.copy(mm, DEST/"src/a3dasm/__main__.py")

    # tests + the shared root conftest
    shutil.copytree(SRC/"tests/agentic", DEST/"tests", dirs_exist_ok=True)
    shutil.copy(SRC/"tests/conftest.py", DEST/"tests/conftest.py")

    # standalone-repo scaffolding (pyproject, mkdocs, CI, docs shell, meta)
    shutil.copytree(SRC/"packaging/a3dasm", DEST, dirs_exist_ok=True)
    shutil.rmtree(DEST/"extract", ignore_errors=True)  # dev-only, not shipped

    # ported a3dasm docs content (FEATURES/BACKLOG/authoring/example_study/specs)
    shutil.copytree(SRC/"docs/agentic", DEST/"docs", dirs_exist_ok=True)

    # studies: shared modules only (skip heavy run artifacts + non-agentic work)
    (DEST/"studies").mkdir(exist_ok=True)
    for name in ["run_ledger.py", "run_ledger.csv", "audit_run.py",
                 "benchmark_embeddings.py"]:
        p = SRC/"studies"/name
        if p.exists():
            shutil.copy(p, DEST/"studies"/name)
    if (SRC/"studies/_oracle_src").exists():
        shutil.copytree(SRC/"studies/_oracle_src", DEST/"studies/_oracle_src",
                        dirs_exist_ok=True)

    _prune(DEST)
    _rewrite(DEST)

    # a3dasm-specific test rewrites (semantics changed, not just renamed)
    for ov in OVERRIDES.rglob("*.py"):
        rel = ov.relative_to(OVERRIDES)
        (DEST/rel).write_text(ov.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"built a3dasm at {DEST}")


def _prune(dest: Path) -> None:
    for p in list(dest.rglob("__pycache__")):
        shutil.rmtree(p, ignore_errors=True)
    for p in list(dest.rglob("*.pyc")):
        p.unlink()


def _rewrite(dest: Path) -> None:
    # namespace rename (agentic layer -> a3dasm); f3dasm CORE refs are untouched.
    subs = [
        (re.compile(r"\.\._src\.agentic\b"), "._src"),      # relative reach (shim)
        (re.compile(r"f3dasm\._src\.agentic\b"), "a3dasm._src"),
        (re.compile(r"f3dasm\.agentic\b"), "a3dasm"),
        (re.compile(r"\btests\.agentic\b"), "tests"),
    ]
    # tests moved up one level (tests/agentic -> tests) and the layout renamed.
    path_fixes = [
        (re.compile(r"parents\[2\]"), "parents[1]"),
        (re.compile(r"\.parent\.parent\.parent\b"), ".parent.parent"),
        (re.compile(r'"src"\s*/\s*"f3dasm"\s*/\s*"_src"\s*/\s*"agentic"'),
         '"src" / "a3dasm" / "_src"'),
        (re.compile(r'"docs"\s*/\s*"agentic"'), '"docs"'),
    ]
    n = 0
    files = (list((dest/"src").rglob("*.py")) + list((dest/"tests").rglob("*.py"))
             + list((dest/"src").rglob("*.md")) + list((dest/"docs").rglob("*.md")))
    for f in files:
        t = o = f.read_text(encoding="utf-8")
        for rx, rep in subs:
            t = rx.sub(rep, t)
        if "tests" in f.parts and f.suffix == ".py":
            for rx, rep in path_fixes:
                t = rx.sub(rep, t)
        if t != o:
            f.write_text(t, encoding="utf-8")
            n += 1
    print(f"rewrote {n} files")


if __name__ == "__main__":
    build()
