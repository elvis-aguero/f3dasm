"""Instrumented DataGenerator: canonical concurrency-safe eval ledger.

Provides :class:`InstrumentedDataGenerator`, a wrapper that:

1. Delegates execution to an inner ``DataGenerator``.
2. Stamps provenance metadata (delegation_id, source, UTC timestamp)
   onto each returned ``ExperimentSample``.
3. Buffers samples and flushes them to a shared ``ExperimentData`` store
   under a ``FileLock`` so concurrent delegations cannot corrupt the
   ledger.
4. Writes a running eval count to a per-delegation counter file.

Also provides :func:`get_evaluator`, a factory that reads
``run_config.json`` from the delegation workspace and returns a
configured :class:`InstrumentedDataGenerator`.

**Phase 1 scope** — wiring into agent_runtime/nodes is a separate task.
The ``fidelity_column`` parameter is accepted but unused in Phase 1;
it exists for forward-compatibility when fidelity-aware stamping is
added.
"""
from __future__ import annotations

#                                                                      Modules
# ==========================================================================

import json
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from filelock import FileLock

from ..core import DataGenerator
from ..design.domain import Domain
from ..errors import EmptyFileError, ReachMaximumTriesError
from ..experimentdata import ExperimentData
from ..experimentsample import ExperimentSample, JobStatus

#                                                         Authorship & Credits
# ==========================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# ==========================================================================

_DELEGATION_ID_RE = re.compile(r"^D\d+$")


# ==========================================================================


class InstrumentedDataGenerator(DataGenerator):
    """Wrap an inner DataGenerator, stamp provenance, and flush to disk.

    Parameters
    ----------
    inner : DataGenerator
        The wrapped evaluator.  Callers are responsible for decorating a
        plain function with ``@datagenerator`` before passing it here.
    store_dir : Path or str
        Canonical project_dir — the directory that *contains*
        ``experiment_data/``.  Every delegation shares the same
        ``store_dir`` so their rows end up in one ledger.
    delegation_id : str
        Identifier for this delegation, e.g. ``"D003"``.
    source : str, optional
        Human-readable label stamped in the ``source`` provenance column
        (e.g. the evaluator name).  Default ``""``.
    fidelity_column : str or None, optional
        Name of the study's fidelity input column if any.  Unused in
        Phase 1 — accepted only for forward-compatibility.
    counter_path : Path or str or None, optional
        File to overwrite with the running eval count after each
        ``execute``.  If ``None`` no counter file is written.
    lock_path : Path or str or None, optional
        Path for the ``FileLock``.  Defaults to
        ``<store_dir>/experiment_data/.lock``.
    flush_every : int, optional
        Number of samples to buffer before a locked flush.  Default 1
        (flush on every execute).
    """

    def __init__(
        self,
        inner: DataGenerator,
        store_dir: Path | str,
        delegation_id: str,
        *,
        source: str = "",
        fidelity_column: Optional[str] = None,
        counter_path: Optional[Path | str] = None,
        lock_path: Optional[Path | str] = None,
        flush_every: int = 1,
    ) -> None:
        self.inner = inner
        self.store_dir = Path(store_dir)
        self.delegation_id = delegation_id
        self.source = source
        self.fidelity_column = fidelity_column  # unused Phase 1
        self.counter_path = (
            Path(counter_path) if counter_path is not None else None
        )
        self.flush_every = flush_every

        if lock_path is None:
            lock_path = (
                self.store_dir / "experiment_data" / ".lock"
            )
        self.lock_path = Path(lock_path)

        self._buffer: list[ExperimentSample] = []
        # Seed from an existing counter so multiple generator
        # instances within ONE delegation accumulate rather than
        # overwrite (observed live: a worker built one generator per
        # phase and the counter undercounted vs the store).
        self._eval_count: int = 0
        if self.counter_path is not None:
            try:
                self._eval_count = int(
                    self.counter_path.read_text().strip()
                )
            except (FileNotFoundError, ValueError, OSError):
                self._eval_count = 0

    # ------------------------------------------------------------------

    def execute(
        self, experiment_sample: ExperimentSample, **kwargs
    ) -> ExperimentSample:
        """Run inner generator, stamp provenance, buffer, maybe flush.

        Parameters
        ----------
        experiment_sample : ExperimentSample
            Sample to evaluate.
        **kwargs
            Forwarded to ``inner.execute``.

        Returns
        -------
        ExperimentSample
            The evaluated sample (with provenance stamped into
            ``_output_data``).
        """
        out = self.inner.execute(experiment_sample, **kwargs)

        # Stamp provenance into the output dict.
        ts = datetime.now(tz=timezone.utc).isoformat(
            timespec="seconds"
        )
        out._output_data["_delegation_id"] = self.delegation_id
        out._output_data["source"] = self.source
        out._output_data["_ts"] = ts

        self._buffer.append(deepcopy(out))
        self._eval_count += 1

        if len(self._buffer) >= self.flush_every:
            self._flush()

        if self.counter_path is not None:
            self.counter_path.parent.mkdir(parents=True, exist_ok=True)
            self.counter_path.write_text(str(self._eval_count))

        return out

    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush any remaining buffered samples to the store.

        Call at the end of a delegation to ensure no samples are lost.
        """
        if self._buffer:
            self._flush()

    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """Flush the current buffer to disk under a FileLock.

        The entire read → merge → write sequence is executed inside the
        lock so concurrent threads/processes cannot interleave.
        """
        if not self._buffer:
            return

        # Ensure the lock parent directory exists.
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        batch_domain = self._build_batch_domain()
        batch = self._build_batch_experimentdata(batch_domain)

        with FileLock(str(self.lock_path)):
            # Absent store (FileNotFoundError) OR a torn/empty CSV from an
            # interrupted prior write (EmptyFileError / retry-exhausted):
            # treat as fresh and let this locked write heal it. We do NOT
            # catch broader errors — a populated store that fails to parse
            # must propagate, never be silently overwritten with the batch.
            try:
                canon = ExperimentData.from_file(
                    project_dir=self.store_dir
                )
            except (
                FileNotFoundError,
                EmptyFileError,
                ReachMaximumTriesError,
            ):
                canon = ExperimentData(domain=batch_domain)

            # Ensure provenance columns are declared on the canon domain.
            for col in ("_delegation_id", "source", "_ts"):
                canon._domain.add_output(col, exist_ok=True)

            merged = canon + batch
            merged.store(project_dir=self.store_dir)

        self._buffer.clear()

    # ------------------------------------------------------------------

    def _build_batch_domain(self) -> Domain:
        """Build a Domain that covers inner outputs + provenance cols."""
        d = Domain()
        # Collect all output keys from the buffer.
        all_keys: set[str] = set()
        for sample in self._buffer:
            all_keys.update(sample._output_data.keys())
        for key in sorted(all_keys):
            d.add_output(key, exist_ok=True)
        return d

    def _build_batch_experimentdata(
        self, domain: Domain
    ) -> ExperimentData:
        """Build an ExperimentData from the current buffer."""
        data: dict[int, ExperimentSample] = {
            i: sample for i, sample in enumerate(self._buffer)
        }
        return ExperimentData.from_data(data=data, domain=domain)


# ==========================================================================


def load_inner_evaluator(
    run_config: dict, study_dir: Path
) -> "DataGenerator | None":
    """Resolve and instantiate the inner evaluator from run_config.

    Resolution order
    ----------------
    1. ``evaluator_lookup`` present → build a
       :class:`.LookupDataGenerator` over the named pool.
    2. ``evaluator_entrypoint`` present → load via file-location
       import.  Entrypoint format: ``"path/to/file.py:AttrName"``
       (path is relative to *study_dir*; no package structure needed).

       - If the resolved attr is a :class:`.DataGenerator` subclass →
         instantiate no-args.
       - If it is a callable (bare function) →
         wrap with ``@datagenerator(output_names=...)`` applied
         functionally.  The callable **must** accept ``**kwargs``
         whose keys are the input column names of the sample.
         ``evaluator_output_names`` in run_config is required in this
         case.
    3. Neither present → return ``None``.

    Parameters
    ----------
    run_config : dict
        The dict loaded from ``run_config.json``.
    study_dir : Path
        Root of the study tree (pool paths and entrypoint paths are
        resolved relative to this directory).

    Returns
    -------
    DataGenerator or None
    """
    import importlib.util
    import sys

    lookup_cfg = run_config.get("evaluator_lookup")
    if lookup_cfg:
        from .lookup import LookupDataGenerator

        pool_rel = lookup_cfg["pool"]
        pool_project = study_dir / pool_rel
        pool = ExperimentData.from_file(project_dir=pool_project)
        return LookupDataGenerator(
            pool=pool,
            input_columns=lookup_cfg["input_columns"],
            output_columns=lookup_cfg.get("output_columns"),
        )

    entrypoint = run_config.get("evaluator_entrypoint")
    if entrypoint:
        if ":" not in entrypoint:
            raise ValueError(
                f"evaluator_entrypoint must be 'path/to/file.py:attr', "
                f"got {entrypoint!r}"
            )
        file_part, attr = entrypoint.rsplit(":", 1)
        abs_file = (study_dir / file_part).resolve()
        if not abs_file.exists():
            raise FileNotFoundError(
                f"Evaluator file not found: {abs_file} "
                f"(entrypoint={entrypoint!r})"
            )
        module_name = (
            "_f3dasm_eval_"
            + abs_file.stem.replace("-", "_").replace(".", "_")
        )
        spec = importlib.util.spec_from_file_location(
            module_name, abs_file
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Cannot load module from {abs_file}"
            )
        mod = importlib.util.module_from_spec(spec)
        # Temporarily add study_dir to sys.path so the loaded module
        # can perform its own relative imports if needed.
        _injected = str(study_dir) not in sys.path
        if _injected:
            sys.path.insert(0, str(study_dir))
        try:
            spec.loader.exec_module(mod)
        finally:
            if _injected and str(study_dir) in sys.path:
                sys.path.remove(str(study_dir))

        obj = getattr(mod, attr)

        # DataGenerator subclass → instantiate
        try:
            if isinstance(obj, type) and issubclass(obj, DataGenerator):
                return obj()
        except TypeError:
            pass

        # Callable (bare function) → wrap with @datagenerator.
        # The contract: the callable accepts **kwargs whose keys are
        # the ExperimentSample's input column names.  We create a thin
        # adapter that passes all _input_data as kwargs so that both
        # VAR_KEYWORD (**kwargs) and named-parameter callables work.
        if callable(obj):
            output_names = run_config.get("evaluator_output_names")
            if not output_names:
                raise ValueError(
                    f"evaluator_entrypoint {entrypoint!r} resolves to a "
                    "callable (not a DataGenerator subclass).  "
                    "Set 'evaluator_output_names' in config.yaml "
                    "(e.g. output_names: [f])."
                )
            _fn = obj
            _out_names = list(output_names)

            class _BareCallableGen(DataGenerator):
                def execute(
                    self,
                    experiment_sample: ExperimentSample,
                    **kwargs,
                ) -> ExperimentSample:
                    result = _fn(**experiment_sample._input_data)
                    if isinstance(result, dict):
                        # map by declared output name, not order
                        result = [result[n] for n in _out_names]
                    elif not isinstance(result, (list, tuple)):
                        result = [result]
                    for name, val in zip(_out_names, result):
                        experiment_sample._output_data[name] = val
                    experiment_sample.job_status = (
                        JobStatus.FINISHED
                    )
                    return experiment_sample

            return _BareCallableGen()

        raise ValueError(
            f"Resolved attr {attr!r} from {entrypoint!r} is neither a "
            "DataGenerator subclass nor a callable."
        )

    return None


# ==========================================================================


def get_evaluator(inner: Optional[DataGenerator] = None) -> (
    InstrumentedDataGenerator
):
    """Factory that creates an InstrumentedDataGenerator for a delegation.

    Locates ``run_config.json`` by walking up from ``Path.cwd()``, reads
    all configuration from it, and derives the delegation ID from the
    current working directory name (expected pattern ``D###``).

    Parameters
    ----------
    inner : DataGenerator or None, optional
        If provided, used as the wrapped evaluator.  If ``None``, Phase 2
        entrypoint resolution is required — this raises
        ``NotImplementedError`` with a TODO.

    Returns
    -------
    InstrumentedDataGenerator

    Raises
    ------
    ValueError
        If the cwd is not a ``D###`` directory and the env var
        ``F3DASM_DELEGATION_ID`` is not set.
    FileNotFoundError
        If ``run_config.json`` cannot be found by walking up from cwd.
    NotImplementedError
        If ``inner`` is ``None`` (Phase 2 not yet implemented).
    """
    delegation_id = _resolve_delegation_id()
    run_config = _load_run_config()

    store_dir = Path(run_config["store_dir"])
    counter_dir = Path(run_config["counter_dir"])
    counter_path = counter_dir / f"{delegation_id}.count"
    lock_path_str = run_config.get("lock_path")
    lock_path = (
        Path(lock_path_str)
        if lock_path_str
        else store_dir / "experiment_data" / ".lock"
    )
    source = run_config.get(
        "source",
        run_config.get("evaluator_name", ""),
    )
    fidelity_column = run_config.get("fidelity_column")

    if inner is None:
        study_dir_str = run_config.get("study_dir")
        if study_dir_str is None:
            raise ValueError(
                "run_config.json is missing 'study_dir' key; "
                "re-run your study to regenerate it."
            )
        resolved = load_inner_evaluator(
            run_config, Path(study_dir_str)
        )
        if resolved is None:
            raise ValueError(
                "This study declares no evaluator entrypoint in its "
                "config.yaml.  Either add an 'evaluator:' block to "
                "config.yaml (see docs), author your own DataGenerator "
                "and pass it via get_evaluator(inner=...), or report "
                "evaluation counts manually via ReportEvals."
            )
        inner = resolved

    return InstrumentedDataGenerator(
        inner=inner,
        store_dir=store_dir,
        delegation_id=delegation_id,
        source=source,
        fidelity_column=fidelity_column,
        counter_path=counter_path,
        lock_path=lock_path,
    )


# --------------------------------------------------------------------------
# Private helpers
# --------------------------------------------------------------------------


def _resolve_delegation_id() -> str:
    """Return delegation ID from cwd name or env var, or raise."""
    cwd_name = Path.cwd().name
    if _DELEGATION_ID_RE.match(cwd_name):
        return cwd_name

    env_id = os.environ.get("F3DASM_DELEGATION_ID", "")
    if env_id and _DELEGATION_ID_RE.match(env_id):
        return env_id

    raise ValueError(
        "get_evaluator must run inside a delegation workspace (cwd "
        "D###) or with F3DASM_DELEGATION_ID set to a D### value. "
        f"Got cwd='{cwd_name}', "
        f"F3DASM_DELEGATION_ID='{env_id or '(unset)'}'."
    )


def _load_run_config() -> dict:
    """Walk up from cwd until run_config.json is found."""
    current = Path.cwd()
    for _ in range(10):  # guard against infinite walk
        candidate = current / "run_config.json"
        if candidate.exists():
            return json.loads(candidate.read_text())
        parent = current.parent
        if parent == current:
            break
        current = parent
    raise FileNotFoundError(
        "run_config.json not found by walking up from "
        f"'{Path.cwd()}'."
    )


# ==========================================================================

__all__ = [
    "InstrumentedDataGenerator",
    "get_evaluator",
    "load_inner_evaluator",
]
