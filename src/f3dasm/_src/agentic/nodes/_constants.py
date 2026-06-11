"""Shared module-level constants for the nodes package — kept here (not in
strategizer.py) so both strategizer.py and lifecycle.py (and routing.py) can
import them without an import cycle."""
import os as _os

# Time budget is a SOFT constraint (warnings only). This multiple is the
# run-level cost backstop: a run is aborted once it exceeds
# RUN_BACKSTOP_MULTIPLE x the time budget, to bound runaway cost.
# Wall-clock is a poor proxy for cost when a single oracle eval can take days
# (the SOTA problems), so the cap is configurable and can be DISABLED: set
# F3DASM_RUN_BACKSTOP_MULTIPLE <= 0 to turn the hard cap off entirely.
RUN_BACKSTOP_MULTIPLE = float(
    _os.environ.get("F3DASM_RUN_BACKSTOP_MULTIPLE", "2.0")
)
_BACKSTOP_ENABLED = RUN_BACKSTOP_MULTIPLE > 0
