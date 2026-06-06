"""System prompts for the two-agent agentic-f3dasm runtime.

This module ships all string constants that bootstrap the Strategizer
and Implementer Claude Agent SDK sessions used by
``f3dasm.agentic.run``.  They are kept in one place so the prompts can be
versioned, reviewed, and improved independently of the routing runtime.

Constants
---------
STRATEGIZER_SYSTEM_PROMPT
    System prompt for the long-running Strategizer (thinker) session.
IMPLEMENTER_SYSTEM_PROMPT
    System prompt for the long-running Implementer (doer) session.
CHECKPOINT_STRATEGIZER_PROMPT
    User-message injected into the Strategizer every 30 delegations.
IMPLEMENTER_RESET_PROMPT_TEMPLATE
    ``.format()``-ready template for the opening user message sent to a
    freshly-reset Implementer after a checkpoint.  Single placeholder:
    ``{checkpoint_summary}``.
RUN_PATHS_PREAMBLE_TEMPLATE
    ``.format()``-ready preamble prepended to the Strategizer system
    prompt at the start of every run.  Placeholders: ``{study_dir}``,
    ``{notes_dir}``.
WORKSPACE_PREAMBLE_TEMPLATE
    ``.format()``-ready preamble prepended to the Implementer system
    prompt at the start of every run.  Placeholder: ``{workspace_dir}``.
IMPLEMENTER_REPORT_RETRY_PROMPT
    Static correction message sent to the Implementer when its first
    reply lacks a parseable ``## Report`` block.
REFLECT_DIAGNOSIS_SHORT
    REFLECT diagnosis text for unusually short Implementer responses.
REFLECT_DIAGNOSIS_CAPABILITY_LIMIT
    REFLECT diagnosis text when the Implementer reports a capability
    limit.
REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE
    ``.format()``-ready REFLECT diagnosis text for a Report block that
    is present but missing required subsections.  Placeholder:
    ``{missing_subsections}``.
REFLECT_DIAGNOSIS_NO_REPORT_HEADING
    REFLECT diagnosis text when the Implementer never starts a
    ``## Report`` block.
REFLECT_DIAGNOSIS_DEFAULT
    Fallback REFLECT diagnosis text for unrecognised malformation.

Notes
-----
Line-length rule: all Python source lines are <= 79 chars.  The string
content of each constant may contain longer lines; that is intentional
and correct.
"""
#                                                                       Modules
# =============================================================================

from __future__ import annotations

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================
#
# =============================================================================

__all__ = [
    "STRATEGIZER_SYSTEM_PROMPT",
    "IMPLEMENTER_SYSTEM_PROMPT",
    "DEBUGGER_SYSTEM_PROMPT",
    "LITERATURE_REVIEW_SYSTEM_PROMPT",
    "ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT",
    "CHECKPOINT_STRATEGIZER_PROMPT",
    "IMPLEMENTER_RESET_PROMPT_TEMPLATE",
    "RUN_PATHS_PREAMBLE_TEMPLATE",
    "WORKSPACE_PREAMBLE_TEMPLATE",
    "IMPLEMENTER_REPORT_RETRY_PROMPT",
    "REFLECT_DIAGNOSIS_SHORT",
    "REFLECT_DIAGNOSIS_CAPABILITY_LIMIT",
    "REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE",
    "REFLECT_DIAGNOSIS_NO_REPORT_HEADING",
    "REFLECT_DIAGNOSIS_DEFAULT",
    "IMPLEMENTER_SYSTEM_PROMPT_OLLAMA",
]

# Re-export agent system prompts from their canonical locations so that
# existing code importing from agent_prompts continues to work.
from .agents.strategizer import STRATEGIZER_SYSTEM_PROMPT  # noqa: E402
from .agents.implementer import IMPLEMENTER_SYSTEM_PROMPT  # noqa: E402
from .agents.debugger import DEBUGGER_SYSTEM_PROMPT  # noqa: E402
from .agents.literature import LITERATURE_REVIEW_SYSTEM_PROMPT  # noqa: E402
from .agents.critic import ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT  # noqa: E402

# =============================================================================

CHECKPOINT_STRATEGIZER_PROMPT = """\
CHECKPOINT — do not generate new hypotheses or delegate new tasks.

The runtime has reached a delegation checkpoint.  Your job right now is
to synthesise what has been learned so far and produce a structured
summary that a human reviewer can read and that a fresh Implementer
session can use as a project briefing.

Produce a report under a ## Checkpoint heading with exactly these four
sections.  Be specific: cite numbers from prior Reports.  Do not
speculate beyond what the data supports.

## Checkpoint

### What we have learned
<Bullet list.  Each bullet: a finding supported by at least one Report
number.  Format: "- [Finding]: [evidence] (Report #N, key: value)">

### What we have ruled out
<Bullet list.  Each bullet: a hypothesis or region of the search space
that has been falsified or shown to be suboptimal.  Include the evidence.>

### Open questions
<Bullet list.  Each bullet: an unresolved uncertainty that materially
affects the final design recommendation.  State why it is unresolved
(e.g. no data yet, conflicting Reports, pool coverage gap).>

### Recommended next direction
<One paragraph, <= 100 words.  The single most information-valuable
experiment or analysis to run next.  Justify in terms of which open
question it resolves.  Do not propose more than one direction.>

### Hypothesis ledger digest
<One line per hypothesis from HypothesisList().
Format: `- <id> [<status>] prior <p> → posterior <q>: <statement>`.
The canonical store is hypotheses.json; this is a digest only.>

After producing this report, wait.  Do not delegate until the runtime
resumes the session.
"""

# =============================================================================

IMPLEMENTER_RESET_PROMPT_TEMPLATE = """\
You are starting a new Implementer session.  The prior session has ended
at a checkpoint.  Below is the Strategizer's checkpoint summary, which is
your complete project context.  You have no memory of the prior session's
tool calls; treat the checkpoint summary as your sole briefing.

--- BEGIN CHECKPOINT SUMMARY ---
{checkpoint_summary}
--- END CHECKPOINT SUMMARY ---

From this point on you will receive Task messages from the Strategizer.
Execute each task and return a Report as specified in your system prompt.
The debug/delegations/ directory under the current run may contain artefacts from
the prior session — check before recomputing anything.
"""

# =============================================================================

RUN_PATHS_PREAMBLE_TEMPLATE = """\
<run_paths>
study_dir             = {study_dir}
run_dir               = {run_dir}
debug_dir             = {debug_dir}
strategizer_notes_dir = {notes_dir}
hypotheses_json       = {notes_dir}/hypotheses.json
delegation_log_jsonl  = {debug_dir}/delegation_log.jsonl
diagnostics_jsonl     = {debug_dir}/diagnostics.jsonl
workspace_dir         = {debug_dir}/workspace
Use these absolute paths when calling Read() and WriteNote().
WriteNote also accepts a bare filename such as 'meta_errors.md',
which is anchored under strategizer_notes_dir automatically.
Workers write exclusively inside workspace_dir/D###/.
</run_paths>

"""
"""Run-paths preamble injected at the head of the Strategizer system
prompt for every new run.

Parameters (via ``.format()``)
------------------------------
study_dir : str or Path
    Absolute path to the study root directory.
run_dir : str or Path
    Absolute path to the current run directory (runs/<timestamp>/).
debug_dir : str or Path
    Absolute path to runs/<timestamp>/debug/.
notes_dir : str or Path
    Absolute path to runs/<timestamp>/debug/strategizer_notes/.
"""

# =============================================================================

WORKSPACE_PREAMBLE_TEMPLATE = """\
<workspace>
study_dir     = {study_dir}
workspace_dir = {workspace_dir}
Your task message contains a <workspace_subfolder>D###/</workspace_subfolder>
tag that names the subfolder assigned exclusively to THIS delegation.
Write ALL outputs (code, data, plots, logs) inside that subfolder.
You may Read() files from other delegations' subfolders but may NOT
write outside your own — the Write tool will reject it.
To access study assets (evaluator, lookup pools, etc.) use study_dir.
Do NOT write to /tmp or any path outside workspace_dir — files there
will be lost and are invisible to the Strategizer.
</workspace>

"""
"""Workspace preamble injected at the head of worker system prompts.

Parameters (via ``.format()``)
------------------------------
study_dir : str or Path
    Absolute path to the study root (evaluator, lookup pools, etc. live here).
workspace_dir : str or Path
    Absolute path to runs/<timestamp>/debug/delegations/ for this run.
"""

# =============================================================================

IMPLEMENTER_REPORT_RETRY_PROMPT = (
    "Your previous reply did not contain a parseable "
    "`## Report` block. Re-emit your output now using "
    "EXACTLY this structure, with the literal line "
    "`## Report` on its own line:\n\n"
    "## Report\n\n"
    "### Actions taken\n- <bulleted list>\n\n"
    "### Files touched\n- <absolute paths under "
    "workspace_dir>\n\n"
    "### Conclusions\n<prose, <= 200 words>\n\n"
    "### Numbers\n- <key>: <value>\n\n"
    "Do not skip any subsection. Include the Stage 1 / "
    "Stage 2 / Stage 3 prose ONLY before the `## Report` "
    "heading. After this retry you have no further "
    "chances — a second malformed reply will be recorded "
    "as a delegation failure."
)
"""Correction message sent to the Implementer when its first reply
lacks a parseable ``## Report`` block.

Injected by ``AgenticRun._tool_delegate`` as a focused one-shot retry
before the delegation falls through to a REFLECT failure.  The message
restates the required structure in literal form so the model cannot
misread it.  No placeholders; pure static text.
"""

# =============================================================================

REFLECT_DIAGNOSIS_SHORT = (
    "Implementer's response is unusually short; the task may "
    "have been too vague or unactionable."
)
"""REFLECT diagnosis emitted when the Implementer's response is fewer
than 100 characters.

Used by ``_classify_failed_implementer_response`` as the diagnosis
string in the ``REFLECT: {diagnosis}`` return value when the raw
response text is too short to carry a meaningful Report.
"""

# =============================================================================

REFLECT_DIAGNOSIS_CAPABILITY_LIMIT = (
    "Implementer reports a capability limit. Check whether "
    "the task asked for something outside its tool set."
)
"""REFLECT diagnosis emitted when the Implementer's response contains a
capability-limit phrase (e.g. "I cannot", "I don't have access").

Used by ``_classify_failed_implementer_response`` when any phrase from
``_CAPABILITY_PHRASES`` is found in the lower-cased response text.
"""

# =============================================================================

REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE = (
    "Implementer started a Report but omitted required "
    "subsections: {missing_subsections}."
)
"""REFLECT diagnosis emitted when a ``## Report`` block is present but
one or more required subsections are absent.

Used by ``_classify_failed_implementer_response`` after detecting that
``## Report`` exists but at least one of ``### Actions taken``,
``### Files touched``, ``### Conclusions``, or ``### Numbers`` is
missing.

Parameters (via ``.format()``)
------------------------------
missing_subsections : str
    Comma-separated list of quoted subsection names that are absent,
    e.g. ``"'Files touched', 'Numbers'"``.
"""

# =============================================================================

REFLECT_DIAGNOSIS_NO_REPORT_HEADING = (
    "Implementer wrote a response but never started a "
    "`## Report` block. Likely the instruction format was "
    "ignored."
)
"""REFLECT diagnosis emitted when the Implementer's response is
sufficiently long but contains no ``## Report`` heading at all.

Used by ``_classify_failed_implementer_response`` for responses that
pass the length threshold and the capability-limit check but never
include the required ``## Report`` anchor that the runtime greps for.
"""

# =============================================================================

REFLECT_DIAGNOSIS_DEFAULT = (
    "Implementer's response is malformed; could not produce a "
    "structured diagnosis."
)
"""Fallback REFLECT diagnosis for malformed responses that do not match
any more specific category.

Used by ``_classify_failed_implementer_response`` as the final
else-branch when the response has a ``## Report`` heading with all
required subsections present yet still failed ``_parse_report``.
"""

# =============================================================================

IMPLEMENTER_SYSTEM_PROMPT_OLLAMA: str = """\
You are the Implementer in an agentic research system. You receive a Task from
the Strategizer and must complete it using a single tool: **bash**.

## Your only tool: bash

Use `bash` for everything: reading files, writing files, running Python
scripts, installing nothing (assume the environment is fixed). All work must
stay inside the study directory you were given at the start.

Examples:
- Read a file:   bash(cmd="cat {delegation_id}/results.csv")
- Write a file:  bash(cmd="python3 {delegation_id}/write_output.py")
- Run a script:  bash(cmd="python3 {delegation_id}/optimise.py")

## Your output

When you have finished the task, emit **exactly** the following block and
nothing else after it:

## Report

### Actions taken
- <bullet per action>

### Files touched
- <path>

### Conclusions
<free-form prose — what you found, what worked, what failed, any anomaly>

### Numbers
<key>: <value>

Never omit a section. Use "- none" if a section is empty. Every anomaly,
error, or unexpected result belongs in ### Conclusions.
"""
"""System prompt for the Ollama-backed Implementer agent.

Unlike the Claude backend (which has dedicated Read/Write/RunPython
tools), the Ollama backend exposes only a single ``bash`` tool.  This
prompt teaches the Implementer to do all file I/O and script execution
via shell commands, while retaining the same structured ``## Report``
output format that the runtime parses.

Notes
-----
The ``## Report`` section header and its four subsections are required
by ``_parse_report`` in ``agent_runtime.py`` — changing those headings
will break report extraction.
"""



