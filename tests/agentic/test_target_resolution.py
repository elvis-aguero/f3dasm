"""Forward-compatible delegation-target resolution (friction #3 fix).

Agents repeatedly name a delegation target by capability rather than the exact
graph node name ('pipeline'/'pipeline_executor' for the implementer,
'data_generation' for the datagenerator) and used to bounce off "unknown
target". resolve_target maps such requests to the right node, surviving node
renames by resolving curated synonyms to a ROLE (not a hardcoded node name).
"""
from __future__ import annotations

from f3dasm._src.agentic.nodes.tools.routing import resolve_target

# Canonical 5-node topology; here role == node name.
_OUT = ["implementer", "datagenerator", "literature_reviewer", "critic"]
_ROLES = {n: n for n in _OUT}


def test_exact_and_case_separator_normalization():
    assert resolve_target("implementer", _OUT, _ROLES) == "implementer"
    assert resolve_target("Implementer", _OUT, _ROLES) == "implementer"
    assert resolve_target("literature-reviewer", _OUT, _ROLES) == "literature_reviewer"
    assert resolve_target("LITERATURE_REVIEWER", _OUT, _ROLES) == "literature_reviewer"


def test_pipeline_synonyms_resolve_to_implementer():
    # the recurring hallucinations across all 3 fresh runs
    for syn in ("pipeline", "pipeline_executor", "pipeline-executor",
                "pipeline_execution", "pipelineExecutor"):
        assert resolve_target(syn, _OUT, _ROLES) == "implementer", syn


def test_data_generation_synonyms_resolve_to_datagenerator():
    for syn in ("data_generation", "data-generation", "datagen"):
        assert resolve_target(syn, _OUT, _ROLES) == "datagenerator", syn


def test_unresolvable_returns_none():
    assert resolve_target("nonsense_xyz", _OUT, _ROLES) is None
    assert resolve_target("", _OUT, _ROLES) is None


def test_synonym_maps_to_role_not_node_name():
    """Forward-compatible: a renamed implementer node (different name, same
    role) still resolves the pipeline synonyms."""
    out = ["doer", "critic"]
    roles = {"doer": "implementer", "critic": "critic"}
    assert resolve_target("pipeline_executor", out, roles) == "doer"
    assert resolve_target("implementer", out, roles) == "doer"  # by role


def test_exact_name_wins_over_alias():
    # if a real node is literally named 'pipeline', exact match takes precedence
    out = ["pipeline", "implementer", "critic"]
    roles = {"pipeline": "pipeline", "implementer": "implementer", "critic": "critic"}
    assert resolve_target("pipeline", out, roles) == "pipeline"
