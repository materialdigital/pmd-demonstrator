"""Builders and the two-stage run of the tensile workflows (private helpers).

:mod:`pmd_tensile.workflows` holds the nodes and macros. The functions here
turn them into runnable ``pyiron_workflow`` objects, assemble the batch
workflow (one per-specimen node per specimen ID, see the docstring of
``workflows``) and run both stages. They are private so that sim_atlas does
not catalogue them as nodes; notebooks reach them through
:mod:`pmd_tensile.demo`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import flowrep as fr
import pandas as pd
import pyiron_workflow as pwf

from pmd_tensile.analysis import summarize_by_group
from pmd_tensile.workflows import (
    GROUP_COLUMN,
    comparison_table,
    exploration_workflow,
    results_table,
    specimen_workflow,
    summary_quantities,
)


def build_exploration_workflow(label: str = "exploration"):
    """Stage 1 as a runnable node: ``.run(profile_id=..., source=...)``."""
    return pwf.node(exploration_workflow, label)


def build_specimen_node(label: str = "specimen"):
    """One per-specimen workflow as a runnable node.

    Run with ``.run(profile_id=..., source=..., specimen_id=..., curve_locator=None)``.
    """
    return pwf.node(specimen_workflow, label)


def specimen_input_label(index: int) -> str:
    """Workflow input label of the ``index``-th specimen ID in the batch workflow."""
    return f"specimen_id_{index}"


def build_batch_workflow(specimen_ids: Sequence[str], label: str = "tensile_batch") -> pwf.Workflow:
    """Stage 2: one per-specimen node per specimen ID, plus aggregation nodes.

    The IDs only fix the number of branches; their values are workflow inputs
    ``specimen_id_<i>`` (see :func:`batch_inputs`), so the recipe holds no
    data-specific values. Inputs: ``profile_id``, ``source``,
    ``curve_locator`` and ``specimen_id_0`` ... Outputs: ``results``
    (one row per specimen), ``orientation_summary`` (method-spec §12),
    ``comparison`` (all specimens, method-spec §13) and ``summaries`` (the
    gathered rows).
    """
    ids = [str(specimen_id) for specimen_id in specimen_ids]
    if not ids:
        raise ValueError("the batch workflow needs at least one specimen ID")
    n = len(ids)
    wf = pwf.Workflow(label)
    wf.create_input("profile_id", type_hint=str)
    wf.create_input("source", type_hint=dict)
    wf.create_input("curve_locator")
    gather_summaries = pwf.node(fr.schemas.TransformNto1(n).recipe, "gather_summaries")
    gather_comparisons = pwf.node(fr.schemas.TransformNto1(n).recipe, "gather_comparisons")
    wf.add_node(gather_summaries, gather_comparisons)
    for i in range(n):
        node = pwf.node(specimen_workflow, f"specimen_{i}")
        wf.add_node(node)
        wf.create_input_for(node.inputs.specimen_id, label=specimen_input_label(i))
        wf.connect(wf.inputs.profile_id, node.inputs.profile_id)
        wf.connect(wf.inputs.source, node.inputs.source)
        wf.connect(wf.inputs.curve_locator, node.inputs.curve_locator)
        item = fr.schemas.TransformNto1.input_label(i)
        wf.connect(node.outputs.summary, gather_summaries.inputs[item])
        wf.connect(node.outputs.comparison, gather_comparisons.inputs[item])
    wf.results = pwf.node(results_table)
    wf.connect(gather_summaries.outputs.output_0, wf.results.inputs.summaries)
    wf.summary_quantities = pwf.node(summary_quantities)
    wf.orientation_summary = pwf.node(summarize_by_group, group_column=GROUP_COLUMN)
    wf.connect(wf.results.outputs.table, wf.orientation_summary.inputs.table)
    wf.connect(wf.summary_quantities.outputs.quantities, wf.orientation_summary.inputs.quantities)
    wf.comparison = pwf.node(comparison_table)
    wf.connect(gather_summaries.outputs.output_0, wf.comparison.inputs.summaries)
    wf.connect(gather_comparisons.outputs.output_0, wf.comparison.inputs.comparisons)
    wf.create_output_from(wf.results.outputs.table, "results")
    wf.create_output_from(wf.orientation_summary.outputs.summary, "orientation_summary")
    wf.create_output_from(wf.comparison.outputs.table, "comparison")
    wf.create_output_from(gather_summaries.outputs.output_0, "summaries")
    return wf


def batch_inputs(
    profile_id: str,
    source: Mapping,
    specimen_ids: Sequence[str],
    curve_locator: Mapping | None = None,
) -> dict:
    """Keyword inputs for running a workflow from :func:`build_batch_workflow`."""
    inputs = {
        "profile_id": str(profile_id),
        "source": dict(source),
        "curve_locator": None if curve_locator is None else dict(curve_locator),
    }
    for i, specimen_id in enumerate(specimen_ids):
        inputs[specimen_input_label(i)] = str(specimen_id)
    return inputs


@dataclasses.dataclass(frozen=True, eq=False)
class TensileAnalysis:
    """Runs and result tables of :func:`run_tensile_analysis`.

    ``exploration_run`` and ``batch_run`` are pyiron_workflow runs; their
    ``.result`` is the flowrep run data that
    :func:`pmd_tensile._knowledge_graph.run_knowledge_graph` exports.
    """

    profile_id: str
    specimen_ids: tuple[str, ...]
    exploration_node: object
    exploration_run: object
    batch_workflow: pwf.Workflow
    batch_run: object
    results: pd.DataFrame
    orientation_summary: pd.DataFrame
    comparison: pd.DataFrame


def run_tensile_analysis(
    profile_id: str,
    source: Mapping,
    curve_locator: Mapping | None = None,
    *,
    specimen_ids: Sequence[str] | None = None,
) -> TensileAnalysis:
    """Run both stages: exploration, then the batch over the listed specimens.

    ``specimen_ids`` optionally restricts the batch to a subset of the listed
    IDs (in the given order); an ID not listed by the exploration raises.
    """
    exploration_node = build_exploration_workflow()
    exploration_run = exploration_node.run(profile_id=profile_id, source=dict(source))
    listed = [str(specimen_id) for specimen_id in exploration_run.outputs.specimen_ids]
    if specimen_ids is None:
        selected = listed
    else:
        selected = [str(specimen_id) for specimen_id in specimen_ids]
        unknown = sorted(set(selected) - set(listed))
        if unknown:
            raise LookupError(f"specimen IDs not described in the source: {unknown}; listed: {listed}")
    batch = build_batch_workflow(selected)
    batch_run = batch.run(**batch_inputs(profile_id, source, selected, curve_locator))
    return TensileAnalysis(
        profile_id=str(profile_id),
        specimen_ids=tuple(selected),
        exploration_node=exploration_node,
        exploration_run=exploration_run,
        batch_workflow=batch,
        batch_run=batch_run,
        results=batch_run.outputs.results,
        orientation_summary=batch_run.outputs.orientation_summary,
        comparison=batch_run.outputs.comparison,
    )
