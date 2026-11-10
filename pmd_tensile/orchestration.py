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

from pmd_tensile.contract import Curve
from pmd_tensile.curve_math import _check_s0
from pmd_tensile.nodes.acquisition import fetch_curve
from pmd_tensile.nodes.analysis.comparison import summarize_by_group
from pmd_tensile.nodes.analysis.elastic_line import (
    ElasticLineDiagnostics,
    ProofStrengthDiagnostics,
    ProvisionalLine,
    WindowSensitivityDiagnostics,
    elastic_line,
    proof_strength,
    provisional_elastic_line,
    window_sensitivity,
)
from pmd_tensile.nodes.analysis.extensometer import (
    DroppedSamples,
    ExtensometerDiagnostics,
    count_dropped_samples,
    extensometer_validity,
)
from pmd_tensile.nodes.analysis.maximum_force import (
    ElongationDiagnostics,
    elongation_at_maximum_force,
    maximum_force,
)
from pmd_tensile.nodes.analysis.yielding import YieldDiagnostics, yield_points
from pmd_tensile.nodes.workflows import (
    GROUP_COLUMN,
    comparison_table,
    exploration_workflow,
    results_table,
    specimen_workflow,
    summary_quantities,
)
from pmd_tensile.parameters import (
    AnalysisParameters,
    parameter_values,
    parameters_from_values,
)


def build_exploration_workflow(label: str = "exploration"):
    """Stage 1 as a runnable node: ``.run(profile_id=..., source=...)``."""
    return pwf.node(exploration_workflow, label)


def build_specimen_node(label: str = "specimen"):
    """One per-specimen workflow as a runnable node.

    Run with ``.run(profile_id=..., source=..., specimen_id=...)``, optionally
    with ``curve_locator=...`` and ``parameters=...`` (the dictionary form of
    the analysis parameters; ``None`` means the method defaults).
    """
    return pwf.node(specimen_workflow, label)


def _specimen_node_label(index: int) -> str:
    return f"specimen_{index}"


def specimen_input_label(index: int) -> str:
    """Workflow input label of the ``index``-th specimen ID in the batch workflow."""
    return f"specimen_id_{index}"


def _unique_specimen_ids(specimen_ids: Sequence[str]) -> list[str]:
    """The IDs as strings; a repeated ID raises, since results are keyed by specimen ID."""
    ids = [str(specimen_id) for specimen_id in specimen_ids]
    duplicates = sorted({specimen_id for specimen_id in ids if ids.count(specimen_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate specimen IDs {duplicates}; list each specimen once")
    return ids


def build_batch_workflow(specimen_ids: Sequence[str], label: str = "tensile_batch") -> pwf.Workflow:
    """Stage 2: one per-specimen node per specimen ID, plus aggregation nodes.

    The IDs only fix the number of branches; their values are workflow inputs
    ``specimen_id_<i>`` (see :func:`batch_inputs`), so the recipe holds no
    data-specific values. Inputs: ``profile_id``, ``source``,
    ``curve_locator``, ``parameters`` and ``specimen_id_0`` ... Outputs: ``results``
    (one row per specimen), ``orientation_summary`` (method-spec §12),
    ``comparison`` (all specimens) and ``rows`` (the gathered rows). A repeated specimen ID raises ``ValueError``.

    Every input is required, ``curve_locator`` and ``parameters`` included,
    even where ``None`` means "not used" or "method defaults": in
    pyiron_workflow 0.20.3, workflow inputs cannot hold default values, and a
    run without one of them raises ``InputDataUnavailable``. Build the inputs
    with :func:`batch_inputs`, which fills all of them.
    """
    ids = _unique_specimen_ids(specimen_ids)
    if not ids:
        raise ValueError("the batch workflow needs at least one specimen ID")
    n = len(ids)
    wf = pwf.Workflow(label)
    wf.create_input("profile_id", type_hint=str)
    wf.create_input("source", type_hint=dict)
    wf.create_input("curve_locator")
    wf.create_input("parameters")
    gather_rows = pwf.node(fr.schemas.TransformNto1(n).recipe, "gather_rows")
    gather_comparisons = pwf.node(fr.schemas.TransformNto1(n).recipe, "gather_comparisons")
    wf.add_node(gather_rows, gather_comparisons)
    for i in range(n):
        node = pwf.node(specimen_workflow, _specimen_node_label(i))
        wf.add_node(node)
        wf.create_input_for(node.inputs.specimen_id, label=specimen_input_label(i))
        wf.connect(wf.inputs.profile_id, node.inputs.profile_id)
        wf.connect(wf.inputs.source, node.inputs.source)
        wf.connect(wf.inputs.curve_locator, node.inputs.curve_locator)
        wf.connect(wf.inputs.parameters, node.inputs.parameters)
        item = fr.schemas.TransformNto1.input_label(i)
        wf.connect(node.outputs.row, gather_rows.inputs[item])
        wf.connect(node.outputs.comparison, gather_comparisons.inputs[item])
    wf.results = pwf.node(results_table)
    wf.connect(gather_rows.outputs.output_0, wf.results.inputs.rows)
    wf.summary_quantities = pwf.node(summary_quantities)
    wf.orientation_summary = pwf.node(summarize_by_group, group_column=GROUP_COLUMN)
    wf.connect(wf.results.outputs.table, wf.orientation_summary.inputs.table)
    wf.connect(wf.summary_quantities.outputs.quantities, wf.orientation_summary.inputs.quantities)
    wf.comparison = pwf.node(comparison_table)
    wf.connect(gather_rows.outputs.output_0, wf.comparison.inputs.rows)
    wf.connect(gather_comparisons.outputs.output_0, wf.comparison.inputs.comparisons)
    wf.create_output_from(wf.results.outputs.table, "results")
    wf.create_output_from(wf.orientation_summary.outputs.summary, "orientation_summary")
    wf.create_output_from(wf.comparison.outputs.table, "comparison")
    wf.create_output_from(gather_rows.outputs.output_0, "rows")
    return wf


def batch_inputs(
    profile_id: str,
    source: Mapping,
    specimen_ids: Sequence[str],
    curve_locator: Mapping | None = None,
    parameters: Mapping | None = None,
) -> dict:
    """Keyword inputs for running a workflow from :func:`build_batch_workflow`.

    The returned dictionary holds every batch-workflow input. All of them are
    required, because pyiron_workflow 0.20.3 workflow inputs cannot hold
    default values; use this function rather than writing the inputs by hand,
    so that ``curve_locator`` and ``parameters`` are present even when they
    take their "unused" and "defaults" values.

    ``parameters`` (the dictionary form of the analysis parameters, ``None``
    for the method defaults) is validated here and passed on with *all* field
    values, so the run's knowledge graph records the complete set used.
    ``specimen_ids`` must be in the order given to
    :func:`build_batch_workflow`; a repeated ID raises ``ValueError``.
    """
    specimen_ids = _unique_specimen_ids(specimen_ids)
    inputs = {
        "profile_id": str(profile_id),
        "source": dict(source),
        "curve_locator": None if curve_locator is None else dict(curve_locator),
        "parameters": parameter_values(parameters_from_values(parameters)),
    }
    for i, specimen_id in enumerate(specimen_ids):
        inputs[specimen_input_label(i)] = str(specimen_id)
    return inputs


@dataclasses.dataclass(frozen=True, eq=False)
class SpecimenAnalysis:
    """The analysis of one curve: the primary values of the analysis nodes and their diagnostics.

    Built from direct node calls (:func:`analyse_curve`) or from the node outputs of a
    run (:func:`specimen_analyses`), with the same field names; figures and notebooks
    read it. ``flags`` joins the flags of every step in the order of the steps.
    """

    yield_type: str
    e_gpa: float
    strain_origin: float
    reh_mpa: float
    rel_mpa: float
    rp02_mpa: float
    fm_n: float
    rm_mpa: float
    agt_percent: float
    ag_percent: float
    noise_mpa: float
    k_ext: int
    max_valid_strain: float
    highest_force_n: float
    e0_gpa: float
    intercept0_mpa: float
    reference_force_n: float
    upper_yield_index: int
    end_of_yielding_index: int
    work_hardening: bool
    max_force_index: int
    e_min_gpa: float
    e_max_gpa: float
    rp02_min_mpa: float
    rp02_max_mpa: float
    dropped: DroppedSamples
    extensometer: ExtensometerDiagnostics
    provisional: ProvisionalLine
    yielding: YieldDiagnostics
    elastic: ElasticLineDiagnostics
    proof: ProofStrengthDiagnostics
    maximum_flags: tuple[str, ...]
    elongation: ElongationDiagnostics
    sensitivity: WindowSensitivityDiagnostics
    flags: tuple[str, ...] = ()

    @classmethod
    def from_outputs(cls, outputs: dict) -> SpecimenAnalysis:
        """From a mapping of node output names to values (the names of the macro's variables)."""
        fields = {f.name for f in dataclasses.fields(cls)} - {"flags"}
        values = {name: outputs[name] for name in fields}
        steps = ("extensometer", "elastic", "yielding", "proof", "elongation", "sensitivity")
        flags = [flag for step in steps[:4] for flag in values[step].flags]
        flags += [*values["maximum_flags"], *values["elongation"].flags, *values["sensitivity"].flags]
        return cls(**values, flags=tuple(dict.fromkeys(flags)))


@dataclasses.dataclass(frozen=True, eq=False)
class TensileAnalysis:
    """Runs and result tables of :func:`run_tensile_analysis`.

    ``exploration_run`` and ``batch_run`` are pyiron_workflow runs; their
    ``.result`` is the flowrep run data that
    :func:`pmd_tensile.knowledge_graph.run_knowledge_graph` exports.

    ``curves`` and ``specimen_analyses`` map each specimen ID (in the order of
    ``specimen_ids``) to the neutral :class:`~pmd_tensile.contract.Curve`
    that the specimen's ``fetch_curve`` step returned and to the
    :class:`SpecimenAnalysis` of its node outputs. Both come from the batch
    run's own data, not from a second download, so a figure drawn from them
    shows exactly the arrays the analysis used.
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
    curves: dict[str, Curve]
    specimen_analyses: dict[str, SpecimenAnalysis]


def _specimen_data(batch_run, index: int, specimen_id: str):
    """The run data of the ``index``-th per-specimen node, checked against its specimen ID."""
    data = batch_run.result.nodes[_specimen_node_label(index)]
    recorded = data.input_ports["specimen_id"].value
    if recorded != specimen_id:
        raise RuntimeError(f"{_specimen_node_label(index)} ran for {recorded!r}, expected {specimen_id!r}")
    return data


_STEP_OUTPUTS = {
    fetch_curve: {},
    count_dropped_samples: {},
    extensometer_validity: {"diagnostics": "extensometer"},
    provisional_elastic_line: {"diagnostics": "provisional"},
    yield_points: {"diagnostics": "yielding"},
    elastic_line: {"diagnostics": "elastic"},
    proof_strength: {"diagnostics": "proof"},
    maximum_force: {"flags": "maximum_flags"},
    elongation_at_maximum_force: {"diagnostics": "elongation"},
    window_sensitivity: {"diagnostics": "sensitivity"},
}
"""The steps whose outputs make up a :class:`SpecimenAnalysis`, and the field name of each
output whose name is shared by several steps."""


def _step_outputs(specimen_data) -> dict:
    """The outputs of the analysis steps of a per-specimen run, by field name of :class:`SpecimenAnalysis`."""
    outputs = {}
    for node in specimen_data.nodes.values():
        renames = _STEP_OUTPUTS.get(getattr(node, "function", None))
        if renames is not None:
            for name, port in node.output_ports.items():
                outputs[renames.get(name, name)] = port.value
    return outputs


def specimen_analyses(batch_run, specimen_ids: Sequence[str]) -> tuple[dict[str, Curve], dict[str, SpecimenAnalysis]]:
    """Curves and specimen analyses per specimen ID from a finished batch run (no download).

    ``specimen_ids`` are the IDs in batch order, as passed to
    :func:`build_batch_workflow` and :func:`batch_inputs`; a repeated ID raises
    ``ValueError``.
    """
    curves, analyses = {}, {}
    for index, specimen_id in enumerate(_unique_specimen_ids(specimen_ids)):
        outputs = _step_outputs(_specimen_data(batch_run, index, specimen_id))
        curves[specimen_id] = outputs["curve"]
        analyses[specimen_id] = SpecimenAnalysis.from_outputs(outputs)
    return curves, analyses


def run_tensile_analysis(
    profile_id: str,
    source: Mapping,
    curve_locator: Mapping | None = None,
    *,
    specimen_ids: Sequence[str] | None = None,
    parameters: Mapping | None = None,
) -> TensileAnalysis:
    """Run both stages: exploration, then the batch over the listed specimens.

    ``specimen_ids`` optionally restricts the batch to a subset of the listed
    IDs (in the given order); an ID not listed by the exploration raises, and
    so does a repeated ID (before anything runs).
    ``parameters`` is the dictionary form of the analysis parameters (missing
    keys keep the method defaults); it is validated before anything runs.
    """
    parameters_from_values(parameters)  # fail early on invalid parameters
    if specimen_ids is not None:
        specimen_ids = _unique_specimen_ids(specimen_ids)  # fail early on repeated IDs
    exploration_node = build_exploration_workflow()
    exploration_run = exploration_node.run(profile_id=profile_id, source=dict(source))
    listed = [str(specimen_id) for specimen_id in exploration_run.outputs.specimen_ids]
    if specimen_ids is None:
        selected = listed
    else:
        selected = specimen_ids
        unknown = sorted(set(selected) - set(listed))
        if unknown:
            raise LookupError(f"specimen IDs not described in the source: {unknown}; listed: {listed}")
    batch = build_batch_workflow(selected)
    batch_run = batch.run(**batch_inputs(profile_id, source, selected, curve_locator, parameters))
    curves, analyses = specimen_analyses(batch_run, selected)
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
        curves=curves,
        specimen_analyses=analyses,
    )




def analyse_curve(
    force_n,
    strain,
    crosshead_mm,
    s0_mm2: float,
    l0_mm: float,
    *,
    a_percent: float | None = None,
    rate_switch_strain: float | None = None,
    params: AnalysisParameters | None = None,
) -> SpecimenAnalysis:
    """Analyse one curve by calling the analysis nodes directly, in the order of :func:`specimen_workflow`.

    Force (N), strain (ratio) and crosshead (mm) arrays may contain NaN.
    ``a_percent`` (A from the test record) enables the check whether the
    extensometer ended early; ``rate_switch_strain`` (ratio) only sets
    ``near_rate_switch`` flags. The numbers equal those of the workflow on the
    same arrays.
    """
    _check_s0(s0_mm2)
    out = {"dropped": count_dropped_samples(force_n, strain, crosshead_mm)}
    out["k_ext"], out["max_valid_strain"], out["extensometer"] = extensometer_validity(
        strain, crosshead_mm, l0_mm, a_percent, rate_switch_strain, params
    )
    k_ext = out["k_ext"]
    (out["highest_force_n"], out["e0_gpa"], out["intercept0_mpa"], out["noise_mpa"],
     out["provisional"]) = provisional_elastic_line(force_n, strain, s0_mm2, k_ext, params)
    (out["yield_type"], out["reh_mpa"], out["rel_mpa"], out["upper_yield_index"], out["end_of_yielding_index"],
     out["work_hardening"], out["yielding"]) = yield_points(
        force_n, strain, s0_mm2, k_ext, out["highest_force_n"], out["e0_gpa"], out["intercept0_mpa"],
        out["noise_mpa"], rate_switch_strain, params,
    )
    out["e_gpa"], out["strain_origin"], out["reference_force_n"], out["elastic"] = elastic_line(
        force_n, strain, s0_mm2, k_ext, out["upper_yield_index"], out["highest_force_n"], out["e0_gpa"],
        out["intercept0_mpa"], params,
    )
    out["rp02_mpa"], out["proof"] = proof_strength(
        force_n, strain, s0_mm2, k_ext, out["e_gpa"], out["strain_origin"], out["reference_force_n"],
        out["upper_yield_index"], out["end_of_yielding_index"], rate_switch_strain, params,
    )
    out["fm_n"], out["rm_mpa"], out["max_force_index"], out["maximum_flags"] = maximum_force(
        force_n, s0_mm2, out["yield_type"], out["work_hardening"], out["end_of_yielding_index"]
    )
    out["agt_percent"], out["ag_percent"], out["elongation"] = elongation_at_maximum_force(
        force_n, strain, s0_mm2, k_ext, out["e_gpa"], out["strain_origin"], out["fm_n"], out["max_force_index"],
        out["noise_mpa"], params,
    )
    out["e_min_gpa"], out["e_max_gpa"], out["rp02_min_mpa"], out["rp02_max_mpa"], out["sensitivity"] = window_sensitivity(
        force_n, strain, s0_mm2, k_ext, out["reference_force_n"], out["e_gpa"], out["strain_origin"],
        out["agt_percent"], out["fm_n"], params,
    )
    return SpecimenAnalysis.from_outputs(out)
