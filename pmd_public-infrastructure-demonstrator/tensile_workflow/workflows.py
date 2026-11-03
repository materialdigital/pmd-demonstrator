"""Workflow assembly: exploration stage and per-specimen batch stage (decision D6).

The tensile analysis runs in two stages, both free of flow controllers
(``for``, ``while``, ``if``, ``try``) so that semantikon can export them as
knowledge graphs (upstream limitation pyiron/semantikon#541):

1. **Exploration** (:func:`exploration_workflow`, an ``@fr.workflow``):
   checks that the source conforms to the ontology profile and lists the
   specimen IDs it describes.
2. **Batch** (:func:`build_batch_workflow`, a programmatically assembled
   ``pyiron_workflow.Workflow``): one :func:`specimen_workflow` node per
   specimen ID, each fed by its own workflow input (convention IC3), gathered
   with ``flowrep.schemas.TransformNto1`` into aggregation nodes: the results
   table, the per-orientation summary (:func:`~tensile_workflow.analysis.summarize_by_group`)
   and the combined comparison with lab-reported values.

The per-specimen workflow is a flat DAG: acquisition (record, reported
properties, curve fetch), the analysis step nodes in the order of
:func:`~tensile_workflow.analysis.analyse_curve`, A, Z, S0 scatter, and the
comparison with reported values. Its numbers equal ``analyse_curve`` on the
same arrays, with ``a_percent`` = A from the metadata (L0, Lu) and the
record's ``rate_switch_strain``.

Workflow inputs are JSON-able only (T1 finding, IC3): ``profile_id`` (str),
``source`` (dict, see :mod:`tensile_workflow.sources`), ``curve_locator``
(dict or ``None``; only for profiles whose graph does not link the curves) and
one ``specimen_id_<i>`` (str) per specimen. The analysis parameters are the
method defaults (:data:`tensile_workflow.analysis.DEFAULT_PARAMETERS`) and are
not exposed as workflow inputs. Credentials are never inputs; the Ontodocker
access path reads them from the environment (decision D5).

Node roles (used by :mod:`tensile_workflow.knowledge_graph`):

- ontology-aware: :mod:`tensile_workflow.exploration`,
  :mod:`tensile_workflow.acquisition`;
- ontology-agnostic analysis: :mod:`tensile_workflow.analysis`;
- ontology-agnostic contract adapters in this module
  (:func:`curve_channels`, :func:`specimen_geometry`, :func:`percent_value`,
  :func:`computed_properties`, :func:`specimen_summary`,
  :func:`summary_quantities`, :func:`results_table`,
  :func:`comparison_table`): they only unpack or collect the neutral contract
  dataclasses and analysis results, and give the arrays and scalars their
  unit annotations;
- workflow compositions: :func:`exploration_workflow`, :func:`specimen_workflow`.

Stage nodes
-----------
- Exploration: ``check_conformance`` -> ``list_specimens``.
- Per specimen (``specimen_<i>``): ``get_specimen_record``,
  ``get_reported_properties``, ``fetch_curve``; ``curve_channels``,
  ``specimen_geometry``; ``check_time_order``, ``elongation_after_fracture``
  (A), ``reduction_of_area`` (Z), ``s0_scatter``, ``percent_value``;
  ``count_dropped_samples``, ``extensometer_validity``, ``bootstrap_fit``,
  ``yield_analysis``, ``final_elastic_fit``, ``proof_strength``,
  ``maximum_force``, ``elongation_at_maximum_force``, ``window_sensitivity``,
  ``assemble_curve_analysis``; ``computed_properties``,
  ``compare_with_reported``, ``specimen_summary``.
- Batch aggregation: ``gather_summaries`` and ``gather_comparisons``
  (``TransformNto1``), ``results`` (:func:`results_table`),
  ``summary_quantities``, ``orientation_summary`` (``summarize_by_group``),
  ``comparison`` (:func:`comparison_table`).

Public helpers
--------------
- :func:`build_exploration_workflow` -> a node; ``.run(profile_id=..., source=...)``.
- :func:`build_specimen_node` -> a node for one specimen.
- :func:`build_batch_workflow` (``specimen_ids``) -> ``pwf.Workflow``.
- :func:`batch_inputs` -> the keyword inputs for running the batch workflow.
- :func:`run_tensile_analysis` (``profile_id``, ``source``, ``curve_locator=None``)
  -> :class:`TensileAnalysis`: runs both stages and returns the run objects
  and the result tables.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import flowrep as fr
import pandas as pd
import pyiron_workflow as pwf

from tensile_workflow.acquisition import fetch_curve, get_reported_properties, get_specimen_record
from tensile_workflow.analysis import (
    AreaMM2,
    CrossheadArray,
    CurveAnalysis,
    ForceArray,
    OptionalAreaMM2,
    OptionalLengthMM,
    OptionalStrainPercent,
    OptionalStrainRatio,
    ScalarResult,
    S0Scatter,
    StrainArray,
    TimeArray,
    assemble_curve_analysis,
    bootstrap_fit,
    check_time_order,
    compare_with_reported,
    count_dropped_samples,
    elongation_after_fracture,
    elongation_at_maximum_force,
    extensometer_validity,
    final_elastic_fit,
    maximum_force,
    proof_strength,
    reduction_of_area,
    s0_scatter,
    summarize_by_group,
    window_sensitivity,
    yield_analysis,
)
from tensile_workflow.contract import Curve, SpecimenRecord
from tensile_workflow.exploration import check_conformance, list_specimens

GROUP_COLUMN = "orientation"
"""Grouping column of the per-orientation summary (method-spec §12)."""

SUMMARY_QUANTITIES: tuple[str, ...] = (
    "e_gpa",
    "reh_mpa",
    "rel_mpa",
    "rp02_mpa",
    "fm_n",
    "rm_mpa",
    "ag_percent",
    "agt_percent",
    "a_percent",
    "z_percent",
)
"""Quantities summarized per orientation (method-spec §12)."""

# ---------------------------------------------------------------------------
# Contract adapters (ontology-agnostic)
# ---------------------------------------------------------------------------


@fr.atomic
def curve_channels(
    curve: Curve,
) -> tuple[TimeArray, CrossheadArray, ForceArray, StrainArray, str, str]:
    """Unpack a neutral :class:`~tensile_workflow.contract.Curve` into annotated channels.

    Also returns the raw-curve SHA-256 and source URL as plain strings (IC3),
    because array values in a knowledge graph are lossy.
    """
    time_s = curve.time_s
    crosshead_mm = curve.crosshead_mm
    force_n = curve.force_n
    strain = curve.strain
    curve_sha256 = str(curve.sha256)
    curve_url = str(curve.source_url)
    return time_s, crosshead_mm, force_n, strain, curve_sha256, curve_url


@fr.atomic
def specimen_geometry(
    record: SpecimenRecord,
) -> tuple[AreaMM2, list, OptionalLengthMM, OptionalLengthMM, OptionalAreaMM2, OptionalStrainRatio]:
    """Unpack the geometry of a neutral :class:`~tensile_workflow.contract.SpecimenRecord`.

    Returns S0, the individual cross-section measurements, L0, Lu, Su and the
    rate-switch strain (diagnostic only), in contract units.
    """
    s0_mm2 = float(record.s0_mm2)
    measurements_mm2 = [float(value) for value in record.cross_section_measurements_mm2]
    l0_mm = record.l0_mm
    lu_mm = record.lu_mm
    su_mm2 = record.su_mm2
    rate_switch_strain = record.rate_switch_strain
    return s0_mm2, measurements_mm2, l0_mm, lu_mm, su_mm2, rate_switch_strain


@fr.atomic
def percent_value(result: ScalarResult) -> OptionalStrainPercent:
    """The value of A (percent) from :func:`~tensile_workflow.analysis.elongation_after_fracture`.

    Feeds the M3 end check of the extensometer (NaN if A is not available).
    """
    a_percent = float(result.value)
    return a_percent


@fr.atomic
def computed_properties(analysis: CurveAnalysis, elongation: ScalarResult, reduction: ScalarResult) -> dict:
    """The computed values in the keys of :func:`~tensile_workflow.analysis.compare_with_reported`."""
    computed = {
        "fm_n": analysis.fm_n,
        "rm_mpa": analysis.rm_mpa,
        "reh_mpa": analysis.reh_mpa,
        "rp02_mpa": analysis.rp02_mpa,
        "e_gpa": analysis.e_gpa,
        "a_percent": elongation.value,
        "z_percent": reduction.value,
    }
    return computed


@fr.atomic
def specimen_summary(
    record: SpecimenRecord,
    analysis: CurveAnalysis,
    elongation: ScalarResult,
    reduction: ScalarResult,
    scatter: S0Scatter,
    curve_sha256: str,
    curve_url: str,
) -> dict:
    """One result row per specimen: values, uncertainty components, flags, provenance.

    The S0 scatter ``s0_u_rel`` applies to every stress-based value (M14);
    the window ranges are the M15 sensitivity. Flags of A, Z and S0 are added
    to the curve flags.
    """
    flags = (*analysis.flags, *elongation.flags, *reduction.flags, *scatter.flags)
    summary = {
        "specimen_id": record.specimen_id,
        "orientation": record.orientation,
        "notes": "; ".join(record.notes),
        "yield_type": analysis.yield_type,
        "e_gpa": analysis.e_gpa,
        "reh_mpa": analysis.reh_mpa,
        "rel_mpa": analysis.rel_mpa,
        "rp02_mpa": analysis.rp02_mpa,
        "fm_n": analysis.fm_n,
        "rm_mpa": analysis.rm_mpa,
        "ag_percent": analysis.ag_percent,
        "agt_percent": analysis.agt_percent,
        "a_percent": elongation.value,
        "z_percent": reduction.value,
        "s0_mm2": record.s0_mm2,
        "s0_u_rel": scatter.u_rel,
        "s0_dof": scatter.dof,
        "e_window_min_gpa": analysis.e_window_range_gpa[0],
        "e_window_max_gpa": analysis.e_window_range_gpa[1],
        "rp02_window_min_mpa": analysis.rp02_window_range_mpa[0],
        "rp02_window_max_mpa": analysis.rp02_window_range_mpa[1],
        "e_stderr_gpa": analysis.e_stderr_gpa,
        "flags": "; ".join(flags),
        "curve_sha256": curve_sha256,
        "curve_url": curve_url,
    }
    return summary


@fr.atomic
def summary_quantities() -> list[str]:
    """The quantities of the per-orientation summary (:data:`SUMMARY_QUANTITIES`).

    A node rather than a constant: pyiron_workflow types a constant list as
    ``list``, which its edge check does not accept for the ``Sequence[str]``
    input of :func:`~tensile_workflow.analysis.summarize_by_group`.
    """
    quantities = list(SUMMARY_QUANTITIES)
    return quantities


@fr.atomic
def results_table(summaries: list) -> pd.DataFrame:
    """One row per specimen from the gathered :func:`specimen_summary` rows."""
    table = pd.DataFrame(list(summaries))
    return table


@fr.atomic
def comparison_table(summaries: list, comparisons: list) -> pd.DataFrame:
    """All per-specimen comparisons with reported values, labelled by specimen and orientation.

    ``summaries`` and ``comparisons`` are gathered in the same specimen order.
    """
    frames = [
        comparison.assign(specimen_id=summary["specimen_id"], orientation=summary["orientation"])
        for summary, comparison in zip(summaries, comparisons, strict=True)
    ]
    leading = ["specimen_id", "orientation"]
    combined = pd.concat(frames, ignore_index=True)
    table = combined[leading + [c for c in combined.columns if c not in leading]]
    return table


# ---------------------------------------------------------------------------
# Stage 1: exploration
# ---------------------------------------------------------------------------


@fr.workflow
def exploration_workflow(profile_id: str, source: dict):
    """Conformance check, then the specimen IDs described in the source.

    The listing depends on the checked profile ID, so it runs only after the
    conformance check has passed (a mismatch fails the run early).
    """
    checked_profile_id = check_conformance(profile_id, source)
    specimen_ids = list_specimens(checked_profile_id, source)
    return checked_profile_id, specimen_ids


# ---------------------------------------------------------------------------
# Stage 2: per-specimen workflow
# ---------------------------------------------------------------------------


@fr.workflow
def specimen_workflow(profile_id: str, source: dict, specimen_id: str, curve_locator: dict | None = None):
    """Acquisition -> analysis -> A, Z, S0 scatter -> comparison for one specimen (flat DAG)."""
    # Ontology-aware acquisition into the neutral contract.
    record = get_specimen_record(profile_id, source, specimen_id, curve_locator)
    reported = get_reported_properties(profile_id, source, specimen_id)
    curve = fetch_curve(record)
    # Contract -> annotated arrays and scalars.
    time_s, crosshead_mm, force_n, strain, curve_sha256, curve_url = curve_channels(curve)
    s0_mm2, measurements_mm2, l0_mm, lu_mm, su_mm2, rate_switch_strain = specimen_geometry(record)
    # Ontology-agnostic analysis (method-spec M1-M16).
    n_samples = check_time_order(time_s)
    elongation = elongation_after_fracture(l0_mm, lu_mm)
    reduction = reduction_of_area(s0_mm2, su_mm2)
    scatter = s0_scatter(s0_mm2, measurements_mm2)
    a_percent = percent_value(elongation)
    dropped_samples = count_dropped_samples(force_n, strain, crosshead_mm)
    extensometer, k_ext, extensometer_max_valid_strain = extensometer_validity(
        strain, crosshead_mm, l0_mm, a_percent, rate_switch_strain
    )
    bootstrap, noise_mpa = bootstrap_fit(force_n, strain, s0_mm2, k_ext)
    yielding, yield_type, reh_mpa, rel_mpa = yield_analysis(
        force_n, strain, s0_mm2, k_ext, bootstrap, rate_switch_strain
    )
    elastic, e_gpa, strain_origin = final_elastic_fit(force_n, strain, s0_mm2, k_ext, bootstrap, yielding)
    proof, rp02_mpa = proof_strength(force_n, strain, s0_mm2, k_ext, elastic, yielding, rate_switch_strain)
    maximum, fm_n, rm_mpa = maximum_force(force_n, s0_mm2, yielding)
    elongation_fm, agt_percent, ag_percent = elongation_at_maximum_force(
        force_n, strain, k_ext, bootstrap, elastic, maximum
    )
    sensitivity = window_sensitivity(force_n, strain, s0_mm2, k_ext, elastic, maximum, elongation_fm)
    analysis = assemble_curve_analysis(
        dropped_samples, extensometer, bootstrap, yielding, elastic, proof, maximum, elongation_fm, sensitivity
    )
    # Comparison with lab-reported values (consistency check, method-spec §13).
    computed = computed_properties(analysis, elongation, reduction)
    comparison = compare_with_reported(computed, reported)
    summary = specimen_summary(record, analysis, elongation, reduction, scatter, curve_sha256, curve_url)
    return (
        analysis,
        summary,
        comparison,
        curve_sha256,
        curve_url,
        n_samples,
        e_gpa,
        reh_mpa,
        rel_mpa,
        rp02_mpa,
        fm_n,
        rm_mpa,
        ag_percent,
        agt_percent,
    )


# ---------------------------------------------------------------------------
# Builders and the two-stage run
# ---------------------------------------------------------------------------


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
    :func:`tensile_workflow.knowledge_graph.run_knowledge_graph` exports.
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
