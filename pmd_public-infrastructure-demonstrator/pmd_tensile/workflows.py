"""Workflow assembly: exploration stage and per-specimen batch stage (decision D6).

The tensile analysis runs in two stages, both free of flow controllers
(``for``, ``while``, ``if``, ``try``) so that semantikon can export them as
knowledge graphs (upstream limitation pyiron/semantikon#541):

1. **Exploration** (:func:`exploration_workflow`, an ``@fr.workflow``):
   checks that the source conforms to the ontology profile and lists the
   specimen IDs it describes.
2. **Batch** (``build_batch_workflow``, a programmatically assembled
   ``pyiron_workflow.Workflow``): one :func:`specimen_workflow` node per
   specimen ID, each fed by its own workflow input (convention IC3), gathered
   with ``flowrep.schemas.TransformNto1`` into aggregation nodes: the results
   table, the per-orientation summary (:func:`~pmd_tensile.analysis.summarize_by_group`)
   and the combined comparison with lab-reported values.

The per-specimen workflow is a flat DAG: acquisition (record, reported
properties, curve fetch), the analysis step nodes in the order of
``analyse_curve``, A, Z, S0 scatter, and the
comparison with reported values. Its numbers equal ``analyse_curve`` on the
same arrays, with ``a_percent`` = A from the metadata (L0, Lu) and the
record's ``rate_switch_strain``.

Workflow inputs are JSON-able only (T1 finding, IC3): ``profile_id`` (str),
``source`` (dict, see ``pmd_tensile._sources``), ``curve_locator``
(dict or ``None``; only for profiles whose graph does not link the curves),
``parameters`` (dict or ``None``) and one ``specimen_id_<i>`` (str) per
specimen. Credentials are never inputs; the Ontodocker access path reads them
from the environment (decision D5).

``parameters`` holds the method parameters (decision D4) as a dictionary of
:class:`~pmd_tensile.analysis.AnalysisParameters` field values; missing keys
keep the method defaults, and ``None`` means all defaults (the form is
described in ``pmd_tensile._parameters``). The node :func:`analysis_parameters`
validates and converts it, and passes the dataclass to every analysis node
that takes ``params``, so a run's knowledge graph records the values used
(as the value of that node's output, and, for the batch, as the workflow
input, which ``batch_inputs`` fills with all field values).

Node roles (used by ``pmd_tensile._knowledge_graph``):

- ontology-aware: :mod:`pmd_tensile.exploration`,
  :mod:`pmd_tensile.acquisition`;
- ontology-agnostic analysis: :mod:`pmd_tensile.analysis`;
- ontology-agnostic contract adapters in this module
  (:func:`analysis_parameters`, :func:`curve_channels`,
  :func:`specimen_geometry`, :func:`percent_value`,
  :func:`computed_properties`, :func:`specimen_summary`,
  :func:`summary_quantities`, :func:`results_table`,
  :func:`comparison_table`): they only convert the JSON-able parameters,
  unpack or collect the neutral contract dataclasses and analysis results, and
  give the arrays and scalars their unit annotations;
- workflow compositions: :func:`exploration_workflow`, :func:`specimen_workflow`.

Stage nodes
-----------
- Exploration: ``check_conformance`` -> ``list_specimens``.
- Per specimen (``specimen_<i>``): ``analysis_parameters``;
  ``get_specimen_record``, ``get_reported_properties``, ``fetch_curve``;
  ``curve_channels``,
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

Helpers
-------
This module holds only nodes and macros, so that sim_atlas catalogues
nothing else. The builders and the two-stage run are plain functions in the
private module ``pmd_tensile._orchestration``; notebooks reach them through
:mod:`pmd_tensile.demo`:

- ``build_exploration_workflow`` -> a node; ``.run(profile_id=..., source=...)``.
- ``build_specimen_node`` -> a node for one specimen.
- ``build_batch_workflow`` (``specimen_ids``) -> ``pwf.Workflow``.
- ``batch_inputs`` -> the keyword inputs for running the batch workflow.
- ``run_tensile_analysis`` (``profile_id``, ``source``, ``curve_locator=None``,
  ``*``, ``specimen_ids=None``, ``parameters=None``) -> ``TensileAnalysis``:
  runs both stages and returns the run objects, the result tables and, per
  specimen, the curve and curve analysis the run used.
- ``specimen_curves`` (``batch_run``, ``specimen_ids``) -> the curves and
  curve analyses of a finished batch run, from its run data.
- ``DEFAULT_PARAMETERS``, ``parameter_values`` and ``parameters_from_values``
  (``pmd_tensile._parameters``): the method defaults and the conversion
  between the dataclass and the dictionary form.
"""

from __future__ import annotations

import flowrep as fr
import pandas as pd

from pmd_tensile._parameters import parameters_from_values
from pmd_tensile.acquisition import fetch_curve, get_reported_properties, get_specimen_record
from pmd_tensile.analysis import (
    AnalysisParameters,
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
    window_sensitivity,
    yield_analysis,
)
from pmd_tensile.contract import Curve, SpecimenRecord
from pmd_tensile.exploration import check_conformance, list_specimens

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
def analysis_parameters(parameters: dict | None = None) -> AnalysisParameters:
    """The method parameters from their JSON-able dictionary form (decision D4).

    Keys are :class:`~pmd_tensile.analysis.AnalysisParameters` field names;
    missing keys keep the method defaults, ``None`` gives all defaults.
    Unknown keys and invalid values raise ``ValueError`` or ``TypeError``
    (rules in ``pmd_tensile._parameters``).
    """
    params = parameters_from_values(parameters)
    return params


@fr.atomic
def curve_channels(
    curve: Curve,
) -> tuple[TimeArray, CrossheadArray, ForceArray, StrainArray, str, str]:
    """Unpack a neutral :class:`~pmd_tensile.contract.Curve` into annotated channels.

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
    """Unpack the geometry of a neutral :class:`~pmd_tensile.contract.SpecimenRecord`.

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
    """The value of A (percent) from :func:`~pmd_tensile.analysis.elongation_after_fracture`.

    Feeds the M3 end check of the extensometer (NaN if A is not available).
    """
    a_percent = float(result.value)
    return a_percent


@fr.atomic
def computed_properties(analysis: CurveAnalysis, elongation: ScalarResult, reduction: ScalarResult) -> dict:
    """The computed values in the keys of :func:`~pmd_tensile.analysis.compare_with_reported`."""
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
    to the curve flags. ``*_range_*`` entries are (min, max) tuples,
    ``e_window_mpa`` is (lo, hi) and ``dropped_samples`` is a dictionary
    channel -> count, as in :class:`~pmd_tensile.analysis.CurveAnalysis`.
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
        "ag_location_halfwidth_percent": analysis.ag_location_halfwidth_percent,
        "ag_window_range_percent": analysis.ag_window_range_percent,
        "agt_window_range_percent": analysis.agt_window_range_percent,
        "e_window_range_gpa": analysis.e_window_range_gpa,
        "e_linearity_rel_diff": analysis.e_linearity_rel_diff,
        "e_lag1_rho": analysis.e_lag1_rho,
        "e_n_eff": analysis.e_n_eff,
        "e_n": analysis.e_n,
        "e_r2": analysis.e_r2,
        "e_window_mpa": analysis.e_window_mpa,
        "rel_incl_transient_mpa": analysis.rel_incl_transient_mpa,
        "reh_sample_increment_mpa": analysis.reh_sample_increment_mpa,
        "strain_origin": analysis.strain_origin,
        "dropped_samples": analysis.dropped_samples,
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
    input of :func:`~pmd_tensile.analysis.summarize_by_group`.
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
def specimen_workflow(
    profile_id: str,
    source: dict,
    specimen_id: str,
    curve_locator: dict | None = None,
    parameters: dict | None = None,
):
    """Acquisition -> analysis -> A, Z, S0 scatter -> comparison for one specimen (flat DAG).

    ``parameters`` is the dictionary form of the method parameters (``None``:
    the method defaults); see the module docstring.
    """
    params = analysis_parameters(parameters)
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
        strain, crosshead_mm, l0_mm, a_percent, rate_switch_strain, params
    )
    bootstrap, noise_mpa = bootstrap_fit(force_n, strain, s0_mm2, k_ext, params)
    yielding, yield_type, reh_mpa, rel_mpa = yield_analysis(
        force_n, strain, s0_mm2, k_ext, bootstrap, rate_switch_strain, params
    )
    elastic, e_gpa, strain_origin = final_elastic_fit(
        force_n, strain, s0_mm2, k_ext, bootstrap, yielding, params
    )
    proof, rp02_mpa = proof_strength(
        force_n, strain, s0_mm2, k_ext, elastic, yielding, rate_switch_strain, params
    )
    maximum, fm_n, rm_mpa = maximum_force(force_n, s0_mm2, yielding)
    elongation_fm, agt_percent, ag_percent = elongation_at_maximum_force(
        force_n, strain, k_ext, bootstrap, elastic, maximum, params
    )
    sensitivity = window_sensitivity(force_n, strain, s0_mm2, k_ext, elastic, maximum, elongation_fm, params)
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
