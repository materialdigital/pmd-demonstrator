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
(dict or ``None``; a non-semantic locator, accepted by ``pmdco3-tto`` and
rejected by ``pmdco2-tto``: it replaces the curve tables of any PMDco3 graph,
the hosted slices included, and is needed for the original TTO graph, which
has no curve tables; see ``acquisition.get_specimen_record``),
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
  between the dataclass and the dictionary form."""

from __future__ import annotations

import dataclasses

import flowrep as fr
import pandas as pd

from pmd_tensile.contract import Curve, SpecimenRecord
from pmd_tensile.curve_math import NAN, _unique
from pmd_tensile.nodes.acquisition import (
    fetch_curve,
    get_reported_properties,
    get_specimen_record,
)
from pmd_tensile.nodes.analysis.comparison import compare_with_reported
from pmd_tensile.nodes.analysis.elastic_line import (
    bootstrap_fit,
    final_elastic_fit,
    proof_strength,
    window_sensitivity,
)
from pmd_tensile.nodes.analysis.extensometer import (
    check_time_order,
    count_dropped_samples,
    extensometer_validity,
)
from pmd_tensile.nodes.analysis.geometry import (
    S0Scatter,
    ScalarResult,
    elongation_after_fracture,
    reduction_of_area,
    s0_scatter,
)
from pmd_tensile.nodes.analysis.maximum_force import (
    elongation_at_maximum_force,
    maximum_force,
)
from pmd_tensile.nodes.analysis.steps import (
    Bootstrap,
    ElasticResult,
    Elongation,
    ExtensometerValidity,
    MaximumForce,
    ProofStrength,
    WindowSensitivity,
    Yielding,
)
from pmd_tensile.nodes.analysis.yielding import yield_analysis
from pmd_tensile.nodes.exploration import check_conformance, list_specimens
from pmd_tensile.parameters import AnalysisParameters, parameters_from_values
from pmd_tensile.units import (
    AreaMM2,
    CrossheadArray,
    ForceArray,
    OptionalAreaMM2,
    OptionalLengthMM,
    OptionalStrainPercent,
    OptionalStrainRatio,
    StrainArray,
    TimeArray,
)


@dataclasses.dataclass(frozen=True)
class CurveAnalysis:
    """Per-specimen curve analysis (fields per analysis-api.md, then extras).

    Undefined values are NaN. ``k_ext`` equals the number of samples if the
    extensometer is valid to the end. ``flags`` entries are
    ``"<property>:<flag>"``. Fields after ``flags`` are additional
    diagnostics: the reference stress of the final E window, the final
    residual noise (1.4826*MAD, M5 step 5), the number of E-window
    iterations, the ReL plateau level and extent (M9), and sample indices
    (-1 if undefined) for plotting.
    """

    yield_type: str
    noise_mpa: float
    e_gpa: float
    e_intercept_mpa: float
    e_n: int
    e_r2: float
    e_stderr_gpa: float
    e_lag1_rho: float
    e_n_eff: float
    e_window_mpa: tuple[float, float]
    strain_origin: float
    reh_mpa: float
    reh_sample_increment_mpa: float
    rel_mpa: float
    rel_incl_transient_mpa: float
    rp02_mpa: float
    fm_n: float
    rm_mpa: float
    agt_percent: float
    ag_percent: float
    ag_location_halfwidth_percent: float
    k_ext: int
    extensometer_max_valid_strain: float
    e_window_range_gpa: tuple[float, float]
    rp02_window_range_mpa: tuple[float, float]
    ag_window_range_percent: tuple[float, float]
    agt_window_range_percent: tuple[float, float]
    e_linearity_rel_diff: float
    dropped_samples: dict
    flags: tuple[str, ...]
    e_reference_stress_mpa: float = NAN
    e_residual_noise_mpa: float = NAN
    e_window_iterations: int = 0
    rel_plateau_level_mpa: float = NAN
    rel_plateau_extent: float = NAN
    indices: dict = dataclasses.field(default_factory=dict)



@fr.atomic
def assemble_curve_analysis(
    dropped_samples: dict,
    extensometer: ExtensometerValidity,
    bootstrap: Bootstrap,
    yielding: Yielding,
    elastic: ElasticResult,
    proof: ProofStrength,
    maximum: MaximumForce,
    elongation: Elongation,
    sensitivity: WindowSensitivity,
) -> CurveAnalysis:
    """Collect the step results into one :class:`CurveAnalysis`."""
    fit = elastic.fit
    flags = _unique(
        extensometer.flags
        + elastic.flags
        + yielding.flags
        + proof.flags
        + maximum.flags
        + elongation.flags
        + sensitivity.flags
    )
    analysis = CurveAnalysis(
        yield_type=yielding.yield_type,
        noise_mpa=bootstrap.noise_mpa,
        e_gpa=fit.e_gpa,
        e_intercept_mpa=fit.intercept_mpa,
        e_n=fit.n,
        e_r2=fit.r2,
        e_stderr_gpa=fit.stderr_gpa,
        e_lag1_rho=fit.lag1_rho,
        e_n_eff=fit.n_eff,
        e_window_mpa=(fit.lo_mpa, fit.hi_mpa),
        strain_origin=elastic.strain_origin,
        reh_mpa=yielding.reh_mpa,
        reh_sample_increment_mpa=yielding.reh_sample_increment_mpa,
        rel_mpa=yielding.rel_mpa,
        rel_incl_transient_mpa=yielding.rel_incl_transient_mpa,
        rp02_mpa=proof.rp02_mpa,
        fm_n=maximum.fm_n,
        rm_mpa=maximum.rm_mpa,
        agt_percent=elongation.agt_percent,
        ag_percent=elongation.ag_percent,
        ag_location_halfwidth_percent=elongation.ag_location_halfwidth_percent,
        k_ext=extensometer.k_ext,
        extensometer_max_valid_strain=extensometer.max_valid_strain,
        e_window_range_gpa=sensitivity.e_range_gpa,
        rp02_window_range_mpa=sensitivity.rp02_range_mpa,
        ag_window_range_percent=sensitivity.ag_range_percent,
        agt_window_range_percent=sensitivity.agt_range_percent,
        e_linearity_rel_diff=sensitivity.linearity_rel_diff,
        dropped_samples=dict(dropped_samples),
        flags=flags,
        e_reference_stress_mpa=elastic.reference_mpa,
        e_residual_noise_mpa=elastic.residual_noise_mpa,
        e_window_iterations=elastic.iterations,
        rel_plateau_level_mpa=yielding.plateau_level_mpa,
        rel_plateau_extent=yielding.plateau_extent,
        indices={
            "reh": yielding.i_p,
            "rel": yielding.i_rel,
            "work_hardening_start": yielding.i_h,
            "rp02": proof.index,
            "fm": maximum.i_m,
            "ag_band": elongation.band,
        },
    )
    return analysis


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
