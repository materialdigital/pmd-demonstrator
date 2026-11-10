"""Workflow assembly: the exploration stage and the per-specimen stage.

The tensile analysis runs in two stages without loops or branches in the
workflow, so that semantikon can export both as knowledge graphs:

1. :func:`exploration_workflow` checks that the source conforms to the
   ontology profile and lists the specimen IDs it describes.
2. :func:`specimen_workflow` runs acquisition, analysis and the comparison
   with the lab values for one specimen. ``pmd_tensile.orchestration`` builds
   a batch with one such node per specimen and gathers the rows of the results
   table, the statistics per orientation and the comparisons.

Workflow inputs are JSON-able: ``profile_id``, ``source`` (see
``pmd_tensile.sources``), ``curve_locator`` (a non-semantic curve location,
see :func:`~pmd_tensile.nodes.acquisition.get_specimen_record`),
``specimen_id`` and ``parameters`` (field values of
:class:`~pmd_tensile.parameters.AnalysisParameters`; missing keys keep the
method defaults). Credentials are never inputs.

The nodes here only convert between the neutral records and the annotated
arrays and values of the analysis nodes, or collect results into tables.
"""

from __future__ import annotations

import dataclasses

import flowrep as fr
import pandas as pd

from pmd_tensile.contract import Curve, SpecimenRecord
from pmd_tensile.curve_math import _unique
from pmd_tensile.nodes.acquisition import (
    fetch_curve,
    get_reported_properties,
    get_specimen_record,
)
from pmd_tensile.nodes.analysis.comparison import compare_with_reported
from pmd_tensile.nodes.analysis.elastic_line import (
    ElasticLineDiagnostics,
    ProofStrengthDiagnostics,
    WindowSensitivityDiagnostics,
    elastic_line,
    proof_strength,
    provisional_elastic_line,
    window_sensitivity,
)
from pmd_tensile.nodes.analysis.extensometer import (
    DroppedSamples,
    ExtensometerDiagnostics,
    check_time_order,
    count_dropped_samples,
    extensometer_validity,
)
from pmd_tensile.nodes.analysis.geometry import (
    elongation_after_fracture,
    reduction_of_area,
    s0_scatter,
)
from pmd_tensile.nodes.analysis.maximum_force import (
    ElongationDiagnostics,
    elongation_at_maximum_force,
    maximum_force,
)
from pmd_tensile.nodes.analysis.yielding import YieldDiagnostics, yield_points
from pmd_tensile.nodes.exploration import check_conformance, list_specimens
from pmd_tensile.parameters import AnalysisParameters, parameters_from_values
from pmd_tensile.units import (
    AreaMM2,
    CrossheadArray,
    ForceArray,
    OptionalAreaMM2,
    OptionalLengthMM,
    OptionalStrainRatio,
    StrainArray,
    TimeArray,
)

GROUP_COLUMN = "orientation"
"""Grouping column of the statistics per orientation."""

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
"""Quantities summarized per orientation."""


@fr.atomic
def analysis_parameters(parameters: dict | None = None) -> AnalysisParameters:
    """The method parameters from their dictionary form.

    Keys are :class:`~pmd_tensile.parameters.AnalysisParameters` field names;
    missing keys keep the method defaults, ``None`` gives all defaults. Unknown
    keys and invalid values raise ``ValueError`` or ``TypeError``.
    """
    params = parameters_from_values(parameters)
    return params


@fr.atomic
def curve_channels(curve: Curve) -> tuple[TimeArray, CrossheadArray, ForceArray, StrainArray, str, str]:
    """The channels of a neutral curve as annotated arrays: time (s), crosshead (mm),
    force (N), strain (ratio); and the SHA-256 and URL of the raw curve file, as text,
    because array values in a knowledge graph are lossy."""
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
    """The geometry of a neutral specimen record: S0 (mm²), the individual
    cross-section measurements (mm²), L0, Lu (mm), Su (mm²) and the strain at which
    the test rate changes (ratio; used only for flags)."""
    s0_mm2 = float(record.s0_mm2)
    measurements_mm2 = [float(value) for value in record.cross_section_measurements_mm2]
    l0_mm = record.l0_mm
    lu_mm = record.lu_mm
    su_mm2 = record.su_mm2
    rate_switch_strain = record.rate_switch_strain
    return s0_mm2, measurements_mm2, l0_mm, lu_mm, su_mm2, rate_switch_strain


@fr.atomic
def specimen_row(
    record: SpecimenRecord, curve_sha256: str, curve_url: str, yield_type: str, e_gpa: float, strain_origin: float,
    reh_mpa: float, rel_mpa: float, rp02_mpa: float, fm_n: float, rm_mpa: float, agt_percent: float,
    ag_percent: float, a_percent: float, z_percent: float, s0_u_rel: float, s0_dof: int, e_min_gpa: float,
    e_max_gpa: float, rp02_min_mpa: float, rp02_max_mpa: float, dropped: DroppedSamples,
    extensometer: ExtensometerDiagnostics, yielding: YieldDiagnostics, elastic: ElasticLineDiagnostics,
    proof: ProofStrengthDiagnostics, maximum_flags: tuple, elongation: ElongationDiagnostics,
    sensitivity: WindowSensitivityDiagnostics, a_flags: tuple, z_flags: tuple, s0_flags: tuple,
) -> dict:
    """One row of the results table: values, uncertainty components, flags, provenance.

    The S0 scatter ``s0_u_rel`` applies to every stress and to E; the window
    ranges describe the sensitivity to the elastic window. ``flags`` joins the
    flags of all steps with "; ".
    """
    flags = (
        *_unique((*extensometer.flags, *elastic.flags, *yielding.flags, *proof.flags, *maximum_flags,
                  *elongation.flags, *sensitivity.flags)),
        *a_flags, *z_flags, *s0_flags,
    )
    row = {
        "specimen_id": record.specimen_id,
        "orientation": record.orientation,
        "notes": "; ".join(record.notes),
        "yield_type": yield_type,
        "e_gpa": e_gpa,
        "reh_mpa": reh_mpa,
        "rel_mpa": rel_mpa,
        "rp02_mpa": rp02_mpa,
        "fm_n": fm_n,
        "rm_mpa": rm_mpa,
        "ag_percent": ag_percent,
        "agt_percent": agt_percent,
        "a_percent": a_percent,
        "z_percent": z_percent,
        "s0_mm2": record.s0_mm2,
        "s0_u_rel": s0_u_rel,
        "s0_dof": s0_dof,
        "e_window_min_gpa": e_min_gpa,
        "e_window_max_gpa": e_max_gpa,
        "rp02_window_min_mpa": rp02_min_mpa,
        "rp02_window_max_mpa": rp02_max_mpa,
        "e_stderr_gpa": elastic.stderr_gpa,
        "ag_location_halfwidth_percent": elongation.halfwidth_percent,
        "ag_window_range_percent": sensitivity.ag_range_percent,
        "agt_window_range_percent": sensitivity.agt_range_percent,
        "e_linearity_rel_diff": sensitivity.linearity_rel_diff,
        "e_lag1_rho": elastic.lag1_rho,
        "e_n_eff": elastic.n_eff,
        "e_n": elastic.n,
        "e_r2": elastic.r2,
        "e_window_mpa": elastic.window_mpa,
        "rel_incl_transient_mpa": yielding.rel_incl_transient_mpa,
        "reh_sample_increment_mpa": yielding.reh_sample_increment_mpa,
        "strain_origin": strain_origin,
        "dropped_samples": dataclasses.asdict(dropped),
        "flags": "; ".join(flags),
        "curve_sha256": curve_sha256,
        "curve_url": curve_url,
    }
    return row


@fr.atomic
def summary_quantities() -> list[str]:
    """The quantities of the statistics per orientation (:data:`SUMMARY_QUANTITIES`).

    A node rather than a constant: pyiron_workflow types a constant list as
    ``list``, which its edge check does not accept for the ``Sequence[str]``
    input of :func:`~pmd_tensile.nodes.analysis.comparison.summarize_by_group`.
    """
    quantities = list(SUMMARY_QUANTITIES)
    return quantities


@fr.atomic
def results_table(rows: list) -> pd.DataFrame:
    """One row per specimen from the gathered :func:`specimen_row` rows."""
    table = pd.DataFrame(list(rows))
    return table


@fr.atomic
def comparison_table(rows: list, comparisons: list) -> pd.DataFrame:
    """All comparisons with the lab values, labelled by specimen and orientation.

    ``rows`` and ``comparisons`` are gathered in the same specimen order.
    """
    frames = [
        comparison.assign(specimen_id=row["specimen_id"], orientation=row["orientation"])
        for row, comparison in zip(rows, comparisons, strict=True)
    ]
    leading = ["specimen_id", "orientation"]
    combined = pd.concat(frames, ignore_index=True)
    table = combined[leading + [c for c in combined.columns if c not in leading]]
    return table


@fr.workflow
def exploration_workflow(profile_id: str, source: dict):
    """Conformance check, then the specimen IDs described in the source.

    The listing depends on the checked profile ID, so it runs only after the
    conformance check has passed (a mismatch fails the run early).
    """
    checked_profile_id = check_conformance(profile_id, source)
    specimen_ids = list_specimens(checked_profile_id, source)
    return checked_profile_id, specimen_ids


@fr.workflow
def specimen_workflow(
    profile_id: str,
    source: dict,
    specimen_id: str,
    curve_locator: dict | None = None,
    parameters: dict | None = None,
):
    """Acquisition, analysis, A, Z and the S0 scatter, and the comparison with the lab values for one specimen.

    ``parameters`` is the dictionary form of the method parameters (``None``:
    the method defaults).
    """
    params = analysis_parameters(parameters)
    # Ontology-aware acquisition into the neutral records.
    record = get_specimen_record(profile_id, source, specimen_id, curve_locator)
    reported = get_reported_properties(profile_id, source, specimen_id)
    curve = fetch_curve(record)
    # Neutral records -> annotated arrays and values.
    time_s, crosshead_mm, force_n, strain, curve_sha256, curve_url = curve_channels(curve)
    s0_mm2, measurements_mm2, l0_mm, lu_mm, su_mm2, rate_switch_strain = specimen_geometry(record)
    # Ontology-agnostic analysis.
    n_samples = check_time_order(time_s)
    a_percent, a_flags = elongation_after_fracture(l0_mm, lu_mm)
    z_percent, z_flags = reduction_of_area(s0_mm2, su_mm2)
    s0_u_rel, s0_dof, s0_flags = s0_scatter(s0_mm2, measurements_mm2)
    dropped = count_dropped_samples(force_n, strain, crosshead_mm)
    k_ext, max_valid_strain, extensometer = extensometer_validity(
        strain, crosshead_mm, l0_mm, a_percent, rate_switch_strain, params
    )
    highest_force_n, e0_gpa, intercept0_mpa, noise_mpa, provisional = provisional_elastic_line(
        force_n, strain, s0_mm2, k_ext, params
    )
    yield_type, reh_mpa, rel_mpa, upper_yield_index, end_of_yielding_index, work_hardening, yielding = yield_points(
        force_n, strain, s0_mm2, k_ext, highest_force_n, e0_gpa, intercept0_mpa, noise_mpa, rate_switch_strain, params
    )
    e_gpa, strain_origin, reference_force_n, window_lo_n, elastic = elastic_line(
        force_n, strain, s0_mm2, k_ext, upper_yield_index, highest_force_n, e0_gpa, intercept0_mpa, params
    )
    rp02_mpa, proof = proof_strength(
        force_n, strain, s0_mm2, k_ext, e_gpa, strain_origin, window_lo_n, upper_yield_index,
        end_of_yielding_index, rate_switch_strain, params,
    )
    fm_n, rm_mpa, max_force_index, maximum_flags = maximum_force(
        force_n, s0_mm2, yield_type, work_hardening, end_of_yielding_index
    )
    agt_percent, ag_percent, elongation = elongation_at_maximum_force(
        force_n, strain, s0_mm2, k_ext, e_gpa, strain_origin, fm_n, max_force_index, noise_mpa, params
    )
    e_min_gpa, e_max_gpa, rp02_min_mpa, rp02_max_mpa, sensitivity = window_sensitivity(
        force_n, strain, s0_mm2, k_ext, reference_force_n, e_gpa, strain_origin, agt_percent, fm_n, params
    )
    # Comparison with the lab-reported values (a consistency check).
    comparison = compare_with_reported(fm_n, rm_mpa, reh_mpa, rp02_mpa, e_gpa, a_percent, z_percent, reported)
    row = specimen_row(
        record, curve_sha256, curve_url, yield_type, e_gpa, strain_origin, reh_mpa, rel_mpa, rp02_mpa, fm_n,
        rm_mpa, agt_percent, ag_percent, a_percent, z_percent, s0_u_rel, s0_dof, e_min_gpa, e_max_gpa,
        rp02_min_mpa, rp02_max_mpa, dropped, extensometer, yielding, elastic, proof,
        maximum_flags, elongation, sensitivity, a_flags, z_flags, s0_flags,
    )
    return (
        row,
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
