"""Plain-function composition of the analysis nodes (ontology-agnostic).

:mod:`pmd_tensile.analysis` holds the step nodes; :func:`analyse_curve` calls
them directly in the M5 bootstrap order. It is private so that sim_atlas does
not catalogue it as a node; notebooks reach it through :mod:`pmd_tensile.demo`.
Like ``analysis``, this module imports no ontology-aware module.
"""

from __future__ import annotations

from pmd_tensile.analysis import (
    AnalysisParameters,
    CurveAnalysis,
    _check_s0,
    assemble_curve_analysis,
    bootstrap_fit,
    count_dropped_samples,
    elongation_at_maximum_force,
    extensometer_validity,
    final_elastic_fit,
    maximum_force,
    proof_strength,
    window_sensitivity,
    yield_analysis,
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
) -> CurveAnalysis:
    """Analyse one curve (M1-M12, M15-M16) in the M5 bootstrap order.

    Arrays may contain NaN (per-channel handling, M1). ``a_percent`` (A from
    metadata) enables the M3 end check; ``rate_switch_strain`` (ratio) only
    sets ``near_rate_switch`` flags. This is a plain composition of the step
    nodes listed in the docstring of :mod:`pmd_tensile.analysis`.
    """
    _check_s0(s0_mm2)
    dropped_samples = count_dropped_samples(force_n, strain, crosshead_mm)
    extensometer, k_ext, _ = extensometer_validity(
        strain, crosshead_mm, l0_mm, a_percent, rate_switch_strain, params
    )
    bootstrap, _ = bootstrap_fit(force_n, strain, s0_mm2, k_ext, params)
    yielding, _, _, _ = yield_analysis(force_n, strain, s0_mm2, k_ext, bootstrap, rate_switch_strain, params)
    elastic, _, _ = final_elastic_fit(force_n, strain, s0_mm2, k_ext, bootstrap, yielding, params)
    proof, _ = proof_strength(force_n, strain, s0_mm2, k_ext, elastic, yielding, rate_switch_strain, params)
    maximum, _, _ = maximum_force(force_n, s0_mm2, yielding)
    elongation, _, _ = elongation_at_maximum_force(force_n, strain, k_ext, bootstrap, elastic, maximum, params)
    sensitivity = window_sensitivity(force_n, strain, s0_mm2, k_ext, elastic, maximum, elongation, params)
    analysis = assemble_curve_analysis(
        dropped_samples, extensometer, bootstrap, yielding, elastic, proof, maximum, elongation, sensitivity
    )
    return analysis
