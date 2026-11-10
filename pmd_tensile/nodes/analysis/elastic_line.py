"""Elastic line, proof strength and window sensitivity (ontology-agnostic analysis nodes)."""

from __future__ import annotations

import math

import flowrep as fr
import numpy as np

from pmd_tensile.curve_math import (
    _CROSSING_ALREADY_BEYOND,
    _CROSSING_BELOW_OFFSET,
    _CROSSING_UNDEFINED,
    _OLS_UNDEFINED,
    NAN,
    _channels,
    _check_s0,
    _near_switch,
    _Ols,
    _ols,
    _params,
    _proof_crossing,
    _strain_at,
    _strain_valid,
    _value_range,
    _window_ols,
)
from pmd_tensile.nodes.analysis.steps import (
    Bootstrap,
    ElasticFit,
    ElasticResult,
    Elongation,
    MaximumForce,
    ProofStrength,
    WindowSensitivity,
    Yielding,
)
from pmd_tensile.nodes.analysis.yielding import DISCONTINUOUS
from pmd_tensile.parameters import DEFAULT_PARAMETERS, AnalysisParameters
from pmd_tensile.units import (
    AreaMM2,
    ForceArray,
    ModulusGPa,
    OptionalStrainRatio,
    StrainArray,
    StrainRatio,
    StressArray,
    StressMPa,
)

# ---------------------------------------------------------------------------
# M4: elastic fit (public helper)
# ---------------------------------------------------------------------------


@fr.atomic
def elastic_fit(
    stress_mpa: StressArray,
    strain: StrainArray,
    lo_mpa: StressMPa,
    hi_mpa: StressMPa,
    *,
    min_points: int = DEFAULT_PARAMETERS.min_elastic_points,
) -> ElasticFit:
    """OLS sigma = E*e + b on the samples with ``lo_mpa <= sigma <= hi_mpa`` (M4).

    The caller passes the first-loading-branch arrays. Samples where either
    channel is non-finite are ignored. With fewer than ``min_points`` samples
    in the window (M4: 10), all fit values are NaN and ``n`` is still given.
    """
    stress = np.asarray(stress_mpa, dtype=float)
    eps = np.asarray(strain, dtype=float)
    mask = np.isfinite(stress) & np.isfinite(eps) & (stress >= lo_mpa) & (stress <= hi_mpa)
    ols = _ols(eps[mask], stress[mask], min_points)
    fit = ElasticFit(
        e_gpa=ols.slope / 1000.0,
        intercept_mpa=ols.intercept,
        n=ols.n,
        r2=ols.r2,
        stderr_gpa=ols.stderr / 1000.0,
        lag1_rho=ols.lag1_rho,
        n_eff=ols.n_eff,
        lo_mpa=float(lo_mpa),
        hi_mpa=float(hi_mpa),
    )
    return fit


_PROVISIONAL_MIN_POINTS = 3
"""M5 step 2: the provisional fit needs residuals for the noise level."""



@fr.atomic
def engineering_stress(force_n: ForceArray, s0_mm2: AreaMM2) -> StressArray:
    """sigma = F/S0 in MPa (for display; the analysis decides in force space)."""
    stress_mpa = np.asarray(force_n, dtype=float) / _check_s0(s0_mm2)
    return stress_mpa



@fr.atomic
def bootstrap_fit(
    force_n: ForceArray,
    strain: StrainArray,
    s0_mm2: AreaMM2,
    k_ext: int,
    params: AnalysisParameters | None = None,
) -> tuple[Bootstrap, StressMPa]:
    """M5 steps 1-2: Fm from force alone, provisional E0/b0 and the noise level s_n.

    The provisional window is [f_lo, f_hi]*(0.5*Rm) on the samples before
    sigma first reaches 0.5*Rm, with Rm = max over all finite forces / S0.
    s_n = 1.4826*MAD of its residuals; it is fixed here for M7, M9 and M12.
    """
    p = _params(params)
    s0 = _check_s0(s0_mm2)
    force, eps, _ = _channels(force_n, strain)
    finite_force = force[np.isfinite(force)]
    fm = float(finite_force.max()) if len(finite_force) else NAN
    valid = _strain_valid(force, eps, k_ext)
    reference = p.provisional_fraction_of_rm * fm
    ols, _, _ = _window_ols(force, eps, valid, reference, p.f_lo, p.f_hi, _PROVISIONAL_MIN_POINTS)
    bootstrap = Bootstrap(
        fm_n=fm,
        slope_n=ols.slope,
        intercept_n=ols.intercept,
        noise_n=ols.residual_noise,
        n=ols.n,
        noise_mpa=ols.residual_noise / s0,
    )
    noise_mpa = bootstrap.noise_mpa
    return bootstrap, noise_mpa



def _continuous_window(
    force: np.ndarray, eps: np.ndarray, valid: np.ndarray, boot: Bootstrap, p: AnalysisParameters
) -> tuple[_Ols, float, float, float, int, str]:
    """M5 step 4 for continuous yielding: fixed point sigma_ref = Rp0.2(E).

    Returns (fit, lo, hi, reference, iterations, status) in force units;
    status is ``converged``, ``not_converged``, ``insufficient`` or
    ``no_reference``.
    """
    undefined = _Ols(n=0, **_OLS_UNDEFINED)
    slope, intercept = boot.slope_n, boot.intercept_n
    if not math.isfinite(slope):
        return undefined, NAN, NAN, NAN, 0, "insufficient"
    start = p.f_lo * p.provisional_fraction_of_rm * boot.fm_n
    rp, _, _ = _proof_crossing(force, eps, valid, slope, intercept, start, p.offset_plastic_strain)
    ols, lo, hi, reference = undefined, NAN, NAN, NAN
    iterations, status = 0, "no_reference"
    for _ in range(p.max_window_iterations):
        if not math.isfinite(rp):
            break
        reference = rp
        ols, lo, hi = _window_ols(force, eps, valid, reference, p.f_lo, p.f_hi, p.min_elastic_points)
        iterations += 1
        if not math.isfinite(ols.slope):
            status = "insufficient"
            break
        change = abs(ols.slope - slope) / abs(slope)
        slope = ols.slope
        rp, _, _ = _proof_crossing(force, eps, valid, ols.slope, ols.intercept, lo, p.offset_plastic_strain)
        if change < p.window_rel_tol:
            status = "converged"
            break
        status = "not_converged"
    return ols, lo, hi, reference, iterations, status



@fr.atomic
def final_elastic_fit(
    force_n: ForceArray,
    strain: StrainArray,
    s0_mm2: AreaMM2,
    k_ext: int,
    bootstrap: Bootstrap,
    yielding: Yielding,
    params: AnalysisParameters | None = None,
) -> tuple[ElasticResult, ModulusGPa, StrainRatio]:
    """M4, M5 steps 4-5, M6, M16: final E window fit and strain origin e0 = -b/E.

    sigma_ref = ReH for discontinuous yielding; for continuous yielding the
    fixed point sigma_ref = Rp0.2(E) from the Rp0.2 with E0 (relative change
    of E < ``window_rel_tol``, at most ``max_window_iterations``). Flags:
    ``e:insufficient_elastic_points`` (fewer than ``min_elastic_points``)
    and ``e:e_window_not_converged``. E is labelled :data:`E_LABEL`.
    """
    p = _params(params)
    s0 = _check_s0(s0_mm2)
    force, eps, _ = _channels(force_n, strain)
    valid = _strain_valid(force, eps, k_ext)
    flags = []
    if yielding.yield_type == DISCONTINUOUS:
        reference = float(force[yielding.i_p])
        ols, lo, hi = _window_ols(force, eps, valid, reference, p.f_lo, p.f_hi, p.min_elastic_points)
        iterations = 1
        status = "converged" if math.isfinite(ols.slope) else "insufficient"
    else:
        ols, lo, hi, reference, iterations, status = _continuous_window(force, eps, valid, bootstrap, p)
    if status == "insufficient":
        flags.append("e:insufficient_elastic_points")
    elif status in ("not_converged", "no_reference"):
        flags.append("e:e_window_not_converged")
    fit = ElasticFit(
        e_gpa=ols.slope / s0 / 1000.0,
        intercept_mpa=ols.intercept / s0,
        n=ols.n,
        r2=ols.r2,
        stderr_gpa=ols.stderr / s0 / 1000.0,
        lag1_rho=ols.lag1_rho,
        n_eff=ols.n_eff,
        lo_mpa=lo / s0,
        hi_mpa=hi / s0,
    )
    elastic = ElasticResult(
        fit=fit,
        slope_n=ols.slope,
        intercept_n=ols.intercept,
        strain_origin=-ols.intercept / ols.slope if math.isfinite(ols.slope) else NAN,
        reference_n=reference,
        reference_mpa=reference / s0,
        lo_n=lo,
        hi_n=hi,
        residual_noise_mpa=ols.residual_noise / s0,
        iterations=iterations,
        flags=tuple(flags),
    )
    e_gpa = elastic.fit.e_gpa
    strain_origin = elastic.strain_origin
    return elastic, e_gpa, strain_origin



@fr.atomic
def proof_strength(
    force_n: ForceArray,
    strain: StrainArray,
    s0_mm2: AreaMM2,
    k_ext: int,
    elastic: ElasticResult,
    yielding: Yielding,
    rate_switch_strain: OptionalStrainRatio = None,
    params: AnalysisParameters | None = None,
) -> tuple[ProofStrength, StressMPa]:
    """M10: Rp0.2 at the first crossing of e_p = 0.002 (linear interpolation).

    Flags: ``rp02:extensometer_invalid`` (crossing not reached before
    ``k_ext``, with the extensometer ending early), ``rp02:not_reached`` (not
    reached in a record with valid strain to its end, or the first in-window
    sample already exceeds the offset), ``rp02:no_elastic_modulus`` (E is NaN,
    so Rp0.2 cannot be evaluated), ``rp02:in_yield_plateau``
    (discontinuous yielding, crossing in [i_p, i_h]) and
    ``rp02:near_rate_switch``.
    """
    p = _params(params)
    s0 = _check_s0(s0_mm2)
    force, eps, _ = _channels(force_n, strain)
    valid = _strain_valid(force, eps, k_ext)
    crossing, index, status = _proof_crossing(
        force, eps, valid, elastic.slope_n, elastic.intercept_n, elastic.lo_n, p.offset_plastic_strain
    )
    flags = []
    # A continuous record without a first crossing has no E window ("no reference").
    no_reference = status == _CROSSING_UNDEFINED and "e:e_window_not_converged" in elastic.flags
    if status == _CROSSING_BELOW_OFFSET or no_reference:
        flags.append("rp02:extensometer_invalid" if k_ext < len(force) else "rp02:not_reached")
    elif status == _CROSSING_ALREADY_BEYOND:
        flags.append("rp02:not_reached")
    elif not math.isfinite(elastic.slope_n):
        flags.append("rp02:no_elastic_modulus")
    if index >= 0:
        if yielding.yield_type == DISCONTINUOUS and yielding.i_p <= index <= yielding.i_h:
            flags.append("rp02:in_yield_plateau")
        if _near_switch(_strain_at(eps, index), rate_switch_strain, p):
            flags.append("rp02:near_rate_switch")
    proof = ProofStrength(rp02_mpa=crossing / s0, index=index, flags=tuple(flags))
    rp02_mpa = proof.rp02_mpa
    return proof, rp02_mpa



@fr.atomic
def window_sensitivity(
    force_n: ForceArray,
    strain: StrainArray,
    s0_mm2: AreaMM2,
    k_ext: int,
    elastic: ElasticResult,
    maximum: MaximumForce,
    elongation: Elongation,
    params: AnalysisParameters | None = None,
) -> WindowSensitivity:
    """M15: E-window sensitivity with sigma_ref fixed at its nominal value.

    Ranges (min, max) of E, Rp0.2, Ag and Agt over the grid f_lo x f_hi, and
    the linearity diagnostic (E_b - E_a)/E_a for the windows 0.05-0.25 (a)
    and 0.25-0.50 (b). Describes sensitivity only; never selects a window.
    Flag ``window:insufficient_elastic_points`` if a grid or linearity window
    has too few points.
    """
    p = _params(params)
    s0 = _check_s0(s0_mm2)
    force, eps, _ = _channels(force_n, strain)
    valid = _strain_valid(force, eps, k_ext)
    reference = elastic.reference_n
    e_values, rp_values, ag_values, agt_values = [], [], [], []
    short = False
    for f_lo in p.window_grid_lo:
        for f_hi in p.window_grid_hi:
            ols, lo, _ = _window_ols(force, eps, valid, reference, f_lo, f_hi, p.min_elastic_points)
            if not math.isfinite(ols.slope):
                short = short or math.isfinite(reference)
                continue
            e_values.append(ols.slope / s0 / 1000.0)
            crossing, _, _ = _proof_crossing(
                force, eps, valid, ols.slope, ols.intercept, lo, p.offset_plastic_strain
            )
            rp_values.append(crossing / s0)
            if math.isfinite(elongation.strain_mid) and math.isfinite(maximum.fm_n):
                agt = (elongation.strain_mid + ols.intercept / ols.slope) * 100.0
                agt_values.append(agt)
                ag_values.append(agt - maximum.fm_n / ols.slope * 100.0)
    moduli = []
    for f_lo, f_hi in p.linearity_windows:
        ols, _, _ = _window_ols(force, eps, valid, reference, f_lo, f_hi, p.min_elastic_points)
        short = short or (math.isfinite(reference) and not math.isfinite(ols.slope))
        moduli.append(ols.slope)
    linearity = (moduli[1] - moduli[0]) / moduli[0] if len(moduli) == 2 else NAN
    sensitivity = WindowSensitivity(
        e_range_gpa=_value_range(e_values),
        rp02_range_mpa=_value_range(rp_values),
        ag_range_percent=_value_range(ag_values),
        agt_range_percent=_value_range(agt_values),
        linearity_rel_diff=float(linearity),
        flags=("window:insufficient_elastic_points",) if short else (),
    )
    return sensitivity
