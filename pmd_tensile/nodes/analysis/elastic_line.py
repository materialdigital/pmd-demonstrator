"""Elastic line, proof strength and window sensitivity (ontology-agnostic analysis nodes).

The nodes take and return stresses in MPa, E in GPa and strains as ratios; inside,
the decisions are made on the force array (slope dF/de = E * S0), so that every
threshold is relative to a force and a change of S0 cannot change a decision.
"""

from __future__ import annotations

import dataclasses
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
    _first,
    _near_switch,
    _Ols,
    _params,
    _proof_crossing,
    _second_difference_noise,
    _strain_at,
    _strain_valid,
    _value_range,
    _window_mask,
    _window_ols,
)
from pmd_tensile.parameters import AnalysisParameters
from pmd_tensile.units import (
    AreaMM2,
    ForceArray,
    ForceN,
    ModulusGPa,
    OptionalStrainRatio,
    StrainArray,
    StrainPercent,
    StrainRatio,
    StressArray,
    StressMPa,
)

_PROVISIONAL_MIN_POINTS = 3
"""The provisional line needs at least three samples."""


@dataclasses.dataclass(frozen=True)
class ProvisionalLine:
    """How :func:`provisional_elastic_line` fitted: window (MPa), number of samples,
    end of the first loading branch (sample index) and residual scatter (MPa)."""

    window_mpa: tuple[float, float]
    n: int
    branch_end_index: int
    residual_scatter_mpa: float


@dataclasses.dataclass(frozen=True)
class ElasticLineDiagnostics:
    """The fit of :func:`elastic_line`: window (MPa), end of the first loading branch (sample
    index: the first sample at the reference force), sample count, R², standard
    error of E (GPa, conditional on the window and on independent residuals, not an
    uncertainty), lag-1 autocorrelation of the residuals and the effective sample count
    n(1 - rho)/(1 + rho), iterations of the window, residual scatter (MPa), flags."""

    window_mpa: tuple[float, float]
    branch_end_index: int
    n: int
    r2: float
    stderr_gpa: float
    lag1_rho: float
    n_eff: float
    iterations: int
    residual_scatter_mpa: float
    flags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ProofStrengthDiagnostics:
    """Sample index after the crossing of :func:`proof_strength` (-1 if none), and flags."""

    index: int
    flags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class WindowSensitivityDiagnostics:
    """E for every window of the grid (rows: lower bound, columns: upper bound; NaN where
    too few samples), ranges (min, max) of Ag and Agt (%), the linearity (E_b - E_a)/E_a,
    and flags."""

    grid_e_gpa: tuple[tuple[float, ...], ...]
    ag_range_percent: tuple[float, float]
    agt_range_percent: tuple[float, float]
    linearity_rel_diff: float
    flags: tuple[str, ...] = ()


def _force_line(e_gpa: float, intercept_mpa: float, s0: float) -> tuple[float, float]:
    """A line sigma = E*e + b in force units: (dF/de in N, intercept in N)."""
    return e_gpa * 1000.0 * s0, intercept_mpa * s0


@fr.atomic
def engineering_stress(force_n: ForceArray, s0_mm2: AreaMM2) -> StressArray:
    """Engineering stress sigma = F/S0 (MPa) from force (N) and S0 (mm²), for display."""
    stress_mpa = np.asarray(force_n, dtype=float) / _check_s0(s0_mm2)
    return stress_mpa


@fr.atomic
def provisional_elastic_line(
    force_n: ForceArray,
    strain: StrainArray,
    s0_mm2: AreaMM2,
    k_ext: int,
    params: AnalysisParameters | None = None,
) -> tuple[ForceN, ModulusGPa, StressMPa, StressMPa, ProvisionalLine]:
    """Highest force, a provisional elastic line and the force noise level.

    The provisional line sigma = E0*e + b0 is the least-squares line through the
    samples with valid strain on the first loading branch (before the force first
    reaches R = ``provisional_fraction_of_rm`` times the highest force) with force
    between ``f_lo``*R and ``f_hi``*R. The noise level s_n is estimated from the
    same samples as 1.4826 * MAD(second differences of F) / sqrt(6) / S0: a straight
    trend contributes nothing to a second difference and a smooth bend almost
    nothing, so a curved loading part or a seating event does not raise it, unlike
    the scatter around the line. The line and s_n serve the yield detection, which
    must not depend on the final elastic line.

    Inputs: force (N), strain (ratio), S0 (mm²), ``k_ext`` (first sample without
    valid strain), ``params``.
    Outputs: ``highest_force_n`` (N, maximum over all finite forces),
    ``e0_gpa`` (GPa), ``intercept0_mpa`` (MPa), ``noise_mpa`` (s_n, MPa),
    ``diagnostics``.
    Limits: s_n assumes uncorrelated noise; filtered (correlated) noise gives a
    lower value. It is undefined with fewer than ten second differences, and the
    line with fewer than three samples.
    """
    p = _params(params)
    s0 = _check_s0(s0_mm2)
    force, eps, _ = _channels(force_n, strain)
    finite_force = force[np.isfinite(force)]
    highest_force_n = float(finite_force.max()) if len(finite_force) else NAN
    valid = _strain_valid(force, eps, k_ext)
    reference = p.provisional_fraction_of_rm * highest_force_n
    ols, lo, hi = _window_ols(force, eps, valid, reference, p.f_lo, p.f_hi, _PROVISIONAL_MIN_POINTS)
    noise_n = _second_difference_noise(force[_window_mask(force, valid, reference, p.f_lo, p.f_hi)])
    reached = _first(np.isfinite(force) & (force >= reference))
    e0_gpa = ols.slope / s0 / 1000.0
    intercept0_mpa = ols.intercept / s0
    noise_mpa = noise_n / s0
    diagnostics = ProvisionalLine(
        window_mpa=(lo / s0, hi / s0),
        n=ols.n,
        branch_end_index=len(force) if reached < 0 else reached,
        residual_scatter_mpa=ols.residual_noise / s0,
    )
    return highest_force_n, e0_gpa, intercept0_mpa, noise_mpa, diagnostics


def _continuous_window(
    force: np.ndarray, eps: np.ndarray, valid: np.ndarray, slope: float, intercept: float, highest_force: float,
    p: AnalysisParameters,
) -> tuple[_Ols, float, float, float, int, str]:
    """Window of the elastic line for continuous yielding: the fixed point sigma_ref = Rp0.2(E).

    Starts from Rp0.2 with the provisional line and refits until E changes by less
    than ``window_rel_tol``, at most ``max_window_iterations`` times. Returns
    (fit, window bounds and reference force in N, iterations, status); the status is
    ``converged``, ``not_converged``, ``insufficient`` or ``no_reference``.
    """
    undefined = _Ols(n=0, **_OLS_UNDEFINED)
    if not math.isfinite(slope):
        return undefined, NAN, NAN, NAN, 0, "insufficient"
    start = p.f_lo * p.provisional_fraction_of_rm * highest_force
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
def elastic_line(
    force_n: ForceArray,
    strain: StrainArray,
    s0_mm2: AreaMM2,
    k_ext: int,
    upper_yield_index: int,
    highest_force_n: ForceN,
    e0_gpa: ModulusGPa,
    intercept0_mpa: StressMPa,
    params: AnalysisParameters | None = None,
) -> tuple[ModulusGPa, StrainRatio, ForceN, ElasticLineDiagnostics]:
    """E, the slope of the elastic line, and the strain origin.

    E and b are the least-squares line sigma = E*e + b through the samples with
    valid strain on the first loading branch with stress between ``f_lo`` and
    ``f_hi`` times a reference (10-40 % by default). The reference is the force at the
    upper yield point for discontinuous yielding; for continuous yielding it is the fixed point
    sigma_ref = Rp0.2(E), found by iteration from the provisional line. The strain
    origin e0 = -b/E removes the seating offset of the strain channel. E is the
    slope of the elastic line, not a material modulus: the extensometer class and
    its averaging are unknown.

    Inputs: force (N), strain (ratio), S0 (mm²), ``k_ext``, the sample index of the
    upper yield point (-1 for continuous yielding), the highest force (N) and the
    provisional line E0 (GPa), b0 (MPa), ``params``.
    Outputs: ``e_gpa`` (GPa), ``strain_origin`` (e0, ratio), ``reference_force_n``
    (the reference of the window, N; the force sample itself for discontinuous
    yielding, so that the first loading branch ends exactly there),
    ``diagnostics``.
    Flags: ``e:insufficient_elastic_points`` (fewer than ``min_elastic_points``
    samples in the window), ``e:e_window_not_converged``.
    """
    p = _params(params)
    s0 = _check_s0(s0_mm2)
    force, eps, _ = _channels(force_n, strain)
    valid = _strain_valid(force, eps, k_ext)
    if upper_yield_index >= 0:
        reference = float(force[upper_yield_index])
        ols, lo, hi = _window_ols(force, eps, valid, reference, p.f_lo, p.f_hi, p.min_elastic_points)
        iterations = 1
        status = "converged" if math.isfinite(ols.slope) else "insufficient"
    else:
        slope0, intercept0 = _force_line(e0_gpa, intercept0_mpa, s0)
        ols, lo, hi, reference, iterations, status = _continuous_window(
            force, eps, valid, slope0, intercept0, highest_force_n, p
        )
    flags = []
    if status == "insufficient":
        flags.append("e:insufficient_elastic_points")
    elif status in ("not_converged", "no_reference"):
        flags.append("e:e_window_not_converged")
    e_gpa = ols.slope / s0 / 1000.0
    strain_origin = -ols.intercept / ols.slope if math.isfinite(ols.slope) else NAN
    reference_force_n = reference
    reached = _first(np.isfinite(force) & (force >= reference))
    diagnostics = ElasticLineDiagnostics(
        window_mpa=(lo / s0, hi / s0),
        branch_end_index=len(force) if reached < 0 else reached,
        n=ols.n,
        r2=ols.r2,
        stderr_gpa=ols.stderr / s0 / 1000.0,
        lag1_rho=ols.lag1_rho,
        n_eff=ols.n_eff,
        iterations=iterations,
        residual_scatter_mpa=ols.residual_noise / s0,
        flags=tuple(flags),
    )
    return e_gpa, strain_origin, reference_force_n, diagnostics


@fr.atomic
def proof_strength(
    force_n: ForceArray,
    strain: StrainArray,
    s0_mm2: AreaMM2,
    k_ext: int,
    e_gpa: ModulusGPa,
    strain_origin: StrainRatio,
    reference_force_n: ForceN,
    upper_yield_index: int,
    end_of_yielding_index: int,
    rate_switch_strain: OptionalStrainRatio = None,
    params: AnalysisParameters | None = None,
) -> tuple[StressMPa, ProofStrengthDiagnostics]:
    """Proof strength Rp0.2: the stress at a plastic strain of ``offset_plastic_strain`` (0.2 %).

    The plastic strain is e_p = e - e0 - sigma/E. Searching the samples with valid
    strain from the first one at or above the lower bound ``f_lo`` of the elastic window
    (so that a curved start below the window cannot cross), Rp0.2 is the stress at
    the first crossing of e_p = 0.2 %, interpolated linearly between the two
    samples around it.

    Inputs: force (N), strain (ratio), S0 (mm²), ``k_ext``, E (GPa), e0 (ratio), the
    reference force of the elastic window (N), the sample indices of the upper yield
    point and of the end of yielding (-1 for continuous yielding),
    ``rate_switch_strain`` (ratio, optional), ``params``.
    Outputs: ``rp02_mpa`` (MPa), ``diagnostics``.
    Flags: ``rp02:extensometer_invalid`` (not reached before the valid strain
    ends early), ``rp02:not_reached`` (not reached in a record with valid strain to
    its end, or already exceeded at the first searched sample),
    ``rp02:no_elastic_modulus``, ``rp02:in_yield_plateau`` (the crossing lies in
    the yield range of discontinuous yielding, where Rp0.2 is not a meaningful
    property), ``rp02:near_rate_switch``.
    """
    p = _params(params)
    s0 = _check_s0(s0_mm2)
    force, eps, _ = _channels(force_n, strain)
    valid = _strain_valid(force, eps, k_ext)
    slope = e_gpa * 1000.0 * s0
    crossing, index, status = _proof_crossing(
        force, eps, valid, slope, -strain_origin * slope, p.f_lo * reference_force_n, p.offset_plastic_strain
    )
    flags = []
    # Without a window, continuous yielding found no first crossing of the provisional line.
    no_reference = status == _CROSSING_UNDEFINED and not math.isfinite(reference_force_n)
    if status == _CROSSING_BELOW_OFFSET or no_reference:
        flags.append("rp02:extensometer_invalid" if k_ext < len(force) else "rp02:not_reached")
    elif status == _CROSSING_ALREADY_BEYOND:
        flags.append("rp02:not_reached")
    elif not math.isfinite(slope):
        flags.append("rp02:no_elastic_modulus")
    if index >= 0:
        if 0 <= upper_yield_index <= index <= end_of_yielding_index:
            flags.append("rp02:in_yield_plateau")
        if _near_switch(_strain_at(eps, index), rate_switch_strain, p):
            flags.append("rp02:near_rate_switch")
    rp02_mpa = crossing / s0
    diagnostics = ProofStrengthDiagnostics(index=index, flags=tuple(flags))
    return rp02_mpa, diagnostics


@fr.atomic
def window_sensitivity(
    force_n: ForceArray,
    strain: StrainArray,
    s0_mm2: AreaMM2,
    k_ext: int,
    reference_force_n: ForceN,
    e_gpa: ModulusGPa,
    strain_origin: StrainRatio,
    agt_percent: StrainPercent,
    fm_n: ForceN,
    params: AnalysisParameters | None = None,
) -> tuple[ModulusGPa, ModulusGPa, StressMPa, StressMPa, WindowSensitivityDiagnostics]:
    """How much E, Rp0.2, Ag and Agt depend on the choice of the elastic window.

    For every window of the grid (lower bounds ``window_grid_lo``, upper bounds
    ``window_grid_hi``, as fractions of the reference of the elastic line)
    the line is fitted again and Rp0.2, Ag and Agt are evaluated with it; the node
    reports the ranges and never selects a window. The linearity diagnostic
    (E_b - E_a)/E_a compares the windows of ``linearity_windows``.

    Inputs: force (N), strain (ratio), S0 (mm²), ``k_ext``, the reference force of
    the elastic window (N), E (GPa), e0 (ratio), Agt (%), Fm (N), ``params``.
    Outputs: ``e_min_gpa``, ``e_max_gpa`` (GPa), ``rp02_min_mpa``,
    ``rp02_max_mpa`` (MPa), ``diagnostics``.
    Flags: ``e:window_sensitive`` (the range of E exceeds ``e_window_range_max``
    of E, 5 % by default: the loading curve is not straight across the grid),
    ``window:insufficient_elastic_points`` (a window has too few samples).
    """
    p = _params(params)
    s0 = _check_s0(s0_mm2)
    force, eps, _ = _channels(force_n, strain)
    valid = _strain_valid(force, eps, k_ext)
    reference = reference_force_n
    strain_mid = agt_percent / 100.0 + strain_origin
    grid, e_values, rp_values, ag_values, agt_values = [], [], [], [], []
    short = False
    for f_lo in p.window_grid_lo:
        row = []
        for f_hi in p.window_grid_hi:
            ols, lo, _ = _window_ols(force, eps, valid, reference, f_lo, f_hi, p.min_elastic_points)
            row.append(ols.slope / s0 / 1000.0)
            if not math.isfinite(ols.slope):
                short = short or math.isfinite(reference)
                continue
            e_values.append(ols.slope / s0 / 1000.0)
            crossing, _, _ = _proof_crossing(force, eps, valid, ols.slope, ols.intercept, lo, p.offset_plastic_strain)
            rp_values.append(crossing / s0)
            if math.isfinite(strain_mid) and math.isfinite(fm_n):
                agt = (strain_mid + ols.intercept / ols.slope) * 100.0
                agt_values.append(agt)
                ag_values.append(agt - fm_n / ols.slope * 100.0)
        grid.append(tuple(row))
    moduli = []
    for f_lo, f_hi in p.linearity_windows:
        ols, _, _ = _window_ols(force, eps, valid, reference, f_lo, f_hi, p.min_elastic_points)
        short = short or (math.isfinite(reference) and not math.isfinite(ols.slope))
        moduli.append(ols.slope)
    linearity = (moduli[1] - moduli[0]) / moduli[0] if len(moduli) == 2 else NAN
    flags = ["window:insufficient_elastic_points"] if short else []
    if len(e_values) > 1 and (max(e_values) - min(e_values)) / e_gpa > p.e_window_range_max:
        flags.append("e:window_sensitive")
    e_min_gpa, e_max_gpa = _value_range(e_values)
    rp02_min_mpa, rp02_max_mpa = _value_range(rp_values)
    diagnostics = WindowSensitivityDiagnostics(
        grid_e_gpa=tuple(grid),
        ag_range_percent=_value_range(ag_values),
        agt_range_percent=_value_range(agt_values),
        linearity_rel_diff=float(linearity),
        flags=tuple(flags),
    )
    return e_min_gpa, e_max_gpa, rp02_min_mpa, rp02_max_mpa, diagnostics
