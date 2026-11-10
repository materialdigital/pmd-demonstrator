"""Result types of the analysis steps, shared by the analysis nodes."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ElasticFit:
    """OLS fit sigma = E*e + b in a stress window (M4, M16).

    ``e_gpa`` is NaN if fewer than the minimum number of points are in the
    window. ``stderr_gpa`` is the OLS standard error of the slope
    (see :data:`E_STDERR_LABEL`), ``lag1_rho`` the lag-1 autocorrelation of
    the residuals in sample order and ``n_eff = n(1 - rho)/(1 + rho)``.
    """

    e_gpa: float
    intercept_mpa: float
    n: int
    r2: float
    stderr_gpa: float
    lag1_rho: float
    n_eff: float
    lo_mpa: float
    hi_mpa: float



@dataclasses.dataclass(frozen=True)
class ExtensometerValidity:
    """Result of :func:`extensometer_validity` (M1-M3)."""

    k_ext: int
    max_valid_strain: float
    n_windows: int
    flags: tuple[str, ...] = ()



@dataclasses.dataclass(frozen=True)
class Bootstrap:
    """Result of :func:`bootstrap_fit` (M5 steps 1-2), in force units.

    ``slope_n`` is the provisional dF/de (E0*S0, N per unit strain);
    ``noise_n`` is the force noise s_F = s_n*S0 (1.4826*MAD of the
    provisional residuals), fixed for M7, M9 and M12.
    """

    fm_n: float
    slope_n: float
    intercept_n: float
    noise_n: float
    n: int
    noise_mpa: float



@dataclasses.dataclass(frozen=True)
class Yielding:
    """Result of :func:`yield_analysis` (M7-M9). Indices are -1 if undefined."""

    yield_type: str
    i_p: int
    i_h: int
    i_rel: int
    work_hardening: bool
    reh_mpa: float
    reh_sample_increment_mpa: float
    rel_mpa: float
    rel_incl_transient_mpa: float
    plateau_level_mpa: float
    plateau_extent: float
    flags: tuple[str, ...] = ()



@dataclasses.dataclass(frozen=True)
class ElasticResult:
    """Result of :func:`final_elastic_fit` (M4-M6, M16).

    ``fit`` is in MPa/GPa; ``slope_n``, ``intercept_n``, ``reference_n``,
    ``lo_n`` and ``hi_n`` are the same quantities in force units.
    """

    fit: ElasticFit
    slope_n: float
    intercept_n: float
    strain_origin: float
    reference_n: float
    reference_mpa: float
    lo_n: float
    hi_n: float
    residual_noise_mpa: float
    iterations: int
    flags: tuple[str, ...] = ()



@dataclasses.dataclass(frozen=True)
class ProofStrength:
    """Result of :func:`proof_strength` (M10); ``index`` is the sample after the crossing."""

    rp02_mpa: float
    index: int
    flags: tuple[str, ...] = ()



@dataclasses.dataclass(frozen=True)
class MaximumForce:
    """Result of :func:`maximum_force` (M11)."""

    fm_n: float
    rm_mpa: float
    i_m: int
    flags: tuple[str, ...] = ()



@dataclasses.dataclass(frozen=True)
class Elongation:
    """Result of :func:`elongation_at_maximum_force` (M12).

    ``band`` holds the first and last sample index of the near-maximum band;
    ``strain_mid`` is its strain mid-point (raw strain, ratio).
    """

    agt_percent: float
    ag_percent: float
    ag_location_halfwidth_percent: float
    band: tuple[int, int]
    strain_mid: float
    flags: tuple[str, ...] = ()



@dataclasses.dataclass(frozen=True)
class WindowSensitivity:
    """Result of :func:`window_sensitivity` (M15): ranges (min, max), never sd."""

    e_range_gpa: tuple[float, float]
    rp02_range_mpa: tuple[float, float]
    ag_range_percent: tuple[float, float]
    agt_range_percent: tuple[float, float]
    linearity_rel_diff: float
    flags: tuple[str, ...] = ()
