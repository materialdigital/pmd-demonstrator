"""Numerical helpers shared by the analysis nodes: channels, fits in a force window, crossings."""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from pmd_tensile.parameters import DEFAULT_PARAMETERS, AnalysisParameters

NAN = float("nan")



# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------

_MAD_TO_SD = 1.4826



def _is_number(value) -> bool:
    return value is not None and isinstance(value, (int, float, np.floating, np.integer)) and math.isfinite(value)



@dataclasses.dataclass(frozen=True)
class _Ols:
    """OLS of y on x, in the units of the inputs."""

    slope: float
    intercept: float
    n: int
    r2: float
    stderr: float
    lag1_rho: float
    n_eff: float
    residual_noise: float  # 1.4826 * MAD of the residuals



_OLS_UNDEFINED = dict(slope=NAN, intercept=NAN, r2=NAN, stderr=NAN, lag1_rho=NAN, n_eff=NAN, residual_noise=NAN)



def _ols(x: np.ndarray, y: np.ndarray, min_points: int) -> _Ols:
    """Ordinary least squares y = slope*x + intercept on the given samples (in order)."""
    n = int(len(x))
    if n < max(min_points, 2):
        return _Ols(n=n, **_OLS_UNDEFINED)
    x_mean, y_mean = x.mean(), y.mean()
    dx, dy = x - x_mean, y - y_mean
    sxx = float(np.dot(dx, dx))
    if sxx == 0.0:
        return _Ols(n=n, **_OLS_UNDEFINED)
    slope = float(np.dot(dx, dy)) / sxx
    intercept = float(y_mean - slope * x_mean)
    residuals = y - (slope * x + intercept)
    ssr = float(np.dot(residuals, residuals))
    sst = float(np.dot(dy, dy))
    r2 = 1.0 - ssr / sst if sst > 0.0 else NAN
    stderr = math.sqrt(ssr / (n - 2) / sxx) if n > 2 else NAN
    if n > 2 and ssr > 0.0:
        rho = float(np.dot(residuals[:-1], residuals[1:])) / ssr
        n_eff = n * (1.0 - rho) / (1.0 + rho) if rho > -1.0 else NAN
    else:
        rho, n_eff = NAN, NAN
    noise = _MAD_TO_SD * float(np.median(np.abs(residuals - np.median(residuals))))
    return _Ols(slope, intercept, n, r2, stderr, rho, n_eff, noise)



# ---------------------------------------------------------------------------
# Curve analysis: helpers (force space)
# ---------------------------------------------------------------------------


def _params(params: AnalysisParameters | None) -> AnalysisParameters:
    return DEFAULT_PARAMETERS if params is None else params



def _channels(force_n, strain, crosshead_mm=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Float arrays of equal length; a missing crosshead becomes all-NaN."""
    force = np.asarray(force_n, dtype=float)
    eps = np.asarray(strain, dtype=float)
    if force.ndim != 1 or eps.shape != force.shape:
        raise ValueError(f"force and strain must be 1-D of equal length, got {force.shape}, {eps.shape}")
    if crosshead_mm is None:
        cross = np.full(force.shape, NAN)
    else:
        cross = np.asarray(crosshead_mm, dtype=float)
        if cross.shape != force.shape:
            raise ValueError(f"crosshead must have the shape of force, got {cross.shape}")
    return force, eps, cross



def _check_s0(s0_mm2) -> float:
    if not (_is_number(s0_mm2) and s0_mm2 > 0):
        raise ValueError(f"S0 must be a positive finite number, got {s0_mm2!r}")
    return float(s0_mm2)



def _first(mask: np.ndarray) -> int:
    """Index of the first True entry, or -1."""
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else -1



def _strain_valid(force: np.ndarray, eps: np.ndarray, k_ext: int) -> np.ndarray:
    """M1/M2: samples with finite force and strain below ``k_ext``."""
    valid = np.isfinite(force) & np.isfinite(eps)
    valid[max(int(k_ext), 0):] = False
    return valid



def _near_switch(strain_value: float, rate_switch_strain, p: AnalysisParameters) -> bool:
    return (
        _is_number(rate_switch_strain)
        and _is_number(strain_value)
        and abs(strain_value - rate_switch_strain) <= p.rate_switch_proximity
    )



def _strain_at(eps: np.ndarray, index: int) -> float:
    return float(eps[index]) if 0 <= index < len(eps) else NAN



def _window_ols(
    force: np.ndarray,
    eps: np.ndarray,
    valid: np.ndarray,
    reference_n: float,
    f_lo: float,
    f_hi: float,
    min_points: int,
) -> tuple[_Ols, float, float]:
    """OLS F = slope*e + b on the first loading branch below ``reference_n``.

    The branch is every sample before the force first reaches the reference;
    the window is ``f_lo*ref <= F <= f_hi*ref`` on strain-valid samples.
    """
    lo, hi = f_lo * reference_n, f_hi * reference_n
    if not math.isfinite(reference_n):
        return _Ols(n=0, **_OLS_UNDEFINED), lo, hi
    mask = _window_mask(force, valid, reference_n, f_lo, f_hi)
    return _ols(eps[mask], force[mask], min_points), lo, hi


def _window_mask(force: np.ndarray, valid: np.ndarray, reference_n: float, f_lo: float, f_hi: float) -> np.ndarray:
    """Strain-valid samples of the first loading branch with ``f_lo*ref <= F <= f_hi*ref``.

    The first loading branch is every sample before the force first reaches the reference.
    """
    reached = _first(np.isfinite(force) & (force >= reference_n))
    branch = np.zeros(len(force), dtype=bool)
    branch[: len(force) if reached < 0 else reached] = True
    return valid & branch & (force >= f_lo * reference_n) & (force <= f_hi * reference_n)


_MIN_SECOND_DIFFERENCES = 10
"""Fewer second differences leave the noise level undefined."""


def _second_difference_noise(values: np.ndarray) -> float:
    """Noise level of a sampled signal from its second differences.

    For independent noise of standard deviation sigma, a second difference
    x[i+1] - 2 x[i] + x[i-1] has variance 6 sigma**2, while a straight trend
    contributes nothing and a smooth bend almost nothing. The estimate is
    1.4826 * MAD(second differences) / sqrt(6), robust to single outliers.
    It assumes uncorrelated noise; correlated (filtered) noise is
    underestimated. NaN with fewer than ten second differences.
    """
    second = np.diff(np.asarray(values, dtype=float), 2)
    if len(second) < _MIN_SECOND_DIFFERENCES:
        return NAN
    return _MAD_TO_SD * float(np.median(np.abs(second - np.median(second)))) / math.sqrt(6.0)



_CROSSING_FOUND = "found"

_CROSSING_UNDEFINED = "undefined"

_CROSSING_BELOW_OFFSET = "below_offset"

_CROSSING_ALREADY_BEYOND = "already_beyond"



def _proof_crossing(
    force: np.ndarray,
    eps: np.ndarray,
    valid: np.ndarray,
    slope: float,
    intercept: float,
    start_n: float,
    offset: float,
) -> tuple[float, int, str]:
    """First crossing of e_p = ``offset`` (M10), linearly interpolated in force.

    The search runs over consecutive strain-valid samples from the first one
    with F >= ``start_n``. Returns (force at the crossing, index of the sample
    after it, status). The status is ``_CROSSING_FOUND``,
    ``_CROSSING_UNDEFINED`` (no fit or no sample in the search range),
    ``_CROSSING_BELOW_OFFSET`` (e_p stays below the offset to the end of the
    valid strain) or ``_CROSSING_ALREADY_BEYOND`` (the first searched sample
    is already at or above the offset).
    """
    if not (math.isfinite(slope) and slope != 0.0 and math.isfinite(intercept) and math.isfinite(start_n)):
        return NAN, -1, _CROSSING_UNDEFINED
    indices = np.flatnonzero(valid)
    f_valid = force[indices]
    plastic = eps[indices] - (f_valid - intercept) / slope
    start = _first(f_valid >= start_n)
    if start < 0:
        return NAN, -1, _CROSSING_UNDEFINED
    beyond = _first(plastic[start:] >= offset)
    if beyond < 0:
        return NAN, -1, _CROSSING_BELOW_OFFSET
    if beyond == 0:
        return NAN, -1, _CROSSING_ALREADY_BEYOND
    a, b = start + beyond - 1, start + beyond
    fraction = (offset - plastic[a]) / (plastic[b] - plastic[a])
    crossing = f_valid[a] + fraction * (f_valid[b] - f_valid[a])
    return float(crossing), int(indices[b]), _CROSSING_FOUND



def _value_range(values: list[float]) -> tuple[float, float]:
    finite = [v for v in values if math.isfinite(v)]
    return (min(finite), max(finite)) if finite else (NAN, NAN)



def _unique(flags) -> tuple[str, ...]:
    return tuple(dict.fromkeys(flags))
