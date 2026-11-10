"""Yield type, upper and lower yield strength (ontology-agnostic analysis nodes).

All decisions are made on the force array; S0 only converts the results into
stresses, so every threshold is relative and a change of S0 cannot change a
decision through rounding.
"""

from __future__ import annotations

import dataclasses
import math

import flowrep as fr
import numpy as np

from pmd_tensile.curve_math import (
    NAN,
    _channels,
    _check_s0,
    _first,
    _near_switch,
    _params,
    _strain_at,
    _strain_valid,
    _unique,
)
from pmd_tensile.parameters import AnalysisParameters
from pmd_tensile.units import (
    AreaMM2,
    ForceArray,
    ForceN,
    ModulusGPa,
    OptionalStrainRatio,
    StrainArray,
    StressMPa,
)

CONTINUOUS = "continuous"
DISCONTINUOUS = "discontinuous"

SEARCH_FRACTION_OF_FM = 0.5
"""The yield drop is searched only where the force exceeds this fraction of the highest force."""

_NO_UPPER_YIELD = ("reh:no_upper_yield", "rel:no_upper_yield")


@dataclasses.dataclass(frozen=True)
class YieldDiagnostics:
    """How :func:`yield_points` decided; stresses in MPa, strains as ratio, indices -1 if undefined.

    ``drop_threshold_mpa`` is the drop d required at ReH; ``largest_valid_drop_mpa``
    the largest drop below the running maximum where the drop is searched and
    the strain is valid; ``first_drop_index`` the first sample whose drop
    qualified. The plateau quantities describe the lower yield strength.
    """

    rel_index: int
    rel_incl_transient_mpa: float
    plateau_level_mpa: float
    plateau_extent: float
    drop_threshold_mpa: float
    largest_valid_drop_mpa: float
    first_drop_index: int
    reh_sample_increment_mpa: float
    flags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class _LowerYield:
    """The lower-yield rule on one yield range Y; positions are within Y, -1 if undefined."""

    rel_index: int = -1
    rel_incl_transient: float = NAN
    level: float = NAN
    extent: float = NAN
    recovery: int = -1
    band_upper: float | None = None
    flags: tuple[str, ...] = ()


def _no_valid_peak_cause(drop_has_valid_strain: bool) -> str:
    """Flag for a yield peak without usable strain.

    If even the first qualifying drop lies beyond the valid strain record,
    no drop was found while the extensometer measured, which the flag says;
    otherwise the drop was found but its peak lies outside the valid record.
    """
    return "reh:extensometer_invalid" if drop_has_valid_strain else "reh:no_drop_in_valid_strain"


def _first_qualifying_drop(f: np.ndarray, highest_force: float, noise: float, p: AnalysisParameters) -> tuple[int, np.ndarray]:
    """First finite sample, above the search fraction of the highest force, that lies at
    least d(j) = max(drop_rel * M(j), drop_noise_factor * noise) below the running maximum M(j).

    Returns its position in ``f`` (-1 if none) and the running maximum.
    """
    running = np.maximum.accumulate(f)
    required = np.maximum(p.drop_rel * running, p.drop_noise_factor * noise)
    first = _first((f > SEARCH_FRACTION_OF_FM * highest_force) & (f <= running - required))
    return first, running


def _plateau_level(y_force: np.ndarray, reh: float) -> float:
    """Median of the yield-range samples in the lower half between their minimum and ReH.

    The bound keeps the rise towards ReH out of the level.
    """
    lowest = float(y_force.min())
    return float(np.median(y_force[y_force <= 0.5 * (lowest + reh)]))


def _recovery(y_force: np.ndarray, level: float, threshold: float) -> int:
    """End of the transient: position of the first sample back at the plateau level.

    The transient begins at the first sample below level - threshold (a hysteresis,
    so that noise on the falling flank does not end it early); 0 if no sample is
    that low (no transient); -1 if the stress never recovers to the level.
    """
    below = _first(y_force < level - threshold)
    if below < 0:
        return 0
    back = _first(y_force[below + 1 :] >= level)
    return -1 if back < 0 else below + 1 + back


def _lower_yield(
    y_force: np.ndarray, y_index: np.ndarray, reh: float, threshold: float, valid: np.ndarray,
    plastic: np.ndarray, p: AnalysisParameters,
) -> _LowerYield:
    """Lower yield strength over the yield range Y (finite-force samples after the peak, in order).

    After the transient, the candidate ReL is the lowest sample; the plateau band
    reaches from it to the level plus the same distance above (it widens with the
    serration depth). The candidate is ReL if the strain range of the in-band samples
    is at least ``plateau_min_extent``; otherwise ReL is the minimum before recovery,
    with ``rel:single_minimum``. A plateau that continues beyond the valid strain
    has an unknown extent, so ReL is undefined with ``rel:extensometer_invalid``.
    """
    if not len(y_force):
        return _LowerYield()
    rel_incl_transient = float(y_force.min())
    level = _plateau_level(y_force, reh)
    start = _recovery(y_force, level, threshold)
    common = dict(rel_incl_transient=rel_incl_transient, level=level, recovery=start)
    if start >= 0:
        candidate = start + int(np.argmin(y_force[start:]))
        band_upper = 2.0 * level - float(y_force[candidate])
        if not valid[y_index[start]]:
            return _LowerYield(band_upper=band_upper, flags=("rel:extensometer_invalid",), **common)
        in_band = y_force[start:] <= band_upper
        strain_valid = valid[y_index[start:]]
        last_valid = int(np.flatnonzero(strain_valid)[-1])
        if in_band[last_valid] and np.any(in_band[last_valid + 1 :]):
            return _LowerYield(band_upper=band_upper, flags=("rel:extensometer_invalid",), **common)
        inside = np.flatnonzero(in_band & strain_valid)
        last = start + int(inside[-1]) if len(inside) else start
        extent = float(plastic[y_index[last]] - plastic[y_index[start]])
        if extent >= p.plateau_min_extent:
            return _LowerYield(rel_index=int(y_index[candidate]), extent=extent, band_upper=band_upper, **common)
        common["extent"] = extent
        common["band_upper"] = band_upper
    transient = y_force[:start] if start > 0 else y_force
    return _LowerYield(rel_index=int(y_index[int(np.argmin(transient))]), flags=("rel:single_minimum",), **common)


def _end_without_return(
    f: np.ndarray, finite_idx: np.ndarray, peak: int, reh: float, threshold: float, highest_force: float,
    valid: np.ndarray, plastic: np.ndarray, p: AnalysisParameters,
) -> tuple[int, bool, _LowerYield]:
    """End of yielding when the stress never returns to ReH.

    The remaining record ends at the last sample above the search fraction of the
    highest force, which excludes the force drop of fracture or unloading. The
    lower-yield rule on it gives the plateau band; work hardening begins only when
    the stress, after having been in the band, rises above it by more than the drop
    threshold d. Then the end of yielding is the highest sample from that rise on,
    and the lower-yield rule is applied again up to it. Returns (end position in
    ``f``, whether work hardening was reached, lower-yield result).
    """
    above = np.flatnonzero(f[peak + 1 :] > SEARCH_FRACTION_OF_FM * highest_force)
    end = peak + 1 + int(above[-1]) + 1
    y_force, y_index = f[peak + 1 : end], finite_idx[peak + 1 : end]
    lower = _lower_yield(y_force, y_index, reh, threshold, valid, plastic, p)
    if lower.band_upper is not None:
        tail = y_force[lower.recovery :]
        inside = _first(tail <= lower.band_upper)
        if inside >= 0:
            rise = _first(tail[inside + 1 :] > lower.band_upper + threshold)
            if rise >= 0:
                rise_pos = lower.recovery + inside + 1 + rise
                end = peak + 1 + rise_pos + int(np.argmax(y_force[rise_pos:]))
                lower = _lower_yield(f[peak + 1 : end], finite_idx[peak + 1 : end], reh, threshold, valid, plastic, p)
                return end, True, lower
    return end, False, lower


def _yield_result(force_n, strain, s0_mm2, k_ext, highest_force_n, e0_gpa, intercept0_mpa, noise_mpa,
                  rate_switch_strain, params) -> tuple[str, float, float, int, int, bool, YieldDiagnostics]:
    """The computation of :func:`yield_points`, with early returns for continuous yielding."""
    p = _params(params)
    s0 = _check_s0(s0_mm2)
    force, eps, _ = _channels(force_n, strain)
    valid = _strain_valid(force, eps, k_ext)
    slope, intercept = e0_gpa * 1000.0 * s0, intercept0_mpa * s0
    noise = noise_mpa * s0 if math.isfinite(noise_mpa) else 0.0
    finite_idx = np.flatnonzero(np.isfinite(force))
    f = force[finite_idx]

    def continuous(*flags: str, first_drop: int = -1, largest: float = NAN):
        diagnostics = YieldDiagnostics(-1, NAN, NAN, NAN, NAN, largest, first_drop, NAN, _unique((*flags, *_NO_UPPER_YIELD)))
        return CONTINUOUS, NAN, NAN, -1, -1, False, diagnostics

    if len(f) < 2 or not math.isfinite(highest_force_n):
        return continuous()
    first, running = _first_qualifying_drop(f, highest_force_n, noise, p)
    searched = (f > SEARCH_FRACTION_OF_FM * highest_force_n) & valid[finite_idx]
    largest = float((running - f)[searched].max()) / s0 if np.any(searched) else NAN
    if first < 0:
        return continuous(largest=largest)
    first_drop = int(finite_idx[first])
    peak = int(np.argmax(f[: first + 1]))
    i_p = int(finite_idx[peak])
    if not (math.isfinite(slope) and slope > 0 and math.isfinite(intercept)):
        return continuous("reh:insufficient_elastic_points", first_drop=first_drop, largest=largest)
    if not valid[i_p]:
        return continuous(_no_valid_peak_cause(valid[first_drop]), first_drop=first_drop, largest=largest)
    plastic = eps - (force - intercept) / slope
    if plastic[i_p] > p.peak_max_plastic_strain:
        return continuous(first_drop=first_drop, largest=largest)
    exceeded = _first(f[peak + 1 :] > f[peak])
    stop = len(force) if exceeded < 0 else int(finite_idx[peak + 1 + exceeded])
    flow = np.arange(i_p + 1, stop)
    flow = flow[valid[flow]]
    if len(flow) == 0:
        return continuous(_no_valid_peak_cause(valid[first_drop]), first_drop=first_drop, largest=largest)
    if plastic[flow].max() - plastic[i_p] < p.min_flow_plastic_strain:
        return continuous(first_drop=first_drop, largest=largest)

    reh = float(f[peak])
    threshold = max(p.drop_rel * reh, p.drop_noise_factor * noise)
    flags = []
    back_at_reh = _first(f[peak + 1 :] >= reh)
    if back_at_reh >= 0:
        end, work_hardening = peak + 1 + back_at_reh, True
        lower = _lower_yield(f[peak + 1 : end], finite_idx[peak + 1 : end], reh, threshold, valid, plastic, p)
    else:
        flags.append("fm:no_return_to_reh")
        end, work_hardening, lower = _end_without_return(f, finite_idx, peak, reh, threshold, highest_force_n, valid, plastic, p)
        if not work_hardening:
            flags.append("rel:no_work_hardening")
    flags += lower.flags
    rel = float(force[lower.rel_index]) if lower.rel_index >= 0 else NAN
    if _near_switch(_strain_at(eps, i_p), rate_switch_strain, p):
        flags.append("reh:near_rate_switch")
    if lower.rel_index >= 0 and valid[lower.rel_index] and _near_switch(_strain_at(eps, lower.rel_index), rate_switch_strain, p):
        flags.append("rel:near_rate_switch")
    yield_type = DISCONTINUOUS
    reh_mpa, rel_mpa = reh / s0, rel / s0
    upper_yield_index = i_p
    end_of_yielding_index = int(finite_idx[end]) if end < len(f) else len(force)
    diagnostics = YieldDiagnostics(
        rel_index=lower.rel_index,
        rel_incl_transient_mpa=lower.rel_incl_transient / s0,
        plateau_level_mpa=lower.level / s0,
        plateau_extent=lower.extent,
        drop_threshold_mpa=threshold / s0,
        largest_valid_drop_mpa=largest,
        first_drop_index=first_drop,
        reh_sample_increment_mpa=float(f[peak] - f[peak - 1]) / s0 if peak > 0 else NAN,
        flags=_unique(flags),
    )
    return yield_type, reh_mpa, rel_mpa, upper_yield_index, end_of_yielding_index, work_hardening, diagnostics


@fr.atomic
def yield_points(
    force_n: ForceArray,
    strain: StrainArray,
    s0_mm2: AreaMM2,
    k_ext: int,
    highest_force_n: ForceN,
    e0_gpa: ModulusGPa,
    intercept0_mpa: StressMPa,
    noise_mpa: StressMPa,
    rate_switch_strain: OptionalStrainRatio = None,
    params: AnalysisParameters | None = None,
) -> tuple[str, StressMPa, StressMPa, int, int, bool, YieldDiagnostics]:
    """Yield type, upper yield strength ReH and lower yield strength ReL.

    Upper yield point: the first sample, above half the highest force, whose force
    lies d = max(``drop_rel`` * M, ``drop_noise_factor`` * s_F) below the running
    maximum M marks a drop; the peak before it is the upper yield point, ReH =
    F(peak)/S0 (the raw sample, no smoothing). It counts only if the plastic strain
    at the peak is at most ``peak_max_plastic_strain`` and grows by at least
    ``min_flow_plastic_strain`` before the force exceeds the peak again (plastic
    strain = strain - (sigma - b0)/E0, from the provisional line). Otherwise the
    yielding is continuous and ReH and ReL are undefined.

    Lower yield point, over the yield range from the peak to the first sample back
    at ReH (the end of yielding): the plateau level P is the median of the samples
    in the lower half between their minimum and ReH; a transient begins below P - d
    and ends when the stress is back at P. After it, the lowest sample is the
    candidate ReL, and the plateau band reaches from it to as far above P. The
    candidate is ReL if the in-band samples span at least ``plateau_min_extent`` of
    plastic strain; otherwise ReL is the minimum before recovery
    (``rel:single_minimum``). If the stress never returns to ReH, the range ends at
    the last sample above half the highest force (excluding fracture), and work
    hardening begins only where the stress rises more than d above the band.

    Inputs: force (N), strain (ratio), S0 (mm²), ``k_ext``, the highest force (N),
    the provisional line E0 (GPa) and b0 (MPa) and the noise level s_n (MPa) of
    :func:`~pmd_tensile.nodes.analysis.elastic_line.provisional_elastic_line`,
    ``rate_switch_strain`` (ratio, optional), ``params``.
    Outputs: ``yield_type`` (``continuous`` or ``discontinuous``), ``reh_mpa``,
    ``rel_mpa`` (MPa), ``upper_yield_index`` and ``end_of_yielding_index`` (sample
    indices, -1 for continuous yielding), ``work_hardening`` (whether the stress
    rises again after yielding), ``diagnostics``.
    Flags: ``reh:no_upper_yield`` and ``rel:no_upper_yield`` (continuous yielding),
    ``reh:no_drop_in_valid_strain`` (no drop while the extensometer measured),
    ``reh:extensometer_invalid``, ``reh:insufficient_elastic_points`` (no
    provisional line), ``rel:single_minimum``, ``rel:extensometer_invalid``,
    ``fm:no_return_to_reh``, ``rel:no_work_hardening``, ``reh:near_rate_switch``,
    ``rel:near_rate_switch``.
    Limits: only the first qualifying drop is examined; a noisy plateau without an
    upper peak can be taken for discontinuous yielding; ReL on a smoothly rising
    plateau is not the plateau start; every result of the branch without return to
    ReH is flagged ``fm:no_return_to_reh`` because a gradual decline after the
    plateau corrupts it.
    """
    yield_type, reh_mpa, rel_mpa, upper_yield_index, end_of_yielding_index, work_hardening, diagnostics = _yield_result(
        force_n, strain, s0_mm2, k_ext, highest_force_n, e0_gpa, intercept0_mpa, noise_mpa, rate_switch_strain, params
    )
    return yield_type, reh_mpa, rel_mpa, upper_yield_index, end_of_yielding_index, work_hardening, diagnostics
