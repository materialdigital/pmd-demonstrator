"""Yield type, upper and lower yield strength (ontology-agnostic analysis nodes)."""

from __future__ import annotations

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
from pmd_tensile.nodes.analysis.steps import Bootstrap, Yielding
from pmd_tensile.parameters import AnalysisParameters
from pmd_tensile.units import (
    AreaMM2,
    ForceArray,
    OptionalStrainRatio,
    StrainArray,
    StressMPa,
)

# ---------------------------------------------------------------------------
# Curve analysis: result types
# ---------------------------------------------------------------------------

CONTINUOUS = "continuous"

DISCONTINUOUS = "discontinuous"


_YIELD_SEARCH_FRACTION_OF_RM = 0.5
"""M7: the drop is searched only where sigma > 0.5*Rm (fixed by the method)."""



def _lower_yield(
    y_force: np.ndarray,
    y_index: np.ndarray,
    reh: float,
    threshold: float,
    valid: np.ndarray,
    plastic: np.ndarray,
    p: AnalysisParameters,
) -> dict:
    """M9 (option A) over the yield range Y; see the module docstring.

    ``y_force``/``y_index`` are the finite-force samples of Y in order,
    ``threshold`` is the M7 drop threshold d at ReH. Returns the ReL sample
    index (-1 if undefined), ReL incl. transient, plateau level P, extent, the
    recovery position ``start`` in Y (-1 if none), the plateau band's upper
    bound 2P - candidate (``None`` without a candidate) and flags.
    """
    out = dict(i_rel=-1, rel_incl=NAN, level=NAN, extent=NAN, start=-1, band_upper=None, flags=[])
    if not len(y_force):
        return out
    out["rel_incl"] = float(y_force.min())
    level = float(np.median(y_force[y_force <= 0.5 * (out["rel_incl"] + reh)]))
    out["level"] = level
    # Hysteresis against noise: the transient begins below level - d; recovery
    # is complete at sigma >= level.
    below = _first(y_force < level - threshold)
    if below < 0:
        start = 0
    else:
        back = _first(y_force[below + 1 :] >= level)
        start = -1 if back < 0 else below + 1 + back
    out["start"] = start
    single = start < 0
    if not single:
        candidate_pos = start + int(np.argmin(y_force[start:]))
        upper = 2.0 * level - float(y_force[candidate_pos])
        out["band_upper"] = upper
        if not valid[y_index[start]]:
            out["flags"].append("rel:extensometer_invalid")
            return out
        in_band = y_force[start:] <= upper
        strain_valid = valid[y_index[start:]]
        last_valid = int(np.flatnonzero(strain_valid)[-1])
        if in_band[last_valid] and np.any(in_band[last_valid + 1 :]):
            # R8-2, M2: the plateau continues beyond k_ext, so its extent is unknown.
            out["flags"].append("rel:extensometer_invalid")
            return out
        inside = np.flatnonzero(in_band & strain_valid)
        last = start + int(inside[-1]) if len(inside) else start
        out["extent"] = float(plastic[y_index[last]] - plastic[y_index[start]])
        if out["extent"] >= p.plateau_min_extent:
            out["i_rel"] = int(y_index[candidate_pos])
        else:
            single = True
    if single:
        transient = y_force[:start] if start > 0 else y_force
        out["i_rel"] = int(y_index[int(np.argmin(transient))])
        out["flags"].append("rel:single_minimum")
    return out



def _yield_points(
    force: np.ndarray,
    eps: np.ndarray,
    valid: np.ndarray,
    boot: Bootstrap,
    rate_switch_strain,
    p: AnalysisParameters,
) -> dict:
    """M7-M9 in force units; see the module docstring for the M9 rule."""
    result = dict(
        yield_type=CONTINUOUS, i_p=-1, i_h=-1, i_rel=-1, work_hardening=False,
        reh=NAN, increment=NAN, rel=NAN, rel_incl=NAN, level=NAN, extent=NAN, flags=[],
    )
    no_upper = ["reh:no_upper_yield", "rel:no_upper_yield"]

    finite_idx = np.flatnonzero(np.isfinite(force))
    f = force[finite_idx]
    if len(f) < 2 or not math.isfinite(boot.fm_n):
        result["flags"] = no_upper
        return result

    # M7: running-maximum drawdown.
    noise = boot.noise_n if math.isfinite(boot.noise_n) else 0.0
    running = np.maximum.accumulate(f)
    drop = np.maximum(p.drop_rel * running, p.drop_noise_factor * noise)
    j_star = _first((f > _YIELD_SEARCH_FRACTION_OF_RM * boot.fm_n) & (f <= running - drop))
    if j_star < 0:
        result["flags"] = no_upper
        return result
    peak = int(np.argmax(f[: j_star + 1]))
    i_p = int(finite_idx[peak])

    slope, intercept = boot.slope_n, boot.intercept_n
    if not (math.isfinite(slope) and slope > 0 and math.isfinite(intercept)):
        result["flags"] = ["reh:insufficient_elastic_points", *no_upper]
        return result
    if not valid[i_p]:
        result["flags"] = ["reh:extensometer_invalid", *no_upper]
        return result
    plastic = eps - (force - intercept) / slope
    if plastic[i_p] > p.peak_max_plastic_strain:
        result["flags"] = no_upper
        return result
    recovered = _first(f[peak + 1 :] > f[peak])
    stop = len(force) if recovered < 0 else int(finite_idx[peak + 1 + recovered])
    flow = np.arange(i_p + 1, stop)
    flow = flow[valid[flow]]
    if len(flow) == 0:
        result["flags"] = ["reh:extensometer_invalid", *no_upper]
        return result
    if plastic[flow].max() - plastic[i_p] < p.min_flow_plastic_strain:
        result["flags"] = no_upper
        return result

    # Discontinuous yielding: ReH (M8) and the end of yielding i_h.
    flags = []
    reh = float(f[peak])
    increment = float(f[peak] - f[peak - 1]) if peak > 0 else NAN
    threshold = max(p.drop_rel * reh, p.drop_noise_factor * noise)  # M7 d at ReH
    hardening = _first(f[peak + 1 :] >= reh)
    work_hardening = hardening >= 0
    if work_hardening:
        h = peak + 1 + hardening
        m9 = _lower_yield(f[peak + 1 : h], finite_idx[peak + 1 : h], reh, threshold, valid, plastic, p)
    else:
        # v2.2 amendment (M11, T4-D1): the stress never returns to ReH. M9 over
        # the remaining record gives the plateau band [ReL, 2P - ReL]; work
        # hardening begins only when the stress, after having been inside the
        # band, rises above it by more than the margin d (v2.3:
        # sigma > 2P - ReL + d, d the M7 drop threshold at ReH). Then
        # i_h = i_m is the highest sample from that rise on, and M9 is
        # repeated with i_h as the plateau end.
        #
        # v2.5 truncation (R8-1): in this branch only, the remainder ends at the
        # last sample above the M7 search threshold, which excludes the terminal
        # force drop of fracture or unloading. j* lies after the peak and above
        # that threshold, so the search below always finds a sample.
        #
        # v2.6 accepted limit: a gradual decline above that threshold stays in,
        # so every result from this branch carries fm:no_return_to_reh.
        flags.append("fm:no_return_to_reh")
        above =np.flatnonzero(f[peak + 1 :] > _YIELD_SEARCH_FRACTION_OF_RM * boot.fm_n)
        h = peak + 1 + int(above[-1]) + 1
        y_force, y_index = f[peak + 1 : h], finite_idx[peak + 1 : h]
        m9 = _lower_yield(y_force, y_index, reh, threshold, valid, plastic, p)
        if m9["band_upper"] is not None:
            tail = y_force[m9["start"] :]
            inside = _first(tail <= m9["band_upper"])
            if inside >= 0:
                rise = _first(tail[inside + 1 :] > m9["band_upper"] + threshold)
                if rise >= 0:
                    rise_pos = m9["start"] + inside + 1 + rise
                    h = peak + 1 + rise_pos + int(np.argmax(y_force[rise_pos:]))
                    work_hardening = True
        if work_hardening:
            m9 = _lower_yield(f[peak + 1 : h], finite_idx[peak + 1 : h], reh, threshold, valid, plastic, p)
        else:
            flags.append("rel:no_work_hardening")
    i_h = int(finite_idx[h]) if h < len(f) else len(force)
    flags += m9["flags"]
    i_rel = m9["i_rel"]
    rel = float(force[i_rel]) if i_rel >= 0 else NAN
    rel_incl, level, extent = m9["rel_incl"], m9["level"], m9["extent"]

    if _near_switch(_strain_at(eps, i_p), rate_switch_strain, p):
        flags.append("reh:near_rate_switch")
    if i_rel >= 0 and valid[i_rel] and _near_switch(_strain_at(eps, i_rel), rate_switch_strain, p):
        flags.append("rel:near_rate_switch")
    result.update(
        yield_type=DISCONTINUOUS, i_p=i_p, i_h=i_h, i_rel=i_rel, work_hardening=work_hardening,
        reh=reh, increment=increment, rel=rel, rel_incl=rel_incl, level=level, extent=extent,
        flags=flags,
    )
    return result



@fr.atomic
def yield_analysis(
    force_n: ForceArray,
    strain: StrainArray,
    s0_mm2: AreaMM2,
    k_ext: int,
    bootstrap: Bootstrap,
    rate_switch_strain: OptionalStrainRatio = None,
    params: AnalysisParameters | None = None,
) -> tuple[Yielding, str, StressMPa, StressMPa]:
    """M5 step 3, M7-M9: yield type, ReH (raw sample) and ReL (option A).

    Uses only the bootstrap E0, b0 and s_n. Continuous yielding gives NaN
    ReH/ReL with flags ``reh:no_upper_yield`` and ``rel:no_upper_yield``. If
    the stress never returns to ReH, the result has ``fm:no_return_to_reh``.
    """
    p = _params(params)
    s0 = _check_s0(s0_mm2)
    force, eps, _ = _channels(force_n, strain)
    valid = _strain_valid(force, eps, k_ext)
    points = _yield_points(force, eps, valid, bootstrap, rate_switch_strain, p)
    yielding = Yielding(
        yield_type=points["yield_type"],
        i_p=points["i_p"],
        i_h=points["i_h"],
        i_rel=points["i_rel"],
        work_hardening=points["work_hardening"],
        reh_mpa=points["reh"] / s0,
        reh_sample_increment_mpa=points["increment"] / s0,
        rel_mpa=points["rel"] / s0,
        rel_incl_transient_mpa=points["rel_incl"] / s0,
        plateau_level_mpa=points["level"] / s0,
        plateau_extent=points["extent"],
        flags=_unique(points["flags"]),
    )
    yield_type = yielding.yield_type
    reh_mpa = yielding.reh_mpa
    rel_mpa = yielding.rel_mpa
    return yielding, yield_type, reh_mpa, rel_mpa
