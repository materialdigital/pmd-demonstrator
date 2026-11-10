"""Sample checks and the valid range of the extensometer (ontology-agnostic analysis nodes)."""

from __future__ import annotations

import flowrep as fr
import numpy as np

from pmd_tensile.curve_math import (
    NAN,
    _channels,
    _first,
    _is_number,
    _near_switch,
    _params,
)
from pmd_tensile.nodes.analysis.steps import ExtensometerValidity
from pmd_tensile.parameters import AnalysisParameters
from pmd_tensile.units import (
    ForceArray,
    OptionalCrossheadArray,
    OptionalLengthMM,
    OptionalStrainPercent,
    OptionalStrainRatio,
    StrainArray,
    StrainRatio,
    TimeArray,
)

# ---------------------------------------------------------------------------
# Step nodes
# ---------------------------------------------------------------------------


@fr.atomic
def check_time_order(time_s: TimeArray) -> int:
    """M1: the sample order must be the time order; a decrease raises ValueError.

    Non-finite times are ignored; equal consecutive times are allowed.
    Returns the number of samples.
    """
    times = np.asarray(time_s, dtype=float)
    finite = times[np.isfinite(times)]
    if np.any(np.diff(finite) < 0.0):
        first = int(np.flatnonzero(np.diff(finite) < 0.0)[0])
        raise ValueError(f"time is not monotonic (decreases after finite sample {first})")
    n_samples = int(len(times))
    return n_samples



@fr.atomic
def count_dropped_samples(
    force_n: ForceArray, strain: StrainArray, crosshead_mm: OptionalCrossheadArray = None
) -> dict:
    """M1: number of non-finite samples per channel (``force``, ``strain``, ``crosshead``).

    Also checks that the channels have equal length (ValueError otherwise).
    """
    force, eps, cross = _channels(force_n, strain, crosshead_mm)
    dropped_samples = {
        "force": int(np.sum(~np.isfinite(force))),
        "strain": int(np.sum(~np.isfinite(eps))),
        "crosshead": int(np.sum(~np.isfinite(cross))),
    }
    return dropped_samples



def _window_invalid(eps: np.ndarray, cross: np.ndarray, a: int, b: int, l0: float, p: AnalysisParameters) -> bool:
    segment = eps[a : b + 1]
    if not np.all(np.isfinite(segment)) or np.any(segment == 0.0):
        return True
    if np.any(np.diff(segment) < -p.ext_jump):
        return True
    ratio = (eps[b] - eps[a]) / ((cross[b] - cross[a]) / l0)
    return bool(ratio < p.ext_ratio_min)



def _extensometer_end(eps: np.ndarray, cross: np.ndarray, l0_mm, p: AnalysisParameters) -> tuple[int, int, bool]:
    """(k_ext, number of crosshead windows, whether the window rule could run)."""
    n = len(eps)
    finite = np.isfinite(eps)
    first_finite = _first(finite)
    if first_finite < 0:
        k_nan = 0
    else:
        after = _first(~finite[first_finite:])
        k_nan = n if after < 0 else first_finite + after

    crosshead_samples = np.flatnonzero(np.isfinite(cross))
    usable = _is_number(l0_mm) and l0_mm > 0 and len(crosshead_samples) >= 2
    windows: list[tuple[int, int]] = []
    if usable:
        start = int(crosshead_samples[0])
        for i in crosshead_samples[1:]:
            if abs(cross[i] - cross[start]) >= p.ext_window_mm:
                windows.append((start, int(i)))
                start = int(i)
        last = int(crosshead_samples[-1])
        if windows and windows[-1][1] != last:
            windows[-1] = (windows[-1][0], last)

    k_window = n
    for a, b in reversed(windows):
        if not _window_invalid(eps, cross, a, b, float(l0_mm), p):
            break
        k_window = a
    return min(k_nan, k_window), len(windows), usable



@fr.atomic
def extensometer_validity(
    strain: StrainArray,
    crosshead_mm: OptionalCrossheadArray,
    l0_mm: OptionalLengthMM,
    a_percent: OptionalStrainPercent = None,
    rate_switch_strain: OptionalStrainRatio = None,
    params: AnalysisParameters | None = None,
) -> tuple[ExtensometerValidity, int, StrainRatio]:
    """M1-M3: extensometer end ``k_ext`` (rule in the module docstring) and checks.

    Flags: ``extensometer:missing_input`` (no crosshead or L0, window rule
    skipped), ``extensometer:no_complete_window`` (no complete crosshead
    window, so M3 could not be evaluated),
    ``extensometer:extensometer_ended_early`` (largest valid strain below
    0.5*A/100 when A is given) and ``extensometer:near_rate_switch``.
    A strain channel continued from the crosshead cannot be detected
    (:data:`RESIDUAL_RISKS`).
    """
    p = _params(params)
    eps = np.asarray(strain, dtype=float)
    _, _, cross = _channels(eps, eps, crosshead_mm)
    k_end, n_windows, usable = _extensometer_end(eps, cross, l0_mm, p)
    flags = []
    if not usable:
        flags.append("extensometer:missing_input")
    elif n_windows == 0:
        flags.append("extensometer:no_complete_window")
    head = eps[:k_end]
    head = head[np.isfinite(head)]
    max_valid = float(head.max()) if len(head) else NAN
    if _is_number(a_percent) and not max_valid >= p.ext_end_fraction_of_a * a_percent / 100.0:
        flags.append("extensometer:extensometer_ended_early")
    if k_end < len(eps) and _near_switch(max_valid, rate_switch_strain, p):
        flags.append("extensometer:near_rate_switch")
    extensometer = ExtensometerValidity(
        k_ext=int(k_end), max_valid_strain=max_valid, n_windows=n_windows, flags=tuple(flags)
    )
    k_ext = extensometer.k_ext
    extensometer_max_valid_strain = extensometer.max_valid_strain
    return extensometer, k_ext, extensometer_max_valid_strain
