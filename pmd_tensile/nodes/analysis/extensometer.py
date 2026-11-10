"""Sample checks and the range of valid extensometer strain (ontology-agnostic analysis nodes)."""

from __future__ import annotations

import dataclasses

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


@dataclasses.dataclass(frozen=True)
class DroppedSamples:
    """Number of non-finite samples per channel."""

    force: int
    strain: int
    crosshead: int


@dataclasses.dataclass(frozen=True)
class ExtensometerDiagnostics:
    """How :func:`extensometer_validity` decided: number of crosshead windows checked, and flags."""

    n_windows: int
    flags: tuple[str, ...] = ()


@fr.atomic
def check_time_order(time_s: TimeArray) -> int:
    """Number of samples, after checking that the sample order is the time order.

    Inputs: ``time_s`` (s). Output: ``n_samples``. A decrease of time between
    two finite samples raises ``ValueError``; non-finite times are ignored and
    equal consecutive times are allowed.
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
) -> DroppedSamples:
    """Number of non-finite samples in force (N), strain (ratio) and crosshead (mm).

    The analysis skips such samples channel by channel. Channels of unequal
    length raise ``ValueError``; a missing crosshead counts as all non-finite.
    """
    force, eps, cross = _channels(force_n, strain, crosshead_mm)
    dropped = DroppedSamples(
        force=int(np.sum(~np.isfinite(force))),
        strain=int(np.sum(~np.isfinite(eps))),
        crosshead=int(np.sum(~np.isfinite(cross))),
    )
    return dropped


def _window_invalid(eps: np.ndarray, cross: np.ndarray, a: int, b: int, l0: float, p: AnalysisParameters) -> bool:
    """A crosshead window [a, b] whose strain cannot come from a working extensometer."""
    segment = eps[a : b + 1]
    if not np.all(np.isfinite(segment)) or np.any(segment == 0.0):
        return True
    if np.any(np.diff(segment) < -p.ext_jump):
        return True
    ratio = (eps[b] - eps[a]) / ((cross[b] - cross[a]) / l0)
    return bool(ratio < p.ext_ratio_min)


def _crosshead_windows(cross: np.ndarray, window_mm: float) -> list[tuple[int, int]]:
    """Consecutive sample ranges over which the crosshead travels ``window_mm``.

    Neighbouring windows share their boundary sample; samples without a finite
    crosshead never close a window; a remainder shorter than one window joins
    the last window.
    """
    samples = np.flatnonzero(np.isfinite(cross))
    windows: list[tuple[int, int]] = []
    start = int(samples[0])
    for i in samples[1:]:
        if abs(cross[i] - cross[start]) >= window_mm:
            windows.append((start, int(i)))
            start = int(i)
    if windows and windows[-1][1] != int(samples[-1]):
        windows[-1] = (windows[-1][0], int(samples[-1]))
    return windows


def _extensometer_end(eps: np.ndarray, cross: np.ndarray, l0_mm, p: AnalysisParameters) -> tuple[int, int, bool]:
    """(first sample without valid strain, number of crosshead windows, whether the window rule could run)."""
    n = len(eps)
    finite = np.isfinite(eps)
    first_finite = _first(finite)
    if first_finite < 0:
        k_nan = 0
    else:
        after = _first(~finite[first_finite:])
        k_nan = n if after < 0 else first_finite + after
    usable = _is_number(l0_mm) and l0_mm > 0 and np.count_nonzero(np.isfinite(cross)) >= 2
    windows = _crosshead_windows(cross, p.ext_window_mm) if usable else []
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
) -> tuple[int, StrainRatio, ExtensometerDiagnostics]:
    """The samples with valid extensometer strain: all samples before ``k_ext``.

    Extensometers are often removed before fracture, and the strain channel
    then freezes, drops to zero, becomes undefined or stops following the
    crosshead. The record is cut into consecutive windows over which the
    crosshead travels ``params.ext_window_mm``. A window is invalid if its
    strain is not finite, is exactly zero, falls by more than
    ``params.ext_jump`` between two samples, or rises by less than
    ``params.ext_ratio_min`` times the crosshead strain (travel / L0).
    ``k_ext`` is the start of the earliest window from which every window to
    the end of the record is invalid, and at most the first undefined strain
    after a defined one.

    Inputs: ``strain`` (ratio), ``crosshead_mm`` (mm), ``l0_mm`` (mm),
    ``a_percent`` (A, %, optional), ``rate_switch_strain`` (ratio, optional),
    ``params``.
    Outputs: ``k_ext`` (sample index), ``max_valid_strain`` (largest valid
    strain, ratio), ``diagnostics``.
    Flags: ``extensometer:missing_input`` (no crosshead or L0, so only the
    rule on undefined strain applies), ``extensometer:no_complete_window``,
    ``extensometer:extensometer_ended_early`` (largest valid strain below
    ``params.ext_end_fraction_of_a`` times A), ``extensometer:near_rate_switch``
    (the valid strain ends near the strain at which the test rate changes).
    Limits: a strain channel that the testing software continues from the
    crosshead after the extensometer is removed cannot be detected.
    """
    p = _params(params)
    eps = np.asarray(strain, dtype=float)
    _, _, cross = _channels(eps, eps, crosshead_mm)
    k_ext, n_windows, usable = _extensometer_end(eps, cross, l0_mm, p)
    flags = []
    if not usable:
        flags.append("extensometer:missing_input")
    elif n_windows == 0:
        flags.append("extensometer:no_complete_window")
    head = eps[:k_ext]
    head = head[np.isfinite(head)]
    max_valid_strain = float(head.max()) if len(head) else NAN
    if _is_number(a_percent) and not max_valid_strain >= p.ext_end_fraction_of_a * a_percent / 100.0:
        flags.append("extensometer:extensometer_ended_early")
    if k_ext < len(eps) and _near_switch(max_valid_strain, rate_switch_strain, p):
        flags.append("extensometer:near_rate_switch")
    k_ext = int(k_ext)
    diagnostics = ExtensometerDiagnostics(n_windows=n_windows, flags=tuple(flags))
    return k_ext, max_valid_strain, diagnostics
