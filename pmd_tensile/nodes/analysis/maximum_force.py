"""Maximum force and elongation at maximum force (ontology-agnostic analysis nodes)."""

from __future__ import annotations

import dataclasses
import math

import flowrep as fr
import numpy as np

from pmd_tensile.curve_math import NAN, _channels, _check_s0, _params
from pmd_tensile.nodes.analysis.yielding import DISCONTINUOUS
from pmd_tensile.parameters import AnalysisParameters
from pmd_tensile.units import (
    AreaMM2,
    ForceArray,
    ForceN,
    ModulusGPa,
    StrainArray,
    StrainPercent,
    StrainRatio,
    StressMPa,
)


@dataclasses.dataclass(frozen=True)
class ElongationDiagnostics:
    """The near-maximum band of :func:`elongation_at_maximum_force`: first and last sample
    index ((-1, -1) if none), half its strain width (percentage points), and flags."""

    band: tuple[int, int]
    halfwidth_percent: float
    flags: tuple[str, ...] = ()


@fr.atomic
def maximum_force(
    force_n: ForceArray,
    s0_mm2: AreaMM2,
    yield_type: str,
    work_hardening: bool,
    end_of_yielding_index: int,
) -> tuple[ForceN, StressMPa, int, tuple[str, ...]]:
    """Maximum force Fm and tensile strength Rm = Fm/S0.

    For continuous yielding Fm is the largest finite force. For discontinuous
    yielding it is the largest force after the end of yielding, during work
    hardening, so that an upper yield peak above the later maximum is not taken.

    Inputs: force (N), S0 (mm²), the yield type, whether work hardening was
    reached, and the sample index of the end of yielding.
    Outputs: ``fm_n`` (N), ``rm_mpa`` (MPa), ``max_force_index`` (sample index,
    -1 if undefined), ``flags``:
    ``fm:no_work_hardening`` (discontinuous yielding without work
    hardening: Fm and Rm are undefined).
    Limits: the force channel only, as recorded (no correction for the machine).
    """
    s0 = _check_s0(s0_mm2)
    force = np.asarray(force_n, dtype=float)
    candidates = np.flatnonzero(np.isfinite(force))
    found = []
    if yield_type == DISCONTINUOUS:
        if work_hardening:
            candidates = candidates[candidates >= end_of_yielding_index]
        else:
            candidates = candidates[:0]
            found.append("fm:no_work_hardening")
    if len(candidates):
        max_force_index = int(candidates[int(np.argmax(force[candidates]))])
        fm_n = float(force[max_force_index])
    else:
        max_force_index, fm_n = -1, NAN
    rm_mpa = fm_n / s0
    flags = tuple(found)
    return fm_n, rm_mpa, max_force_index, flags


@fr.atomic
def elongation_at_maximum_force(
    force_n: ForceArray,
    strain: StrainArray,
    s0_mm2: AreaMM2,
    k_ext: int,
    e_gpa: ModulusGPa,
    strain_origin: StrainRatio,
    fm_n: ForceN,
    max_force_index: int,
    noise_mpa: StressMPa,
    params: AnalysisParameters | None = None,
) -> tuple[StrainPercent, StrainPercent, ElongationDiagnostics]:
    """Total and plastic extension at maximum force, Agt and Ag.

    Near the maximum the force is flat and noisy, so the strain at Fm is taken as
    the mid-point of the near-maximum band: the contiguous samples around the
    maximum with F >= Fm (1 - delta), delta = max(``band_rel_min``,
    ``band_noise_factor`` * s_F/Fm) with s_F = s_n * S0, the noise level as a force.
    Agt = (e_mid - e0) * 100 %, Ag = Agt - Rm/E * 100 %.

    Inputs: force (N), strain (ratio), S0 (mm²), ``k_ext``, E (GPa), e0 (ratio),
    Fm (N), the sample index of Fm, the noise level s_n (MPa), ``params``.
    Outputs: ``agt_percent``, ``ag_percent`` (%), ``diagnostics``.
    Flags (on ``ag:`` and ``agt:``): ``extensometer_invalid`` (a band sample has
    no valid strain), ``no_maximum_force``, ``no_elastic_modulus``.
    Limits: undefined whenever the extensometer is removed before the maximum.
    """
    p = _params(params)
    s0 = _check_s0(s0_mm2)
    force, eps, _ = _channels(force_n, strain)
    slope = e_gpa * 1000.0 * s0
    flags = []
    agt_percent = ag_percent = halfwidth = NAN
    band = (-1, -1)
    if max_force_index < 0:
        flags += ["ag:no_maximum_force", "agt:no_maximum_force"]
    if not math.isfinite(slope):
        flags += ["ag:no_elastic_modulus", "agt:no_elastic_modulus"]
    if max_force_index >= 0:
        noise = noise_mpa * s0 if math.isfinite(noise_mpa) else 0.0
        delta = max(p.band_rel_min, p.band_noise_factor * noise / fm_n)
        threshold = fm_n * (1.0 - delta)
        finite_idx = np.flatnonzero(np.isfinite(force))
        centre = int(np.searchsorted(finite_idx, max_force_index))
        lo = centre
        while lo > 0 and force[finite_idx[lo - 1]] >= threshold:
            lo -= 1
        hi = centre
        while hi + 1 < len(finite_idx) and force[finite_idx[hi + 1]] >= threshold:
            hi += 1
        members = finite_idx[lo : hi + 1]
        band = (int(members[0]), int(members[-1]))
        band_strain = eps[members]
        if members[-1] >= k_ext or not np.all(np.isfinite(band_strain)):
            flags += ["ag:extensometer_invalid", "agt:extensometer_invalid"]
        else:
            strain_mid = 0.5 * (float(band_strain.min()) + float(band_strain.max()))
            halfwidth = 0.5 * (float(band_strain.max()) - float(band_strain.min())) * 100.0
            if math.isfinite(slope):
                agt_percent = (strain_mid - strain_origin) * 100.0
                ag_percent = agt_percent - fm_n / slope * 100.0
    diagnostics = ElongationDiagnostics(band=band, halfwidth_percent=halfwidth, flags=tuple(flags))
    return agt_percent, ag_percent, diagnostics
