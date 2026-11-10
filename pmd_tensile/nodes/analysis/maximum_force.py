"""Maximum force and elongation at maximum force (ontology-agnostic analysis nodes)."""

from __future__ import annotations

import math

import flowrep as fr
import numpy as np

from pmd_tensile.curve_math import NAN, _channels, _check_s0, _params
from pmd_tensile.nodes.analysis.steps import (
    Bootstrap,
    ElasticResult,
    Elongation,
    MaximumForce,
    Yielding,
)
from pmd_tensile.nodes.analysis.yielding import DISCONTINUOUS
from pmd_tensile.parameters import AnalysisParameters
from pmd_tensile.units import (
    AreaMM2,
    ForceArray,
    ForceN,
    StrainArray,
    StrainPercent,
    StressMPa,
)


@fr.atomic
def maximum_force(
    force_n: ForceArray, s0_mm2: AreaMM2, yielding: Yielding
) -> tuple[MaximumForce, ForceN, StressMPa]:
    """M11: Fm and Rm = Fm/S0.

    Continuous yielding: maximum over all finite forces. Discontinuous: the
    maximum from i_h on (work hardening); without work hardening, NaN with
    flag ``fm:no_work_hardening``.
    """
    s0 = _check_s0(s0_mm2)
    force = np.asarray(force_n, dtype=float)
    candidates = np.flatnonzero(np.isfinite(force))
    flags = []
    if yielding.yield_type == DISCONTINUOUS:
        if yielding.work_hardening:
            candidates = candidates[candidates >= yielding.i_h]
        else:
            candidates = candidates[:0]
            flags.append("fm:no_work_hardening")
    if len(candidates):
        i_m = int(candidates[int(np.argmax(force[candidates]))])
        fm = float(force[i_m])
    else:
        i_m, fm = -1, NAN
    maximum = MaximumForce(fm_n=fm, rm_mpa=fm / s0, i_m=i_m, flags=tuple(flags))
    fm_n = maximum.fm_n
    rm_mpa = maximum.rm_mpa
    return maximum, fm_n, rm_mpa



@fr.atomic
def elongation_at_maximum_force(
    force_n: ForceArray,
    strain: StrainArray,
    k_ext: int,
    bootstrap: Bootstrap,
    elastic: ElasticResult,
    maximum: MaximumForce,
    params: AnalysisParameters | None = None,
) -> tuple[Elongation, StrainPercent, StrainPercent]:
    """M12: Agt and Ag at the strain mid-point of the near-maximum band.

    The band is the contiguous run of finite-force samples around i_m with
    F >= Fm*(1 - delta_F), delta_F = max(band_rel_min, 3*s_F/Fm). Agt =
    (e_mid - e0)*100 %, Ag = Agt - Rm/E*100 %. If any band sample is at or
    beyond ``k_ext`` (or has no finite strain), both are NaN with flags
    ``ag:extensometer_invalid`` and ``agt:extensometer_invalid``. Without
    Fm or E both are NaN with the ``ag:`` and ``agt:`` flags
    ``no_maximum_force`` or ``no_elastic_modulus``.
    """
    p = _params(params)
    force, eps, _ = _channels(force_n, strain)
    flags = []
    agt = ag = halfwidth = strain_mid = NAN
    band = (-1, -1)
    if maximum.i_m < 0:
        flags += ["ag:no_maximum_force", "agt:no_maximum_force"]
    if not math.isfinite(elastic.slope_n):
        flags += ["ag:no_elastic_modulus", "agt:no_elastic_modulus"]
    if maximum.i_m >= 0:
        noise = bootstrap.noise_n if math.isfinite(bootstrap.noise_n) else 0.0
        delta = max(p.band_rel_min, p.band_noise_factor * noise / maximum.fm_n)
        threshold = maximum.fm_n * (1.0 - delta)
        finite_idx = np.flatnonzero(np.isfinite(force))
        centre = int(np.searchsorted(finite_idx, maximum.i_m))
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
            if math.isfinite(elastic.slope_n):
                agt = (strain_mid - elastic.strain_origin) * 100.0
                ag = agt - maximum.fm_n / elastic.slope_n * 100.0
    elongation = Elongation(
        agt_percent=agt,
        ag_percent=ag,
        ag_location_halfwidth_percent=halfwidth,
        band=band,
        strain_mid=strain_mid,
        flags=tuple(flags),
    )
    agt_percent = elongation.agt_percent
    ag_percent = elongation.ag_percent
    return elongation, agt_percent, ag_percent
