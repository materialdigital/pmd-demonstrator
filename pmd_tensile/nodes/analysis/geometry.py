"""Values from the specimen geometry: S0 scatter, A and Z (ontology-agnostic analysis nodes)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import flowrep as fr
import numpy as np

from pmd_tensile.curve_math import NAN, _is_number
from pmd_tensile.units import (
    AreaMM2,
    OptionalAreaMM2,
    OptionalLengthMM,
    Percent,
    RelativeValue,
    StrainPercent,
)


@fr.atomic
def s0_scatter(s0_mm2: AreaMM2, measurements_mm2: Sequence[float]) -> tuple[RelativeValue, int, tuple[str, ...]]:
    """Relative standard uncertainty of S0 from the scatter of its measurements.

    u_rel(S0) = sd(s_1 ... s_n) / sqrt(n) / S0 with n - 1 degrees of freedom: the
    within-specimen dimensional scatter of the mean, excluding instrument bias.
    Every stress, and E, carries the same relative uncertainty from S0; strains do
    not.

    Inputs: S0 (mm²), the individual cross-section measurements (mm²).
    Outputs: ``s0_u_rel`` (ratio), ``s0_dof``, ``flags``: ``s0:no_s0_spread``
    (fewer than two finite measurements; then ``s0_u_rel`` is undefined).
    Limits: the scatter of the measurements only; the uncertainty of the
    measuring instrument is not included.
    """
    values = np.asarray([m for m in measurements_mm2 if _is_number(m)], dtype=float)
    if len(values) < 2 or not _is_number(s0_mm2) or s0_mm2 <= 0:
        s0_u_rel, s0_dof, flags = NAN, max(len(values) - 1, 0), ("s0:no_s0_spread",)
    else:
        s0_u_rel = float(np.std(values, ddof=1)) / math.sqrt(len(values)) / float(s0_mm2)
        s0_dof, flags = len(values) - 1, ()
    return s0_u_rel, s0_dof, flags


@fr.atomic
def elongation_after_fracture(l0_mm: OptionalLengthMM, lu_mm: OptionalLengthMM) -> tuple[StrainPercent, tuple[str, ...]]:
    """Percentage elongation after fracture A = (Lu - L0) / L0 * 100 %.

    Inputs: original and final gauge length L0, Lu (mm), from the test record.
    Outputs: ``a_percent`` (%), ``flags``: ``a:missing_input`` (L0 or Lu missing;
    then A is undefined).
    """
    if _is_number(l0_mm) and _is_number(lu_mm) and l0_mm > 0:
        a_percent, flags = (float(lu_mm) - float(l0_mm)) / float(l0_mm) * 100.0, ()
    else:
        a_percent, flags = NAN, ("a:missing_input",)
    return a_percent, flags


@fr.atomic
def reduction_of_area(s0_mm2: OptionalAreaMM2, su_mm2: OptionalAreaMM2) -> tuple[Percent, tuple[str, ...]]:
    """Percentage reduction of area Z = (S0 - Su) / S0 * 100 %.

    Inputs: original and smallest final cross-section S0, Su (mm²), from the
    test record.
    Outputs: ``z_percent`` (%), ``flags``: ``z:missing_input`` (S0 or Su
    missing; then Z is undefined).
    """
    if _is_number(s0_mm2) and _is_number(su_mm2) and s0_mm2 > 0:
        z_percent, flags = (float(s0_mm2) - float(su_mm2)) / float(s0_mm2) * 100.0, ()
    else:
        z_percent, flags = NAN, ("z:missing_input",)
    return z_percent, flags
