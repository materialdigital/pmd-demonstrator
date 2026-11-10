"""Values from the specimen geometry: S0 scatter, A and Z (ontology-agnostic analysis nodes)."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import flowrep as fr
import numpy as np

from pmd_tensile.curve_math import NAN, _is_number
from pmd_tensile.units import AreaMM2, OptionalAreaMM2, OptionalLengthMM


@dataclasses.dataclass(frozen=True)
class S0Scatter:
    """Relative standard uncertainty of S0 from the scatter of s_i (M14)."""

    u_rel: float
    dof: int
    flags: tuple[str, ...] = ()



@dataclasses.dataclass(frozen=True)
class ScalarResult:
    """A single value with its flags (M13)."""

    value: float
    flags: tuple[str, ...] = ()



# ---------------------------------------------------------------------------
# M14: S0 scatter; M13: A and Z
# ---------------------------------------------------------------------------


@fr.atomic
def s0_scatter(s0_mm2: AreaMM2, measurements_mm2: Sequence[float]) -> S0Scatter:
    """u_rel(S0) = sd(s_1..s_n)/sqrt(n)/S0 with nu = n - 1 (M14).

    Labelled :data:`S0_SCATTER_LABEL`. Propagation is exact: every
    stress-based property (including E) has the same relative uncertainty,
    and strain outputs are unaffected. Fewer than 2 finite measurements give
    NaN with flag ``"s0:no_s0_spread"``.
    """
    values = np.asarray([m for m in measurements_mm2 if _is_number(m)], dtype=float)
    if len(values) < 2 or not _is_number(s0_mm2) or s0_mm2 <= 0:
        scatter = S0Scatter(u_rel=NAN, dof=max(len(values) - 1, 0), flags=("s0:no_s0_spread",))
    else:
        u_abs = float(np.std(values, ddof=1)) / math.sqrt(len(values))
        scatter = S0Scatter(u_rel=u_abs / float(s0_mm2), dof=len(values) - 1)
    return scatter



@fr.atomic
def elongation_after_fracture(l0_mm: OptionalLengthMM, lu_mm: OptionalLengthMM) -> ScalarResult:
    """A = (Lu - L0)/L0 * 100 % from metadata (M13); NaN, ``"a:missing_input"`` if missing."""
    if _is_number(l0_mm) and _is_number(lu_mm) and l0_mm > 0:
        result = ScalarResult(value=(float(lu_mm) - float(l0_mm)) / float(l0_mm) * 100.0)
    else:
        result = ScalarResult(value=NAN, flags=("a:missing_input",))
    return result



@fr.atomic
def reduction_of_area(s0_mm2: OptionalAreaMM2, su_mm2: OptionalAreaMM2) -> ScalarResult:
    """Z = (S0 - Su)/S0 * 100 % from metadata (M13); NaN, ``"z:missing_input"`` if missing."""
    if _is_number(s0_mm2) and _is_number(su_mm2) and s0_mm2 > 0:
        result = ScalarResult(value=(float(s0_mm2) - float(su_mm2)) / float(s0_mm2) * 100.0)
    else:
        result = ScalarResult(value=NAN, flags=("z:missing_input",))
    return result
