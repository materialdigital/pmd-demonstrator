"""Comparison with lab-reported values and statistics per group (ontology-agnostic analysis nodes)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import flowrep as fr
import numpy as np
import pandas as pd

from pmd_tensile.contract import ReportedProperties
from pmd_tensile.curve_math import NAN, _is_number

# ---------------------------------------------------------------------------
# Labels for outputs and reports (method-spec §1, §11)
# ---------------------------------------------------------------------------

METHOD_LABEL = "method-defined, ISO 6892-1-inspired"

E_LABEL = "E: regression slope of the elastic part (mE-type), not Annex-G-conformant"

E_STDERR_LABEL = "OLS standard error (conditional on window and independence; not an uncertainty)"

S0_SCATTER_LABEL = (
    "within-specimen dimensional scatter of the mean (nu = 2); excludes instrument bias"
)

OMITTED_UNCERTAINTY: tuple[str, ...] = (
    "force class (ISO 7500-1) and extensometer class (ISO 9513, unknown here)",
    "instrument calibration",
    "uncertainty in Lu and Su",
    "ReH sampling resolution (reported as a diagnostic instead)",
    "min/max noise bias of ReL, ReH and Fm",
    "rate effects",
    "material variability (shown only through the group statistics)",
)

RESIDUAL_RISKS: tuple[str, ...] = (
    "A strain channel that vendor software continues from the crosshead after "
    "extensometer removal cannot be detected (M3).",
)



# ---------------------------------------------------------------------------
# §13: comparison with lab-reported values; §12: aggregation
# ---------------------------------------------------------------------------

# (computed key, ReportedProperties field, tolerance kind, relative part, absolute part, text)
_COMPARISONS: tuple[tuple[str, str, str, float, float, str], ...] = (
    ("fm_n", "fm_n", "relative", 0.001, 0.0, "±0.1 %"),
    ("rm_mpa", "rm_mpa", "relative", 0.001, 0.5, "±(0.1 % + 0.5 MPa)"),
    ("reh_mpa", "reh_mpa", "relative", 0.02, 0.0, "±2 %"),
    ("rp02_mpa", "rp02_mpa", "relative", 0.02, 0.0, "±2 %"),
    ("e_gpa", "e_gpa", "relative", 0.10, 0.0, "±10 %"),
    ("a_percent", "a_percent", "rounding", 0.0, 0.5, "rounding-aware: ±0.5 pp (for an integer lab value: the computed value rounds to it, ties within)"),
    ("z_percent", "z_percent", "rounding", 0.0, 0.5, "rounding-aware: ±0.5 pp (for an integer lab value: the computed value rounds to it, ties within)"),
)
"""Tolerances fixed at G1 (method-spec §13). ReL has no lab values and no row."""



@fr.atomic
def compare_with_reported(
    computed: Mapping[str, float | None], reported: ReportedProperties
) -> pd.DataFrame:
    """Consistency check of computed against lab-reported values (method-spec §13).

    One row per quantity (``fm_n``, ``rm_mpa``, ``reh_mpa``, ``rp02_mpa``,
    ``e_gpa``, ``a_percent``, ``z_percent``) with columns ``quantity``,
    ``computed``, ``reported``, ``rel_deviation``, ``outcome`` and
    ``tolerance``. ``outcome`` is ``"not_comparable"`` if either value is
    missing or NaN. A and Z are ``"within"`` if the computed value rounds to
    the lab's integer value, which with ties counted as within is
    ``|computed - reported| <= 0.5``. This is a consistency check with the
    lab's software, not a validation.
    """
    rows = []
    for key, field, kind, rel, abs_, text in _COMPARISONS:
        value = computed.get(key)
        lab = getattr(reported, field)
        value = float(value) if _is_number(value) else NAN
        lab = float(lab) if _is_number(lab) else NAN
        if math.isnan(value) or math.isnan(lab):
            outcome, deviation = "not_comparable", NAN
        else:
            deviation = (value - lab) / lab if lab != 0.0 else NAN
            if kind == "relative":
                allowed = rel * abs(lab) + abs_
            else:
                allowed = abs_
            outcome = "within" if abs(value - lab) <= allowed else "outside"
        rows.append(
            {
                "quantity": key,
                "computed": value,
                "reported": lab,
                "rel_deviation": deviation,
                "outcome": outcome,
                "tolerance": text,
            }
        )
    comparison = pd.DataFrame(
        rows, columns=["quantity", "computed", "reported", "rel_deviation", "outcome", "tolerance"]
    )
    return comparison



@fr.atomic
def summarize_by_group(
    table: pd.DataFrame, group_column: str, quantities: Sequence[str]
) -> pd.DataFrame:
    """Descriptive statistics per group and quantity (method-spec §12).

    Long format with columns ``group_column``, ``quantity``, ``n`` (non-NaN
    count), ``mean``, ``sd`` (ddof = 1; NaN for n < 2), ``min`` and ``max``.
    Groups appear in sorted order, quantities in the given order. No
    hypothesis tests and no exclusions.
    """
    rows = []
    for group, frame in table.groupby(group_column, sort=True):
        for quantity in quantities:
            values = pd.to_numeric(frame[quantity], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            n = int(len(values))
            rows.append(
                {
                    group_column: group,
                    "quantity": quantity,
                    "n": n,
                    "mean": float(values.mean()) if n else NAN,
                    "sd": float(values.std(ddof=1)) if n >= 2 else NAN,
                    "min": float(values.min()) if n else NAN,
                    "max": float(values.max()) if n else NAN,
                }
            )
    summary = pd.DataFrame(rows, columns=[group_column, "quantity", "n", "mean", "sd", "min", "max"])
    return summary
