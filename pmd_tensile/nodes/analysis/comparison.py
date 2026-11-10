"""Comparison with lab-reported values and statistics per group (ontology-agnostic analysis nodes)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import flowrep as fr
import numpy as np
import pandas as pd

from pmd_tensile.contract import ReportedProperties
from pmd_tensile.curve_math import NAN, _is_number
from pmd_tensile.units import ForceN, ModulusGPa, Percent, StrainPercent, StressMPa

# (quantity, unit, lab resolution, tolerance kind, relative part, absolute part, rule as text).
# The lab resolution is the step of the lab's reported values; the ten S355 lab values are
# multiples of it. ReL has no lab values and no row.
COMPARISON_RULES: tuple[tuple[str, str, float, str, float, float, str], ...] = (
    ("fm_n", "N", 1.0, "relative", 0.001, 0.0, "±0.1 %"),
    ("rm_mpa", "MPa", 1.0, "relative", 0.001, 0.5, "±(0.1 % + 0.5 MPa)"),
    ("reh_mpa", "MPa", 1.0, "relative", 0.02, 0.0, "±2 %"),
    ("rp02_mpa", "MPa", 1.0, "relative", 0.02, 0.0, "±2 %"),
    ("e_gpa", "GPa", 1.0, "relative", 0.10, 0.0, "±10 %"),
    ("a_percent", "%", 0.5, "rounding", 0.0, 0.5, "rounding-aware: ±0.5 pp (for an integer lab value: the computed value rounds to it, ties within)"),
    ("z_percent", "%", 1.0, "rounding", 0.0, 0.5, "rounding-aware: ±0.5 pp (for an integer lab value: the computed value rounds to it, ties within)"),
)

COMPARISON_COLUMNS = (
    "quantity", "unit", "computed", "reported", "deviation", "lab_resolution", "deviation_in_resolution",
    "allowed_deviation", "deviation_in_tolerance", "outcome", "tolerance",
)


@fr.atomic
def compare_with_reported(
    fm_n: ForceN,
    rm_mpa: StressMPa,
    reh_mpa: StressMPa,
    rp02_mpa: StressMPa,
    e_gpa: ModulusGPa,
    a_percent: StrainPercent,
    z_percent: Percent,
    reported: ReportedProperties,
) -> pd.DataFrame:
    """Computed values next to the lab-reported ones: a consistency check, not a validation.

    The lab's values come from its own software with unknown settings, so close
    agreement shows that the two evaluations construct the quantities alike.
    For each of Fm, Rm, ReH, Rp0.2, E, A and Z the table gives the deviation
    computed - reported, the deviation in steps of the lab's resolution, the
    allowed deviation of the comparison rule and the deviation in units of it.
    The rule is relative for Fm, Rm, ReH, Rp0.2 and E; for A and Z the computed
    value must round to the lab's integer value (|deviation| <= 0.5 pp, ties within).

    Inputs: the computed Fm (N), Rm, ReH, Rp0.2 (MPa), E (GPa), A, Z (%), and the
    lab-reported values.
    Output: ``comparison``, one row per quantity with the columns
    :data:`COMPARISON_COLUMNS`; ``outcome`` is ``within``, ``outside`` or
    ``not_comparable`` (either value missing).
    """
    computed = {"fm_n": fm_n, "rm_mpa": rm_mpa, "reh_mpa": reh_mpa, "rp02_mpa": rp02_mpa, "e_gpa": e_gpa,
                "a_percent": a_percent, "z_percent": z_percent}
    rows = []
    for key, unit, resolution, kind, rel, abs_, text in COMPARISON_RULES:
        value = float(computed[key]) if _is_number(computed[key]) else NAN
        lab = float(getattr(reported, key)) if _is_number(getattr(reported, key)) else NAN
        allowed = rel * abs(lab) + abs_ if kind == "relative" else abs_
        deviation = value - lab
        if math.isnan(deviation):
            outcome = "not_comparable"
        else:
            outcome = "within" if abs(deviation) <= allowed else "outside"
        rows.append((key, unit, value, lab, deviation, resolution, deviation / resolution, allowed,
                     deviation / allowed if allowed > 0 else NAN, outcome, text))
    comparison = pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))
    return comparison


@fr.atomic
def summarize_by_group(
    table: pd.DataFrame, group_column: str, quantities: Sequence[str]
) -> pd.DataFrame:
    """Descriptive statistics per group and quantity, without hypothesis tests or exclusions.

    Output ``summary``, long format with the columns ``group_column``,
    ``quantity``, ``n`` (count of defined values), ``mean``, ``sd`` (n - 1 in the
    denominator; undefined for n < 2), ``min`` and ``max``. Groups appear sorted,
    quantities in the given order.
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
