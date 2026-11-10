"""Neutral data contract between the ontology-aware and the ontology-agnostic parts.

Everything here is free of ontology IRIs, SPARQL and graph access. The only
IRI-valued field is ``SpecimenRecord.provenance_iri``, which is opaque and used
for reporting only.

Contract units (fixed; acquisition converts into them):

=====================  ======  ===========================================
quantity kind          unit    used for
=====================  ======  ===========================================
``time``               s       curve time
``length``             mm      crosshead separation, L0, Lu
``area``               mm2     S0, s1..sn, Su
``force``              N       curve force, reported Fm
``strain``             1       curve strain, rate-switch strain (ratio)
``stress``             MPa     reported Rp0.2, ReH, ReL, Rm
``modulus``            GPa     reported E
``percent``            %       reported A, Z
=====================  ======  ===========================================

Source units are given as neutral unit symbols (``"s"``, ``"mm"``, ``"mm2"``,
``"N"``, ``"kN"``, ``"%"``, ``"1"``, ``"MPa"``, ``"GPa"``). Profiles map their
vocabulary (e.g. QUDT IRIs) to these symbols. :func:`to_contract_unit` converts
with an explicit table and raises :class:`UnitError` for anything it does not
know; it never only warns.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

import numpy as np

CONTRACT_UNITS: dict[str, str] = {
    "time": "s",
    "length": "mm",
    "area": "mm2",
    "force": "N",
    "strain": "1",
    "stress": "MPa",
    "modulus": "GPa",
    "percent": "%",
}
"""Contract unit symbol per quantity kind."""

# Multiplicative factors from a source unit symbol to the contract unit, per
# quantity kind. A unit that is not listed for a kind is a mismatch.
_TO_CONTRACT_FACTORS: dict[str, dict[str, float]] = {
    "time": {"s": 1.0},
    "length": {"mm": 1.0},
    "area": {"mm2": 1.0},
    "force": {"N": 1.0, "kN": 1000.0},
    "strain": {"1": 1.0, "%": 0.01},
    "stress": {"MPa": 1.0},
    "modulus": {"GPa": 1.0, "MPa": 0.001},
    "percent": {"%": 1.0},
}


class UnitError(ValueError):
    """An unknown unit, or a unit that does not fit the expected quantity kind."""


def check_unit(unit: str, kind: str) -> None:
    """Raise :class:`UnitError` unless ``unit`` is convertible for ``kind``."""
    if kind not in _TO_CONTRACT_FACTORS:
        raise UnitError(f"unknown quantity kind {kind!r}")
    if unit not in _TO_CONTRACT_FACTORS[kind]:
        allowed = sorted(_TO_CONTRACT_FACTORS[kind])
        raise UnitError(
            f"unit {unit!r} is not valid for quantity kind {kind!r} "
            f"(contract unit {CONTRACT_UNITS[kind]!r}; accepted: {allowed})"
        )


def to_contract_unit(value, unit: str, kind: str):
    """Convert a float or array from ``unit`` to the contract unit of ``kind``."""
    check_unit(unit, kind)
    factor = _TO_CONTRACT_FACTORS[kind][unit]
    if factor == 1.0:
        return value
    return value * factor


__all__ = [
    "CONTRACT_UNITS",
    "CURVE_ROLES",
    "CURVE_ROLE_KINDS",
    "ColumnSpec",
    "Curve",
    "ReportedProperties",
    "SpecimenRecord",
    "UnitError",
    "check_unit",
    "to_contract_unit",
]

CURVE_ROLES: tuple[str, ...] = ("time", "crosshead", "force", "strain")
"""Channels of a raw tensile curve, in contract order."""

CURVE_ROLE_KINDS: dict[str, str] = {
    "time": "time",
    "crosshead": "length",
    "force": "force",
    "strain": "strain",
}
"""Quantity kind of each curve channel."""


@dataclasses.dataclass(frozen=True)
class ColumnSpec:
    """Where a curve channel is in the raw CSV, and its source unit.

    A column is located either by its 0-based ``index`` or by its ``header``
    name in the first header row. ``unit`` is a neutral unit symbol.
    """

    unit: str
    index: int | None = None
    header: str | None = None

    def __post_init__(self) -> None:
        if self.index is None and self.header is None:
            raise ValueError("a column needs an index or a header name")
        if self.index is not None and self.index < 0:
            raise ValueError(f"column index must be >= 0, got {self.index}")


@dataclasses.dataclass(frozen=True)
class SpecimenRecord:
    """Ontology-free description of one tested specimen.

    ``specimen_id`` is the test-piece name (e.g. ``"Zx1"``) and the key across
    ontology profiles. ``provenance_iri`` is opaque (reporting only).
    ``curve_columns`` maps each of :data:`CURVE_ROLES` to a :class:`ColumnSpec`.
    Optional values are ``None`` when the graph does not provide them.

    ``rate_switch_strain`` is the strain (ratio, not percent) at which the lab
    switched the test speed. It is diagnostic only (method-spec M3/A2): the
    analysis uses it for the ``near_rate_switch`` flag and never to change a
    value.
    """

    specimen_id: str
    provenance_iri: str
    orientation: str
    s0_mm2: float
    cross_section_measurements_mm2: tuple[float, ...]
    l0_mm: float | None
    lu_mm: float | None
    su_mm2: float | None
    notes: tuple[str, ...]
    curve_url: str
    curve_columns: Mapping[str, ColumnSpec]
    rate_switch_strain: float | None = None

    def __post_init__(self) -> None:
        if not self.specimen_id:
            raise ValueError("specimen_id must be non-empty")
        if not (math.isfinite(self.s0_mm2) and self.s0_mm2 > 0):
            raise ValueError(f"S0 must be a positive finite number, got {self.s0_mm2}")
        if set(self.curve_columns) != set(CURVE_ROLES):
            raise ValueError(
                f"curve_columns must have exactly the roles {CURVE_ROLES}, "
                f"got {tuple(self.curve_columns)}"
            )
        for role, column in self.curve_columns.items():
            check_unit(column.unit, CURVE_ROLE_KINDS[role])


@dataclasses.dataclass(frozen=True, eq=False)
class Curve:
    """Raw tensile curve in contract units, with its provenance.

    ``strain`` is a ratio (not percent). ``sha256`` is the hex digest of the
    exact downloaded bytes; ``source_url`` is where they came from.
    """

    time_s: np.ndarray
    crosshead_mm: np.ndarray
    force_n: np.ndarray
    strain: np.ndarray
    source_url: str
    sha256: str

    def __post_init__(self) -> None:
        lengths = {
            len(self.time_s),
            len(self.crosshead_mm),
            len(self.force_n),
            len(self.strain),
        }
        if len(lengths) != 1:
            raise ValueError(f"curve channels differ in length: {sorted(lengths)}")


@dataclasses.dataclass(frozen=True)
class ReportedProperties:
    """Lab-reported values in contract units; used only for comparison.

    These values are never inputs to the analysis. ``None`` means the graph
    does not report the value (absent, or given as NaN).
    """

    specimen_id: str
    e_gpa: float | None = None
    rp02_mpa: float | None = None
    reh_mpa: float | None = None
    rel_mpa: float | None = None
    rm_mpa: float | None = None
    fm_n: float | None = None
    a_percent: float | None = None
    z_percent: float | None = None
