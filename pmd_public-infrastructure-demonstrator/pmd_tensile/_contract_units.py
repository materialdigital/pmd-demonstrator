"""Contract units and the explicit conversion into them (private helpers).

:mod:`pmd_tensile.contract` re-exports everything here; import it from there.
The module is private so that sim_atlas does not catalogue the plain functions
:func:`check_unit` and :func:`to_contract_unit` as workflow nodes.
"""

from __future__ import annotations

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
