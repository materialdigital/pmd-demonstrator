"""QUDT unit IRIs -> neutral unit symbols of the data contract.

The table is explicit. An IRI that is not listed (after applying a profile's
declared aliases) raises ``contract.UnitError``; whether the symbol fits the
expected quantity kind is checked by ``contract.check_unit``.

The canonical IRIs below were checked against the QUDT unit vocabulary bundled
with semantikon 1.4.4 (``semantikon/data/qudt.ttl``).
"""

from __future__ import annotations

from collections.abc import Mapping

from pmd_tensile.contract import UnitError

QUDT_UNIT = "http://qudt.org/vocab/unit/"

QUDT_UNIT_SYMBOLS: dict[str, str] = {
    QUDT_UNIT + "SEC": "s",
    QUDT_UNIT + "MilliM": "mm",
    QUDT_UNIT + "MilliM2": "mm2",
    QUDT_UNIT + "N": "N",
    QUDT_UNIT + "KiloN": "kN",
    QUDT_UNIT + "PERCENT": "%",
    QUDT_UNIT + "UNITLESS": "1",
    QUDT_UNIT + "MegaPA": "MPa",
    QUDT_UNIT + "GigaPA": "GPa",
}
"""Canonical QUDT unit IRI -> contract unit symbol."""


def qudt_unit_symbol(unit_iri: str, aliases: Mapping[str, str] | None = None) -> str:
    """Return the neutral unit symbol for a QUDT unit IRI.

    ``aliases`` maps declared non-canonical IRIs to canonical QUDT IRIs.
    Raises ``UnitError`` for an unknown IRI.
    """
    canonical = (aliases or {}).get(unit_iri, unit_iri)
    try:
        return QUDT_UNIT_SYMBOLS[canonical]
    except KeyError:
        raise UnitError(f"unknown unit IRI {unit_iri!r}") from None
