"""Plain-function forms behind the acquisition nodes (ontology-aware).

:mod:`pmd_tensile.acquisition` holds the nodes; they call the functions here.
The outputs are the ontology-free dataclasses of ``pmd_tensile.contract``, in
contract units. Every unit is converted through an explicit table; an unknown
or mismatched unit raises. Notebooks reach these functions through
:mod:`pmd_tensile.demo`.
"""

from __future__ import annotations

import math
from collections import defaultdict

from pmd_tensile import _sources as sources
from pmd_tensile._exploration import specimen_table
from pmd_tensile.contract import (
    CURVE_ROLE_KINDS,
    CURVE_ROLES,
    ColumnSpec,
    Curve,
    ReportedProperties,
    SpecimenRecord,
    UnitError,
    check_unit,
    to_contract_unit,
)
from pmd_tensile.profiles import CurveLocator, OntologyProfile, get_profile
from pmd_tensile.profiles.base import QUANTITY_KINDS

# A measured value from the graph: (individual IRI, value, unit symbol or None).
_Value = tuple[str, float, str | None]


def _specimen_row(profile_id: str, source: dict, specimen_id: str) -> dict:
    table = specimen_table(profile_id, source)
    match = table[table["specimen_id"] == specimen_id]
    if match.empty:
        raise LookupError(
            f"specimen {specimen_id!r} not found with profile {profile_id!r}; "
            f"available: {list(table['specimen_id'])}"
        )
    return match.iloc[0].to_dict()


def _quantity_values(profile: OntologyProfile, source: dict, process: str) -> dict[str, list[_Value]]:
    """Values per neutral quantity key for one test, with unit symbols."""
    rows = sources.select(source, profile.quantities.query, profile.quantities.columns)
    rows = rows[rows["process"] == process]
    values: dict[str, list[_Value]] = defaultdict(list)
    seen = set()
    for row in rows.itertuples(index=False):
        if row.quantity not in QUANTITY_KINDS:
            raise ValueError(
                f"profile {profile.profile_id!r} returned unrecognized quantity "
                f"{row.quantity!r} for {row.individual}"
            )
        key = (row.quantity, row.individual)
        if key in seen:
            raise ValueError(f"{row.quantity} of {row.individual} has several values or units")
        seen.add(key)
        unit = None if row.unit is None else profile.unit_symbol(row.unit)
        values[row.quantity].append((row.individual, _parse_float(row.value), unit))
    for items in values.values():
        items.sort(key=lambda item: item[0])
    return values


def _parse_float(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        raise ValueError(f"not a numeric value: {text!r}") from None


def _single_value(
    values: dict[str, list[_Value]], key: str, specimen_id: str, *, required: bool
) -> float | None:
    """The one value of ``key`` in contract units, or ``None`` if absent/NaN."""
    items = values.get(key, [])
    if len(items) > 1:
        raise ValueError(f"{key} of {specimen_id!r}: expected one value, found {len(items)}")
    if not items:
        if required:
            raise LookupError(f"{key} of {specimen_id!r} is missing in the graph")
        return None
    individual, value, unit = items[0]
    if unit is None:
        raise UnitError(f"{key} of {specimen_id!r} ({individual}) has no unit")
    converted = float(to_contract_unit(value, unit, QUANTITY_KINDS[key]))
    if math.isnan(converted):
        if required:
            raise ValueError(f"{key} of {specimen_id!r} is NaN in the graph")
        return None
    return converted


def _measurements(values: dict[str, list[_Value]], key: str, specimen_id: str) -> tuple[float, ...]:
    """All finite values of ``key`` (e.g. s1..sn) in contract units."""
    result = []
    for individual, value, unit in values.get(key, []):
        if unit is None:
            raise UnitError(f"{key} of {specimen_id!r} ({individual}) has no unit")
        converted = float(to_contract_unit(value, unit, QUANTITY_KINDS[key]))
        if math.isfinite(converted):
            result.append(converted)
    return tuple(result)


def _notes(profile: OntologyProfile, source: dict, process: str) -> tuple[str, ...]:
    if profile.notes is None:
        return ()
    rows = sources.select(source, profile.notes.query, profile.notes.columns)
    return tuple(sorted({note for note in rows[rows["process"] == process]["note"] if note}))


def _semantic_curve_columns(
    profile: OntologyProfile, source: dict, process: str, specimen_id: str
) -> tuple[str, dict[str, ColumnSpec]]:
    query = profile.curve_columns
    rows = sources.select(source, query.query, query.columns)
    rows = rows[rows["process"] == process]
    urls = sorted(set(rows["url"]))
    if len(urls) != 1:
        raise LookupError(f"specimen {specimen_id!r} needs exactly one curve URL, found {urls}")
    columns = {}
    for role in CURVE_ROLES:
        matches = rows[rows["role"] == role]
        if len(matches) != 1:
            raise LookupError(
                f"curve of {specimen_id!r} needs exactly one {role!r} column, found {len(matches)}"
            )
        row = matches.iloc[0]
        if row["unit"] is None:
            raise UnitError(f"{role!r} column of {specimen_id!r} has no unit")
        unit = profile.unit_symbol(row["unit"])
        check_unit(unit, CURVE_ROLE_KINDS[role])
        columns[role] = ColumnSpec(unit=unit, index=int(float(row["column_number"])) - 1)
    return urls[0], columns


def _located_curve_columns(
    profile: OntologyProfile, specimen_id: str, curve_locator: dict | None
) -> tuple[str, dict[str, ColumnSpec]]:
    locator = profile.curve_locator
    if curve_locator is not None:
        locator = CurveLocator.from_dict(curve_locator)
    return locator.url_for(specimen_id), dict(locator.columns)


def build_specimen_record(
    profile_id: str, source: dict, specimen_id: str, curve_locator: dict | None = None
) -> SpecimenRecord:
    """Plain-function form of :func:`pmd_tensile.acquisition.get_specimen_record`."""
    profile = get_profile(profile_id)
    if curve_locator is not None and profile.links_curves_semantically:
        raise ValueError(
            f"profile {profile_id!r} links curves semantically; "
            "a non-semantic curve_locator is not accepted"
        )
    row = _specimen_row(profile_id, source, specimen_id)
    process = row["process"]
    values = _quantity_values(profile, source, process)
    if profile.links_curves_semantically:
        curve_url, curve_columns = _semantic_curve_columns(profile, source, process, specimen_id)
    else:
        curve_url, curve_columns = _located_curve_columns(profile, specimen_id, curve_locator)
    return SpecimenRecord(
        specimen_id=specimen_id,
        provenance_iri=process,
        orientation=row["orientation"],
        s0_mm2=_single_value(values, "S0", specimen_id, required=True),
        cross_section_measurements_mm2=_measurements(values, "s", specimen_id),
        l0_mm=_single_value(values, "L0", specimen_id, required=False),
        lu_mm=_single_value(values, "Lu", specimen_id, required=False),
        su_mm2=_single_value(values, "Su", specimen_id, required=False),
        notes=_notes(profile, source, process),
        curve_url=curve_url,
        curve_columns=curve_columns,
        rate_switch_strain=_single_value(values, "eps_sw", specimen_id, required=False),
    )


def build_reported_properties(profile_id: str, source: dict, specimen_id: str) -> ReportedProperties:
    """Plain-function form of :func:`pmd_tensile.acquisition.get_reported_properties`."""
    profile = get_profile(profile_id)
    process = _specimen_row(profile_id, source, specimen_id)["process"]
    values = _quantity_values(profile, source, process)

    def reported(key: str) -> float | None:
        return _single_value(values, key, specimen_id, required=False)

    return ReportedProperties(
        specimen_id=specimen_id,
        e_gpa=reported("E"),
        rp02_mpa=reported("Rp0.2"),
        reh_mpa=reported("ReH"),
        rel_mpa=reported("ReL"),
        rm_mpa=reported("Rm"),
        fm_n=reported("Fm"),
        a_percent=reported("A"),
        z_percent=reported("Z"),
    )


def download_curve(record: SpecimenRecord) -> Curve:
    """Plain-function form of :func:`pmd_tensile.acquisition.fetch_curve`."""
    data, sha256 = sources.fetch_bytes(record.curve_url)
    curve = sources.parse_curve_csv(data, record.curve_columns, source_url=record.curve_url)
    if curve.sha256 != sha256:
        raise RuntimeError(f"SHA-256 mismatch for {record.curve_url}")
    return curve
