"""Offline acquisition tests for the pmdco2-tto profile on the local Turtle files.

The expected values below were read from ``datasets/pmdco2_tto_example_*.ttl``
(read-only assets of the demonstrator).
"""

from __future__ import annotations

import re
import statistics

import pyiron_workflow as pwf
import pytest
from conftest import PMDCO2_ORIENTATION_FILES

from pmd_tensile import acquisition, exploration, sources
from pmd_tensile.contract import CURVE_ROLES, SpecimenRecord, UnitError, to_contract_unit
from pmd_tensile.profiles import available_profiles, get_profile
from pmd_tensile.profiles.base import QUANTITY_KINDS
from pmd_tensile.profiles.units import qudt_unit_symbol

PROFILE = "pmdco2-tto"

EXPECTED_IDS = {
    "parallel": ["Zx1", "Zx2", "Zx3", "Zx4"],
    "perpendicular": ["Zy1", "Zy2", "Zy3", "Zy4"],
    "diagonal": ["Zd2", "Zd3"],
}
EXPECTED_ORIENTATION = {
    "parallel": "in rolling direction",
    "perpendicular": "perpendicular to rolling direction",
    "diagonal": "diagonal to rolling direction",
}
ALL_IDS = [sid for ids in EXPECTED_IDS.values() for sid in ids]
ORIENTATION_OF = {
    sid: EXPECTED_ORIENTATION[orientation]
    for orientation, ids in EXPECTED_IDS.items()
    for sid in ids
}
EXPECTED_COLUMNS = {"time": (0, "s"), "crosshead": (1, "mm"), "force": (2, "kN"), "strain": (3, "%")}


def _source_with(tmp_path, orientation: str, text: str) -> dict:
    """A one-file Turtle source with modified content of a local pmdco2 file."""
    path = tmp_path / f"{orientation}.ttl"
    path.write_text(text, encoding="utf-8")
    return {"kind": "turtle", "locations": [str(path)]}


def _replace_once(text: str, pattern: str, replacement: str) -> str:
    new_text, count = re.subn(pattern, replacement, text, count=0, flags=re.DOTALL)
    assert count == 1, f"pattern matched {count} times"
    return new_text


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------


def test_profile_is_registered():
    assert PROFILE in available_profiles()
    with pytest.raises(KeyError):
        get_profile("no-such-profile")


def test_pmdco2_files_conform_to_pmdco2_profile(pmdco2_source):
    results = exploration.conformance_results(PROFILE, pmdco2_source)
    assert results
    assert all(all(answers.values()) for answers in results.values())
    run = pwf.node(exploration.check_conformance, "conformance").run(
        profile_id=PROFILE, source=pmdco2_source
    )
    assert run.status == "finished", run.exception
    assert run.outputs.checked_profile_id == PROFILE


def test_lists_ten_specimens_split_4_4_2(pmdco2_source):
    run = pwf.node(exploration.list_specimens, "specimens").run(
        profile_id=PROFILE, source=pmdco2_source
    )
    assert run.status == "finished", run.exception
    assert sorted(run.outputs.specimen_ids) == sorted(ALL_IDS)

    table = exploration.specimen_table(PROFILE, pmdco2_source)
    counts = table.groupby("orientation")["specimen_id"].count().to_dict()
    assert counts == {
        EXPECTED_ORIENTATION["parallel"]: 4,
        EXPECTED_ORIENTATION["perpendicular"]: 4,
        EXPECTED_ORIENTATION["diagonal"]: 2,
    }


@pytest.mark.parametrize("orientation", list(PMDCO2_ORIENTATION_FILES))
def test_each_file_holds_its_orientation(orientation):
    source = {"kind": "turtle", "locations": [str(PMDCO2_ORIENTATION_FILES[orientation])]}
    table = exploration.specimen_table(PROFILE, source)
    assert list(table["specimen_id"]) == EXPECTED_IDS[orientation]
    assert set(table["orientation"]) == {EXPECTED_ORIENTATION[orientation]}


# ---------------------------------------------------------------------------
# Specimen records
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("specimen_id", ALL_IDS)
def test_specimen_record(pmdco2_source, specimen_id):
    record = acquisition.build_specimen_record(PROFILE, pmdco2_source, specimen_id)
    assert isinstance(record, SpecimenRecord)
    assert record.specimen_id == specimen_id
    assert record.orientation == ORIENTATION_OF[specimen_id]
    assert record.provenance_iri.startswith("https://w3id.org/pmd/ao/tte/")

    # S0 and the individual measurements s1..s3 (S0 is their mean in these data)
    assert len(record.cross_section_measurements_mm2) == 3
    assert record.s0_mm2 == pytest.approx(
        statistics.mean(record.cross_section_measurements_mm2), abs=5e-4
    )
    assert 100.0 < record.s0_mm2 < 140.0

    assert record.l0_mm == 60.0
    assert record.su_mm2 is not None and 0.0 < record.su_mm2 < record.s0_mm2

    assert record.curve_url == f"https://data.pmds-mpcdf-2.pyiron.com/{specimen_id}.csv"
    assert set(record.curve_columns) == set(CURVE_ROLES)
    for role, (index, unit) in EXPECTED_COLUMNS.items():
        assert record.curve_columns[role].index == index
        assert record.curve_columns[role].unit == unit

    assert len(record.notes) == 1


def test_specimen_record_known_values(pmdco2_source):
    zd3 = acquisition.build_specimen_record(PROFILE, pmdco2_source, "Zd3")
    assert zd3.s0_mm2 == 120.707
    assert zd3.cross_section_measurements_mm2 == (120.867, 120.544, 120.71)
    assert (zd3.l0_mm, zd3.lu_mm, zd3.su_mm2) == (60.0, 79.97, 43.924)
    assert zd3.notes == ("Proben aus Stahlblech entnommen, diagonal der Walzrichtung",)

    # Zx1: Lu is "NaN" in the graph -> None; its lab note is carried through.
    zx1 = acquisition.build_specimen_record(PROFILE, pmdco2_source, "Zx1")
    assert zx1.lu_mm is None
    assert "Weggeschwindigkeit wurde erhöht" in zx1.notes[0]
    zx2 = acquisition.build_specimen_record(PROFILE, pmdco2_source, "Zx2")
    assert "Extensometer nach 3,9 % Dehnung" in zx2.notes[0]


def test_specimen_record_node(pmdco2_source):
    run = pwf.node(acquisition.get_specimen_record, "record").run(
        profile_id=PROFILE, source=pmdco2_source, specimen_id="Zy3"
    )
    assert run.status == "finished", run.exception
    assert run.outputs.record == acquisition.build_specimen_record(PROFILE, pmdco2_source, "Zy3")


def test_test_with_two_specimen_ids_raises(tmp_path):
    # Give Zx1's test piece a second name; the test then maps to two IDs.
    text = PMDCO2_ORIENTATION_FILES["parallel"].read_text(encoding="utf-8")
    text += (
        "\n<https://w3id.org/pmd/ao/tte/testpiece_name/Zx1testpiece/Zx1> "
        "<https://w3id.org/pmd/co/characteristic> <urn:test:second-name> .\n"
        "<urn:test:second-name> a <https://w3id.org/pmd/co/TestPieceName> ;\n"
        '    <https://w3id.org/pmd/co/value> "Zx1-alias" .\n'
    )
    source = _source_with(tmp_path, "parallel", text)
    with pytest.raises(ValueError, match="several specimen IDs"):
        exploration.specimen_table(PROFILE, source)


def test_unknown_specimen_raises(pmdco2_source):
    with pytest.raises(LookupError, match="Zq9"):
        acquisition.build_specimen_record(PROFILE, pmdco2_source, "Zq9")


def test_semantic_profile_rejects_curve_locator(pmdco2_source):
    locator = {"url_template": "https://example.org/{specimen_id}.csv", "columns": {}}
    with pytest.raises(ValueError, match="links curves semantically"):
        acquisition.build_specimen_record(PROFILE, pmdco2_source, "Zx1", curve_locator=locator)


# ---------------------------------------------------------------------------
# Reported properties
# ---------------------------------------------------------------------------


def test_reported_properties_known_values(pmdco2_source):
    zx1 = acquisition.build_reported_properties(PROFILE, pmdco2_source, "Zx1")
    assert zx1.specimen_id == "Zx1"
    assert zx1.e_gpa == 194.0
    assert zx1.rp02_mpa == 375.0
    assert zx1.reh_mpa == 380.0
    assert zx1.rm_mpa == 514.0
    assert zx1.fm_n == pytest.approx(62007.0)  # graph: 62.007 kiloN
    assert zx1.a_percent is None  # "NaN" in the graph
    assert zx1.z_percent == 56.0
    assert zx1.rel_mpa is None  # ReL is not described in pmdco2 graphs

    zd3 = acquisition.build_reported_properties(PROFILE, pmdco2_source, "Zd3")
    assert zd3.rp02_mpa is None  # "NaN" in the graph
    assert (zd3.e_gpa, zd3.reh_mpa, zd3.rm_mpa) == (195.0, 396.0, 511.0)
    assert (zd3.a_percent, zd3.z_percent) == (33.0, 64.0)


def test_reported_properties_present_or_none(pmdco2_source):
    reported = {
        sid: acquisition.build_reported_properties(PROFILE, pmdco2_source, sid) for sid in ALL_IDS
    }
    # Rp0.2 is missing exactly for S355-7 (Zy3) and S355-10 (Zd3).
    assert {sid for sid, r in reported.items() if r.rp02_mpa is None} == {"Zy3", "Zd3"}
    for r in reported.values():
        assert r.rel_mpa is None
        assert 190.0 <= r.e_gpa <= 220.0
        assert 370.0 <= r.reh_mpa <= 410.0
        assert 500.0 <= r.rm_mpa <= 520.0
        assert 59_000.0 <= r.fm_n <= 64_000.0


def _rate_switch_strain(source: dict, specimen_id: str) -> float:
    """The profile's ``eps_sw`` value of one specimen, in contract units (ratio).

    This checks the profile's query and its unit conversion directly,
    independently of ``acquisition.build_specimen_record``.
    """
    profile = get_profile(PROFILE)
    table = exploration.specimen_table(PROFILE, source)
    process = table[table["specimen_id"] == specimen_id]["process"].item()
    rows = sources.select(source, profile.quantities.query, profile.quantities.columns)
    rows = rows[(rows["process"] == process) & (rows["quantity"] == "eps_sw")]
    assert len(rows) == 1
    row = rows.iloc[0]
    return to_contract_unit(float(row["value"]), profile.unit_symbol(row["unit"]), QUANTITY_KINDS["eps_sw"])


@pytest.mark.parametrize(
    ("specimen_id", "expected"),
    [("Zx1", 0.016), ("Zx2", 0.038), ("Zd2", 0.040)],  # graph: 1.6 %, 3.8 %, 4.0 %
)
def test_rate_switch_strain_is_a_ratio(pmdco2_source, specimen_id, expected):
    assert _rate_switch_strain(pmdco2_source, specimen_id) == pytest.approx(expected, rel=1e-12)


def test_specimen_record_rate_switch_strain(pmdco2_source):
    # graph: 1.6 % (Zx1) and 4.0 % (Zd2), arriving as ratios in the record
    for specimen_id, expected in (("Zx1", 0.016), ("Zd2", 0.040)):
        record = acquisition.build_specimen_record(PROFILE, pmdco2_source, specimen_id)
        assert record.rate_switch_strain == pytest.approx(expected, rel=1e-12)


def test_reported_properties_node(pmdco2_source):
    run = pwf.node(acquisition.get_reported_properties, "reported").run(
        profile_id=PROFILE, source=pmdco2_source, specimen_id="Zx4"
    )
    assert run.status == "finished", run.exception
    assert run.outputs.reported.rm_mpa == 519.0


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_unit_tables():
    assert qudt_unit_symbol("http://qudt.org/vocab/unit/KiloN") == "kN"
    assert qudt_unit_symbol("http://qudt.org/vocab/unit/PERCENT") == "%"
    with pytest.raises(UnitError):
        qudt_unit_symbol("http://qudt.org/vocab/unit/kiloN")  # alias only within a profile
    assert get_profile(PROFILE).unit_symbol("http://qudt.org/vocab/unit/kiloN") == "kN"
    assert to_contract_unit(2.5, "kN", "force") == 2500.0
    assert to_contract_unit(4.0, "%", "strain") == pytest.approx(0.04)
    with pytest.raises(UnitError):
        to_contract_unit(1.0, "mm", "area")


S0_UNIT_PATTERN = (
    r"(pmdao-tto-tt-S355-1_crossSectionArea_S0> a [^.]*?"
    r"ns1:unit <http://qudt\.org/vocab/unit/)MilliM2>"
)


def test_unknown_unit_raises(tmp_path):
    text = PMDCO2_ORIENTATION_FILES["parallel"].read_text(encoding="utf-8")
    source = _source_with(tmp_path, "parallel", _replace_once(text, S0_UNIT_PATTERN, r"\1FURLONG2>"))
    with pytest.raises(UnitError, match="FURLONG2"):
        acquisition.build_specimen_record(PROFILE, source, "Zx1")


def test_mismatched_unit_raises(tmp_path):
    # A valid QUDT unit of the wrong kind (length instead of area) for S0.
    text = PMDCO2_ORIENTATION_FILES["parallel"].read_text(encoding="utf-8")
    source = _source_with(tmp_path, "parallel", _replace_once(text, S0_UNIT_PATTERN, r"\1MilliM>"))
    with pytest.raises(UnitError, match="'mm' is not valid for quantity kind 'area'"):
        acquisition.build_specimen_record(PROFILE, source, "Zx1")


def test_mismatched_curve_column_unit_raises(tmp_path):
    # Declare the force column of the Zx1 curve table in millimetres.
    text = PMDCO2_ORIENTATION_FILES["parallel"].read_text(encoding="utf-8")
    graph_text = _replace_once(
        text,
        # <_:t711> is the force column of Zx1's curve table; stay within its block.
        r"(<_:t711> a csvw:Column,\s+ns1:Force ;(?:(?!\n\n).)*?"
        r"ns1:unit <http://qudt\.org/vocab/unit/)KiloN>",
        r"\1MilliM>",
    )
    source = _source_with(tmp_path, "parallel", graph_text)
    with pytest.raises(UnitError, match="'force'"):
        acquisition.build_specimen_record(PROFILE, source, "Zx1")


def test_missing_unit_raises(tmp_path):
    text = PMDCO2_ORIENTATION_FILES["parallel"].read_text(encoding="utf-8")
    graph_text = _replace_once(
        text,
        r"(pmdao-tto-tt-S355-1_crossSectionArea_S0> a [^.]*?)"
        r"ns1:unit <http://qudt\.org/vocab/unit/MilliM2> ;",
        r"\1",
    )
    source = _source_with(tmp_path, "parallel", graph_text)
    with pytest.raises(UnitError, match="has no unit"):
        acquisition.build_specimen_record(PROFILE, source, "Zx1")


def test_unrecognized_cross_section_name_raises(tmp_path):
    # The pmdco2 naming convention is strict: an unexpected suffix is an error.
    text = PMDCO2_ORIENTATION_FILES["parallel"].read_text(encoding="utf-8")
    graph_text = text.replace("pmdao-tto-tt-S355-1_crossSectionArea_s3", "pmdao-tto-tt-S355-1_crossSectionArea_x3")
    source = _source_with(tmp_path, "parallel", graph_text)
    with pytest.raises(ValueError, match="UNRECOGNIZED"):
        acquisition.build_specimen_record(PROFILE, source, "Zx1")


# ---------------------------------------------------------------------------
# Source descriptions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        {"kind": "turtle", "locations": []},
        {"kind": "turtle", "locations": "datasets/x.ttl"},
        {"kind": "sparql", "locations": ["x"]},
        {"kind": "ontodocker", "datasets": ["a"], "token": "secret"},
        {"kind": "turtle", "locations": ["x.ttl"], "address": "host"},
    ],
)
def test_invalid_sources_are_rejected(source):
    with pytest.raises(sources.SourceError):
        sources.validate_source(source)
