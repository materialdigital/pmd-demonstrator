"""Tests of the pmdco3-tto profile.

Offline tests use a hand-written STRUCTURAL FIXTURE: a tiny Turtle graph that
reproduces the modelling pattern of the pinned PMDco3/TTO S355 graph with the
same IRIs the profile queries, including its distractors (extensometer gauge
length with the L0 class, load-cell force range with the Fm class, process
identifier next to the process note). Its numbers are made up; it is not real
data.

The test against the real pinned graph needs network access and is marked
``network`` (skipped by default).
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pyiron_workflow as pwf
import pytest

from tensile_workflow import acquisition, exploration, sources
from tensile_workflow.contract import UnitError
from tensile_workflow.profiles import available_profiles, get_profile, pmdco3_tto

PROFILE = "pmdco3-tto"

_PREFIXES = """\
@prefix ex:    <https://example.org/fixture/> .
@prefix obo:   <http://purl.obolibrary.org/obo/> .
@prefix pmdco: <https://w3id.org/pmd/co/> .
@prefix tto:   <https://w3id.org/pmd/tto/> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix unit:  <http://qudt.org/vocab/unit/> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
"""


def _measured(test: str, name: str, cls: str, bearer: str, value: str, unit: str | None) -> str:
    """A quality, its measurement datum (output of the test) and its value spec."""
    unit_line = f"    obo:IAO_0000039 {unit} ;\n" if unit else ""
    return f"""
ex:{test}_{name} a {cls} .
{bearer} obo:RO_0000086 ex:{test}_{name} .
ex:{test} obo:OBI_0000299 ex:{test}_{name}_datum .
ex:{test}_{name}_datum a obo:IAO_0000109 ;
    obo:IAO_0000136 ex:{test}_{name} ;
    obo:OBI_0001938 ex:{test}_{name}_value .
ex:{test}_{name}_value a obo:OBI_0001931 ;
{unit_line}    obo:OBI_0001927 ex:{test}_{name} ;
    pmdco:PMD_0000006 "{value}"^^xsd:float .
"""


def _test(test: str, piece: str, orientation: str, s0_unit: str = "unit:MilliM2") -> str:
    tp, broken, material = f"ex:{piece}", f"ex:{piece}_after", f"ex:{piece}_material"
    extensometer, load_cell = f"ex:{test}_extensometer", f"ex:{test}_load_cell"
    text = f"""
ex:{test} a pmdco:PMD_0000974 ;
    obo:OBI_0000293 {tp} ;
    obo:OBI_0000299 {broken} .
{tp} a obo:BFO_0000030 ;
    pmdco:PMD_0020005 {material} .
ex:{piece}_id obo:IAO_0000219 {tp} ;
    rdfs:label "Test piece identifier"^^xsd:string ;
    obo:OBI_0001938 ex:{piece}_id_value .
ex:{piece}_id_value pmdco:PMD_0000006 "{piece}"^^xsd:string .
ex:{piece}_material_id obo:IAO_0000219 {material} ;
    obo:OBI_0001938 ex:{piece}_material_id_value .
ex:{piece}_material_id_value pmdco:PMD_0000006 "S355"^^xsd:string .
{material} obo:RO_0000086 ex:{piece}_rolling_direction .
ex:{piece}_rolling_direction a obo:BFO_0000019 ;
    obo:OBI_0001938 ex:{piece}_rolling_direction_value .
ex:{piece}_rolling_direction_value pmdco:PMD_0000006 "{orientation}"^^xsd:string .
ex:{test}_identifier obo:IAO_0000219 ex:{test} ;
    rdfs:label "Process identifier"^^xsd:string ;
    obo:OBI_0001938 ex:{test}_identifier_value .
ex:{test}_identifier_value pmdco:PMD_0000006 "process-{test}"^^xsd:string .
ex:{test}_note obo:IAO_0000219 ex:{test} ;
    rdfs:label "process note"^^xsd:string ;
    obo:OBI_0001938 ex:{test}_note_value .
ex:{test}_note_value pmdco:PMD_0000006 "note of {piece}"^^xsd:string .
"""
    text += _measured(test, "S0", "tto:TTO_0000026", tp, "100.2", s0_unit)
    text += _measured(test, "s1", "tto:TTO_0000012", tp, "100.0", "unit:MilliM2")
    text += _measured(test, "s2", "tto:TTO_0000012", tp, "100.3", "unit:MilliM2")
    text += _measured(test, "s3", "tto:TTO_0000012", tp, "100.3", "unit:MilliM2")
    text += _measured(test, "L0", "tto:TTO_0000028", tp, "60.0", "unit:MilliM")
    text += _measured(test, "ext_gauge", "tto:TTO_0000028", extensometer, "50.0", "unit:MilliM")
    text += _measured(test, "Lu", "tto:TTO_0000018", broken, "80.5", "unit:MilliM")
    text += _measured(test, "Su", "tto:TTO_0000024", broken, "45.0", "unit:MilliM2")
    text += _measured(test, "E", "pmdco:PMD_0000618", material, "205.0", "unit:GigaPa")
    text += _measured(test, "Rp02", "tto:TTO_0000047", material, "NaN", "unit:MegaPa")
    text += _measured(test, "ReH", "tto:TTO_0000059", material, "400.0", "unit:MegaPa")
    text += _measured(test, "ReL", "tto:TTO_0000022", material, "380.0", "unit:MegaPa")
    text += _measured(test, "Rm", "tto:TTO_0000053", material, "510.0", "unit:MegaPa")
    text += _measured(test, "Fm", "tto:TTO_0000023", material, "51.1", "unit:KiloN")
    text += _measured(test, "load_range", "tto:TTO_0000023", load_cell, "100.0", "unit:KiloN")
    text += _measured(test, "A", "tto:TTO_0000033", material, "34.0", "unit:PERCENT")
    text += _measured(test, "Z", "tto:TTO_0000038", material, "55.0", "unit:PERCENT")
    return text


FIXTURE = (
    _PREFIXES
    + _test("test_1", "Za1", "in rolling direction")
    + _test("test_2", "Zb7", "diagonal to rolling direction")
)

CSV = (
    b"Time;Crosshead separation;Load;Extensometer elongation\n"
    b"(sec);(mm);(kN);(%)\n"
    b"0,0;0,00;0,00;0,000\n"
    b"0,5;0,02;2,50;0,050\n"
)


def _source(tmp_path, text: str = FIXTURE) -> dict:
    path = tmp_path / "pmdco3_structural_fixture.ttl"
    path.write_text(text, encoding="utf-8")
    return {"kind": "turtle", "locations": [str(path)]}


@pytest.fixture
def fixture_source(tmp_path) -> dict:
    return _source(tmp_path)


# ---------------------------------------------------------------------------
# Profile and conformance
# ---------------------------------------------------------------------------


def test_profile_is_registered_and_non_semantic_for_curves():
    assert PROFILE in available_profiles()
    profile = get_profile(PROFILE)
    assert not profile.links_curves_semantically
    assert profile.curve_locator is pmdco3_tto.CURVE_LOCATOR
    assert profile.curve_locator.url_for("Zx1") == (
        "https://raw.githubusercontent.com/materialdigital/tensile-test-ontology/"
        "08a7aec359f1072b823362bd67fbb93a88643e6f/tensile_test_data/primary_data/Zx1.csv"
    )


def test_conformance_on_structural_fixture(fixture_source):
    assert exploration.is_conformant(PROFILE, fixture_source)
    assert not exploration.is_conformant("pmdco2-tto", fixture_source)
    run = pwf.node(exploration.check_conformance, "c").run(
        profile_id=PROFILE, source=fixture_source
    )
    assert run.outputs.checked_profile_id == PROFILE


def test_pmdco2_files_do_not_conform_to_pmdco3(pmdco2_source):
    results = exploration.conformance_results(PROFILE, pmdco2_source)
    assert not any(any(answers.values()) for answers in results.values())
    assert not exploration.is_conformant(PROFILE, pmdco2_source)
    with pytest.raises(exploration.ConformanceError, match="pmdco3-tto"):
        pwf.node(exploration.check_conformance, "c").run(profile_id=PROFILE, source=pmdco2_source)


# ---------------------------------------------------------------------------
# Specimens, records, reported values
# ---------------------------------------------------------------------------


def test_list_specimens(fixture_source):
    run = pwf.node(exploration.list_specimens, "s").run(profile_id=PROFILE, source=fixture_source)
    assert run.outputs.specimen_ids == ["Za1", "Zb7"]
    table = exploration.specimen_table(PROFILE, fixture_source)
    assert dict(zip(table["specimen_id"], table["orientation"])) == {
        "Za1": "in rolling direction",
        "Zb7": "diagonal to rolling direction",
    }


def test_specimen_record(fixture_source):
    run = pwf.node(acquisition.get_specimen_record, "r").run(
        profile_id=PROFILE, source=fixture_source, specimen_id="Zb7"
    )
    record = run.outputs.record
    assert record.specimen_id == "Zb7"
    assert record.provenance_iri == "https://example.org/fixture/test_2"
    assert record.orientation == "diagonal to rolling direction"
    assert record.s0_mm2 == pytest.approx(100.2)
    assert record.cross_section_measurements_mm2 == pytest.approx((100.0, 100.3, 100.3))
    assert record.l0_mm == 60.0  # not the extensometer gauge length (same class)
    assert record.lu_mm == 80.5
    assert record.su_mm2 == 45.0
    assert record.notes == ("note of Zb7",)  # not the process identifier
    assert record.curve_url == pmdco3_tto.CURVE_LOCATOR.url_for("Zb7")
    assert {role: (c.header, c.unit, c.index) for role, c in record.curve_columns.items()} == {
        "time": ("Time", "s", None),
        "crosshead": ("Crosshead separation", "mm", None),
        "force": ("Load", "kN", None),
        "strain": ("Extensometer elongation", "%", None),
    }


def test_explicit_curve_locator_overrides_default(fixture_source):
    locator = pmdco3_tto.CURVE_LOCATOR.as_dict()
    locator["url_template"] = "https://curves.example.org/s355/{specimen_id}.csv"
    record = acquisition.build_specimen_record(PROFILE, fixture_source, "Za1", curve_locator=locator)
    assert record.curve_url == "https://curves.example.org/s355/Za1.csv"
    with pytest.raises(ValueError, match="specimen_id"):
        acquisition.build_specimen_record(
            PROFILE, fixture_source, "Za1", curve_locator=dict(locator, url_template="https://x/a.csv")
        )


def test_reported_properties(fixture_source):
    run = pwf.node(acquisition.get_reported_properties, "p").run(
        profile_id=PROFILE, source=fixture_source, specimen_id="Za1"
    )
    reported = run.outputs.reported
    assert reported.e_gpa == 205.0  # unit:GigaPa via declared alias
    assert reported.rp02_mpa is None  # NaN in the graph
    assert (reported.reh_mpa, reported.rel_mpa, reported.rm_mpa) == (400.0, 380.0, 510.0)
    assert reported.fm_n == pytest.approx(51100.0)  # not the load-cell range (same class)
    assert (reported.a_percent, reported.z_percent) == (34.0, 55.0)


def test_unknown_unit_raises(tmp_path):
    text = _PREFIXES + _test("test_1", "Za1", "in rolling direction", s0_unit="unit:FURLONG2")
    with pytest.raises(UnitError, match="FURLONG2"):
        acquisition.build_specimen_record(PROFILE, _source(tmp_path, text), "Za1")


def test_mismatched_unit_raises(tmp_path):
    text = _PREFIXES + _test("test_1", "Za1", "in rolling direction", s0_unit="unit:MilliM")
    with pytest.raises(UnitError, match="'area'"):
        acquisition.build_specimen_record(PROFILE, _source(tmp_path, text), "Za1")


def test_fetch_curve_by_header_with_stubbed_fetch(monkeypatch, fixture_source):
    record = acquisition.build_specimen_record(PROFILE, fixture_source, "Za1")
    monkeypatch.setattr(sources, "fetch_bytes", lambda url: (CSV, hashlib.sha256(CSV).hexdigest()))
    curve = pwf.node(acquisition.fetch_curve, "f").run(record=record).outputs.curve
    np.testing.assert_allclose(curve.force_n, [0.0, 2500.0])
    np.testing.assert_allclose(curve.strain, [0.0, 0.0005])
    assert curve.source_url == record.curve_url
    assert curve.sha256 == hashlib.sha256(CSV).hexdigest()


def test_iri_label_table_covers_every_iri_used():
    profile = get_profile(PROFILE)
    texts = [profile.specimens.query, profile.quantities.query, profile.notes.query]
    texts += list(profile.conformance_asks.values())
    used = set(re.findall(r"\b(?:pmdco|obo|tto):[A-Za-z]+_[0-9]+\b", " ".join(texts)))
    listed = {iri for iri, _label, _note in pmdco3_tto.IRI_LABELS}
    assert used == listed


# ---------------------------------------------------------------------------
# Real pinned graph (network)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_real_pinned_graph_matches_pmdco2(pmdco2_source):
    source = {"kind": "turtle", "locations": [pmdco3_tto.PINNED_GRAPH_URL]}
    assert exploration.is_conformant(PROFILE, source)
    ids = list(exploration.specimen_table(PROFILE, source)["specimen_id"])
    assert sorted(ids) == sorted(exploration.specimen_table("pmdco2-tto", pmdco2_source)["specimen_id"])
    assert len(ids) == 10

    compared = ["orientation", "s0_mm2", "cross_section_measurements_mm2", "l0_mm", "lu_mm", "su_mm2"]
    for sid in ids:
        r3 = acquisition.build_specimen_record(PROFILE, source, sid)
        r2 = acquisition.build_specimen_record("pmdco2-tto", pmdco2_source, sid)
        for field in compared:
            assert getattr(r3, field) == getattr(r2, field), (sid, field)
        q3 = acquisition.build_reported_properties(PROFILE, source, sid)
        q2 = acquisition.build_reported_properties("pmdco2-tto", pmdco2_source, sid)
        assert q3 == q2, sid
