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
from workflow_helpers import PMDCO3_PREFIXES, pmdco3_test

from tensile_workflow import acquisition, exploration, sources
from tensile_workflow.contract import UnitError, to_contract_unit
from tensile_workflow.profiles import available_profiles, get_profile, pmdco3_tto
from tensile_workflow.profiles.base import QUANTITY_KINDS

PROFILE = "pmdco3-tto"

FIXTURE = (
    PMDCO3_PREFIXES
    + pmdco3_test("test_1", "Za1", "in rolling direction", switch="1.6")
    + pmdco3_test("test_2", "Zb7", "diagonal to rolling direction", switch="4.0")
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


def _rate_switch_strain(profile_id: str, source: dict, specimen_id: str) -> float | None:
    """The profile's ``eps_sw`` value of one specimen as a ratio (``None`` if absent).

    ``acquisition.build_specimen_record`` does not fill
    ``SpecimenRecord.rate_switch_strain`` yet, so this checks the profile's
    query and its unit conversion directly.
    """
    profile = get_profile(profile_id)
    table = exploration.specimen_table(profile_id, source)
    process = table[table["specimen_id"] == specimen_id]["process"].item()
    rows = sources.select(source, profile.quantities.query, profile.quantities.columns)
    rows = rows[(rows["process"] == process) & (rows["quantity"] == "eps_sw")]
    if rows.empty:
        return None
    assert len(rows) == 1
    row = rows.iloc[0]
    return to_contract_unit(float(row["value"]), profile.unit_symbol(row["unit"]), QUANTITY_KINDS["eps_sw"])


def test_rate_switch_strain_is_a_ratio(fixture_source):
    assert _rate_switch_strain(PROFILE, fixture_source, "Za1") == pytest.approx(0.016, rel=1e-12)
    assert _rate_switch_strain(PROFILE, fixture_source, "Zb7") == pytest.approx(0.040, rel=1e-12)
    # Acquisition still succeeds with the extra quantity in the graph.
    assert acquisition.build_specimen_record(PROFILE, fixture_source, "Za1").specimen_id == "Za1"


def test_unknown_unit_raises(tmp_path):
    text = PMDCO3_PREFIXES + pmdco3_test("test_1", "Za1", "in rolling direction", s0_unit="unit:FURLONG2")
    with pytest.raises(UnitError, match="FURLONG2"):
        acquisition.build_specimen_record(PROFILE, _source(tmp_path, text), "Za1")


def test_mismatched_unit_raises(tmp_path):
    text = PMDCO3_PREFIXES + pmdco3_test("test_1", "Za1", "in rolling direction", s0_unit="unit:MilliM")
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
        switch3 = _rate_switch_strain(PROFILE, source, sid)
        assert switch3 is not None, sid
        assert switch3 == _rate_switch_strain("pmdco2-tto", pmdco2_source, sid), sid
