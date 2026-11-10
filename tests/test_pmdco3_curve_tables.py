"""The pmdco3-tto profile reads the curve tables of the PMDco3 slices (decision D8.3), offline (C7).

The graphs are the committed PMDco3 files: the three orientation slices and the
full file (each test with one CSVW curve table, decision D8.6), and the pinned
original without curve tables. Curve downloads are stubbed with synthetic CSV
bytes served by file name (``workflow_helpers``), so pmdco2 and pmdco3 runs
analyse identical bytes. The slicing itself is tested in
``test_pmdco3_slices.py``.
"""

from __future__ import annotations

import dataclasses

import pytest
import rdflib
from conftest import PMDCO3_FULL_FILE, PMDCO3_ORIGINAL_FILE, PMDCO3_SLICE_FILES
from workflow_helpers import (
    PMDCO2,
    PMDCO2_SOURCE,
    PMDCO3,
    curve_bytes_for,
    run_with_stubbed_curves,
)

from pmd_tensile import profiles, sources
from pmd_tensile.nodes import acquisition
from pmd_tensile.profiles import get_profile, pmdco3_tto
from pmd_tensile.profiles.base import (
    CURVE_COLUMN_COLUMNS,
    CURVE_DIALECT_COLUMNS,
    SelectQuery,
)

DATA_HOST = "https://data.pmds-mpcdf-2.pyiron.com/"
TTO_RAW = "https://raw.githubusercontent.com/materialdigital/tensile-test-ontology/"
LOCATOR = pmdco3_tto.CURVE_LOCATOR.as_dict()


def turtle(*paths) -> dict:
    return {"kind": "turtle", "locations": [str(path) for path in paths]}


SLICES = turtle(*PMDCO3_SLICE_FILES.values())
FULL = turtle(PMDCO3_FULL_FILE)
ORIGINAL = turtle(PMDCO3_ORIGINAL_FILE)
SLICE_IDS = {
    "parallel": ["Zx1", "Zx2", "Zx3", "Zx4"],
    "perpendicular": ["Zy1", "Zy2", "Zy3", "Zy4"],
    "diagonal": ["Zd2", "Zd3"],
}
ALL_IDS = [sid for ids in SLICE_IDS.values() for sid in ids]
RUN_IDS = ("Zx1", "Zx2", "Zy1", "Zd2")  # the specimens with synthetic curves, from all three slices
EXPECTED_COLUMNS = {"time": (0, "s"), "crosshead": (1, "mm"), "force": (2, "kN"), "strain": (3, "%")}
TABLES = ("results", "orientation_summary", "comparison")

# semantikon's data hashing calls networkx, which warns that its graph hashes
# changed in networkx 3.5; an upstream notice, not a test concern.
pytestmark = pytest.mark.filterwarnings("ignore:The hashes produced for directed graphs:UserWarning")


def cell_texts(table) -> dict[str, list[str]]:
    return {column: [repr(value) for value in table[column]] for column in table.columns}


# ---------------------------------------------------------------------------
# Profile, conformance and specimens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", [*PMDCO3_SLICE_FILES.values(), PMDCO3_FULL_FILE, PMDCO3_ORIGINAL_FILE], ids=lambda p: p.stem
)
def test_conformance_asks_pass(path):
    results = profiles.conformance_results(PMDCO3, turtle(path))
    assert results and all(all(answers.values()) for answers in results.values()), results
    assert not profiles.is_conformant(PMDCO2, turtle(path))


def test_slices_list_their_specimens():
    for orientation, path in PMDCO3_SLICE_FILES.items():
        assert sorted(profiles.specimen_table(PMDCO3, turtle(path))["specimen_id"]) == SLICE_IDS[orientation]
    assert sorted(profiles.specimen_table(PMDCO3, FULL)["specimen_id"]) == sorted(ALL_IDS)


def test_profile_contract_for_curve_columns_and_locators():
    profile = get_profile(PMDCO3)
    assert tuple(profile.curve_columns.columns) == CURVE_COLUMN_COLUMNS + CURVE_DIALECT_COLUMNS
    # Without the dialect columns is still valid (pmdco2-tto), any other binding is not.
    assert tuple(get_profile(PMDCO2).curve_columns.columns) == CURVE_COLUMN_COLUMNS
    with pytest.raises(ValueError, match="curve_columns query must bind"):
        dataclasses.replace(profile, curve_columns=SelectQuery(profile.curve_columns.query, ("process", "url")))
    # Accepting an explicit locator only makes sense for a profile that reads curve tables.
    with pytest.raises(ValueError, match="accepts_curve_locator applies only"):
        dataclasses.replace(profile, curve_columns=None, curve_locator=pmdco3_tto.CURVE_LOCATOR)
    assert not get_profile(PMDCO2).accepts_curve_locator


def test_curve_columns_query_reads_roles_numbers_units_and_dialect():
    profile = get_profile(PMDCO3)
    rows = sources.select(turtle(PMDCO3_SLICE_FILES["diagonal"]), profile.curve_columns.query, profile.curve_columns.columns)
    assert len(rows) == 2 * 4  # Zd2 and Zd3, four roles each; the lab table is not a time series
    columns = {
        role: (int(float(number)), profile.unit_symbol(unit))
        for role, number, unit in rows[["role", "column_number", "unit"]].itertuples(index=False)
    }
    assert {role: (index + 1, unit) for role, (index, unit) in EXPECTED_COLUMNS.items()} == columns
    assert set(rows["unit"]) == {
        "http://qudt.org/vocab/unit/" + name for name in ("SEC", "MilliM", "KiloN", "PERCENT")
    }
    assert set(rows["delimiter"]) == {";"} and set(rows["decimal_char"]) == {","}
    assert {int(float(value)) for value in rows["header_row_count"]} == {2}
    assert set(rows["url"]) == {f"{DATA_HOST}Zd2.csv", f"{DATA_HOST}Zd3.csv"}


# ---------------------------------------------------------------------------
# Records from the curve tables, without a locator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", [SLICES, FULL], ids=["slices", "full"])
def test_records_take_the_curves_from_the_graph(source, pmdco2_source):
    for sid in ALL_IDS:
        r3 = acquisition.get_specimen_record(PMDCO3, source, sid)
        assert r3.curve_url == f"{DATA_HOST}{sid}.csv"
        assert {role: (c.index, c.unit) for role, c in r3.curve_columns.items()} == EXPECTED_COLUMNS
        # Everything the analysis sees equals the pmdco2 record; only the provenance IRI differs.
        r2 = acquisition.get_specimen_record(PMDCO2, pmdco2_source, sid)
        assert dataclasses.replace(r3, provenance_iri=r2.provenance_iri) == r2, sid
        q3 = acquisition.get_reported_properties(PMDCO3, source, sid)
        assert q3 == acquisition.get_reported_properties(PMDCO2, pmdco2_source, sid), sid


def test_an_explicit_locator_replaces_the_curve_tables():
    record = acquisition.get_specimen_record(PMDCO3, SLICES, "Zx3", curve_locator=LOCATOR)
    assert record.curve_url == pmdco3_tto.CURVE_LOCATOR.url_for("Zx3")
    assert {role: c.header for role, c in record.curve_columns.items()} == {
        role: spec["header"] for role, spec in LOCATOR["columns"].items()
    }
    with pytest.raises(ValueError, match="not accepted"):  # pmdco2 still rejects a locator
        acquisition.get_specimen_record(PMDCO2, PMDCO2_SOURCE, "Zx3", curve_locator=LOCATOR)


CSVW = rdflib.Namespace("http://www.w3.org/ns/csvw#")


@pytest.mark.parametrize(
    ("predicate", "value", "message"),
    [
        (CSVW.delimiter, rdflib.Literal(","), r"declares delimiter \[','\].*';'"),
        (CSVW.headerRowCount, rdflib.Literal(1), r"declares header_row_count \['1'\].*'2'"),
        (CSVW.decimalChar, rdflib.Literal("."), r"declares decimal_char \['.'\].*','"),
    ],
)
def test_a_declared_dialect_the_parser_cannot_read_is_rejected(tmp_path, predicate, value, message):
    graph = rdflib.Graph().parse(PMDCO3_SLICE_FILES["diagonal"])
    subjects = set(graph.subjects(predicate, None))
    assert subjects  # the slice declares it
    for subject in subjects:
        graph.set((subject, predicate, value))
    path = tmp_path / "edited_diagonal.ttl"
    graph.serialize(destination=path, format="turtle")
    with pytest.raises(ValueError, match=message):
        acquisition.get_specimen_record(PMDCO3, turtle(path), "Zd3")


# ---------------------------------------------------------------------------
# The original graph: explicit locator only
# ---------------------------------------------------------------------------


def test_the_original_graph_needs_the_explicit_locator():
    with pytest.raises(LookupError, match="links no curve table for specimen 'Zx3'.*curve_locator"):
        acquisition.get_specimen_record(PMDCO3, ORIGINAL, "Zx3")
    with pytest.raises(ExceptionGroup) as excinfo:  # a batch reports every failed specimen
        run_with_stubbed_curves(curve_bytes_for(["Zx1", "Zy1"]), PMDCO3, ORIGINAL, specimen_ids=["Zx1", "Zy1"])
    assert excinfo.group_contains(LookupError, match="links no curve table for specimen 'Zx1'")
    record = acquisition.get_specimen_record(PMDCO3, ORIGINAL, "Zx3", curve_locator=LOCATOR)
    assert record.curve_url.startswith(TTO_RAW)


# ---------------------------------------------------------------------------
# Analysis: pmdco3 equals pmdco2 on identical curve bytes (acceptance criterion 6)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def curves() -> dict:
    return curve_bytes_for(RUN_IDS)


@pytest.fixture(scope="module")
def pmdco2_run(curves):
    return run_with_stubbed_curves(curves, PMDCO2, PMDCO2_SOURCE, specimen_ids=RUN_IDS)


@pytest.mark.parametrize("source", [SLICES, FULL], ids=["slices", "full"])
def test_slices_give_the_pmdco2_results_without_a_locator(source, curves, pmdco2_run):
    run2, fetched2 = pmdco2_run
    run3, fetched3 = run_with_stubbed_curves(curves, PMDCO3, source, specimen_ids=RUN_IDS)
    assert sorted(fetched3) == sorted(fetched2) == sorted(f"{DATA_HOST}{sid}.csv" for sid in RUN_IDS)
    assert run3.batch_workflow.recipe == run2.batch_workflow.recipe  # only profile ID and source differ
    for name in TABLES:  # every cell, including the curve URL and hash
        assert cell_texts(getattr(run3, name)) == cell_texts(getattr(run2, name)), name
    assert {sid: c.sha256 for sid, c in run3.curves.items()} == {sid: c.sha256 for sid, c in run2.curves.items()}


def test_original_with_the_locator_gives_the_pmdco2_results(curves, pmdco2_run):
    run2, _ = pmdco2_run
    run3, fetched3 = run_with_stubbed_curves(curves, PMDCO3, ORIGINAL, curve_locator=LOCATOR, specimen_ids=RUN_IDS)
    assert all(url.startswith(TTO_RAW) for url in fetched3) and len(fetched3) == len(RUN_IDS)
    for name in TABLES:
        old, new = cell_texts(getattr(run2, name)), cell_texts(getattr(run3, name))
        assert {column for column in old if old[column] != new[column]} <= {"curve_url"}, name
