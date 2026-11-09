"""Offline checks of the PMDco3 slices written by ``scripts/slice_pmdco3.py``.

The input is the pinned original TTO graph ``datasets/S355_data_tto_pmdco3.ttl``.
The script adds one curve table per test, relinks the lab table, and writes the
full file ``datasets/pmdco3_tto_example.ttl`` and the slices
``datasets/pmdco3_tto_example_{parallel,perpendicular,diagonal}.ttl``.

The checks follow decision D8.6 of the tensile-notebooks plan (modelling
approved at gate G1) and acceptance criterion 4. They use rdflib only: the
script runs in a subprocess (its write helper is also imported directly),
and nothing is read from the network. Expected
values (tests, orientations, counts, the curve-table Turtle) are written out
here instead of being derived with the script's own rules.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from rdflib import RDF, BNode, Graph, Literal, URIRef
from rdflib.compare import isomorphic
from rdflib.namespace import CSVW, DC, RDFS

DEMO_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = DEMO_DIR / "datasets"
SCRIPT = DEMO_DIR / "scripts" / "slice_pmdco3.py"
ORIGINAL = DATASETS_DIR / "S355_data_tto_pmdco3.ttl"
ORIGINAL_SHA256 = "b67d95710ab607bc0c93ec3df83f072a20bc07ac58adc020a2c1b666ddc8f25d"

SLICES = ("parallel", "perpendicular", "diagonal")
FILE_NAMES = {"full": "pmdco3_tto_example.ttl"}
FILE_NAMES.update({name: f"pmdco3_tto_example_{name}.ttl" for name in SLICES})

ABOX = "https://w3id.org/pmd/demodata/tensiletest_S355/"
UNIT = "http://qudt.org/vocab/unit/"
# Plain IRIs: rdflib's XSD namespace object makes pytest's collection warn.
XSD_STRING = URIRef("http://www.w3.org/2001/XMLSchema#string")
XSD_ANY_URI = URIRef("http://www.w3.org/2001/XMLSchema#anyURI")
HEADER = URIRef(ABOX)
LAB_TABLE = URIRef(ABOX + "S355_data_csvw_table")
OLD_LAB_URL = Literal(
    "https://raw.githubusercontent.com/materialdigital/tensile-test-ontology/"
    "refs/heads/main/tensile_test_data/S355_data.csv",
    datatype=XSD_ANY_URI,
)
NEW_LAB_URL = Literal("https://data.pmds-mpcdf-2.pyiron.com/S355_data.csv", datatype=XSD_ANY_URI)
LAB_COMMENT = Literal(
    "csvw:url relinked to the PMD data host by the PMD demonstrator "
    "(materialdigital/pmd-demonstrator, scripts/slice_pmdco3.py); file unchanged "
    "from the TTO repository at commit 08a7aec.",
    datatype=XSD_STRING,
)
ORIGINAL_TITLE = "TTO/PMDco 3.0 A-Box mapping example of S355 steel tensile test data"
ORIENTATION_TEXT = {
    "parallel": "in rolling direction",
    "perpendicular": "perpendicular to rolling direction",
    "diagonal": "diagonal to rolling direction",
}

# Specimen id -> (test number in the process IRIs, slice).
TESTS = {
    "Zx1": (1, "parallel"),
    "Zx2": (2, "parallel"),
    "Zx3": (3, "parallel"),
    "Zx4": (4, "parallel"),
    "Zy1": (5, "perpendicular"),
    "Zy2": (6, "perpendicular"),
    "Zy3": (7, "perpendicular"),
    "Zy4": (8, "perpendicular"),
    "Zd2": (9, "diagonal"),
    "Zd3": (10, "diagonal"),
}

# Triple counts (T3 proposal, section 6; D8.6).
ORIGINAL_TRIPLES = 6876
SHARED_ORIGINAL_TRIPLES = 246  # header 4, lab table 5, schema 58, 57 columns x 3, units 8
PER_TEST_ORIGINAL_TRIPLES = 663
ADDED_PER_TEST = 63
SHARED_TRIPLES = SHARED_ORIGINAL_TRIPLES + 1  # + the lab-table comment
EXPECTED_COUNTS = {
    "full": SHARED_TRIPLES + len(TESTS) * (PER_TEST_ORIGINAL_TRIPLES + ADDED_PER_TEST),
    "parallel": SHARED_TRIPLES + 4 * (PER_TEST_ORIGINAL_TRIPLES + ADDED_PER_TEST),
    "perpendicular": SHARED_TRIPLES + 4 * (PER_TEST_ORIGINAL_TRIPLES + ADDED_PER_TEST),
    "diagonal": SHARED_TRIPLES + 2 * (PER_TEST_ORIGINAL_TRIPLES + ADDED_PER_TEST),
}

PREFIXES = """\
@prefix abox: <https://w3id.org/pmd/demodata/tensiletest_S355/> .
@prefix csvw: <http://www.w3.org/ns/csvw#> .
@prefix obo: <http://purl.obolibrary.org/obo/> .
@prefix pmdco: <https://w3id.org/pmd/co/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix tto: <https://w3id.org/pmd/tto/> .
@prefix unit: <http://qudt.org/vocab/unit/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""

# The approved Turtle for Zx3 (T3 proposal, section 3), with <…> written out.
ZX3_CURVE_TABLE = """\
abox:pmdao-tto-tt-S355-3_process obo:OBI_0000299 abox:Zx3_csvw_table .

abox:Zx3_csvw_table a csvw:Table, pmdco:PMD_0000992 ;
    rdfs:label "Zx3 test record"^^xsd:string ;
    rdfs:comment "Added by the PMD demonstrator (materialdigital/pmd-demonstrator, scripts/slice_pmdco3.py); not part of the TTO graph at commit 08a7aec."^^xsd:string ;
    csvw:url "https://data.pmds-mpcdf-2.pyiron.com/Zx3.csv"^^xsd:anyURI ;
    csvw:dialect abox:Zx3_csvw_dialect ;
    csvw:tableSchema abox:Zx3_csvw_schema .
abox:Zx3_csvw_dialect a csvw:Dialect ; csvw:delimiter ";"^^xsd:string ;
    csvw:headerRowCount "2"^^xsd:nonNegativeInteger ; csvw:encoding "utf-8"^^xsd:string .
abox:Zx3_csvw_datatype a csvw:Datatype ; csvw:base "decimal"^^xsd:string ; csvw:format abox:Zx3_csvw_numeric_format .
abox:Zx3_csvw_numeric_format a csvw:NumericFormat ; csvw:decimalChar ","^^xsd:string .
abox:Zx3_csvw_schema a csvw:Schema ; csvw:column ( <https://w3id.org/pmd/demodata/tensiletest_S355/Zx3_csvw_table/col/Time> <https://w3id.org/pmd/demodata/tensiletest_S355/Zx3_csvw_table/col/Crosshead_separation> <https://w3id.org/pmd/demodata/tensiletest_S355/Zx3_csvw_table/col/Load> <https://w3id.org/pmd/demodata/tensiletest_S355/Zx3_csvw_table/col/Extensometer_elongation> ) .

<https://w3id.org/pmd/demodata/tensiletest_S355/Zx3_csvw_table/col/Time> a csvw:Column, obo:IAO_0000109 ; csvw:name "Time"^^xsd:string ; csvw:title "Time"^^xsd:string ;
    csvw:datatype abox:Zx3_csvw_datatype ; obo:IAO_0000039 unit:SEC ; obo:IAO_0000136 abox:pmdao-tto-tt-S355-3_process_testing_interval .
<https://w3id.org/pmd/demodata/tensiletest_S355/Zx3_csvw_table/col/Crosshead_separation> a csvw:Column, obo:IAO_0000109 ; csvw:name "Crosshead_separation"^^xsd:string ; csvw:title "Crosshead separation"^^xsd:string ;
    csvw:datatype abox:Zx3_csvw_datatype ; obo:IAO_0000039 unit:MilliM ; obo:IAO_0000136 abox:pmdao-tto-tt-S355-3_crosshead_separation .
<https://w3id.org/pmd/demodata/tensiletest_S355/Zx3_csvw_table/col/Load> a csvw:Column, obo:IAO_0000109 ; csvw:name "Load"^^xsd:string ; csvw:title "Load"^^xsd:string ;
    csvw:datatype abox:Zx3_csvw_datatype ; obo:IAO_0000039 unit:KiloN ; obo:IAO_0000136 abox:pmdao-tto-tt-S355-3_force .
<https://w3id.org/pmd/demodata/tensiletest_S355/Zx3_csvw_table/col/Extensometer_elongation> a csvw:Column, obo:IAO_0000109 ; csvw:name "Extensometer_elongation"^^xsd:string ; csvw:title "Extensometer elongation"^^xsd:string ;
    csvw:datatype abox:Zx3_csvw_datatype ; obo:IAO_0000039 unit:PERCENT ; obo:IAO_0000136 abox:pmdao-tto-tt-S355-3_percentage_extension .

abox:pmdao-tto-tt-S355-3_process_testing obo:BFO_0000199 abox:pmdao-tto-tt-S355-3_process_testing_interval .
abox:pmdao-tto-tt-S355-3_process_testing_interval a obo:BFO_0000202 .
abox:pmdao-tto-tt-S355-3_tensile_testing_machine obo:RO_0000086 abox:pmdao-tto-tt-S355-3_crosshead_separation .
abox:pmdao-tto-tt-S355-3_crosshead_separation a tto:TTO_0000013 .
<https://w3id.org/pmd/demodata/tensiletest_S355/testpiece/Zx3> obo:RO_0000053 abox:pmdao-tto-tt-S355-3_force, abox:pmdao-tto-tt-S355-3_percentage_extension .
abox:pmdao-tto-tt-S355-3_force a pmdco:PMD_0020200 .
abox:pmdao-tto-tt-S355-3_percentage_extension a pmdco:PMD_0060011 .
"""

# Curve-table columns per test, read like the PMDco2 profile's CURVE_COLUMNS:
# position from the csvw:column RDF list, quantity class of the entity the
# column is about, unit, and the dialect and number format.
CURVE_COLUMNS_QUERY = """
PREFIX csvw:  <http://www.w3.org/ns/csvw#>
PREFIX obo:   <http://purl.obolibrary.org/obo/>
PREFIX pmdco: <https://w3id.org/pmd/co/>
PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX tto:   <https://w3id.org/pmd/tto/>

SELECT ?process ?url ?role ?column_number ?unit ?delimiter ?header_row_count ?decimal_char
WHERE {
  VALUES (?class ?role) {
    (obo:BFO_0000202   "time")
    (tto:TTO_0000013   "crosshead")
    (pmdco:PMD_0020200 "force")
    (pmdco:PMD_0060011 "strain")
  }
  {
    # Position of each column in the csvw:column RDF list (1-based).
    SELECT ?process ?table ?url ?column (COUNT(DISTINCT ?position) AS ?column_number)
    WHERE {
      ?process a pmdco:PMD_0000974 ;           # tensile testing process
               obo:OBI_0000299 ?table .        # has specified output
      ?table a csvw:Table, pmdco:PMD_0000992 ; # time series
             csvw:url ?url ;
             csvw:tableSchema ?schema .
      ?schema csvw:column/rdf:rest* ?position .
      ?position rdf:rest* ?cell .
      ?cell rdf:first ?column .
    }
    GROUP BY ?process ?table ?url ?column
  }
  ?column obo:IAO_0000136 ?entity .            # is about
  ?entity a ?class .
  OPTIONAL { ?column obo:IAO_0000039 ?unit }   # has measurement unit label
  OPTIONAL {
    ?table csvw:dialect ?dialect .
    OPTIONAL { ?dialect csvw:delimiter ?delimiter }
    OPTIONAL { ?dialect csvw:headerRowCount ?header_row_count }
  }
  OPTIONAL { ?column csvw:datatype/csvw:format/csvw:decimalChar ?decimal_char }
}
"""

EXPECTED_COLUMNS = {
    ("time", 1, URIRef(UNIT + "SEC")),
    ("crosshead", 2, URIRef(UNIT + "MilliM")),
    ("force", 3, URIRef(UNIT + "KiloN")),
    ("strain", 4, URIRef(UNIT + "PERCENT")),
}


def process_iri(specimen_id: str) -> URIRef:
    return URIRef(f"{ABOX}pmdao-tto-tt-S355-{TESTS[specimen_id][0]}_process")


def curve_table_turtle(specimen_id: str) -> str:
    """The approved Zx3 Turtle, renamed for ``specimen_id``."""
    number = TESTS[specimen_id][0]
    return ZX3_CURVE_TABLE.replace("tt-S355-3_", f"tt-S355-{number}_").replace(
        "Zx3", specimen_id
    )


def curve_table(specimen_id: str) -> Graph:
    return Graph().parse(data=PREFIXES + curve_table_turtle(specimen_id), format="turtle")


def own_iri_pattern(specimen_id: str) -> re.Pattern:
    """IRIs of the nodes that belong to one test, by the graph's naming."""
    number = TESTS[specimen_id][0]
    return re.compile(
        rf"^{re.escape(ABOX)}(pmdao-tto-tt-S355-{number}_"
        rf"|testpiece/{specimen_id}(_|$)"
        rf"|{specimen_id}_csvw_)"
    )


def owner(node) -> str | None:
    """The specimen id whose naming pattern ``node`` matches, if exactly one."""
    owners = [sid for sid in TESTS if own_iri_pattern(sid).match(str(node))]
    assert len(owners) <= 1, (node, owners)
    return owners[0] if owners else None


def slice_title(name: str) -> Literal:
    return Literal(f"{ORIGINAL_TITLE} ({ORIENTATION_TEXT[name]})", datatype=XSD_STRING)


def iri_nodes(graph: Graph) -> set:
    nodes = set()
    for triple in graph:
        nodes.update(term for term in triple[::2] if isinstance(term, URIRef))
    return nodes


@pytest.fixture(scope="module")
def original() -> Graph:
    return Graph().parse(ORIGINAL, format="turtle")


@pytest.fixture(scope="module")
def committed() -> dict[str, Graph]:
    return {key: Graph().parse(DATASETS_DIR / name) for key, name in FILE_NAMES.items()}


@pytest.fixture(scope="module")
def fresh_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("pmdco3_slices")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return out


@pytest.fixture(scope="module")
def shared_subjects(original) -> set:
    """Header, lab table, its schema and 57 columns, and the unit IRIs."""
    schema = original.value(LAB_TABLE, CSVW.tableSchema)
    columns = set(original.objects(schema, CSVW.column))
    assert len(columns) == 57
    units = {s for s in original.subjects() if str(s).startswith(UNIT)}
    return {HEADER, LAB_TABLE, schema} | columns | units


# (a) pinned input


def test_original_is_the_pinned_file():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == ORIGINAL_SHA256


def test_original_counts(original, shared_subjects):
    assert len(original) == ORIGINAL_TRIPLES
    shared = [t for t in original if t[0] in shared_subjects]
    assert len(shared) == SHARED_ORIGINAL_TRIPLES
    for specimen_id in TESTS:
        pattern = own_iri_pattern(specimen_id)
        own =[t for t in original if pattern.match(str(t[0]))]
        assert len(own) == PER_TEST_ORIGINAL_TRIPLES, specimen_id


# (b) fresh run


@pytest.mark.parametrize("key", list(FILE_NAMES))
def test_fresh_run_is_isomorphic_to_committed_file(key, fresh_dir, committed):
    fresh = Graph().parse(fresh_dir / FILE_NAMES[key])
    assert isomorphic(fresh, committed[key])


def test_script_refuses_a_modified_input(tmp_path):
    modified = tmp_path / "input.ttl"
    modified.write_bytes(ORIGINAL.read_bytes() + b"# modified\n")
    out = tmp_path / "out"
    out.mkdir()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(modified), "--output-dir", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "SHA-256" in result.stderr
    assert list(out.iterdir()) == []


def test_script_refuses_to_overwrite_its_input(tmp_path):
    pinned_copy = tmp_path / FILE_NAMES["full"]
    shutil.copyfile(ORIGINAL, pinned_copy)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(pinned_copy), "--output-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "overwrite the input" in result.stderr
    assert hashlib.sha256(pinned_copy.read_bytes()).hexdigest() == ORIGINAL_SHA256
    assert sorted(path.name for path in tmp_path.iterdir()) == [FILE_NAMES["full"]]


def test_script_refuses_to_overwrite_its_input_under_another_name(tmp_path):
    """The same file under an output name (here a hard link; on a case-insensitive file system also a
    name in another letter case) is refused, although its resolved path differs from the input's."""
    pinned_copy = tmp_path / "input.ttl"
    shutil.copyfile(ORIGINAL, pinned_copy)
    out = tmp_path / "out"
    out.mkdir()
    alias = out / FILE_NAMES["perpendicular"]
    try:
        os.link(pinned_copy, alias)
    except OSError as error:
        pytest.skip(f"no hard links here: {error}")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(pinned_copy), "--output-dir", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "overwrite the input" in result.stderr
    assert hashlib.sha256(pinned_copy.read_bytes()).hexdigest() == ORIGINAL_SHA256
    assert os.path.samefile(alias, pinned_copy)
    assert sorted(path.name for path in out.iterdir()) == [alias.name]


def _script_module():
    spec = importlib.util.spec_from_file_location("slice_pmdco3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # the script's dataclasses look their module up while it executes
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


class _FailingGraph:
    def serialize(self, format):
        raise RuntimeError("serialisation failed")


def test_writes_replace_no_file_before_every_graph_is_written(tmp_path):
    script = _script_module()
    targets = [tmp_path / name for name in FILE_NAMES.values()]
    for target in targets:
        target.write_text(f"old {target.name}\n", encoding="utf-8")
    graph = Graph()
    graph.add((URIRef(ABOX + "a"), RDFS.label, Literal("new")))
    # The failure comes last, after three graphs have been written to temporary files.
    graphs = dict(zip(targets, [graph, graph, graph, _FailingGraph()]))
    with pytest.raises(RuntimeError, match="serialisation failed"):
        script.write_turtle_files(graphs)
    assert [target.read_text(encoding="utf-8") for target in targets] == [
        f"old {target.name}\n" for target in targets
    ]
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(FILE_NAMES.values())

    script.write_turtle_files(dict.fromkeys(targets, graph))
    for target in targets:
        assert isomorphic(Graph().parse(target, format="turtle"), graph)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(FILE_NAMES.values())


# (c) full file composition


def test_curve_table_has_63_triples():
    for specimen_id in TESTS:
        assert len(curve_table(specimen_id)) == ADDED_PER_TEST, specimen_id


def test_full_file_is_original_plus_relink_and_curve_tables(original, committed):
    expected = Graph()
    for triple in original:
        expected.add(triple)
    expected.remove((LAB_TABLE, CSVW.url, OLD_LAB_URL))
    expected.add((LAB_TABLE, CSVW.url, NEW_LAB_URL))
    expected.add((LAB_TABLE, RDFS.comment, LAB_COMMENT))
    for specimen_id in TESTS:
        for triple in curve_table(specimen_id):
            expected.add(triple)
    assert len(expected) == EXPECTED_COUNTS["full"]
    assert isomorphic(expected, committed["full"])
    title = Literal(ORIGINAL_TITLE, datatype=XSD_STRING)
    assert set(committed["full"].objects(HEADER, DC.title)) == {title}


# (d) union of the slices


def test_union_of_slices_equals_full_file_with_original_title(committed):
    union = Graph()
    for name in SLICES:
        assert set(committed[name].objects(HEADER, DC.title)) == {slice_title(name)}
        for triple in committed[name]:
            union.add(triple)
        union.remove((HEADER, DC.title, slice_title(name)))
    union.add((HEADER, DC.title, Literal(ORIGINAL_TITLE, datatype=XSD_STRING)))
    assert isomorphic(union, committed["full"])


# (e) one slice per test, matching the orientation literal


def test_orientation_literals_match_expected_slices(original):
    query = """
    PREFIX obo:   <http://purl.obolibrary.org/obo/>
    PREFIX pmdco: <https://w3id.org/pmd/co/>
    SELECT ?specimen_id ?orientation WHERE {
      ?process a pmdco:PMD_0000974 ; obo:OBI_0000293 ?test_piece .
      ?identifier obo:IAO_0000219 ?test_piece ; obo:OBI_0001938 ?id_value .
      ?id_value pmdco:PMD_0000006 ?specimen_id .
      ?test_piece pmdco:PMD_0020005 ?material .
      ?material obo:RO_0000086 ?quality .
      ?quality a obo:BFO_0000019 ; obo:OBI_0001938 ?value_spec .
      ?value_spec pmdco:PMD_0000006 ?orientation .
    }
    """
    found = {str(row.specimen_id): str(row.orientation) for row in original.query(query)}
    assert found == {sid: ORIENTATION_TEXT[name] for sid, (_, name) in TESTS.items()}


@pytest.mark.parametrize("specimen_id", list(TESTS))
def test_each_test_is_in_exactly_its_slice(specimen_id, committed):
    expected_slice = TESTS[specimen_id][1]
    pattern = own_iri_pattern(specimen_id)
    full =committed["full"]
    own_triples = {t for t in full if isinstance(t[0], URIRef) and pattern.match(str(t[0]))}
    own_nodes = {t[0] for t in own_triples}
    assert process_iri(specimen_id) in own_nodes
    # Blank nodes (the column lists) differ between parsed files; they are
    # checked in test_every_non_shared_subject_belongs_to_one_test and by
    # isomorphism of the union.
    own_iri_triples = {t for t in own_triples if not isinstance(t[2], BNode)}
    for name in SLICES:
        graph = committed[name]
        if name == expected_slice:
            assert own_iri_triples <= set(graph)
            assert len(own_triples - own_iri_triples) == 1  # schema csvw:column ( ... )
            schema = URIRef(f"{ABOX}{specimen_id}_csvw_schema")
            assert len(list(graph.objects(schema, CSVW.column))) == 1
        else:
            assert not own_nodes & iri_nodes(graph), name
            assert (process_iri(specimen_id), None, None) not in graph


def test_every_non_shared_subject_belongs_to_one_test(committed, shared_subjects):
    for key in FILE_NAMES:
        graph = committed[key]
        for subject in set(graph.subjects()):
            if subject in shared_subjects:
                continue
            if isinstance(subject, URIRef):
                specimen_id = owner(subject)
                assert specimen_id is not None, subject
                assert key == "full" or TESTS[specimen_id][1] == key, (key, subject)
            else:
                # RDF-list cells of a curve table's column order.
                column = graph.value(subject, RDF.first)
                specimen_id = owner(column)
                assert specimen_id is not None, column
                assert key == "full" or TESTS[specimen_id][1] == key, (key, column)


# (f) shared triples


def test_shared_triples_are_in_all_slices(committed, shared_subjects):
    full = committed["full"]
    shared = {t for t in full if t[0] in shared_subjects}
    assert len(shared) == SHARED_TRIPLES
    assert (LAB_TABLE, CSVW.url, NEW_LAB_URL) in shared
    assert (LAB_TABLE, RDFS.comment, LAB_COMMENT) in shared
    shared_without_title = {t for t in shared if t[:2] != (HEADER, DC.title)}
    for name in SLICES:
        assert shared_without_title <= set(committed[name]), name
        assert (LAB_TABLE, CSVW.url, OLD_LAB_URL) not in committed[name]


# (g) original triples preserved


def test_every_original_triple_is_in_a_slice(original, committed):
    exceptions = {
        (LAB_TABLE, CSVW.url, OLD_LAB_URL),
        (HEADER, DC.title, Literal(ORIGINAL_TITLE, datatype=XSD_STRING)),
    }
    for triple in original:
        in_slices = [name for name in SLICES if triple in committed[name]]
        if triple in exceptions:
            assert in_slices == [], triple
        else:
            assert in_slices, triple


# (h) counts


@pytest.mark.parametrize("key", list(FILE_NAMES))
def test_triple_counts(key, committed):
    assert len(committed[key]) == EXPECTED_COUNTS[key]


# (i) curve-table query


@pytest.mark.parametrize("key", list(FILE_NAMES))
def test_curve_columns_query(key, committed):
    rows = list(committed[key].query(CURVE_COLUMNS_QUERY))
    expected_ids = [sid for sid, (_, name) in TESTS.items() if key in ("full", name)]
    assert len(rows) == 4 * len(expected_ids)
    for row in rows:
        assert str(row.url) != str(NEW_LAB_URL)
        assert str(row.delimiter) == ";"
        assert int(row.header_row_count) == 2
        assert str(row.decimal_char) == ","
    for specimen_id in expected_ids:
        own = [row for row in rows if row.process == process_iri(specimen_id)]
        assert {(str(r.role), int(r.column_number), r.unit) for r in own} == EXPECTED_COLUMNS
        assert {str(r.url) for r in own} == {
            f"https://data.pmds-mpcdf-2.pyiron.com/{specimen_id}.csv"
        }
    assert (LAB_TABLE, RDF.type, URIRef("https://w3id.org/pmd/co/PMD_0000992")) not in committed[
        key
    ]
