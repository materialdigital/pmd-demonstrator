"""Tests of the results node on the committed PMDco 3 file (offline).

The rows come from a batch run on the committed PMDco 3 graph with synthetic
curves (the curve download is stubbed), so the values are made up but every
IRI, quality and curve table is the graph's own.
"""

from __future__ import annotations

import math

import pytest
import rdflib
from conftest import PMDCO3_FULL_FILE
from rdflib import RDF, RDFS, URIRef
from workflow_helpers import PMDCO2, PMDCO2_SOURCE, curve_bytes_for, run_with_stubbed_curves

from pmd_tensile import orchestration
from pmd_tensile.nodes.acquisition import get_reported_properties
from pmd_tensile.nodes.results import MARKING, NAMESPACE, PROPERTIES, results_graph
from pmd_tensile.profiles import pmdco3_tto

PMDCO3 = "pmdco3-tto"
SOURCE = {"kind": "turtle", "locations": [str(PMDCO3_FULL_FILE)]}
IDS = ("Zd2", "Zx1")
OBO, PMD = "http://purl.obolibrary.org/obo/", "https://w3id.org/pmd/co/"
CALCULATION_DATUM, MEASUREMENT_DATUM = URIRef(PMD + "PMD_0025035"), URIRef(OBO + "IAO_0000109")
VALUE, IS_ABOUT, HAS_INPUT = URIRef(PMD + "PMD_0000006"), URIRef(OBO + "IAO_0000136"), URIRef(OBO + "OBI_0000293")

SIDE_BY_SIDE = """
PREFIX obo:   <http://purl.obolibrary.org/obo/>
PREFIX pmdco: <https://w3id.org/pmd/co/>
SELECT ?quality ?computed ?reported WHERE {
    ?c a pmdco:PMD_0025035 ; obo:IAO_0000136 ?quality ; obo:OBI_0001938/pmdco:PMD_0000006 ?computed .
    ?r a obo:IAO_0000109 ; obo:IAO_0000136 ?quality ; obo:OBI_0001938/pmdco:PMD_0000006 ?reported .
}
"""


@pytest.fixture(scope="module")
def run():
    curves = curve_bytes_for(IDS, PMDCO3, SOURCE)
    result, _ = run_with_stubbed_curves(curves, PMDCO3, SOURCE, specimen_ids=IDS, results_graph=True)
    return result


@pytest.fixture(scope="module")
def graphs(run):
    results = rdflib.Graph().parse(data=run.results_ntriples, format="nt")
    data = rdflib.Graph().parse(PMDCO3_FULL_FILE, format="turtle")
    return results, data


def rows_of(run) -> list[dict]:
    return run.results.to_dict("records")


def test_every_defined_value_is_one_calculation_datum(run, graphs):
    results, _ = graphs
    data_values = {float(v) for s in results.subjects(RDF.type, CALCULATION_DATUM)
                   for v in results.objects(results.value(s, URIRef(OBO + "OBI_0001938")), VALUE)}
    expected = [row[key] for row in rows_of(run) for key, *_ in PROPERTIES
                if row[key] is not None and math.isfinite(row[key])]
    assert len(set(results.subjects(RDF.type, CALCULATION_DATUM))) == len(expected)
    assert data_values == set(expected)  # doubles keep every digit


def test_computed_and_reported_values_share_their_quality(run, graphs):
    results, data = graphs
    merged = results + data
    pairs = {str(r.quality).rsplit("/", 1)[-1]: (float(r.computed), float(r.reported)) for r in merged.query(SIDE_BY_SIDE)}
    rows = {row["specimen_id"]: row for row in rows_of(run)}
    for sid in IDS:
        reported = get_reported_properties(PMDCO3, SOURCE, sid)
        for key in ("e_gpa", "rm_mpa", "reh_mpa"):
            hits = [p for p in pairs.values() if p == (rows[sid][key], getattr(reported, key))]
            assert len(hits) == 1, (sid, key)


def test_lab_triples_are_not_changed_and_new_individuals_are_marked(graphs):
    results, data = graphs
    assert not set(results.subjects(RDF.type, MEASUREMENT_DATUM))
    for s, p, o in results:
        if not str(s).startswith(NAMESPACE):  # only: the material bears a new quality
            assert p == URIRef(OBO + "RO_0000086") and str(o).startswith(NAMESPACE), (s, p, o)
            assert (s, None, None) in data
    for s in {s for s in results.subjects() if str(s).startswith(NAMESPACE)}:
        assert (s, RDFS.comment, rdflib.Literal(MARKING)) in results, s


def test_calculation_reads_the_curve_table_and_records_its_checksum(run, graphs):
    results, data = graphs
    rows = {row["specimen_id"]: row for row in rows_of(run)}
    for calculation in results.subjects(RDF.type, URIRef(PMD + "PMD_0025012")):
        inputs = set(results.objects(calculation, HAS_INPUT))
        tables = [i for i in inputs if (i, URIRef("http://www.w3.org/ns/csvw#url"), None) in data]
        checksums = [i for i in inputs if (i, RDF.type, URIRef(OBO + "IAO_0020000")) in results]
        assert len(tables) == 1 and len(checksums) == 1
        url = str(data.value(tables[0], URIRef("http://www.w3.org/ns/csvw#url")))
        sid = url.rsplit("/", 1)[-1].removesuffix(".csv")
        assert str(results.value(checksums[0], VALUE)) == rows[sid]["curve_sha256"]
        assert sid in str(calculation)


def test_new_qualities_only_where_the_graph_has_none(graphs):
    # The graph has a quality for E, ReH, ReL, Rp0.2, Rm, Fm, A and Z of every test (ReL with
    # the value NaN), so new qualities occur only for Ag and Agt, borne by the test's material.
    results, data = graphs
    new = {s for s in results.subjects() if str(s).endswith("/quality")}
    assert {str(s).split("/")[-2] for s in new} <= {"Ag", "Agt"}
    for quality in new:
        cls = results.value(quality, RDF.type)
        assert str(cls) == pmdco3_tto.RESULT_QUALITY_CLASSES[str(quality).split("/")[-2]]
        material = results.value(None, URIRef(OBO + "RO_0000086"), quality)
        assert (material, RDF.type, URIRef(PMD + "PMD_0020096")) in data
    rel = [s for s in results.subjects(RDF.type, CALCULATION_DATUM) if str(s).endswith("/ReL")]
    assert rel and all(str(results.value(s, IS_ABOUT)).endswith("_lower_yield_strength_ReL") for s in rel)


def test_flags_are_textual_entities_about_their_quality(run, graphs):
    results, _ = graphs
    texts = {"/".join(str(s).split("/")[-3:-1]): s for s in results.subjects(RDF.type, URIRef(OBO + "IAO_0000300"))}
    expected = {}
    for row in rows_of(run):
        flags = [flag for flag in row["flags"].split("; ") if flag]
        for _, _, _, prefix, segment in PROPERTIES:
            own = [flag for flag in flags if flag.split(":", 1)[0] == prefix]
            if own:
                expected[f"{row['specimen_id']}/{segment}"] = "; ".join(own)
    assert expected and set(texts) == set(expected)
    for key, text in texts.items():
        assert str(results.value(text, VALUE)) == expected[key]
        assert results.value(text, IS_ABOUT) is not None


def test_output_is_deterministic_and_tied_to_the_parameters(run):
    rows = rows_of(run)
    once = results_graph(PMDCO3, SOURCE, rows)
    assert once == results_graph(PMDCO3, SOURCE, rows) == run.results_ntriples
    other = results_graph(PMDCO3, SOURCE, rows, {"offset_plastic_strain": 0.003})
    assert other != once
    assert {line.split(" ")[0].split("/")[-3] for line in other.splitlines() if "/calculation>" in line.split(" ")[0]} \
        .isdisjoint({line.split(" ")[0].split("/")[-3] for line in once.splitlines() if "/calculation>" in line.split(" ")[0]})


def test_pmdco2_has_no_results_graph():
    with pytest.raises(ValueError, match="PMDco 3 only"):
        results_graph(PMDCO2, PMDCO2_SOURCE, [])


def test_the_batch_without_results_graph_is_unchanged():
    assert "results_graph" not in orchestration.build_batch_workflow(IDS).nodes
    assert "results_graph" in orchestration.build_batch_workflow(IDS, results_graph=True).nodes


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:The hashes produced for directed graphs:UserWarning")
def test_the_run_graph_records_the_results_node_and_conforms(run):
    from pmd_tensile.knowledge_graph import recipe_knowledge_graph, run_knowledge_graph, validate_knowledge_graph

    for graph in (recipe_knowledge_graph(run.batch_workflow), run_knowledge_graph(run.batch_run)):
        conforms, report = validate_knowledge_graph(graph)
        assert conforms, report[:2000]
    run_graph = run_knowledge_graph(run.batch_run)
    names = {str(o) for o in run_graph.objects(None, URIRef(PMD + "PMD_0000006"))}
    assert "results_graph" in names and "pmd_tensile.nodes.results" in names
    sha = {row["curve_sha256"] for row in rows_of(run)}
    assert sha <= names  # the checksums that join the results graph with the run graph are values here
