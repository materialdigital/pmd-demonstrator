"""Separation and knowledge graph (acceptance criteria 2 and 6), offline (C9).

The workflow graphs come from runs on the local pmdco2 Turtle files with a
stubbed curve download (synthetic CSV bytes, see ``workflow_helpers``), and from
a stubbed Ontodocker source with a dummy token for the credential check.

semantikon's own workflow vocabulary uses PMDco 3.x terms in the namespace
``https://w3id.org/pmd/co/PMD_``, bound to the prefix ``pmdco`` in every
exported graph; the tests read that binding from the graph. The pmdco3
*data* profile uses the same namespace for its domain concepts (e.g.
``PMD_0000974`` tensile testing process), so the analysis data nodes are
also checked against the profiles' concept IRIs.
"""

from __future__ import annotations

import ast
import dataclasses
import json

import pandas as pd
import pytest
import rdflib
from conftest import PMDCO2_ORIENTATION_FILES
from rdflib.compare import isomorphic
from workflow_helpers import PMDCO2, PMDCO2_SOURCE, curve_bytes_for, run_with_stubbed_curves

from pmd_tensile import _knowledge_graph as kg
from pmd_tensile import _sources as sources
from pmd_tensile.analysis import QUANTITY_KIND_IRIS
from pmd_tensile.profiles import pmdco3_tto

KG_SPECIMENS = ("Zx1", "Zd2")
PMD = "https://w3id.org/pmd/"
TTO = "https://w3id.org/pmd/tto/"
UNIT_TEN = "http://qudt.org/vocab/unit/TEN"
QK = "http://qudt.org/vocab/quantitykind/"
UNIT = "http://qudt.org/vocab/unit/"

EXPLORATION_FUNCTIONS = {
    ("pmd_tensile.exploration", "check_conformance"),
    ("pmd_tensile.exploration", "list_specimens"),
}
ACQUISITION_FUNCTIONS = {
    ("pmd_tensile.acquisition", "get_specimen_record"),
    ("pmd_tensile.acquisition", "get_reported_properties"),
    ("pmd_tensile.acquisition", "fetch_curve"),
}

# semantikon's data hashing calls networkx, which warns that its graph hashes
# changed in networkx 3.5; an upstream notice, not a test concern.
pytestmark = pytest.mark.filterwarnings("ignore:The hashes produced for directed graphs:UserWarning")


# ---------------------------------------------------------------------------
# Fixtures: one two-specimen run and its graphs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analysis_run():
    result, _ = run_with_stubbed_curves(curve_bytes_for(KG_SPECIMENS), PMDCO2, PMDCO2_SOURCE, specimen_ids=KG_SPECIMENS)
    return result


@pytest.fixture(scope="module")
def graphs(analysis_run) -> dict[str, rdflib.Graph]:
    return {
        "exploration recipe": kg.recipe_knowledge_graph(analysis_run.exploration_node),
        "exploration run": kg.run_knowledge_graph(analysis_run.exploration_run),
        "batch recipe": kg.recipe_knowledge_graph(analysis_run.batch_workflow),
        "batch run": kg.run_knowledge_graph(analysis_run.batch_run),
    }


def _bound_namespace(graph: rdflib.Graph, prefix: str) -> str:
    return {p: str(ns) for p, ns in graph.namespaces()}[prefix]


# ---------------------------------------------------------------------------
# Export, Turtle, validation (criterion 6)
# ---------------------------------------------------------------------------


def test_graphs_export_and_reparse_as_turtle(graphs, tmp_path):
    for name, graph in graphs.items():
        assert len(graph) > 0, name
        reparsed = kg.parse_turtle(kg.to_turtle(graph))
        assert len(reparsed) == len(graph), name
        path = kg.write_turtle(graph, tmp_path / f"{name.replace(' ', '_')}.ttl")
        assert len(kg.read_turtle(path)) == len(graph), name


def _double_values(graph: rdflib.Graph) -> list[str]:
    return sorted(
        str(o) for o in graph.objects() if isinstance(o, rdflib.Literal) and o.datatype == rdflib.XSD.double
    )


def test_default_export_is_lossless_ntriples(graphs, tmp_path):
    """Saved graphs default to N-Triples, which keeps doubles exactly (Turtle keeps 7 digits)."""
    value = 400.26355612345
    graph = rdflib.Graph()
    graph += graphs["batch run"]
    graph.add((rdflib.URIRef("urn:test:item"), rdflib.URIRef("urn:test:value"), rdflib.Literal(value)))
    assert kg.LOSSLESS_FORMAT == "nt"

    # No suffix: reading back falls back to N-Triples, so this also checks the written format.
    path = kg.write_graph(graph, tmp_path / "batch_run_graph")
    reread = kg.read_graph(path)
    (stored,) = reread.objects(rdflib.URIRef("urn:test:item"), rdflib.URIRef("urn:test:value"))
    assert stored.datatype == rdflib.XSD.double
    assert stored.toPython() == value
    assert len(reread) == len(graph)
    assert len(_double_values(graph)) > 1
    assert _double_values(reread) == _double_values(graph)
    # semantikon names some blank nodes after IRIs; they are relabelled, so the graphs are isomorphic.
    assert isomorphic(reread, graph)


def test_validate_values_conforms(graphs):
    for name, graph in graphs.items():
        conforms, report = kg.validate_knowledge_graph(graph)
        assert conforms, (name, report[:2000])


def test_semantikon_vocabulary_namespace_is_pmdco3(graphs):
    for name, graph in graphs.items():
        assert _bound_namespace(graph, "pmdco") == kg.SEMANTIKON_PMD_NAMESPACE, name
        imports = {str(o) for o in graph.objects(None, rdflib.OWL.imports)}
        assert imports <= {"https://w3id.org/pmd/co/3.1.1"}, name


# ---------------------------------------------------------------------------
# Separation in the graph (criterion 2)
# ---------------------------------------------------------------------------

_ANALYSIS_DATA_NODES = """
PREFIX bfo:   <http://purl.obolibrary.org/obo/BFO_>
PREFIX iao:   <http://purl.obolibrary.org/obo/IAO_>
PREFIX ro:    <http://purl.obolibrary.org/obo/RO_>
PREFIX pmdco: <https://w3id.org/pmd/co/PMD_>
SELECT DISTINCT ?data WHERE {
    ?step ro:0000059 ?f ; bfo:0000051 ?port .
    ?f iao:0000235 ?module .
    ?module a pmdco:0000101 ; pmdco:0000006 "pmd_tensile.analysis" .
    ?port ro:0000057 ?data .
}
"""


def _profile_concept_iris() -> set[str]:
    """The pmdco3 profile's IRIs (expanded from its label table), under https://w3id.org/pmd/."""
    expand = {"pmdco:": "https://w3id.org/pmd/co/", "tto:": TTO, "obo:": "http://purl.obolibrary.org/obo/"}
    iris = set()
    for curie, _, _ in pmdco3_tto.IRI_LABELS:
        for prefix, namespace in expand.items():
            if curie.startswith(prefix):
                iris.add(namespace + curie.removeprefix(prefix))
    return {iri for iri in iris if iri.startswith(PMD)}


def _description_of(graph: rdflib.Graph, data: rdflib.URIRef) -> tuple[set[str], list[str]]:
    """IRIs and literal texts on a data node and the nodes it points to (hash, quantity)."""
    iris, literals = set(), []
    subjects = [data] + [o for o in graph.objects(data, None) if isinstance(o, rdflib.URIRef)]
    for subject in subjects:
        for _, p, o in graph.triples((subject, None, None)):
            iris.add(str(p))
            if isinstance(o, rdflib.URIRef):
                iris.add(str(o))
            elif isinstance(o, rdflib.Literal):
                literals.append(str(o))
    return iris, literals


def test_analysis_data_nodes_reference_no_data_description_concepts(graphs):
    graph = graphs["batch run"]
    vocabulary = _bound_namespace(graph, "pmdco")
    # "has value" is the one profile IRI that is also semantikon's own value property.
    domain = _profile_concept_iris() - {vocabulary + "0000006"}
    assert "https://w3id.org/pmd/co/PMD_0000974" in domain  # tensile testing process
    data_nodes = [row.data for row in graph.query(_ANALYSIS_DATA_NODES)]
    assert len(data_nodes) > 50
    seen = set()
    for data in data_nodes:
        iris, literals = _description_of(graph, data)
        pmd = {iri for iri in iris if iri.startswith(PMD)}
        seen |= pmd
        assert all(iri.startswith(vocabulary) for iri in pmd), (data, pmd)
        assert not pmd & domain, (data, pmd & domain)
        assert not any(PMD in text for text in literals), data
    assert seen  # the vocabulary is used, so the check is not vacuous


def test_no_tto_or_pmdco2_concept_iris_anywhere(graphs):
    for name, graph in graphs.items():
        pmd = kg.pmd_iris(graph)
        assert not {iri for iri in pmd if iri.startswith(TTO)}, name
        # pmdco2 concepts are named IRIs such as https://w3id.org/pmd/co/TensileTest.
        foreign = {iri for iri in pmd if not iri.startswith(kg.SEMANTIKON_PMD_NAMESPACE)}
        assert foreign <= {"https://w3id.org/pmd/co/3.1.1"}, (name, foreign)


def test_no_unit_ten_anywhere(graphs):
    for name, graph in graphs.items():
        iris = {str(t) for triple in graph for t in triple if isinstance(t, rdflib.URIRef)}
        assert UNIT_TEN not in iris, name


# ---------------------------------------------------------------------------
# Example queries (criterion 6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["batch recipe", "batch run"])
def test_query_step_roles(graphs, name):
    roles = kg.run_example_query(graphs[name], "step_roles")
    assert "unclassified" not in set(roles["role"])
    aware = {(m, f) for m, f, r in roles.itertuples(index=False) if r.startswith("ontology-aware")}
    assert aware == ACQUISITION_FUNCTIONS
    by_module = roles.groupby("module")["role"].agg(set).to_dict()
    assert by_module["pmd_tensile.analysis"] == {"ontology-agnostic: analysis"}
    analysis_functions = set(roles.loc[roles["module"] == "pmd_tensile.analysis", "function"])
    assert {"yield_analysis", "final_elastic_fit", "maximum_force", "summarize_by_group"} <= analysis_functions
    assert ("pmd_tensile.workflows", "specimen_workflow", "workflow composition") in set(
        roles.itertuples(index=False, name=None)
    )


def test_query_step_roles_exploration(graphs):
    roles = kg.run_example_query(graphs["exploration run"], "step_roles")
    aware = {(m, f) for m, f, r in roles.itertuples(index=False) if r == "ontology-aware: exploration"}
    assert aware == EXPLORATION_FUNCTIONS


EXPECTED_OUTPUT_UNITS = {
    ("yield_analysis", "reh_mpa"): ("MegaPA", "Stress"),
    ("yield_analysis", "rel_mpa"): ("MegaPA", "Stress"),
    ("proof_strength", "rp02_mpa"): ("MegaPA", "Stress"),
    ("maximum_force", "rm_mpa"): ("MegaPA", "Stress"),
    ("maximum_force", "fm_n"): ("N", "Force"),
    ("final_elastic_fit", "e_gpa"): ("GigaPA", "ModulusOfElasticity"),
    ("final_elastic_fit", "strain_origin"): ("UNITLESS", "Strain"),
    ("elongation_at_maximum_force", "ag_percent"): ("PERCENT", "Strain"),
    ("elongation_at_maximum_force", "agt_percent"): ("PERCENT", "Strain"),
    ("extensometer_validity", "extensometer_max_valid_strain"): ("UNITLESS", "Strain"),
    ("bootstrap_fit", "noise_mpa"): ("MegaPA", "Stress"),
}


@pytest.mark.parametrize("name", ["batch recipe", "batch run"])
def test_query_analysis_output_units(graphs, name):
    units = kg.run_example_query(graphs[name], "analysis_output_units")
    found = {
        (f, o): (u.removeprefix(UNIT), q.removeprefix(QK) if q else None)
        for f, o, u, q in units.itertuples(index=False)
    }
    assert found == EXPECTED_OUTPUT_UNITS
    assert {QK + kind for _, kind in found.values()} <= set(QUANTITY_KIND_IRIS)


def test_query_run_inputs(graphs):
    inputs = dict(kg.run_example_query(graphs["batch run"], "run_inputs").itertuples(index=False))
    assert inputs["profile_id"] == PMDCO2
    assert ast.literal_eval(inputs["source"]) == PMDCO2_SOURCE
    assert inputs["curve_locator"] == "None"
    assert {inputs[f"specimen_id_{i}"] for i in range(len(KG_SPECIMENS))} == set(KG_SPECIMENS)
    exploration = dict(kg.run_example_query(graphs["exploration run"], "run_inputs").itertuples(index=False))
    assert exploration == {"profile_id": PMDCO2, "source": str(PMDCO2_SOURCE)}
    assert kg.run_example_query(graphs["batch recipe"], "run_inputs").empty  # a recipe has no values


def test_remove_data_keeps_the_run_inputs(analysis_run):
    graph = kg.run_knowledge_graph(analysis_run.batch_run, remove_data=True)
    inputs = dict(kg.run_example_query(graph, "run_inputs").itertuples(index=False))
    assert inputs["profile_id"] == PMDCO2
    assert not kg.run_example_query(graph, "analysis_output_units").empty


# ---------------------------------------------------------------------------
# Credentials (criterion 6): a dummy token never reaches graphs, outputs or tables
# ---------------------------------------------------------------------------

TOKEN = "DUMMY_ONTODOCKER_TOKEN_7f3a91"
DATASETS = {f"pmdco2_tto_example_{o}": path for o, path in PMDCO2_ORIENTATION_FILES.items()}


class _StubSparql:
    """Answers SPARQL from the local pmdco2 files named like the Ontodocker datasets."""

    def __init__(self):
        self.graphs = {}

    def _graph(self, dataset):
        if dataset not in self.graphs:
            self.graphs[dataset] = rdflib.Graph().parse(DATASETS[dataset])
        return self.graphs[dataset]

    def query_df(self, dataset, query, columns):
        rows = [
            [None if row.asdict().get(c) is None else str(row.asdict()[c]) for c in columns]
            for row in self._graph(dataset).query(query)
        ]
        return pd.DataFrame(rows, columns=columns)

    def query_raw(self, dataset, query, *, accept="application/sparql-results+json"):
        return json.dumps({"head": {}, "boolean": bool(self._graph(dataset).query(query).askAnswer)})


class _StubOntodockerClient:
    tokens: list = []

    def __init__(self, address, *, token=None, **kwargs):
        _StubOntodockerClient.tokens.append(token)
        self.sparql = _StubSparql()


def _texts(value) -> list[str]:
    """Everything an output exposes: repr, dataclass fields and table contents."""
    texts = [repr(value)]
    if isinstance(value, pd.DataFrame):
        texts.append(value.to_csv())
    elif dataclasses.is_dataclass(value):
        texts += [t for f in dataclasses.fields(value) for t in _texts(getattr(value, f.name))]
    elif isinstance(value, (list, tuple)):
        texts += [t for item in value for t in _texts(item)]
    elif isinstance(value, dict):
        texts += [t for item in value.values() for t in _texts(item)]
    return texts


def test_token_appears_in_no_graph_output_or_table(monkeypatch):
    monkeypatch.setenv("ONTODOCKER_ADDRESS", "ontodocker.example.invalid")
    monkeypatch.setenv("ONTODOCKER_TOKEN", TOKEN)
    _StubOntodockerClient.tokens = []
    monkeypatch.setattr(sources, "OntodockerClient", _StubOntodockerClient)
    source = {"kind": "ontodocker", "datasets": list(DATASETS)}
    curves = curve_bytes_for(["Zx1"])

    result, _ = run_with_stubbed_curves(curves, PMDCO2, source, specimen_ids=["Zx1"])

    assert _StubOntodockerClient.tokens and set(_StubOntodockerClient.tokens) == {TOKEN}  # the stub saw it
    graphs = {
        "exploration recipe": kg.recipe_knowledge_graph(result.exploration_node),
        "exploration run": kg.run_knowledge_graph(result.exploration_run),
        "batch recipe": kg.recipe_knowledge_graph(result.batch_workflow),
        "batch run": kg.run_knowledge_graph(result.batch_run),
        "batch run (remove_data)": kg.run_knowledge_graph(result.batch_run, remove_data=True),
    }
    for name, graph in graphs.items():
        assert TOKEN not in kg.to_turtle(graph), name
    for run in (result.exploration_run, result.batch_run):
        for label, value in run.outputs.items():
            assert not any(TOKEN in text for text in _texts(value)), label
        assert TOKEN not in repr(run.result)
    for table in (result.results, result.orientation_summary, result.comparison):
        assert TOKEN not in table.to_csv()
    # The run used the Ontodocker source and recorded it, without credentials.
    inputs = dict(kg.run_example_query(graphs["batch run"], "run_inputs").itertuples(index=False))
    assert ast.literal_eval(inputs["source"]) == source
