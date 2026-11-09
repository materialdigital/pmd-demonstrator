"""Knowledge-graph export of the tensile workflows with semantikon (acceptance criterion 6).

semantikon describes a workflow with its own workflow vocabulary: PMDco 3.x
terms in the namespace ``https://w3id.org/pmd/co/PMD_`` (bound to the prefix
``pmdco`` in the exported graphs, :data:`SEMANTIKON_PMD_NAMESPACE`), plus OBO
terms (BFO, RO, IAO, OBI) and QUDT for units. This vocabulary describes
functions, ports and data; it is independent of the ontology that describes
the tensile-test *data* (the profile). Analysis ports carry only units and
generic QUDT quantity kinds (decision D4).

Graphs
------
- :func:`recipe_knowledge_graph`: the workflow *recipe* (what will run).
- :func:`run_knowledge_graph`: an executed run (what ran, with which inputs
  and results). Options, as decided here:

  - ``hash_data=True`` (default): every data node gets semantikon's
    provenance hash (function metadata, upstream hashes and workflow-input
    values). Equal results of different runs can be recognized only within
    the same networkx (>= 3.5) environment, because these hashes use networkx
    graph hashes, which differ across networkx versions.
  - ``remove_data=False`` (default): values stay in the graph, so results and
    inputs can be queried (example queries (b) and (c)). semantikon writes
    values as literals; arrays and tables become their (shortened, lossy)
    ``repr``, which is why each specimen reports its raw-curve SHA-256 and
    URL as explicit strings (IC3). ``remove_data=True`` drops every value that
    has a hash, i.e. all computed values; workflow inputs (profile ID,
    source, specimen IDs) always stay (T1 finding). Use it for a compact
    graph without results.

  Credentials never enter either graph: they are not node inputs or outputs
  (decision D5, :mod:`pmd_tensile.sources`).

- :func:`validate_knowledge_graph`: ``semantikon.validate_values`` (OWL-RL
  reasoning and SHACL), returning ``(conforms, report text)``.

Saving and display
------------------
- :func:`write_graph` and :func:`read_graph`: save and load a graph. The
  default format is **N-Triples** (:data:`LOSSLESS_FORMAT`), which keeps every
  literal's lexical form and therefore round-trips all values exactly. Use
  these for any graph that is kept, exported or compared later.
  Before writing, blank nodes whose IDs are not valid N-Triples labels are
  relabelled deterministically. semantikon 1.4.4 names some blank nodes after
  IRIs (``http://pyiron.org/ontology/...-not_defined_input_..._uri``), which
  rdflib 7.6.0 writes verbatim, so that a plain N-Triples file of a batch KG
  cannot be parsed again. Blank-node labels are local to a document, so the
  written graph is isomorphic to the original: same triples, same literals.
  JSON-LD would also round-trip, but N-Triples is the decided default
  (handoff r10) and is line-based, which makes diffs easy.
- :func:`to_turtle`, :func:`write_turtle`, :func:`parse_turtle`,
  :func:`read_turtle`: Turtle, for human-readable display only.
  :data:`TURTLE_PRECISION_NOTE` explains why: rdflib 7.6.0 writes
  ``xsd:double`` literals in Turtle in the short numeric form
  ``f"{value:e}"`` (``rdflib/term.py``, line 1681), i.e. with 7 significant
  digits, so 400.26355612345 becomes ``4.002636e+02``. Triple counts survive a
  Turtle round trip; double values do not.

Example SPARQL queries
----------------------
:data:`EXAMPLE_QUERIES` holds three documented queries over a workflow graph;
run them with :func:`run_example_query`. They rely on how semantikon 1.4.4
records a workflow:

- a step (node) is linked to its Python function by ``ro:RO_0000059``;
- the function is denoted (``iao:IAO_0000235``) by its function name
  (a ``pmdco:PMD_0000100``) and by its module (a ``pmdco:PMD_0000101``), each
  with the text as ``pmdco:PMD_0000006`` (has value);
- steps and workflows have their ports as parts (``bfo:BFO_0000051``); a port
  is an input if its class is a subclass of ``pmdco:PMD_0000066`` and an
  output for ``pmdco:PMD_0000067``; its name is ``pmdco:PMD_0000128``;
- a port is linked to its data node by ``ro:RO_0000057``; the data node has
  its value (``pmdco:PMD_0000006``), its unit (``qudt:hasUnit``) and a
  quantity-kind individual (``obi:OBI_0001927``) typed with the QUDT quantity
  kind.

The queries:

(a) ``step_roles``: each function used, its module and its role, from the
    module recorded in the graph (:data:`MODULE_ROLES`): ontology-aware
    (exploration, acquisition), ontology-agnostic analysis, ontology-agnostic
    contract adapters, or flowrep plumbing. A step that has steps of its own
    (the per-specimen and exploration workflows) is a "workflow composition".
(b) ``analysis_output_units``: the units and QUDT quantity kinds carried by
    the outputs of the analysis functions (module
    ``pmd_tensile.analysis``).
(c) ``run_inputs``: the inputs of the top-level workflow and their values,
    e.g. the profile ID and source a run used. Works on run graphs.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import flowrep as fr
import pandas as pd
import rdflib
import semantikon
from rdflib.util import guess_format

SEMANTIKON_PMD_NAMESPACE = "https://w3id.org/pmd/co/PMD_"
"""Namespace of semantikon's own workflow vocabulary (PMDco 3.x terms; prefix ``pmdco``)."""

LOSSLESS_FORMAT = "nt"
"""Default rdflib format for saved or exported graphs: N-Triples (lossless)."""

TURTLE_PRECISION_NOTE = (
    "Turtle is for display only: rdflib 7.6.0 writes xsd:double literals with 7 "
    "significant digits (e.g. 400.26355612345 -> 4.002636e+02), so double values "
    "do not survive a Turtle round trip. Saved graphs use N-Triples, which is lossless."
)
"""Why Turtle output is for display only (see the module docstring)."""

MODULE_ROLES: dict[str, str] = {
    "pmd_tensile.exploration": "ontology-aware: exploration",
    "pmd_tensile.acquisition": "ontology-aware: acquisition",
    "pmd_tensile.analysis": "ontology-agnostic: analysis",
    "pmd_tensile.workflows": "ontology-agnostic: contract adapter",
    "flowrep.prospective.transformers": "workflow plumbing (flowrep)",
}
"""Role of each Python module whose functions appear as workflow steps."""

ONTOLOGY_AWARE_MODULES: tuple[str, ...] = (
    "pmd_tensile.exploration",
    "pmd_tensile.acquisition",
)
ANALYSIS_MODULE = "pmd_tensile.analysis"

_PREFIXES = """\
PREFIX bfo:   <http://purl.obolibrary.org/obo/BFO_>
PREFIX iao:   <http://purl.obolibrary.org/obo/IAO_>
PREFIX obi:   <http://purl.obolibrary.org/obo/OBI_>
PREFIX ro:    <http://purl.obolibrary.org/obo/RO_>
PREFIX pmdco: <https://w3id.org/pmd/co/PMD_>
PREFIX qudt:  <http://qudt.org/schema/qudt/>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
"""


def _role_values() -> str:
    rows = "\n".join(f'        ("{module}" "{role}")' for module, role in MODULE_ROLES.items())
    return f"    VALUES (?module ?known_role) {{\n{rows}\n    }}"


EXAMPLE_QUERIES: dict[str, str] = {
    "step_roles": _PREFIXES
    + """
# (a) Which steps are ontology-aware and which are ontology-agnostic?
# Every step is linked to its function; the function's module decides the role.
SELECT DISTINCT ?module ?function ?role WHERE {
    ?step ro:0000059 ?f .
    ?f iao:0000235 ?module_node, ?name_node .
    ?module_node a pmdco:0000101 ; pmdco:0000006 ?module_value .
    ?name_node a pmdco:0000100 ; pmdco:0000006 ?function_value .
    BIND(STR(?module_value) AS ?module)
    BIND(STR(?function_value) AS ?function)
    OPTIONAL {
"""
    + _role_values()
    + """
    }
    # A step with steps of its own is a workflow composing them.
    BIND(EXISTS { ?step bfo:0000051/ro:0000059 ?inner } AS ?composite)
    BIND(IF(?composite, "workflow composition", COALESCE(?known_role, "unclassified")) AS ?role)
}
ORDER BY ?role ?module ?function
""",
    "analysis_output_units": _PREFIXES
    + """
# (b) Which units and QUDT quantity kinds do the analysis outputs carry?
SELECT DISTINCT ?function ?output ?unit ?quantity_kind WHERE {
    ?step ro:0000059 ?f ;
          bfo:0000051 ?port .
    ?f iao:0000235 ?module_node, ?name_node .
    ?module_node a pmdco:0000101 ; pmdco:0000006 "pmd_tensile.analysis" .
    ?name_node a pmdco:0000100 ; pmdco:0000006 ?function_value .
    ?port a/rdfs:subClassOf pmdco:0000067 ;
          pmdco:0000128 ?output_value ;
          ro:0000057 ?data .
    ?data qudt:hasUnit ?unit .
    OPTIONAL {
        ?data obi:0001927 ?quantity .
        ?quantity a ?quantity_kind .
        FILTER(STRSTARTS(STR(?quantity_kind), "http://qudt.org/vocab/quantitykind/"))
    }
    BIND(STR(?function_value) AS ?function)
    BIND(STR(?output_value) AS ?output)
}
ORDER BY ?function ?output
""",
    "run_inputs": _PREFIXES
    + """
# (c) Which inputs did the run use? Inputs of the top-level workflow, which is
# not itself part of another node.
SELECT DISTINCT ?input ?value WHERE {
    ?workflow bfo:0000051 ?port .
    FILTER NOT EXISTS { ?parent bfo:0000051 ?workflow }
    ?port a/rdfs:subClassOf pmdco:0000066 ;
          pmdco:0000128 ?input_value ;
          ro:0000057 ?data .
    ?data pmdco:0000006 ?data_value .
    BIND(STR(?input_value) AS ?input)
    BIND(STR(?data_value) AS ?value)
}
ORDER BY ?input
""",
}
"""Example SPARQL queries (a)-(c) over a workflow knowledge graph; see the module docstring."""


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _recipe_of(workflow) -> fr.schemas.WorkflowRecipe:
    """The flowrep recipe of a workflow object, a decorated function or a recipe."""
    if isinstance(workflow, fr.schemas.WorkflowRecipe):
        return workflow
    recipe = getattr(workflow, "flowrep_recipe", None)  # @fr.workflow function
    if recipe is None:
        recipe = getattr(workflow, "recipe", None)  # pyiron_workflow workflow or node
    if not isinstance(recipe, fr.schemas.WorkflowRecipe):
        raise TypeError(f"no flowrep workflow recipe found on {type(workflow).__name__}")
    return recipe


def recipe_knowledge_graph(workflow) -> rdflib.Graph:
    """KG of a workflow recipe.

    ``workflow`` is a ``pyiron_workflow`` workflow or node (``.recipe``), an
    ``@fr.workflow`` function (``.flowrep_recipe``) or a flowrep
    ``WorkflowRecipe``.
    """
    return semantikon.get_knowledge_graph(_recipe_of(workflow))


def run_knowledge_graph(run, *, hash_data: bool = True, remove_data: bool = False) -> rdflib.Graph:
    """KG of an executed run (a ``pyiron_workflow`` run, or its flowrep ``DagData``).

    Defaults: provenance hashes on, values kept (see the module docstring).
    """
    data = getattr(run, "result", run)
    return semantikon.get_knowledge_graph(data, hash_data=hash_data, remove_data=remove_data)


def validate_knowledge_graph(graph: rdflib.Graph) -> tuple[bool, str]:
    """``semantikon.validate_values`` on ``graph``: ``(conforms, report text)``.

    Runs on a copy, so ``graph`` is not changed by the OWL-RL reasoner.
    """
    result = semantikon.validate_values(graph, copy_graph=True)
    conforms = bool(result[0])
    text = str(result[-1]) if len(result) > 1 else ""
    return conforms, text


# ---------------------------------------------------------------------------
# Saving (lossless by default)
# ---------------------------------------------------------------------------


_NT_BLANK_NODE_LABEL = re.compile(r"[A-Za-z0-9_](?:[A-Za-z0-9_.\-]*[A-Za-z0-9_\-])?")
"""Blank-node labels valid in N-Triples (ASCII subset of the W3C grammar; no ':' or '/')."""


def _with_valid_blank_node_labels(graph: rdflib.Graph) -> rdflib.Graph:
    """``graph`` with every blank node whose ID is not a valid N-Triples label relabelled.

    semantikon 1.4.4 names some blank nodes after IRIs (``BNode(str(iri) +
    "_uri")`` in ``semantikon/ontology.py``), and rdflib writes such IDs
    verbatim, so the N-Triples file could not be parsed again. Blank-node
    labels are local to a document, so relabelling them (deterministically,
    ``b`` + SHA-256 of the old ID) gives an isomorphic graph. Other blank nodes
    and all IRIs and literals are unchanged.
    """
    relabel = {
        term: rdflib.BNode("b" + hashlib.sha256(str(term).encode("utf-8")).hexdigest())
        for triple in graph
        for term in triple
        if isinstance(term, rdflib.BNode) and not _NT_BLANK_NODE_LABEL.fullmatch(str(term))
    }
    if not relabel:
        return graph
    relabelled = rdflib.Graph()
    for prefix, namespace in graph.namespaces():
        relabelled.bind(prefix, namespace, override=True, replace=True)
    for s, p, o in graph:
        relabelled.add((relabel.get(s, s), p, relabel.get(o, o)))
    return relabelled


def write_graph(graph: rdflib.Graph, path: str | Path, *, format: str = LOSSLESS_FORMAT) -> Path:
    """Write ``graph`` to ``path`` (the caller chooses the location).

    The default is N-Triples (:data:`LOSSLESS_FORMAT`), which round-trips every
    value exactly; Turtle would round double values to 7 significant digits
    (:data:`TURTLE_PRECISION_NOTE`). Blank nodes whose IDs are not valid
    N-Triples labels are relabelled first (an isomorphic graph, see
    :func:`_with_valid_blank_node_labels`), so the file can be read back with
    :func:`read_graph`. ``graph`` itself is not changed.
    """
    path = Path(path)
    _with_valid_blank_node_labels(graph).serialize(destination=path, format=format, encoding="utf-8")
    return path


def read_graph(path: str | Path, *, format: str | None = None) -> rdflib.Graph:
    """Parse a graph file into a new graph.

    Without ``format``, it is guessed from the file suffix (e.g. ``.nt``,
    ``.ttl``), falling back to N-Triples.
    """
    path = Path(path)
    return rdflib.Graph().parse(path, format=format or guess_format(str(path)) or LOSSLESS_FORMAT)


# ---------------------------------------------------------------------------
# Turtle (human-readable display; lossy for doubles)
# ---------------------------------------------------------------------------


def to_turtle(graph: rdflib.Graph) -> str:
    """Serialize ``graph`` as Turtle text, for display.

    Lossy for ``xsd:double`` values (:data:`TURTLE_PRECISION_NOTE`); save
    graphs with :func:`write_graph` instead.
    """
    return graph.serialize(format="turtle")


def write_turtle(graph: rdflib.Graph, path: str | Path) -> Path:
    """Write ``graph`` as a Turtle file (the caller chooses the location).

    Lossy for ``xsd:double`` values (:data:`TURTLE_PRECISION_NOTE`); use
    :func:`write_graph` (N-Triples) for graphs that are kept or exported.
    """
    path = Path(path)
    graph.serialize(destination=path, format="turtle")
    return path


def parse_turtle(text: str) -> rdflib.Graph:
    """Parse Turtle text (e.g. from :func:`to_turtle`) into a new graph."""
    return rdflib.Graph().parse(data=text, format="turtle")


def read_turtle(path: str | Path) -> rdflib.Graph:
    """Parse a Turtle file into a new graph."""
    return rdflib.Graph().parse(Path(path), format="turtle")


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def run_example_query(graph: rdflib.Graph, name: str) -> pd.DataFrame:
    """Run one of :data:`EXAMPLE_QUERIES` and return the rows as a table of strings."""
    if name not in EXAMPLE_QUERIES:
        raise KeyError(f"unknown example query {name!r}; available: {sorted(EXAMPLE_QUERIES)}")
    result = graph.query(EXAMPLE_QUERIES[name])
    columns = [str(variable) for variable in result.vars]
    rows = [[None if value is None else str(value) for value in row] for row in result]
    return pd.DataFrame(rows, columns=columns)


def pmd_iris(graph: rdflib.Graph) -> set[str]:
    """All IRIs in ``graph`` under ``https://w3id.org/pmd/`` (any position)."""
    return {
        str(term)
        for triple in graph
        for term in triple
        if isinstance(term, rdflib.URIRef) and str(term).startswith("https://w3id.org/pmd/")
    }
