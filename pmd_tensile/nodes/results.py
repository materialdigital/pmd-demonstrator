"""Results node (ontology-aware): the computed properties as RDF in the vocabulary of the data graph."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math

import flowrep as fr
import rdflib
from rdflib import RDF, RDFS, XSD, Literal, URIRef

from pmd_tensile import sources
from pmd_tensile.parameters import parameter_values, parameters_from_values
from pmd_tensile.profiles import get_profile, specimen_table

NAMESPACE = "https://github.com/materialdigital/pmd-demonstrator/results/"
"""Namespace of every individual the results node writes."""

MARKING = (
    "Computed by the PMD demonstrator (materialdigital/pmd-demonstrator, pmd_tensile); "
    "not part of the TTO graph."
)
"""The rdfs:comment on every individual the results node writes."""

METHOD_VERSION = "method version 3"

# (row key, quantity name in the profile, QUDT unit, flag prefix, IRI segment)
PROPERTIES: tuple[tuple[str, str, str, str, str], ...] = (
    ("e_gpa", "E", "GigaPA", "e", "E"),
    ("reh_mpa", "ReH", "MegaPA", "reh", "ReH"),
    ("rel_mpa", "ReL", "MegaPA", "rel", "ReL"),
    ("rp02_mpa", "Rp0.2", "MegaPA", "rp02", "Rp02"),
    ("fm_n", "Fm", "N", "fm", "Fm"),
    ("rm_mpa", "Rm", "MegaPA", "fm", "Rm"),
    ("ag_percent", "Ag", "PERCENT", "ag", "Ag"),
    ("agt_percent", "Agt", "PERCENT", "agt", "Agt"),
    ("a_percent", "A", "PERCENT", "a", "A"),
    ("z_percent", "Z", "PERCENT", "z", "Z"),
)

_OBO = rdflib.Namespace("http://purl.obolibrary.org/obo/")
_PMD = rdflib.Namespace("https://w3id.org/pmd/co/")
_UNIT = rdflib.Namespace("http://qudt.org/vocab/unit/")
_RES = rdflib.Namespace(NAMESPACE)
_HAS_INPUT, _HAS_OUTPUT, _OUTPUT_OF = _OBO.OBI_0000293, _OBO.OBI_0000299, _OBO.OBI_0000312
_IS_ABOUT, _HAS_VALUE_SPEC, _SPECIFIES_VALUE_OF = _OBO.IAO_0000136, _OBO.OBI_0001938, _OBO.OBI_0001927
_UNIT_LABEL, _BEARER_OF, _VALUE = _OBO.IAO_0000039, _OBO.RO_0000086, _PMD.PMD_0000006


def _run_id(curve_sha256: str, parameters: dict) -> str:
    """First 16 hex digits of the SHA-256 of curve checksum, method parameters and package version."""
    payload = json.dumps(
        {"curve_sha256": curve_sha256, "parameters": parameters, "pmd_tensile": importlib.metadata.version("pmd-tensile")},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _new(graph: rdflib.Graph, iri: URIRef, cls: URIRef) -> URIRef:
    graph.add((iri, RDF.type, cls))
    graph.add((iri, RDFS.comment, Literal(MARKING)))
    return iri


@fr.atomic
def results_graph(profile_id: str, source: dict, rows: list, parameters: dict | None = None) -> str:
    """The computed properties of each specimen as RDF in the data graph's own vocabulary (N-Triples).

    Per specimen, one analytical calculation (``pmdco:PMD_0025012``) has as
    specified inputs the test's curve table and an identifier holding the curve's
    SHA-256, and as specified outputs one calculation datum (``pmdco:PMD_0025035``)
    per defined property. Each datum is about the quality individual that the
    graph already has for that property of the material, or a new one where it
    has none (class from the profile's results mapping), and has a scalar value
    specification with the value as ``xsd:double`` and its QUDT unit. The flags of
    a property become a textual entity about its quality. The calculation executes
    a workflow definition naming ``specimen_workflow``, the method and package
    versions. Every new individual carries :data:`MARKING`; the lab's triples are
    not repeated. Run identifier (in every IRI): the first 16 hex digits of the
    SHA-256 of curve checksum, method parameters and package version, so the same
    inputs give the same IRIs; the curve checksum joins the result with the run
    graph, which holds it as a value.

    Inputs: ``profile_id``, ``source`` (as for the acquisition nodes), ``rows``
    (the rows of :func:`~pmd_tensile.nodes.workflows.specimen_row`; values in
    GPa, MPa, N and %), ``parameters`` (the dictionary form of the method
    parameters, ``None`` for the defaults).
    Output: ``results_ntriples``, the graph as N-Triples, lines sorted (N-Triples
    keeps every digit of a double; Turtle as written by rdflib does not).
    Raises ``ValueError`` for a profile without a results mapping (PMDco 2), and
    ``LookupError`` for a specimen whose test, material or curve table is missing.
    Limits: undefined values are not written; the modelling is the demonstrator's
    and is not reviewed by the maintainers of PMDco or TTO.
    """
    profile = get_profile(profile_id)
    if profile.results is None:
        raise ValueError(f"profile {profile_id!r} has no results mapping; results graphs are written for PMDco 3 only")
    params = parameter_values(parameters_from_values(parameters))
    process_of = dict(zip(*(specimen_table(profile_id, source)[c] for c in ("specimen_id", "process"))))
    targets = sources.select(source, profile.results.targets.query, profile.results.targets.columns)
    quantities = sources.select(source, profile.quantities.query, profile.quantities.columns)
    quality_of = {(q.process, q.quantity): q.individual for q in quantities.itertuples(index=False)}

    graph = rdflib.Graph()
    definition = _new(graph, _RES["specimen_workflow"], _PMD.PMD_0000012)  # workflow definition
    graph.add((definition, RDFS.label, Literal("pmd_tensile.nodes.workflows.specimen_workflow")))
    version = f"{METHOD_VERSION}; pmd_tensile {importlib.metadata.version('pmd-tensile')}"
    graph.add((definition, _OBO.IAO_0000129, Literal(version)))  # version number
    for row in rows:
        specimen_id = row["specimen_id"]
        process = process_of.get(specimen_id)
        target = targets[targets["process"] == process]
        if process is None or len(target) != 1:
            raise LookupError(f"specimen {specimen_id!r}: expected one test with one material and curve table")
        material, table = URIRef(target.iloc[0]["material"]), URIRef(target.iloc[0]["table"])
        base = f"{_run_id(row['curve_sha256'], params)}/{specimen_id}"
        calculation = _new(graph, _RES[f"{base}/calculation"], _PMD.PMD_0025012)  # analytical calculation
        graph.add((calculation, RDFS.label, Literal(f"calculation of the tensile properties of {specimen_id}")))
        graph.add((calculation, _OBO.STATO_0000102, definition))  # executes
        graph.add((calculation, _HAS_INPUT, table))
        checksum = _new(graph, _RES[f"{base}/curve-sha256"], _OBO.IAO_0020000)  # identifier
        graph.add((checksum, _VALUE, Literal(row["curve_sha256"])))
        graph.add((checksum, _IS_ABOUT, table))
        graph.add((calculation, _HAS_INPUT, checksum))
        flags = [flag for flag in row["flags"].split("; ") if flag]
        for key, quantity, unit, prefix, segment in PROPERTIES:
            value = row[key]
            defined = value is not None and math.isfinite(value)
            own = [flag for flag in flags if flag.split(":", 1)[0] == prefix]
            if not defined and not own:
                continue
            quality = quality_of.get((process, quantity))
            if quality is None:  # the graph has no quality for this property: a new one, borne by the material
                quality = _new(graph, _RES[f"{base}/{segment}/quality"], URIRef(profile.results.quality_classes[quantity]))
                graph.add((material, _BEARER_OF, quality))
            quality = URIRef(quality)
            if defined:
                datum = _new(graph, _RES[f"{base}/{segment}"], _PMD.PMD_0025035)  # calculation datum
                specification = _new(graph, _RES[f"{base}/{segment}/value"], _OBO.OBI_0001931)  # scalar value specification
                graph.add((calculation, _HAS_OUTPUT, datum))
                graph.add((datum, _IS_ABOUT, quality))
                graph.add((datum, _HAS_VALUE_SPEC, specification))
                graph.add((specification, _VALUE, Literal(repr(float(value)), datatype=XSD.double)))
                graph.add((specification, _UNIT_LABEL, _UNIT[unit]))
                graph.add((specification, _SPECIFIES_VALUE_OF, quality))
            if own:
                text = _new(graph, _RES[f"{base}/{segment}/flags"], _OBO.IAO_0000300)  # textual entity
                graph.add((text, _VALUE, Literal("; ".join(own))))
                graph.add((text, _IS_ABOUT, quality))
                graph.add((text, _OUTPUT_OF, calculation))
    lines = graph.serialize(format="nt").splitlines()
    results_ntriples = "\n".join(sorted(line for line in lines if line.strip())) + "\n"
    return results_ntriples
