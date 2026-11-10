"""Cut the PMDco3/TTO S355 tensile-test graph into orientation slices.

The input is the original graph from the tensile-test ontology (TTO)
repository at commit 08a7aec, committed unchanged as
``datasets/S355_data_tto_pmdco3.ttl``. Its SHA-256 is checked before anything
else; the script refuses to run on any other file.

The script

1. adds one CSVW curve table per test, describing the test's curve CSV on the
   PMD data host (``https://data.pmds-mpcdf-2.pyiron.com/{specimen_id}.csv``)
   in PMDco3/TTO terms: each column is a measurement datum about a per-test
   entity (testing interval, crosshead separation, force, percentage
   extension) with its unit, and the table is a time series that the test
   process has as specified output;
2. relinks the ``csvw:url`` of the lab table ``S355_data_csvw_table`` to the
   PMD data host and records this in an ``rdfs:comment``;
3. cuts the graph into the slices parallel, perpendicular and diagonal. The
   ontology header, the lab table and all QUDT unit IRIs are hubs; they and
   their forward closure are shared and copied into every slice. The other
   triples belong to the test whose connected component contains their
   subject;
4. writes the full file ``pmdco3_tto_example.ttl`` (original title) and the
   slices ``pmdco3_tto_example_{parallel,perpendicular,diagonal}.ttl``, whose
   titles get the orientation appended.

The per-test nodes (process, test piece, specimen id, orientation, testing
step and testing machine) are read from the graph. Only rdflib and the
standard library are used, and nothing is read from the network.

Usage::

    python scripts/slice_pmdco3.py [--input PATH] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rdflib import RDF, RDFS, BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import CSVW, DC, OWL, XSD

DEMO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = DEMO_DIR / "datasets" / "S355_data_tto_pmdco3.ttl"
DEFAULT_OUTPUT_DIR = DEMO_DIR / "datasets"
ORIGINAL_SHA256 = "b67d95710ab607bc0c93ec3df83f072a20bc07ac58adc020a2c1b666ddc8f25d"

FULL_FILE_NAME = "pmdco3_tto_example.ttl"
SLICE_NAMES = ("parallel", "perpendicular", "diagonal")
# Orientation literal of the graph -> slice name. The literal is also the
# text appended to the slice title.
ORIENTATIONS = {
    "in rolling direction": "parallel",
    "perpendicular to rolling direction": "perpendicular",
    "diagonal to rolling direction": "diagonal",
}

CURVE_URL_TEMPLATE = "https://data.pmds-mpcdf-2.pyiron.com/{specimen_id}.csv"
LAB_TABLE_FILE_NAME = "S355_data.csv"
LAB_TABLE_URL = "https://data.pmds-mpcdf-2.pyiron.com/S355_data.csv"
ADDED_TABLE_COMMENT = (
    "Added by the PMD demonstrator (materialdigital/pmd-demonstrator, "
    "scripts/slice_pmdco3.py); not part of the TTO graph at commit 08a7aec."
)
LAB_TABLE_COMMENT = (
    "csvw:url relinked to the PMD data host by the PMD demonstrator "
    "(materialdigital/pmd-demonstrator, scripts/slice_pmdco3.py); file unchanged "
    "from the TTO repository at commit 08a7aec."
)

OBO = Namespace("http://purl.obolibrary.org/obo/")
PMDCO = Namespace("https://w3id.org/pmd/co/")
TTO = Namespace("https://w3id.org/pmd/tto/")
UNIT = Namespace("http://qudt.org/vocab/unit/")

TENSILE_TEST_PROCESS = PMDCO.PMD_0000974
TENSILE_TESTING_MACHINE = PMDCO.PMD_0000973
TIME_SERIES = PMDCO.PMD_0000992
MEASUREMENT_DATUM = OBO.IAO_0000109
HAS_VALUE = PMDCO.PMD_0000006
HAS_SPECIFIED_INPUT = OBO.OBI_0000293
HAS_SPECIFIED_OUTPUT = OBO.OBI_0000299
HAS_VALUE_SPECIFICATION = OBO.OBI_0001938
DENOTES = OBO.IAO_0000219
IS_ABOUT = OBO.IAO_0000136
HAS_MEASUREMENT_UNIT_LABEL = OBO.IAO_0000039
HAS_OCCURRENT_PART = OBO.BFO_0000117
HAS_PARTICIPANT = OBO.RO_0000057
HAS_QUALITY = OBO.RO_0000086
HAS_CHARACTERISTIC = OBO.RO_0000053
OCCUPIES_TEMPORAL_REGION = OBO.BFO_0000199
TEMPORAL_INTERVAL = OBO.BFO_0000202
QUALITY = OBO.BFO_0000019
HAS_MATERIAL = PMDCO.PMD_0020005
CROSSHEAD_SEPARATION = TTO.TTO_0000013
FORCE = PMDCO.PMD_0020200
PERCENTAGE_EXTENSION = PMDCO.PMD_0060011

# Curve CSV columns in file order: (header text, entity the column is about, unit).
CURVE_COLUMNS = (
    ("Time", "interval", UNIT.SEC),
    ("Crosshead separation", "crosshead_separation", UNIT.MilliM),
    ("Load", "force", UNIT.KiloN),
    ("Extensometer elongation", "percentage_extension", UNIT.PERCENT),
)

Triple = tuple


class SliceError(RuntimeError):
    """The input does not have the structure the approved modelling assumes."""


@dataclass(frozen=True)
class TensileTest:
    """The nodes of one test that the curve table attaches to."""

    process: URIRef
    specimen_id: str
    orientation: str
    test_piece: URIRef
    testing_step: URIRef
    machine: URIRef


def _one(values: list, what: str):
    """Return the single element of ``values`` or fail loudly."""
    if len(values) != 1:
        raise SliceError(f"expected exactly one {what}, found {len(values)}: {values!r}")
    return values[0]


def read_original(path: Path) -> Graph:
    """Parse the pinned original graph after checking its SHA-256."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != ORIGINAL_SHA256:
        raise SliceError(
            f"{path} has SHA-256 {digest}, expected the pinned original {ORIGINAL_SHA256}"
        )
    # Bind only the prefixes the file declares, so the output keeps them.
    graph = Graph(bind_namespaces="none")
    graph.parse(data=data, format="turtle")
    return graph


def find_header(graph: Graph) -> URIRef:
    return _one(list(graph.subjects(RDF.type, OWL.Ontology)), "owl:Ontology header")


def find_lab_table(graph: Graph) -> URIRef:
    table = _one(list(graph.subjects(RDF.type, CSVW.Table)), "csvw:Table in the original")
    url = _one(list(graph.objects(table, CSVW.url)), f"csvw:url of {table}")
    if not str(url).endswith("/" + LAB_TABLE_FILE_NAME):
        raise SliceError(f"lab table {table} links {url}, not {LAB_TABLE_FILE_NAME}")
    return table


def find_tests(graph: Graph) -> list[TensileTest]:
    """Find each tensile test and the nodes its curve table attaches to."""
    tests = []
    for process in sorted(graph.subjects(RDF.type, TENSILE_TEST_PROCESS)):
        test_piece = _one(
            list(graph.objects(process, HAS_SPECIFIED_INPUT)), f"test piece of {process}"
        )
        specimen_id = str(
            _one(
                [
                    value
                    for identifier in graph.subjects(DENOTES, test_piece)
                    for value_spec in graph.objects(identifier, HAS_VALUE_SPECIFICATION)
                    for value in graph.objects(value_spec, HAS_VALUE)
                ],
                f"specimen id of {test_piece}",
            )
        )
        orientation = str(
            _one(
                [
                    value
                    for material in graph.objects(test_piece, HAS_MATERIAL)
                    for quality in graph.objects(material, HAS_QUALITY)
                    if (quality, RDF.type, QUALITY) in graph
                    for value_spec in graph.objects(quality, HAS_VALUE_SPECIFICATION)
                    for value in graph.objects(value_spec, HAS_VALUE)
                ],
                f"orientation of {test_piece}",
            )
        )
        if orientation not in ORIENTATIONS:
            raise SliceError(f"unexpected orientation {orientation!r} of {test_piece}")
        testing_step = _one(
            [
                part
                for part in graph.objects(process, HAS_OCCURRENT_PART)
                if Literal("testing", datatype=XSD.string) in graph.objects(part, RDFS.label)
            ],
            f"testing step of {process}",
        )
        machine = _one(
            [
                participant
                for participant in graph.objects(testing_step, HAS_PARTICIPANT)
                if (participant, RDF.type, TENSILE_TESTING_MACHINE) in graph
            ],
            f"tensile testing machine of {testing_step}",
        )
        if not str(process).endswith("_process"):
            raise SliceError(f"process IRI {process} does not end with '_process'")
        tests.append(
            TensileTest(process, specimen_id, orientation, test_piece, testing_step, machine)
        )
    if not tests:
        raise SliceError("no tensile test process found")
    ids = [test.specimen_id for test in tests]
    if len(set(ids)) != len(ids):
        raise SliceError(f"specimen ids are not unique: {ids}")
    return tests


def curve_table_triples(test: TensileTest, base: str) -> list[Triple]:
    """The 63 triples of one test's curve table (approved at G1, D8.6)."""
    stem = test.specimen_id
    test_stem = str(test.process).removesuffix("_process")
    table = URIRef(f"{base}{stem}_csvw_table")
    dialect = URIRef(f"{base}{stem}_csvw_dialect")
    datatype = URIRef(f"{base}{stem}_csvw_datatype")
    numeric_format = URIRef(f"{base}{stem}_csvw_numeric_format")
    schema = URIRef(f"{base}{stem}_csvw_schema")
    entities = {
        "interval": URIRef(f"{test.testing_step}_interval"),
        "crosshead_separation": URIRef(f"{test_stem}_crosshead_separation"),
        "force": URIRef(f"{test_stem}_force"),
        "percentage_extension": URIRef(f"{test_stem}_percentage_extension"),
    }

    def string(text: str) -> Literal:
        return Literal(text, datatype=XSD.string)

    triples: list[Triple] = [
        (test.process, HAS_SPECIFIED_OUTPUT, table),
        (table, RDF.type, CSVW.Table),
        (table, RDF.type, TIME_SERIES),
        (table, RDFS.label, string(f"{stem} test record")),
        (table, RDFS.comment, string(ADDED_TABLE_COMMENT)),
        (
            table,
            CSVW.url,
            Literal(CURVE_URL_TEMPLATE.format(specimen_id=stem), datatype=XSD.anyURI),
        ),
        (table, CSVW.dialect, dialect),
        (table, CSVW.tableSchema, schema),
        (dialect, RDF.type, CSVW.Dialect),
        (dialect, CSVW.delimiter, string(";")),
        (dialect, CSVW.headerRowCount, Literal("2", datatype=XSD.nonNegativeInteger)),
        (dialect, CSVW.encoding, string("utf-8")),
        (datatype, RDF.type, CSVW.Datatype),
        (datatype, CSVW.base, string("decimal")),
        (datatype, CSVW.format, numeric_format),
        (numeric_format, RDF.type, CSVW.NumericFormat),
        (numeric_format, CSVW.decimalChar, string(",")),
        (schema, RDF.type, CSVW.Schema),
    ]

    columns = []
    for title, entity_key, unit in CURVE_COLUMNS:
        name = title.replace(" ", "_")
        column = URIRef(f"{table}/col/{name}")
        columns.append(column)
        triples += [
            (column, RDF.type, CSVW.Column),
            (column, RDF.type, MEASUREMENT_DATUM),
            (column, CSVW.name, string(name)),
            (column, CSVW.title, string(title)),
            (column, CSVW.datatype, datatype),
            (column, HAS_MEASUREMENT_UNIT_LABEL, unit),
            (column, IS_ABOUT, entities[entity_key]),
        ]

    # Column order as an RDF list.
    cells = [BNode() for _ in columns]
    triples.append((schema, CSVW.column, cells[0]))
    for index, (cell, column) in enumerate(zip(cells, columns)):
        rest = cells[index + 1] if index + 1 < len(cells) else RDF.nil
        triples += [(cell, RDF.first, column), (cell, RDF.rest, rest)]

    triples += [
        (test.testing_step, OCCUPIES_TEMPORAL_REGION, entities["interval"]),
        (entities["interval"], RDF.type, TEMPORAL_INTERVAL),
        (test.machine, HAS_QUALITY, entities["crosshead_separation"]),
        (entities["crosshead_separation"], RDF.type, CROSSHEAD_SEPARATION),
        (test.test_piece, HAS_CHARACTERISTIC, entities["force"]),
        (test.test_piece, HAS_CHARACTERISTIC, entities["percentage_extension"]),
        (entities["force"], RDF.type, FORCE),
        (entities["percentage_extension"], RDF.type, PERCENTAGE_EXTENSION),
    ]
    return triples


def _is_node(term) -> bool:
    return not isinstance(term, Literal)


def shared_nodes(graph: Graph, header: URIRef, lab_table: URIRef) -> set:
    """Hubs (header, lab table, unit IRIs) and their forward closure.

    The closure skips ``rdf:type`` objects and literals.
    """
    hubs = {header, lab_table}
    for triple in graph:
        hubs.update(term for term in triple if isinstance(term, URIRef) and term in UNIT)
    shared = set()
    todo = list(hubs)
    while todo:
        node = todo.pop()
        if node in shared:
            continue
        shared.add(node)
        for predicate, obj in graph.predicate_objects(node):
            if predicate != RDF.type and _is_node(obj) and obj not in shared:
                todo.append(obj)
    return shared


def assign_subjects(graph: Graph, shared: set, tests: list[TensileTest]) -> dict:
    """Map every non-shared subject to its test via connected components.

    Components are taken over the undirected subject-object graph of the
    non-shared nodes, skipping ``rdf:type`` edges, literals and edges into
    shared nodes. Each component must contain exactly one test process.
    """
    parent: dict = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for subject, predicate, obj in graph:
        if subject in shared:
            continue
        find(subject)
        if predicate == RDF.type or not _is_node(obj) or obj in shared:
            continue
        parent[find(subject)] = find(obj)

    test_of_root = {}
    for test in tests:
        root = find(test.process)
        if root in test_of_root:
            raise SliceError(
                f"{test.process} and {test_of_root[root].process} are in one component"
            )
        test_of_root[root] = test

    assignment = {}
    for subject in set(graph.subjects()) - shared:
        root = find(subject)
        if root not in test_of_root:
            raise SliceError(f"subject {subject} is not connected to any test process")
        assignment[subject] = test_of_root[root]
    return assignment


def build_graphs(original: Graph) -> tuple[Graph, dict[str, Graph], dict[str, int]]:
    """Build the full graph and the three slices from the original."""
    header = find_header(original)
    lab_table = find_lab_table(original)
    tests = find_tests(original)
    base = str(header)
    title = _one(list(original.objects(header, DC.title)), f"dc:title of {header}")

    shared = shared_nodes(original, header, lab_table)
    assignment = assign_subjects(original, shared, tests)

    shared_triples: list[Triple] = []
    per_test: dict[TensileTest, list[Triple]] = {test: [] for test in tests}
    for triple in original:
        if triple[0] in shared:
            shared_triples.append(triple)
        else:
            per_test[assignment[triple[0]]].append(triple)
    counts = {"original": len(original), "shared (original)": len(shared_triples)}
    for test in tests:
        counts[f"per test {test.specimen_id} (original)"] = len(per_test[test])

    # Relink the lab table and record the relink.
    old_url = _one(list(original.objects(lab_table, CSVW.url)), "lab-table csvw:url")
    shared_triples.remove((lab_table, CSVW.url, old_url))
    shared_triples += [
        (lab_table, CSVW.url, Literal(LAB_TABLE_URL, datatype=XSD.anyURI)),
        (lab_table, RDFS.comment, Literal(LAB_TABLE_COMMENT, datatype=XSD.string)),
    ]

    # Add the curve tables. Their new IRIs must not exist yet, and they may
    # only attach to the test's own nodes, its new nodes or shared unit IRIs.
    existing_nodes = set(original.subjects()) | {
        obj for obj in original.objects() if _is_node(obj)
    }
    for test in tests:
        added = curve_table_triples(test, base)
        test_nodes = {s for s, t in assignment.items() if t is test}
        new_nodes = {s for s, _, _ in added if s not in test_nodes}
        collisions = {node for node in new_nodes if node in existing_nodes}
        if collisions:
            raise SliceError(f"new IRIs of {test.specimen_id} already exist: {collisions}")
        allowed = test_nodes | new_nodes | shared | {RDF.nil}
        for subject, predicate, obj in added:
            if subject not in allowed or (
                predicate != RDF.type and _is_node(obj) and obj not in allowed
            ):
                raise SliceError(f"added triple leaves test {test.specimen_id}: {subject} {obj}")
        counts[f"added per test {test.specimen_id}"] = len(added)
        per_test[test] += added

    full = _new_graph(original)
    for triple in shared_triples:
        full.add(triple)
    for triples in per_test.values():
        for triple in triples:
            full.add(triple)
    counts["shared (with relink comment)"] = len(shared_triples)
    counts["full"] = len(full)

    slices = {}
    for name in SLICE_NAMES:
        graph = _new_graph(original)
        for triple in shared_triples:
            graph.add(triple)
        orientation = next(text for text, slice_name in ORIENTATIONS.items() if slice_name == name)
        graph.set((header, DC.title, Literal(f"{title} ({orientation})", datatype=XSD.string)))
        for test in tests:
            if ORIENTATIONS[test.orientation] == name:
                for triple in per_test[test]:
                    graph.add(triple)
        slices[name] = graph
        counts[f"slice {name}"] = len(graph)
        counts[f"slice {name} tests"] = ", ".join(
            test.specimen_id for test in tests if ORIENTATIONS[test.orientation] == name
        )
    return full, slices, counts


def _new_graph(original: Graph) -> Graph:
    graph = Graph(bind_namespaces="none")
    for prefix, namespace in original.namespaces():
        graph.bind(prefix, namespace)
    # The column lists are written inline as ( ... ), but rdflib still declares
    # the rdf namespace for rdf:first/rdf:rest; name it rdf instead of ns1.
    graph.bind("rdf", RDF)
    return graph


def output_paths(output_dir: Path, input_path: Path) -> dict[str, Path]:
    """The four output paths, checked to stay inside ``output_dir``."""
    output_dir = output_dir.resolve()
    if not output_dir.is_dir():
        raise SliceError(f"output directory {output_dir} does not exist")
    input_resolved = input_path.resolve()
    names = {"full": FULL_FILE_NAME}
    names.update({name: f"pmdco3_tto_example_{name}.ttl" for name in SLICE_NAMES})
    paths = {}
    for key, name in names.items():
        path = output_dir / name
        if path.is_symlink():
            raise SliceError(f"refusing to write through the symlink {path}")
        if path.resolve().parent != output_dir:
            raise SliceError(f"{path} is outside {output_dir}")
        if path.resolve() == input_resolved:
            raise SliceError(f"refusing to overwrite the input file {input_path}")
        # The same file under another name, e.g. a hard link, or a different
        # letter case on a case-insensitive file system (resolve() keeps the case).
        if path.exists() and input_path.exists() and os.path.samefile(path, input_path):
            raise SliceError(f"refusing to overwrite the input file {input_path} (same file as {path})")
        paths[key] = path
    return paths


def write_turtle_files(graphs: dict[Path, Graph]) -> None:
    """Write each graph as Turtle to its path, replacing no target before all are written.

    Every graph is first serialised to a temporary file next to its target;
    only then are the targets replaced (``os.replace``, which does not follow
    a symlink at the target). A failure while serialising or writing leaves
    every existing target unchanged and removes the temporary files.
    """
    # mkstemp creates the files with mode 0600; use the usual mode instead.
    umask = os.umask(0)
    os.umask(umask)
    temporaries: dict[Path, str] = {}
    try:
        for path, graph in graphs.items():
            data = graph.serialize(format="turtle").encode("utf-8")
            handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
            temporaries[path] = temporary
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
            os.chmod(temporary, 0o666 & ~umask)
        for path, temporary in temporaries.items():
            os.replace(temporary, path)
    except BaseException:
        for temporary in temporaries.values():
            Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="pinned original graph (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="existing directory for the four output files (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    try:
        paths = output_paths(args.output_dir, args.input)
        original = read_original(args.input)
        full, slices, counts = build_graphs(original)
    except SliceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    write_turtle_files({paths["full"]: full, **{paths[name]: graph for name, graph in slices.items()}})
    for key, value in counts.items():
        print(f"{key}: {value}")
    for path in paths.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
