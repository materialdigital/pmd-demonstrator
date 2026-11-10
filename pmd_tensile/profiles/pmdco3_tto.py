"""Profile ``pmdco3-tto``: PMDco 3.x with the Tensile Test Ontology (TTO).

Graphs:

- The original ``S355_data_tto_pmdco3.ttl`` in
  materialdigital/tensile-test-ontology at commit
  08a7aec359f1072b823362bd67fbb93a88643e6f (:data:`PINNED_GRAPH_URL`; no
  licence was detected for that repository), committed unchanged as
  ``datasets/S355_data_tto_pmdco3.ttl``. It has no per-test curve tables
  (see "Gaps" below), so it is analysed with the explicit
  :data:`CURVE_LOCATOR`.
- The demonstrator's slices ``datasets/pmdco3_tto_example_{parallel,
  perpendicular,diagonal}.ttl`` and the full file
  ``datasets/pmdco3_tto_example.ttl``, written by
  ``scripts/slice_pmdco3.py`` (the slices are meant for Ontodocker as
  ``pmdco3_tto_example_*``). They are the original plus one CSVW curve table
  per test (handoff decision D8.6), read by :data:`CURVE_COLUMNS`; no locator
  is needed.

How the concepts are expressed (observed in the original graph; the slices
keep all of it):

- A test is a ``pmdco:PMD_0000974`` (tensile testing process) whose
  ``obo:OBI_0000293`` (has specified input) is the test piece.
- The test-piece name is the value (``pmdco:PMD_0000006``) of the value
  specification (``obo:OBI_0001938``) of the information entity that
  ``obo:IAO_0000219`` (denotes) the test piece. It is found by this structure;
  the entity's class ``pmdco:PMD_0000017`` has no label in the pinned
  ontology files and is not used.
- Measured values: the test ``obo:OBI_0000299`` (has specified output) a
  ``obo:IAO_0000109`` (measurement datum) that ``obo:IAO_0000136`` (is about)
  a quality individual typed with a TTO/PMDco class, and that has a scalar
  value specification with ``pmdco:PMD_0000006`` (value) and
  ``obo:IAO_0000039`` (unit).
- S0, s1..s3 and L0 are qualities (``obo:RO_0000086``) of the test piece.
  The bearer matters: L0 and the extensometer's nominal gauge length share the
  class ``tto:TTO_0000028``. S0 and the measurements s_i have distinct classes,
  so no IRI naming convention is needed.
- Lu and Su are qualities of the test piece after the test (a specified
  output); the reported results are qualities of the material. The bearer is
  required here too: the load cell's force range is also typed
  ``tto:TTO_0000023`` (maximum force).
- The test-speed switch point is a process attribute
  (``pmdco:PMD_0000009``) of the test typed ``tto:TTO_0000058`` (transition
  point testing rate), measured like the other quantities (unit ``PERCENT``),
  and queried as ``eps_sw``.
- Orientation: the material (``pmdco:PMD_0020005`` from the test piece) bears
  a quality typed only as ``obo:BFO_0000019`` (quality), whose value
  specification holds the rolling-direction text. No specific concept marks it
  as orientation; the profile relies on it being the material's only
  generically typed quality with a textual value (acquisition fails if there
  are several).

- Curves (slices and full file only): the test ``obo:OBI_0000299`` (has
  specified output) a ``csvw:Table`` that is also a ``pmdco:PMD_0000992``
  (time series), with ``csvw:url`` (the data-host CSV) and a
  ``csvw:tableSchema`` whose ``csvw:column`` RDF list gives the column
  order. Each column is a measurement datum (``obo:IAO_0000109``) with its
  unit (``obo:IAO_0000039``), that ``obo:IAO_0000136`` (is about) a per-test
  entity typed ``obo:BFO_0000202`` (temporal interval: time),
  ``tto:TTO_0000013`` (crosshead separation), ``pmdco:PMD_0020200`` (force)
  or ``pmdco:PMD_0060011`` (percentage extension: strain). The table's
  ``csvw:dialect`` and the columns' ``csvw:datatype``/``csvw:format`` declare
  the CSV layout, which acquisition checks against the curve parser's.
  The aggregate lab table ``S355_data.csv`` is not a time series and is not
  matched.

Gaps and non-semantic parts (reported as findings):

- **No curve links in the original graph.** The original links only the
  aggregate table ``S355_data.csv`` via CSVW, not the per-test curve CSVs.
  For it, the curve location comes from :data:`CURVE_LOCATOR`, an explicit,
  NON-SEMANTIC URL template with ``{specimen_id}``, and the column semantics
  from its declared, NON-SEMANTIC header-name map; neither is knowledge from
  the graph. It is not the profile's default: pass
  ``curve_locator=CURVE_LOCATOR.as_dict()`` explicitly; without it,
  acquisition on the original fails with a message saying so. The slices
  close this gap with their curve tables; there, an explicit locator still
  replaces the graph's links if given.
- **Notes by instance label.** Lab notes and the process identifier are both
  information entities that denote the test; only the instance label
  ``"process note"`` tells them apart (NON-SEMANTIC selection).
- ``pmdco:PMD_0020005`` (test piece -> material) is labelled "porosity" (an
  ``owl:Class``) in the pinned ontology files; the profile uses it because the
  data does.
- Several TTO classes used by the data are marked deprecated in the same
  commit's ``src/ontology/tto-full.ttl`` (replaced by PMDco terms), but not in
  the released ``tto-full.ttl``. The profile follows the data. See
  :data:`IRI_LABELS`.
- Unit IRIs ``unit:MegaPa`` and ``unit:GigaPa`` are not QUDT IRIs; they are
  mapped by declared aliases (reported values only).
"""

from __future__ import annotations

from pmd_tensile.contract import ColumnSpec
from pmd_tensile.profiles.base import (
    CURVE_COLUMN_COLUMNS,
    CURVE_DIALECT_COLUMNS,
    NOTE_COLUMNS,
    QUANTITY_COLUMNS,
    SPECIMEN_COLUMNS,
    CurveLocator,
    OntologyProfile,
    SelectQuery,
)
from pmd_tensile.profiles.units import QUDT_UNIT

PROFILE_ID = "pmdco3-tto"

TTO_COMMIT = "08a7aec359f1072b823362bd67fbb93a88643e6f"
_TTO_RAW = f"https://raw.githubusercontent.com/materialdigital/tensile-test-ontology/{TTO_COMMIT}"
PINNED_GRAPH_URL = f"{_TTO_RAW}/tensile_test_data/S355_data_tto_pmdco3.ttl"
"""The S355 PMDco3/TTO graph at the pinned commit."""

# NON-SEMANTIC, for the original graph only, which does not link per-test
# curves: a declared assumption about where the curves are and what their
# columns mean. Not the profile's default; pass it explicitly as
# ``curve_locator=CURVE_LOCATOR.as_dict()``. The slices link the curves in
# their curve tables (:data:`CURVE_COLUMNS`).
CURVE_LOCATOR = CurveLocator(
    url_template=f"{_TTO_RAW}/tensile_test_data/primary_data/{{specimen_id}}.csv",
    columns={
        "time": ColumnSpec(unit="s", header="Time"),
        "crosshead": ColumnSpec(unit="mm", header="Crosshead separation"),
        "force": ColumnSpec(unit="kN", header="Load"),
        "strain": ColumnSpec(unit="%", header="Extensometer elongation"),
    },
)

_PREFIXES = """\
PREFIX obo:   <http://purl.obolibrary.org/obo/>
PREFIX pmdco: <https://w3id.org/pmd/co/>
PREFIX tto:   <https://w3id.org/pmd/tto/>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
"""

SPECIMENS = SelectQuery(
    query=_PREFIXES
    + """
SELECT DISTINCT ?process ?specimen_id ?orientation
WHERE {
  ?process a pmdco:PMD_0000974 ;             # tensile testing process
           obo:OBI_0000293 ?test_piece .     # has specified input
  ?identifier obo:IAO_0000219 ?test_piece ;  # denotes
              obo:OBI_0001938 ?identifier_value .
  ?identifier_value pmdco:PMD_0000006 ?specimen_id .
  OPTIONAL {
    ?test_piece pmdco:PMD_0020005 ?material .
    ?material obo:RO_0000086 ?orientation_quality .   # has quality
    ?orientation_quality a obo:BFO_0000019 ;          # quality (generic)
                         obo:OBI_0001938 ?orientation_value .
    ?orientation_value pmdco:PMD_0000006 ?orientation .
  }
}
""",
    columns=SPECIMEN_COLUMNS,
)

QUANTITIES = SelectQuery(
    query=_PREFIXES
    + """
SELECT DISTINCT ?process ?quantity ?individual ?value ?unit
WHERE {
  ?process a pmdco:PMD_0000974 .
  # Classes alone are ambiguous (L0 and the extensometer gauge length share
  # TTO_0000028; Fm and the load-cell force range share TTO_0000023), so every
  # quantity is also constrained by its bearer.
  {
    # Qualities of the test piece (specified input) before the test.
    VALUES (?class ?quantity) {
      (tto:TTO_0000026 "S0")
      (tto:TTO_0000012 "s")
      (tto:TTO_0000028 "L0")
    }
    ?process obo:OBI_0000293 ?test_piece .
    ?test_piece obo:RO_0000086 ?individual .
  }
  UNION
  {
    # Qualities of the test piece after the test (a specified output).
    VALUES (?class ?quantity) {
      (tto:TTO_0000018 "Lu")
      (tto:TTO_0000024 "Su")
    }
    ?process obo:OBI_0000299 ?broken_piece .
    ?broken_piece obo:RO_0000086 ?individual .
  }
  UNION
  {
    # Lab-reported results: qualities of the test piece's material.
    VALUES (?class ?quantity) {
      (pmdco:PMD_0000618 "E")
      (tto:TTO_0000047   "Rp0.2")
      (tto:TTO_0000059   "ReH")
      (tto:TTO_0000022   "ReL")
      (tto:TTO_0000053   "Rm")
      (tto:TTO_0000023   "Fm")
      (tto:TTO_0000033   "A")
      (tto:TTO_0000038   "Z")
    }
    ?process obo:OBI_0000293 ?test_piece .
    ?test_piece pmdco:PMD_0020005 ?material .
    ?material obo:RO_0000086 ?individual .
  }
  UNION
  {
    # Process attributes of the test (test-control set points).
    VALUES (?class ?quantity) {
      (tto:TTO_0000058 "eps_sw")
    }
    ?process pmdco:PMD_0000009 ?individual .   # has process attribute
  }
  ?individual a ?class .
  ?process obo:OBI_0000299 ?datum .          # has specified output
  ?datum a obo:IAO_0000109 ;                 # measurement datum
         obo:IAO_0000136 ?individual ;       # is about
         obo:OBI_0001938 ?value_spec .       # has value specification
  ?value_spec pmdco:PMD_0000006 ?value .
  OPTIONAL { ?value_spec obo:IAO_0000039 ?unit }   # has measurement unit label
}
""",
    columns=QUANTITY_COLUMNS,
)

NOTES = SelectQuery(
    query=_PREFIXES
    + """
SELECT DISTINCT ?process ?note
WHERE {
  ?process a pmdco:PMD_0000974 .
  ?note_entity obo:IAO_0000219 ?process ;
               rdfs:label ?label ;
               obo:OBI_0001938 ?note_value .
  # NON-SEMANTIC: notes are told apart from the process identifier only by
  # the instance label.
  FILTER(STR(?label) = "process note")
  ?note_value pmdco:PMD_0000006 ?note .
}
""",
    columns=NOTE_COLUMNS,
)

CURVE_COLUMNS = SelectQuery(
    query=_PREFIXES
    + """PREFIX csvw:  <http://www.w3.org/ns/csvw#>
PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT ?process ?url ?role ?column_number ?unit ?delimiter ?header_row_count ?decimal_char
WHERE {
  VALUES (?class ?role) {
    (obo:BFO_0000202   "time")        # temporal interval
    (tto:TTO_0000013   "crosshead")   # crosshead separation
    (pmdco:PMD_0020200 "force")       # force
    (pmdco:PMD_0060011 "strain")      # percentage extension
  }
  {
    # Position of each column in the csvw:column RDF list (1-based).
    SELECT ?process ?table ?url ?column (COUNT(DISTINCT ?position) AS ?column_number)
    WHERE {
      ?process a pmdco:PMD_0000974 ;                # tensile testing process
               obo:OBI_0000299 ?table .             # has specified output
      ?table a csvw:Table, pmdco:PMD_0000992 ;      # time series
             csvw:url ?url ;
             csvw:tableSchema ?schema .
      ?schema csvw:column/rdf:rest* ?position .
      ?position rdf:rest* ?cell .
      ?cell rdf:first ?column .
    }
    GROUP BY ?process ?table ?url ?column
  }
  ?column obo:IAO_0000136 ?entity .                 # is about
  ?entity a ?class .
  OPTIONAL { ?column obo:IAO_0000039 ?unit }        # has measurement unit label
  OPTIONAL {
    ?table csvw:dialect ?dialect .
    OPTIONAL { ?dialect csvw:delimiter ?delimiter }
    OPTIONAL { ?dialect csvw:headerRowCount ?header_row_count }
  }
  OPTIONAL { ?column csvw:datatype/csvw:format/csvw:decimalChar ?decimal_char }
}
""",
    columns=CURVE_COLUMN_COLUMNS + CURVE_DIALECT_COLUMNS,
)
"""Curve columns from the per-test CSVW tables of the slices (D8.3, query of the T4 report).

Units are QUDT IRIs (``unit:SEC``, ``unit:MilliM``, ``unit:KiloN``,
``unit:PERCENT``), mapped through the shared unit table. Column numbers are
1-based list positions.
"""

CONFORMANCE_ASKS = {
    "tensile testing process with test-piece identifier": _PREFIXES
    + """
ASK {
  ?process a pmdco:PMD_0000974 ;
           obo:OBI_0000293 ?test_piece .
  ?identifier obo:IAO_0000219 ?test_piece ;
              obo:OBI_0001938 ?identifier_value .
  ?identifier_value pmdco:PMD_0000006 ?specimen_id .
}
""",
    "original cross-section S0 of the test piece, measured with unit": _PREFIXES
    + """
ASK {
  ?process a pmdco:PMD_0000974 ;
           obo:OBI_0000293 ?test_piece ;
           obo:OBI_0000299 ?datum .
  ?test_piece obo:RO_0000086 ?area .
  ?area a tto:TTO_0000026 .
  ?datum a obo:IAO_0000109 ;
         obo:IAO_0000136 ?area ;
         obo:OBI_0001938 ?value_spec .
  ?value_spec pmdco:PMD_0000006 ?value ;
              obo:IAO_0000039 ?unit .
}
""",
    "orientation quality of the test piece's material": _PREFIXES
    + """
ASK {
  ?process a pmdco:PMD_0000974 ;
           obo:OBI_0000293 ?test_piece .
  ?test_piece pmdco:PMD_0020005 ?material .
  ?material obo:RO_0000086 ?quality .
  ?quality a obo:BFO_0000019 ;
           obo:OBI_0001938 ?value_spec .
  ?value_spec pmdco:PMD_0000006 ?orientation .
}
""",
}

UNIT_ALIASES = {
    # Non-canonical unit IRIs observed in the PMDco3 S355 graph (reported
    # values only); not in the QUDT unit vocabulary.
    QUDT_UNIT + "MegaPa": QUDT_UNIT + "MegaPA",
    QUDT_UNIT + "GigaPa": QUDT_UNIT + "GigaPA",
}

# Every PMDco3/TTO/OBO IRI used above, with its rdfs:label from the pinned
# ontology files at TTO_COMMIT: the released ``tto-full.ttl`` (repository root)
# and ``src/ontology/tto-full.ttl``; OBO and PMDco labels agree with the PMDco
# mirror ``src/ontology/mirror/pmdco.owl``. Columns: IRI, English label, note.
# Extracted with .notes/tensile-workflow/evidence/T3/extract_labels.py and
# inspect_terms.py (not committed).
IRI_LABELS: tuple[tuple[str, str, str], ...] = (
    ("pmdco:PMD_0000974", "tensile testing process", ""),
    ("pmdco:PMD_0000006", "has value", ""),
    ("pmdco:PMD_0000618", "elastic modulus", "reported E"),
    ("pmdco:PMD_0000009", "has process attribute", "test -> rate-switch point"),
    ("pmdco:PMD_0020005", "porosity", "owl:Class in the pinned files, used as test piece -> material property in the data"),
    ("obo:OBI_0000293", "has specified input", ""),
    ("obo:OBI_0000299", "has specified output", ""),
    ("obo:OBI_0001938", "has value specification", ""),
    ("obo:IAO_0000219", "denotes", ""),
    ("obo:IAO_0000109", "measurement datum", ""),
    ("obo:IAO_0000136", "is about", ""),
    ("obo:IAO_0000039", "has measurement unit label", ""),
    ("obo:RO_0000086", "has quality", ""),
    ("obo:BFO_0000019", "quality", "orientation is typed only with this generic class"),
    ("tto:TTO_0000026", "original cross-sectional area of the parallel length", "S0"),
    ("tto:TTO_0000012", "cross section area", "s1..s3; deprecated in src/ontology/tto-full.ttl"),
    ("tto:TTO_0000028", "original gauge length", "L0 (also the extensometer gauge length); deprecated in src"),
    ("tto:TTO_0000018", "final gauge length after fracture", "Lu; deprecated in src"),
    ("tto:TTO_0000024", "minimum cross-sectional area after fracture", "Su"),
    ("tto:TTO_0000047", "proof strength plastic extension rp02", "reported Rp0.2"),
    ("tto:TTO_0000059", "upper yield strength", "reported ReH; deprecated in src"),
    ("tto:TTO_0000022", "lower yield strength", "reported ReL; deprecated in src"),
    ("tto:TTO_0000053", "tensile strength", "reported Rm"),
    ("tto:TTO_0000023", "maximum force", "reported Fm"),
    ("tto:TTO_0000033", "percentage elongation after fracture", "reported A; deprecated in src"),
    ("tto:TTO_0000038", "percentage reduction of area", "reported Z; deprecated in src"),
    ("tto:TTO_0000058", "transition point testing rate", "strain at the test-speed switch (diagnostic)"),
    # Curve tables of the slices (CURVE_COLUMNS; labels as in the T3 proposal, handoff D8.6).
    ("pmdco:PMD_0000992", "time series", "type of the per-test curve tables (slices only)"),
    ("obo:BFO_0000202", "temporal interval", "what the time column is about"),
    ("tto:TTO_0000013", "crosshead separation", "what the crosshead column is about"),
    ("pmdco:PMD_0020200", "force", "what the force column is about"),
    ("pmdco:PMD_0060011", "percentage extension", "what the strain column is about; not in the released tto-full"),
)

PROFILE = OntologyProfile(
    profile_id=PROFILE_ID,
    description=(
        "PMDco 3.x + TTO S355 mapping; specimens, geometry and reported values "
        "from the graph; curves from the per-test CSVW curve tables of the "
        "demonstrator's slices. The original graph has none and needs the "
        "explicit, NON-SEMANTIC CURVE_LOCATOR."
    ),
    conformance_asks=CONFORMANCE_ASKS,
    specimens=SPECIMENS,
    quantities=QUANTITIES,
    notes=NOTES,
    curve_columns=CURVE_COLUMNS,
    unit_aliases=UNIT_ALIASES,
    accepts_curve_locator=True,
)
