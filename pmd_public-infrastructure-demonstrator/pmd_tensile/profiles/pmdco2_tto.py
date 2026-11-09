"""Profile ``pmdco2-tto``: PMDco 2.x with the TTO v2-era mapping of the S355 data.

Graphs: the demonstrator's ``datasets/pmdco2_tto_example_*.ttl`` files and the
Ontodocker datasets ``pmdco2_tto_example_{parallel,perpendicular,diagonal}``.

How the concepts are expressed (observed in the local Turtle files):

- A test is a ``pmd:TensileTest`` whose ``pmd:input`` is a ``pmd:TestPiece``
  carrying a ``pmd:TestPieceName`` (``pmd:value`` e.g. ``"Zx1"``); that name is
  the cross-profile ``specimen_id``.
- Characteristics hang off the test via ``pmd:characteristic`` and carry
  ``pmd:value`` (a plain literal) and ``pmd:unit`` (a QUDT unit IRI).
- Orientation: the ``pmd:MaterialRelated`` characteristic.
- Lab notes: ``pmd:Note`` characteristics.
- Test-speed switch point: the ``pmd:TransitionPointTestingRate``
  characteristic (``..._transitionPoint``, unit ``PERCENT``; e.g. 1.6 % for
  S355-1), queried as ``eps_sw`` (strain kind, so it arrives as a ratio).
- Curves: CSVW. Test -> ``pmd:Dataset`` -> ``pmd:resource`` -> ``csvw:Table``
  with ``csvw:url`` and a ``csvw:tableSchema`` whose ``csvw:column`` RDF list
  holds one column node per channel, typed ``pmd:Duration``,
  ``pmd:CrossheadSeparation``, ``pmd:Force`` or ``pmd:PercentageExtension``
  with a ``pmd:unit``. The column position is the list position.

**Naming convention (pmdco2 only):** S0, Su and the measurements s1..sn all
have class ``pmd:CrossSectionArea`` and differ only by the local name of the
individual (``..._crossSectionArea_S0``, ``..._Su``, ``..._s1`` ...). The
``quantities`` query below encodes that convention; a cross-section individual
that matches none of the patterns is returned as ``"UNRECOGNIZED"`` and makes
acquisition fail rather than be silently ignored. Nothing outside this module
relies on IRI local names.

Lower yield strength (ReL) is not described in these graphs, so the profile
does not query it and the reported ReL is ``None``.

The pmdco2 files write some nodes as pseudo-IRIs ``<_:tNNN>`` (not blank
nodes). Their labels repeat across files, so the files must be queried one by
one and never merged into one graph; ``pmd_tensile._sources`` does that.

The queries adapt patterns of ``query_collection.S355queries`` (notably the
CSVW column-list walk of ``csv_columns``) but are self-contained, so the
package does not depend on that notebook helper module.
"""

from __future__ import annotations

from pmd_tensile.profiles.base import (
    CURVE_COLUMN_COLUMNS,
    NOTE_COLUMNS,
    QUANTITY_COLUMNS,
    SPECIMEN_COLUMNS,
    OntologyProfile,
    SelectQuery,
)
from pmd_tensile.profiles._units import QUDT_UNIT

PROFILE_ID = "pmdco2-tto"

_PREFIXES = """\
PREFIX pmd:  <https://w3id.org/pmd/co/>
PREFIX csvw: <http://www.w3.org/ns/csvw#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""

SPECIMENS = SelectQuery(
    query=_PREFIXES
    + """
SELECT DISTINCT ?process ?specimen_id ?orientation
WHERE {
  ?process a pmd:TensileTest ;
           pmd:input ?test_piece .
  ?test_piece a pmd:TestPiece ;
              pmd:characteristic ?name .
  ?name a pmd:TestPieceName ;
        pmd:value ?specimen_id .
  OPTIONAL {
    ?process pmd:characteristic ?material_related .
    ?material_related a pmd:MaterialRelated ;
                      pmd:value ?orientation .
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
  ?process a pmd:TensileTest ;
           pmd:characteristic ?individual .
  {
    # pmdco2 naming convention: S0, Su and s1..sn share pmd:CrossSectionArea.
    ?individual a pmd:CrossSectionArea .
    BIND(STR(?individual) AS ?iri)
    BIND(
      IF(STRENDS(?iri, "_crossSectionArea_S0"), "S0",
      IF(STRENDS(?iri, "_crossSectionArea_Su"), "Su",
      IF(REGEX(?iri, "_crossSectionArea_s[0-9]+$"), "s",
      "UNRECOGNIZED"))) AS ?quantity)
  }
  UNION
  {
    VALUES (?class ?quantity) {
      (pmd:OriginalGaugeLength               "L0")
      (pmd:FinalGaugeLengthAfterFracture     "Lu")
      (pmd:TransitionPointTestingRate        "eps_sw")
      (pmd:ModulusOfElasticity               "E")
      (pmd:Rp02                              "Rp0.2")
      (pmd:UpperYieldStrength                "ReH")
      (pmd:TensileStrength                   "Rm")
      (pmd:MaximumForce                      "Fm")
      (pmd:PercentageElongationAfterFracture "A")
      (pmd:PercentageReductionOfArea         "Z")
    }
    ?individual a ?class .
  }
  ?individual pmd:value ?value .
  OPTIONAL { ?individual pmd:unit ?unit }
}
""",
    columns=QUANTITY_COLUMNS,
)

NOTES = SelectQuery(
    query=_PREFIXES
    + """
SELECT DISTINCT ?process ?note
WHERE {
  ?process a pmd:TensileTest ;
           pmd:characteristic ?note_node .
  ?note_node a pmd:Note ;
             pmd:value ?note .
}
""",
    columns=NOTE_COLUMNS,
)

CURVE_COLUMNS = SelectQuery(
    query=_PREFIXES
    + """
SELECT ?process ?url ?role ?column_number ?unit
WHERE {
  VALUES (?class ?role) {
    (pmd:Duration            "time")
    (pmd:CrossheadSeparation "crosshead")
    (pmd:Force               "force")
    (pmd:PercentageExtension "strain")
  }
  {
    # Position of each column in the csvw:column RDF list (1-based).
    SELECT ?process ?url ?column (COUNT(DISTINCT ?position) AS ?column_number)
    WHERE {
      ?process a pmd:TensileTest ;
               pmd:characteristic ?dataset .
      ?dataset a pmd:Dataset ;
               pmd:resource ?table .
      ?table a csvw:Table ;
             csvw:url ?url ;
             csvw:tableSchema ?schema .
      ?schema csvw:column/rdf:rest* ?position .
      ?position rdf:rest* ?cell .
      ?cell rdf:first ?column .
    }
    GROUP BY ?process ?url ?column
  }
  ?column a ?class .
  OPTIONAL { ?column pmd:unit ?unit }
}
""",
    columns=CURVE_COLUMN_COLUMNS,
)

CONFORMANCE_ASKS = {
    "tensile test with test-piece name": _PREFIXES
    + """
ASK {
  ?process a pmd:TensileTest ;
           pmd:input ?test_piece .
  ?test_piece a pmd:TestPiece ;
              pmd:characteristic ?name .
  ?name a pmd:TestPieceName ;
        pmd:value ?specimen_id .
}
""",
    "cross-section area with value and unit": _PREFIXES
    + """
ASK {
  ?process a pmd:TensileTest ;
           pmd:characteristic ?area .
  ?area a pmd:CrossSectionArea ;
        pmd:value ?value ;
        pmd:unit ?unit .
}
""",
    "orientation (MaterialRelated)": _PREFIXES
    + """
ASK {
  ?process a pmd:TensileTest ;
           pmd:characteristic ?material_related .
  ?material_related a pmd:MaterialRelated ;
                    pmd:value ?orientation .
}
""",
    "CSVW curve table with column schema": _PREFIXES
    + """
ASK {
  ?process a pmd:TensileTest ;
           pmd:characteristic ?dataset .
  ?dataset a pmd:Dataset ;
           pmd:resource ?table .
  ?table a csvw:Table ;
         csvw:url ?url ;
         csvw:tableSchema ?schema .
  ?schema csvw:column ?list .
}
""",
}

UNIT_ALIASES = {
    # Non-canonical unit IRIs observed in the pmdco2 S355 graphs (reported
    # values only). They are not in the QUDT unit vocabulary; each evidently
    # means the canonical IRI it is mapped to.
    QUDT_UNIT + "kiloN": QUDT_UNIT + "KiloN",
    QUDT_UNIT + "MegaPa": QUDT_UNIT + "MegaPA",
    QUDT_UNIT + "GigaPa": QUDT_UNIT + "GigaPA",
}

PROFILE = OntologyProfile(
    profile_id=PROFILE_ID,
    description=(
        "PMDco 2.x + TTO (v2-era) S355 mapping; curves linked via CSVW; "
        "S0/Su/s1..sn told apart by IRI naming convention."
    ),
    conformance_asks=CONFORMANCE_ASKS,
    specimens=SPECIMENS,
    quantities=QUANTITIES,
    notes=NOTES,
    curve_columns=CURVE_COLUMNS,
    unit_aliases=UNIT_ALIASES,
)
