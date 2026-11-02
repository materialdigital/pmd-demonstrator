"""Declarative structure of an ontology profile.

A profile describes how one ontology (and data-modelling style) expresses the
concepts the workflow needs. It holds SPARQL text, the expected unit
vocabulary and any naming conventions, but no graph-access code: the generic
exploration and acquisition code runs the queries and turns the rows into the
neutral contract (``tensile_workflow.contract``).

Query result contracts (column names are fixed; profiles must bind them):

- ``specimens``: ``process``, ``specimen_id``, ``orientation``. One row per
  tensile test; ``specimen_id`` is the test-piece name.
- ``quantities``: ``process``, ``quantity``, ``individual``, ``value``,
  ``unit``. ``quantity`` is a neutral key from :data:`QUANTITY_KINDS`;
  ``unit`` is a unit IRI or unbound.
- ``notes`` (optional): ``process``, ``note``.
- ``curve_columns`` (semantic curve links): ``process``, ``url``, ``role``,
  ``column_number`` (1-based), ``unit``. ``role`` is one of
  ``contract.CURVE_ROLES``.

A profile whose graph does not link curves semantically declares a
:class:`CurveLocator` instead of ``curve_columns``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from urllib.parse import quote

from tensile_workflow.contract import CURVE_ROLES, ColumnSpec
from tensile_workflow.profiles.units import qudt_unit_symbol

QUANTITY_KINDS: dict[str, str] = {
    # specimen geometry (analysis inputs)
    "S0": "area",
    "s": "area",  # individual cross-section measurements s1..sn
    "Su": "area",
    "L0": "length",
    "Lu": "length",
    # test-control set point (diagnostic only): strain at the test-speed switch
    "eps_sw": "strain",
    # lab-reported results (comparison only)
    "E": "modulus",
    "Rp0.2": "stress",
    "ReH": "stress",
    "ReL": "stress",
    "Rm": "stress",
    "Fm": "force",
    "A": "percent",
    "Z": "percent",
}
"""Neutral quantity keys used in ``quantities`` queries, with their kind."""

SPECIMEN_COLUMNS = ("process", "specimen_id", "orientation")
QUANTITY_COLUMNS = ("process", "quantity", "individual", "value", "unit")
NOTE_COLUMNS = ("process", "note")
CURVE_COLUMN_COLUMNS = ("process", "url", "role", "column_number", "unit")


@dataclasses.dataclass(frozen=True)
class SelectQuery:
    """SPARQL SELECT text and the variable names it binds, in order."""

    query: str
    columns: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class CurveLocator:
    """NON-SEMANTIC curve locator for graphs that do not link the raw curves.

    This is an explicit, declared assumption, not knowledge taken from the
    graph: the curve URL is built from a URL template with ``{specimen_id}``,
    and the column semantics come from a declared header-name map
    (role -> header name and unit symbol). Any profile that uses it has a gap
    in its graph's description of the raw data.
    """

    url_template: str
    columns: Mapping[str, ColumnSpec]

    def __post_init__(self) -> None:
        if "{specimen_id}" not in self.url_template:
            raise ValueError("url_template must contain '{specimen_id}'")
        if set(self.columns) != set(CURVE_ROLES):
            raise ValueError(f"columns must have exactly the roles {CURVE_ROLES}")

    def url_for(self, specimen_id: str) -> str:
        """Build the curve URL for a specimen (the ID is URL-quoted)."""
        return self.url_template.format(specimen_id=quote(specimen_id, safe=""))

    def as_dict(self) -> dict:
        """JSON-able form, e.g. to pass the locator explicitly as a node input."""
        return {
            "url_template": self.url_template,
            "columns": {
                role: {"header": spec.header, "unit": spec.unit}
                for role, spec in self.columns.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> CurveLocator:
        """Inverse of :meth:`as_dict`; rejects unknown keys."""
        unknown = set(data) - {"url_template", "columns"}
        if unknown:
            raise ValueError(f"unknown curve locator keys: {sorted(unknown)}")
        columns = {
            role: ColumnSpec(unit=spec["unit"], header=spec["header"])
            for role, spec in data["columns"].items()
        }
        return cls(url_template=data["url_template"], columns=columns)


@dataclasses.dataclass(frozen=True)
class OntologyProfile:
    """Declarative description of how a graph expresses the needed concepts.

    ``unit_aliases`` maps unit IRIs that occur in the profile's data but are
    not canonical QUDT IRIs to the canonical QUDT IRI they evidently mean.
    The aliases are explicit and per profile; any other unknown unit IRI
    raises ``contract.UnitError``.
    """

    profile_id: str
    description: str
    conformance_asks: Mapping[str, str]
    specimens: SelectQuery
    quantities: SelectQuery
    notes: SelectQuery | None = None
    curve_columns: SelectQuery | None = None
    curve_locator: CurveLocator | None = None
    unit_aliases: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.curve_columns is None) == (self.curve_locator is None):
            raise ValueError(
                "a profile needs exactly one of curve_columns (semantic) "
                "or curve_locator (non-semantic)"
            )
        if not self.conformance_asks:
            raise ValueError("a profile needs at least one conformance ASK query")
        expected = {
            "specimens": SPECIMEN_COLUMNS,
            "quantities": QUANTITY_COLUMNS,
            "notes": NOTE_COLUMNS,
            "curve_columns": CURVE_COLUMN_COLUMNS,
        }
        for name, columns in expected.items():
            query = getattr(self, name)
            if query is not None and tuple(query.columns) != columns:
                raise ValueError(f"{name} query must bind {columns}, got {query.columns}")

    @property
    def links_curves_semantically(self) -> bool:
        """True if the graph itself links each test to its curve and columns."""
        return self.curve_columns is not None

    def unit_symbol(self, unit_iri: str) -> str:
        """Map a unit IRI from this profile's data to a neutral unit symbol."""
        return qudt_unit_symbol(unit_iri, aliases=self.unit_aliases)
