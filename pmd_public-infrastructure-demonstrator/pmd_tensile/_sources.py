"""Graph sources and the raw-curve download.

A *source* is a JSON-able dict:

- ``{"kind": "turtle", "locations": [<path or http(s) URL>, ...]}``: each
  location is parsed with rdflib and queried separately.
- ``{"kind": "ontodocker", "datasets": [<dataset name>, ...]}``: each dataset
  is queried through praeco's ``OntodockerClient``. The address and the
  optional token are read from the environment variables
  ``ONTODOCKER_ADDRESS`` and ``ONTODOCKER_TOKEN`` inside this module only; they
  are never node inputs or outputs, so they cannot enter recipes, run data or
  knowledge graphs.

Parts (files or datasets) are queried one by one and the rows concatenated;
they are never merged into one graph, because some files reuse node labels
(see ``profiles.pmdco2_tto``).

Raw curves and remote Turtle files are fetched with a separate, token-free
praeco ``HttpClient`` bound to the URL's host, never with an authenticated
service client. Downloads stay in memory; nothing is written to disk.

The module holds no nodes. It is private so that sim_atlas does not catalogue
its plain functions as workflow nodes; notebooks reach the functions they need
through :mod:`pmd_tensile.demo`.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import os
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import rdflib
from praeco import HttpClient, OntodockerClient
from rdflib.util import guess_format

from pmd_tensile.contract import (
    CURVE_ROLE_KINDS,
    CURVE_ROLES,
    ColumnSpec,
    Curve,
    UnitError,
    to_contract_unit,
)

ONTODOCKER_ADDRESS_ENV = "ONTODOCKER_ADDRESS"
ONTODOCKER_TOKEN_ENV = "ONTODOCKER_TOKEN"

_SOURCE_KEYS = {"turtle": "locations", "ontodocker": "datasets"}


class SourceError(ValueError):
    """An invalid source description or a source that cannot be accessed."""


# --------------------------------------------------------------------------
# Source descriptions
# --------------------------------------------------------------------------


def reject_url_credentials(url: str) -> None:
    """Raise :class:`SourceError` if a URL has userinfo or a query string.

    Either can carry a credential, and these URLs are recorded in knowledge
    graphs; ``requests`` would also send userinfo as Basic auth. The message
    names only the host, so that it does not repeat a secret.
    """
    parts = urlsplit(url)
    if "@" in parts.netloc:
        raise SourceError(f"URL for host {parts.hostname!r} must not contain userinfo")
    if parts.query:
        raise SourceError(f"URL for host {parts.hostname!r} must not contain a query string")


def validate_source(source: Mapping) -> tuple[str, list[str]]:
    """Check a source dict and return ``(kind, parts)``.

    A source is a node input and ends up in the knowledge graph. Only the
    documented keys are accepted, and URL locations with userinfo or a query
    string are refused (:func:`reject_url_credentials`). This closes the
    regular places for credentials; it cannot recognise a secret elsewhere in
    a path or dataset name.
    """
    if not isinstance(source, Mapping):
        raise SourceError(f"source must be a dict, got {type(source).__name__}")
    kind = source.get("kind")
    if kind not in _SOURCE_KEYS:
        raise SourceError(f"source kind must be one of {sorted(_SOURCE_KEYS)}, got {kind!r}")
    parts_key = _SOURCE_KEYS[kind]
    unknown = set(source) - {"kind", parts_key}
    if unknown:
        raise SourceError(f"unknown keys in {kind} source: {sorted(unknown)}")
    parts = source.get(parts_key)
    if (
        not isinstance(parts, Sequence)
        or isinstance(parts, str)
        or not parts
        or not all(isinstance(p, str) and p.strip() for p in parts)
    ):
        raise SourceError(f"{kind} source needs a non-empty list of strings in {parts_key!r}")
    if kind == "turtle":
        for location in parts:
            if urlsplit(location).netloc:  # A local path may contain '?' or '@'.
                reject_url_credentials(location)
    return kind, [str(p) for p in parts]


def select(source: Mapping, query: str, columns: Sequence[str]) -> pd.DataFrame:
    """Run a SPARQL SELECT on every part of a source and concatenate the rows.

    Returns a DataFrame with the requested ``columns`` (string values, ``None``
    when unbound) plus a ``source_part`` column naming the file or dataset.
    """
    kind, parts = validate_source(source)
    columns = list(columns)
    frames = []
    if kind == "turtle":
        for location in parts:
            frame = _turtle_select(location, query, columns)
            frame["source_part"] = location
            frames.append(frame)
    else:
        client = _ontodocker_client()
        for dataset in parts:
            frame = client.sparql.query_df(dataset, query, columns)
            frame = frame.astype(object).where(frame.notna(), None)
            frame["source_part"] = dataset
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def ask(source: Mapping, query: str) -> dict[str, bool]:
    """Run a SPARQL ASK on every part of a source; returns part -> answer."""
    kind, parts = validate_source(source)
    answers: dict[str, bool] = {}
    if kind == "turtle":
        for location in parts:
            answers[location] = bool(_query_graph(_load_graph(location), query).askAnswer)
    else:
        client = _ontodocker_client()
        for dataset in parts:
            payload = json.loads(client.sparql.query_raw(dataset, query))
            if not isinstance(payload.get("boolean"), bool):
                raise SourceError(f"ASK on dataset {dataset!r} returned no boolean")
            answers[dataset] = payload["boolean"]
    return answers


# --------------------------------------------------------------------------
# Turtle files and URLs (rdflib)
# --------------------------------------------------------------------------

# pyiron_workflow runs nodes in a thread pool, but rdflib's SPARQL grammar is
# module-level pyparsing state, which is not thread-safe: one overlap can break
# a grammar element for the life of the process. The pyparsing maintainer says
# a lock belongs in the calling code
# (https://github.com/pyparsing/pyparsing/issues/89).
_SPARQL_QUERY_LOCK = threading.Lock()


def _query_graph(graph: rdflib.Graph, query: str) -> rdflib.query.Result:
    """Run ``Graph.query`` under the lock; the graph must already be loaded."""
    with _SPARQL_QUERY_LOCK:
        return graph.query(query)


def _is_url(location: str) -> bool:
    return urlsplit(location).scheme in ("http", "https")


def _load_graph(location: str) -> rdflib.Graph:
    """Parse a local file or http(s) URL, cached in memory per location.

    Local files are re-parsed when their size or modification time changes.
    """
    if _is_url(location):
        stamp: tuple = ()
    else:
        path = Path(location)
        if not path.is_file():
            raise SourceError(f"Turtle location is not a file: {location}")
        info = path.stat()
        stamp = (str(path.resolve()), info.st_mtime_ns, info.st_size)
    return _parse_graph(location, stamp)


@functools.lru_cache(maxsize=32)
def _parse_graph(location: str, stamp: tuple) -> rdflib.Graph:
    graph = rdflib.Graph()
    if _is_url(location):
        data, _ = fetch_bytes(location)
        graph.parse(data=data, format=guess_format(urlsplit(location).path) or "turtle")
    else:
        graph.parse(location, format=guess_format(location) or "turtle")
    return graph


def clear_graph_cache() -> None:
    """Drop all parsed graphs held in memory."""
    _parse_graph.cache_clear()


def _turtle_select(location: str, query: str, columns: list[str]) -> pd.DataFrame:
    result = _query_graph(_load_graph(location), query)
    rows = []
    for row in result:
        bindings = row.asdict()
        rows.append([_term_text(bindings.get(column)) for column in columns])
    return pd.DataFrame(rows, columns=columns, dtype=object)


def _term_text(term) -> str | None:
    """Lexical form of an RDF term, matching what SPARQL JSON results carry."""
    return None if term is None else str(term)


# --------------------------------------------------------------------------
# Ontodocker (praeco)
# --------------------------------------------------------------------------


def _ontodocker_client() -> OntodockerClient:
    """Build an Ontodocker client from the environment (never from node inputs)."""
    address = os.environ.get(ONTODOCKER_ADDRESS_ENV, "").strip()
    if not address:
        raise SourceError(f"set {ONTODOCKER_ADDRESS_ENV} to use an ontodocker source")
    token = os.environ.get(ONTODOCKER_TOKEN_ENV) or None
    return OntodockerClient(address, token=token)


# --------------------------------------------------------------------------
# Token-free downloads and curve parsing
# --------------------------------------------------------------------------


def fetch_bytes(url: str) -> tuple[bytes, str]:
    """GET a public http(s) URL into memory; returns ``(data, sha256_hex)``.

    Uses a fresh praeco ``HttpClient`` bound to the URL's host and without a
    token, and refuses URLs with userinfo or a query string
    (:func:`reject_url_credentials`), so no credential is sent along by these
    means.
    """
    reject_url_credentials(url)
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise SourceError(f"not an absolute http(s) URL: {url!r}")
    client = HttpClient(f"{parts.scheme}://{parts.netloc}", token=None)
    response = client.request("GET", url)
    try:
        response.raise_for_status()
        data = response.content
    finally:
        response.close()
    return data, hashlib.sha256(data).hexdigest()


# Layout of the S355 curve CSVs: ';' delimiter, ',' decimal mark, one header
# row with channel names, one row with units in parentheses, then data rows.
# A declared assumption for the pmdco2 graphs and the original PMDco3 graph,
# which state no dialect. The PMDco3 slices declare it in their curve tables
# (csvw:Dialect, csvw:NumericFormat); acquisition checks that declaration
# against these values (``pmd_tensile._acquisition``).
CSV_DELIMITER = ";"
CSV_DECIMAL = ","
CSV_HEADER_ROWS = 2

CSV_UNIT_TOKENS: dict[str, str] = {
    "sec": "s",
    "s": "s",
    "mm": "mm",
    "kN": "kN",
    "N": "N",
    "%": "%",
}
"""Unit-row tokens of the curve CSV (without parentheses) -> unit symbols."""


def parse_curve_csv(data: bytes, columns: Mapping[str, ColumnSpec], *, source_url: str) -> Curve:
    """Parse raw curve CSV bytes into a :class:`Curve` in contract units.

    ``columns`` maps each curve role to its column (by index or header name)
    and declared source unit. The CSV's unit row must agree with the declared
    unit; any disagreement or unknown unit raises ``UnitError``. The SHA-256
    is computed over exactly ``data``.
    """
    sha256 = hashlib.sha256(data).hexdigest()
    text = _decode(data)
    lines = text.splitlines()
    if len(lines) <= CSV_HEADER_ROWS:
        raise ValueError(f"curve CSV has no data rows: {source_url}")
    names = _split_header(lines[0])
    units = _split_header(lines[1])
    table = pd.read_csv(
        io.StringIO("\n".join(lines[CSV_HEADER_ROWS:])),
        sep=CSV_DELIMITER,
        decimal=CSV_DECIMAL,
        header=None,
        dtype=float,
    )

    channels = {}
    for role in CURVE_ROLES:
        spec = columns[role]
        index = _resolve_column(role, spec, names, source_url)
        if index >= table.shape[1] or index >= len(units):
            raise ValueError(f"column {index} for {role!r} is missing in {source_url}")
        _check_unit_row(role, spec.unit, units[index], source_url)
        values = table.iloc[:, index].to_numpy(dtype=float)
        channels[role] = to_contract_unit(values, spec.unit, CURVE_ROLE_KINDS[role])

    return Curve(
        time_s=channels["time"],
        crosshead_mm=channels["crosshead"],
        force_n=channels["force"],
        strain=channels["strain"],
        source_url=source_url,
        sha256=sha256,
    )


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _split_header(line: str) -> list[str]:
    return [cell.strip().strip('"').strip() for cell in line.split(CSV_DELIMITER)]


def _resolve_column(role: str, spec: ColumnSpec, names: list[str], source_url: str) -> int:
    if spec.index is not None:
        index = spec.index
        if spec.header is not None and (index >= len(names) or names[index] != spec.header):
            raise ValueError(
                f"column {index} of {source_url} is not {spec.header!r} (role {role!r})"
            )
        return index
    matches = [i for i, name in enumerate(names) if name == spec.header]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one column named {spec.header!r} for {role!r} in "
            f"{source_url}, found {len(matches)} (header: {names})"
        )
    return matches[0]


def _check_unit_row(role: str, declared: str, token: str, source_url: str) -> None:
    bare = token.strip().removeprefix("(").removesuffix(")").strip()
    if bare not in CSV_UNIT_TOKENS:
        raise UnitError(f"unknown unit {token!r} in the unit row of {source_url} ({role!r})")
    if CSV_UNIT_TOKENS[bare] != declared:
        raise UnitError(
            f"unit mismatch for {role!r} in {source_url}: declared {declared!r}, "
            f"CSV unit row says {token!r}"
        )
