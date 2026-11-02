"""Credential hygiene of the Ontodocker access path (offline, stubbed praeco client).

The stub answers SPARQL from the local pmdco2 Turtle files, named like the
Ontodocker datasets of the demonstrator.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pandas as pd
import pyiron_workflow as pwf
import pytest
import rdflib
from conftest import PMDCO2_ORIENTATION_FILES

from tensile_workflow import acquisition, exploration, sources

TOKEN = "DUMMY_TOKEN_XYZ"
ADDRESS = "ontodocker.example.invalid"
DATASETS = {f"pmdco2_tto_example_{o}": path for o, path in PMDCO2_ORIENTATION_FILES.items()}
CSV = b"Time;Crosshead separation;Load;Extensometer elongation\n(sec);(mm);(kN);(%)\n0,0;0,0;0,0;0,0\n0,1;0,1;1,0;0,1\n"


class _FakeSparql:
    def __init__(self):
        self.graphs = {}

    def _graph(self, dataset):
        if dataset not in self.graphs:
            graph = rdflib.Graph()
            graph.parse(DATASETS[dataset])
            self.graphs[dataset] = graph
        return self.graphs[dataset]

    def query_df(self, dataset, query, columns):
        rows = [
            [None if row.asdict().get(c) is None else str(row.asdict()[c]) for c in columns]
            for row in self._graph(dataset).query(query)
        ]
        return pd.DataFrame(rows, columns=columns)

    def query_raw(self, dataset, query, *, accept="application/sparql-results+json"):
        answer = bool(self._graph(dataset).query(query).askAnswer)
        return json.dumps({"head": {}, "boolean": answer})


class _FakeOntodockerClient:
    instances: list = []

    def __init__(self, address, *, token=None, **kwargs):
        self.address = address
        self.token = token
        self.sparql = _FakeSparql()
        _FakeOntodockerClient.instances.append(self)


@pytest.fixture
def ontodocker_source(monkeypatch):
    monkeypatch.setenv("ONTODOCKER_ADDRESS", ADDRESS)
    monkeypatch.setenv("ONTODOCKER_TOKEN", TOKEN)
    _FakeOntodockerClient.instances = []
    monkeypatch.setattr(sources, "OntodockerClient", _FakeOntodockerClient)
    monkeypatch.setattr(sources, "fetch_bytes", lambda url: (CSV, hashlib.sha256(CSV).hexdigest()))
    return {"kind": "ontodocker", "datasets": list(DATASETS)}


def _text_of(value) -> str:
    """Everything a node output can expose: repr, and field values recursively."""
    parts = [repr(value)]
    if dataclasses.is_dataclass(value):
        parts += [_text_of(getattr(value, f.name)) for f in dataclasses.fields(value)]
    return "\n".join(parts)


def test_no_node_output_contains_the_token(ontodocker_source):
    common = {"profile_id": "pmdco2-tto", "source": ontodocker_source}
    runs = {
        "conformance": pwf.node(exploration.check_conformance, "c").run(**common),
        "specimens": pwf.node(exploration.list_specimens, "s").run(**common),
        "record": pwf.node(acquisition.get_specimen_record, "r").run(**common, specimen_id="Zx1"),
        "reported": pwf.node(acquisition.get_reported_properties, "p").run(
            **common, specimen_id="Zx1"
        ),
    }
    runs["curve"] = pwf.node(acquisition.fetch_curve, "f").run(record=runs["record"].outputs.record)

    for name, run in runs.items():
        assert run.status == "finished", (name, run.exception)
        for label, value in run.outputs.items():
            assert TOKEN not in _text_of(value), (name, label)
        assert TOKEN not in repr(run.result), name

    # The token was read from the environment inside the access path ...
    assert _FakeOntodockerClient.instances
    assert all(c.token == TOKEN and c.address == ADDRESS for c in _FakeOntodockerClient.instances)
    # ... and the Ontodocker path gives the same record as the local files.
    turtle = {"kind": "turtle", "locations": [str(p) for p in DATASETS.values()]}
    assert runs["record"].outputs.record == acquisition.build_specimen_record(
        "pmdco2-tto", turtle, "Zx1"
    )
    assert sorted(runs["specimens"].outputs.specimen_ids) == sorted(
        exploration.specimen_table("pmdco2-tto", turtle)["specimen_id"]
    )


def test_missing_address_fails_without_echoing_the_token(monkeypatch):
    monkeypatch.delenv("ONTODOCKER_ADDRESS", raising=False)
    monkeypatch.setenv("ONTODOCKER_TOKEN", TOKEN)
    with pytest.raises(sources.SourceError) as error:
        exploration.specimen_table(
            "pmdco2-tto", {"kind": "ontodocker", "datasets": ["pmdco2_tto_example_parallel"]}
        )
    assert TOKEN not in str(error.value)
