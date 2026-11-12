"""Tests of the publishing nodes and workflow against stand-ins for Ontodocker and the Dataportal (offline).

The stand-ins record every call; no request leaves the machine. A dummy token
checks that tokens are read from the environment and appear in no input,
output or knowledge graph.
"""

from __future__ import annotations

import hashlib
import types
from pathlib import Path

import pyiron_workflow as pwf
import pytest
from praeco.exceptions import HttpError

from pmd_tensile import sources
from pmd_tensile.nodes import publishing

DUMMY_TOKEN = "dummy-token-4f1c9e"
RECORD = {
    "name": "pmd-demonstrator-results", "title": "Computed tensile properties", "description": "Results.",
    "creators": ["A Person"], "license": "CC-BY-4.0", "keywords": ["tensile test"], "publication_date": "2026-11-10",
    "owner_org": "pmd", "private": False, "extras": {"pmd_profile": "pmdco3-tto"},
}
CONTENTS = {"results.nt": "<urn:a> <urn:b> <urn:c> .\n", "results.csv": "specimen_id;e_gpa\nZx3;200.08\n"}


class FakeOntodocker:
    def __init__(self, existing=()):
        self.existing, self.calls = list(existing), []
        self.datasets = types.SimpleNamespace(list=lambda: list(self.existing), create=self._create, upload_turtlefile=self._upload)

    def _create(self, name):
        self.calls.append(("create", name))
        self.existing.append(name)

    def _upload(self, name, path):
        self.calls.append(("upload", name, Path(path).name, Path(path).read_bytes()))


class FakeDataportal:
    def __init__(self, state=None, error_status=404):
        self.state, self.error_status, self.calls = state, error_status, []
        self.datasets = types.SimpleNamespace(show=self._show, create=self._create)
        self.assets = types.SimpleNamespace(upload_rdf=self._upload_rdf, upload=self._upload)

    def _show(self, name):
        if self.state is None:
            raise HttpError(method="GET", url=f"package_show?id={name}", status_code=self.error_status)
        return types.SimpleNamespace(raw={"state": self.state})

    def _create(self, metadata):
        self.calls.append(("create", metadata.to_payload()))
        return types.SimpleNamespace(name=metadata.name)

    def _upload_rdf(self, dataset, path, *, format, content_type):
        self.calls.append(("upload_rdf", dataset.name, path, format, content_type))

    def _upload(self, dataset, path, *, format):
        self.calls.append(("upload", dataset.name, path, format))


@pytest.fixture
def services(monkeypatch):
    ontodocker, dataportal = FakeOntodocker(), FakeDataportal()
    monkeypatch.setattr(sources, "ontodocker_client", lambda: ontodocker)
    monkeypatch.setattr(sources, "DataportalClient", lambda api_token=None: dataportal if api_token == DUMMY_TOKEN else None)
    monkeypatch.setenv(sources.DATAPORTAL_TOKEN_ENV, DUMMY_TOKEN)
    return ontodocker, dataportal


def test_files_are_written_once_and_never_overwritten(tmp_path):
    paths = publishing.write_publication_files(CONTENTS, str(tmp_path))
    assert [p.rsplit("/", 1)[-1] for p in paths] == sorted(CONTENTS)
    assert (tmp_path / "results.nt").read_text(encoding="utf-8") == CONTENTS["results.nt"]
    with pytest.raises(FileExistsError):
        publishing.write_publication_files(CONTENTS, str(tmp_path))
    with pytest.raises(ValueError, match="not a path"):
        publishing.write_publication_files({"../x.nt": ""}, str(tmp_path))
    with pytest.raises(FileNotFoundError):
        publishing.write_publication_files(CONTENTS, str(tmp_path / "missing"))


def test_ontodocker_dataset_is_created_and_never_merged(tmp_path, services):
    ontodocker, _ = services
    paths = publishing.write_publication_files(CONTENTS, str(tmp_path))
    assert publishing.publish_to_ontodocker("pmd_results", paths, "results.nt") == "pmd_results"
    assert ontodocker.calls == [("create", "pmd_results"), ("upload", "pmd_results", "results.ttl", CONTENTS["results.nt"].encode())]
    with pytest.raises(FileExistsError, match="merge"):
        publishing.publish_to_ontodocker("pmd_results", paths, "results.nt")
    with pytest.raises(LookupError):
        publishing.publish_to_ontodocker("other", paths, "missing.nt")


def test_dataportal_record_carries_the_reviewed_metadata_and_checksums(tmp_path, services):
    _, dataportal = services
    paths = publishing.write_publication_files(CONTENTS, str(tmp_path))
    assert publishing.publish_to_dataportal(RECORD, paths) == RECORD["name"]
    (kind, payload), *uploads = dataportal.calls
    assert kind == "create"
    assert (payload["name"], payload["title"], payload["license_id"], payload["owner_org"], payload["private"]) == (
        RECORD["name"], RECORD["title"], "CC-BY-4.0", "pmd", False)
    extras = {item["key"]: item["value"] for item in payload["extras"]}
    assert extras["pmd_profile"] == "pmdco3-tto"
    for path in paths:
        name = path.rsplit("/", 1)[-1]
        assert extras[f"sha256:{name}"] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    assert sorted(call[0] for call in uploads) == ["upload", "upload_rdf"]
    assert ("upload_rdf", RECORD["name"], str(tmp_path / "results.nt"), "nt", "application/n-triples") in uploads


@pytest.mark.parametrize("state", ["active", "draft"])
def test_dataportal_record_is_not_overwritten(tmp_path, services, state):
    _, dataportal = services
    dataportal.state = state
    with pytest.raises(FileExistsError, match=state):
        publishing.publish_to_dataportal(RECORD, [])
    assert dataportal.calls == []


def test_dataportal_name_of_a_deleted_record_is_reused(services):
    _, dataportal = services
    dataportal.state = "deleted"
    assert publishing.publish_to_dataportal(RECORD, []) == RECORD["name"]


def test_dataportal_errors_other_than_not_found_stop(services):
    _, dataportal = services
    dataportal.error_status = 403
    with pytest.raises(HttpError):
        publishing.publish_to_dataportal(RECORD, [])
    assert dataportal.calls == []


def test_dataportal_needs_a_token_and_a_complete_description(monkeypatch, services):
    with pytest.raises(ValueError, match="lacks"):
        publishing.publish_to_dataportal({"name": "x"}, [])
    monkeypatch.delenv(sources.DATAPORTAL_TOKEN_ENV)
    with pytest.raises(sources.SourceError, match=sources.DATAPORTAL_TOKEN_ENV):
        publishing.publish_to_dataportal(RECORD, [])


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:The hashes produced for directed graphs:UserWarning")
def test_publishing_workflow_runs_and_its_graph_holds_no_token(tmp_path, services):
    # ASCII contents: with non-ASCII contents the export fails upstream (see the xfail test below).
    from pmd_tensile.knowledge_graph import run_knowledge_graph, validate_knowledge_graph

    ontodocker, dataportal = services
    run = pwf.node(publishing.publishing_workflow).run(
        contents=CONTENTS, directory=str(tmp_path), ontodocker_dataset="pmd_results", graph_file="results.nt", record=RECORD
    )
    assert run.outputs.dataset == "pmd_results" and run.outputs.record_name == RECORD["name"]
    assert [call[0] for call in ontodocker.calls] == ["create", "upload"]
    graph = run_knowledge_graph(run)
    assert validate_knowledge_graph(graph)[0]
    assert DUMMY_TOKEN not in graph.serialize(format="nt")


def test_dataportal_creators_may_carry_an_orcid(tmp_path, services):
    _, dataportal = services
    record = dict(RECORD, creators=[{"name": "A Person", "orcid": "0000-0002-1825-0097"}, "B Person"])
    publishing.publish_to_dataportal(record, [])
    payload = dataportal.calls[0][1]
    assert "0000-0002-1825-0097" in str(payload["creator"]) and "B Person" in str(payload["creator"])


def test_files_are_checked_before_any_is_written(tmp_path):
    (tmp_path / "results.nt").write_text("already here", encoding="utf-8")
    with pytest.raises(FileExistsError):
        publishing.write_publication_files(CONTENTS, str(tmp_path))
    assert not (tmp_path / "results.csv").exists()  # sorted first, but not written


def test_files_are_written_as_utf8_bytes(tmp_path):
    text = "Zx1;Weggeschwindigkeit wurde erhöht\r\nmm²\n"
    (path,) = publishing.write_publication_files({"notes.csv": text}, str(tmp_path))
    assert Path(path).read_bytes() == text.encode("utf-8")


def test_anonymous_dataportal_client_ignores_the_token(monkeypatch):
    seen = []
    monkeypatch.setattr(sources, "DataportalClient", lambda api_token=None: seen.append(api_token))
    monkeypatch.setenv(sources.DATAPORTAL_TOKEN_ENV, DUMMY_TOKEN)
    sources.dataportal_client()
    sources.dataportal_client(write=True)
    assert seen == [None, DUMMY_TOKEN]


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="networkx 3.6.1 hashes graph labels as ASCII; the published files hold non-ASCII text")
@pytest.mark.filterwarnings("ignore:The hashes produced for directed graphs:UserWarning")
def test_publishing_run_graph_with_non_ascii_contents(tmp_path, services):
    from pmd_tensile.knowledge_graph import run_knowledge_graph

    run = pwf.node(publishing.publishing_workflow).run(
        contents={"results.nt": "<urn:a> <urn:b> \"erhöht\" .\n"}, directory=str(tmp_path),
        ontodocker_dataset="pmd_results", graph_file="results.nt", record=RECORD,
    )
    run_knowledge_graph(run)
