"""Publishing nodes: the files of a results publication, an Ontodocker dataset and a Dataportal record.

The nodes wrap praeco calls, the same calls notebook 00 makes directly.
Addresses and tokens come from the environment (``ONTODOCKER_ADDRESS``,
``ONTODOCKER_TOKEN``, ``DATAPORTAL_TOKEN``), never from node inputs, so they
cannot enter a recipe, a run or a knowledge graph. Every remote node refuses to
overwrite: an existing Ontodocker dataset (an upload would merge into it) or a
Dataportal record that is not deleted.
"""

from __future__ import annotations

import datetime
import hashlib
import os
from pathlib import Path

import flowrep as fr
from praeco import DataportalClient, Person, PublicationMetadata
from praeco.exceptions import HttpError
from praeco.services.dataportal.metadata import DataportalMetadata

from pmd_tensile import sources

DATAPORTAL_TOKEN_ENV = "DATAPORTAL_TOKEN"

RECORD_FIELDS = (
    "name", "title", "description", "creators", "license", "keywords", "publication_date", "owner_org", "private",
)
"""The keys of a reviewed record description (``extras`` is optional)."""

_RDF_SUFFIXES = {".nt": "nt", ".ttl": "ttl"}


def _dataportal_client() -> DataportalClient:
    """A Dataportal client with the token from the environment (never from node inputs)."""
    token = os.environ.get(DATAPORTAL_TOKEN_ENV) or None
    if token is None:
        raise RuntimeError(f"set {DATAPORTAL_TOKEN_ENV} to create a Dataportal record")
    return DataportalClient(api_token=token)


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@fr.atomic
def write_publication_files(contents: dict, directory: str) -> list:
    """Write each text of ``contents`` (file name -> text) to ``directory`` as UTF-8.

    Inputs: ``contents``, ``directory`` (an existing directory the caller chose).
    Output: ``paths``, the written files in the order of the names. A name with a
    path separator, an existing file or a missing directory raises (nothing is
    overwritten).
    """
    folder = Path(directory)
    if not folder.is_dir():
        raise FileNotFoundError(f"not a directory: {folder}")
    paths = []
    for name in sorted(contents):
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError(f"a file name, not a path, is expected: {name!r}")
        target = folder / name
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        target.write_text(contents[name], encoding="utf-8")
        paths.append(str(target))
    return paths


@fr.atomic
def publish_to_ontodocker(dataset: str, paths: list, file_name: str) -> str:
    """Create a new Ontodocker dataset and upload one RDF file of ``paths`` into it.

    Inputs: ``dataset`` (the new dataset's name), ``paths`` (written files),
    ``file_name`` (which of them; Turtle or N-Triples, which is Turtle too).
    Output: ``ontodocker_dataset``, the dataset's name. An existing dataset of that
    name raises, because an upload would merge into it.
    """
    path = next((p for p in paths if Path(p).name == file_name), None)
    if path is None:
        raise LookupError(f"{file_name!r} is not among the written files")
    client = sources.ontodocker_client()
    if dataset in client.datasets.list():
        raise FileExistsError(f"Ontodocker dataset {dataset!r} exists; an upload would merge into it")
    client.datasets.create(dataset)
    client.datasets.upload_turtlefile(dataset, path)
    ontodocker_dataset = dataset
    return ontodocker_dataset


@fr.atomic
def publish_to_dataportal(record: dict, paths: list) -> str:
    """Create a Dataportal record from a reviewed description and upload the files as its assets.

    ``record`` holds :data:`RECORD_FIELDS`: ``name``, ``title``, ``description``,
    ``creators`` (names), ``license`` (an identifier such as ``CC-BY-4.0``),
    ``keywords``, ``publication_date`` (ISO date), ``owner_org``, ``private``, and
    optionally ``extras`` (text values). The SHA-256 of every file is added to the
    extras as ``sha256:<file name>``. RDF files (``.nt``, ``.ttl``) are uploaded as
    RDF, others with their suffix as format.
    Output: ``record_name``. A record of that name that is not deleted raises.
    """
    missing = [key for key in RECORD_FIELDS if key not in record]
    if missing:
        raise ValueError(f"the record description lacks {missing}")
    client = _dataportal_client()
    try:
        state = client.datasets.show(record["name"]).raw.get("state")
    except HttpError as error:
        if error.status_code != 404:  # anything but "not found" (e.g. no permission) stops here
            raise
        state = None  # not found: the name is free
    if state not in (None, "deleted"):
        raise FileExistsError(f"Dataportal record {record['name']!r} exists ({state})")
    metadata = PublicationMetadata(
        title=record["title"],
        description=record["description"],
        creators=tuple(Person(name=name) for name in record["creators"]),
        publication_date=datetime.date.fromisoformat(record["publication_date"]),
        keywords=tuple(record["keywords"]),
        license=record["license"],
    )
    extras = dict(record.get("extras", {}))
    extras.update({f"sha256:{Path(p).name}": _sha256(p) for p in paths})
    dataset = client.datasets.create(
        DataportalMetadata(metadata=metadata, name=record["name"], owner_org=record["owner_org"],
                           private=bool(record["private"]), extras=extras)
    )
    for path in paths:
        suffix = Path(path).suffix.lower()
        if suffix in _RDF_SUFFIXES:
            client.assets.upload_rdf(dataset, path, format=_RDF_SUFFIXES[suffix],
                                     content_type="application/n-triples" if suffix == ".nt" else "text/turtle")
        else:
            client.assets.upload(dataset, path, format=suffix.lstrip(".") or None)
    record_name = dataset.name
    return record_name


@fr.workflow
def publishing_workflow(contents: dict, directory: str, ontodocker_dataset: str, graph_file: str, record: dict):
    """Write the files, then publish the graph file on Ontodocker and all files as a Dataportal record."""
    paths = write_publication_files(contents, directory)
    dataset = publish_to_ontodocker(ontodocker_dataset, paths, graph_file)
    record_name = publish_to_dataportal(record, paths)
    return paths, dataset, record_name
