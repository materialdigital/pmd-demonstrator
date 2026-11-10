"""Curve download and parsing, offline (synthetic CSV bytes, stubbed HTTP)."""

from __future__ import annotations

import hashlib

import numpy as np
import pyiron_workflow as pwf
import pytest

from pmd_tensile import sources
from pmd_tensile.contract import ColumnSpec, UnitError
from pmd_tensile.nodes import acquisition

# Synthetic CSV in the layout of the S355 curve files: ';' delimiter, ','
# decimal mark, a header row and a unit row. The numbers are made up.
CSV = (
    "Time;Crosshead separation;Load;Extensometer elongation\r\n"
    "(sec);(mm);(kN);(%)\r\n"
    "0,0;0,000;0,010;0,0000\r\n"
    "0,1;0,005;1,500;0,0100\r\n"
    "0,2;0,010;3,000;0,0200\r\n"
    "0,3;0,020;4,250;1,5000\r\n"
).encode("utf-8")

BY_INDEX = {
    "time": ColumnSpec(unit="s", index=0),
    "crosshead": ColumnSpec(unit="mm", index=1),
    "force": ColumnSpec(unit="kN", index=2),
    "strain": ColumnSpec(unit="%", index=3),
}
BY_HEADER = {
    "time": ColumnSpec(unit="s", header="Time"),
    "crosshead": ColumnSpec(unit="mm", header="Crosshead separation"),
    "force": ColumnSpec(unit="kN", header="Load"),
    "strain": ColumnSpec(unit="%", header="Extensometer elongation"),
}
URL = "https://data.example.org/curves/Zx1.csv"


def _check_curve(curve):
    np.testing.assert_allclose(curve.time_s, [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(curve.crosshead_mm, [0.0, 0.005, 0.010, 0.020])
    np.testing.assert_allclose(curve.force_n, [10.0, 1500.0, 3000.0, 4250.0])  # kN -> N
    np.testing.assert_allclose(curve.strain, [0.0, 1e-4, 2e-4, 0.015])  # % -> ratio
    assert curve.source_url == URL
    assert curve.sha256 == hashlib.sha256(CSV).hexdigest()


@pytest.mark.parametrize("columns", [BY_INDEX, BY_HEADER], ids=["by-index", "by-header"])
def test_parse_curve_csv(columns):
    _check_curve(sources.parse_curve_csv(CSV, columns, source_url=URL))


def test_sha256_is_over_the_exact_bytes():
    # Same numbers, different line endings: different bytes, different hash.
    other = CSV.replace(b"\r\n", b"\n")
    a = sources.parse_curve_csv(CSV, BY_INDEX, source_url=URL)
    b = sources.parse_curve_csv(other, BY_INDEX, source_url=URL)
    np.testing.assert_array_equal(a.force_n, b.force_n)
    assert a.sha256 == hashlib.sha256(CSV).hexdigest()
    assert b.sha256 == hashlib.sha256(other).hexdigest()
    assert a.sha256 != b.sha256


def test_declared_unit_must_match_unit_row():
    columns = dict(BY_INDEX, force=ColumnSpec(unit="N", index=2))
    with pytest.raises(UnitError, match="unit mismatch"):
        sources.parse_curve_csv(CSV, columns, source_url=URL)


def test_unknown_unit_in_unit_row_raises():
    data = CSV.replace(b"(kN)", b"(lbf)")
    with pytest.raises(UnitError, match="lbf"):
        sources.parse_curve_csv(data, BY_INDEX, source_url=URL)


def test_declared_unit_of_wrong_kind_raises():
    columns = dict(BY_INDEX, strain=ColumnSpec(unit="mm", index=3))
    with pytest.raises(UnitError):
        sources.parse_curve_csv(CSV, columns, source_url=URL)


def test_missing_header_raises():
    columns = dict(BY_HEADER, force=ColumnSpec(unit="kN", header="Force"))
    with pytest.raises(ValueError, match="'Force'"):
        sources.parse_curve_csv(CSV, columns, source_url=URL)


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content
        self.closed = False

    def raise_for_status(self):
        pass

    def close(self):
        self.closed = True


class _RecordingHttpClient:
    """Stand-in for praeco.HttpClient that records how it is built and used."""

    instances: list = []

    def __init__(self, address, *, token=None, **kwargs):
        self.address = address
        self.token = token
        self.requests = []
        self.response = _FakeResponse(CSV)
        _RecordingHttpClient.instances.append(self)

    def request(self, method, url, **kwargs):
        self.requests.append((method, url))
        return self.response


def test_fetch_bytes_uses_token_free_client_bound_to_host(monkeypatch):
    monkeypatch.setenv("ONTODOCKER_TOKEN", "DUMMY_TOKEN_XYZ")
    _RecordingHttpClient.instances = []
    monkeypatch.setattr(sources, "HttpClient", _RecordingHttpClient)

    data, sha256 = sources.fetch_bytes(URL)

    assert data == CSV
    assert sha256 == hashlib.sha256(CSV).hexdigest()
    (client,) = _RecordingHttpClient.instances
    assert client.address == "https://data.example.org"
    assert client.token is None
    assert client.requests == [("GET", URL)]
    assert client.response.closed


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/x.csv", "Zx1.csv"])
def test_fetch_bytes_rejects_non_http_urls(url):
    with pytest.raises(sources.SourceError):
        sources.fetch_bytes(url)


def test_fetch_curve_node_with_stubbed_fetch(monkeypatch, pmdco2_source):
    record = acquisition.get_specimen_record("pmdco2-tto", pmdco2_source, "Zx1")
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return CSV, hashlib.sha256(CSV).hexdigest()

    monkeypatch.setattr(sources, "fetch_bytes", fake_fetch)
    run = pwf.node(acquisition.fetch_curve, "curve").run(record=record)

    assert run.status == "finished", run.exception
    curve = run.outputs.curve
    assert calls == [record.curve_url]
    assert curve.source_url == record.curve_url
    assert curve.sha256 == hashlib.sha256(CSV).hexdigest()
    np.testing.assert_allclose(curve.force_n, [10.0, 1500.0, 3000.0, 4250.0])
    np.testing.assert_allclose(curve.strain, [0.0, 1e-4, 2e-4, 0.015])
