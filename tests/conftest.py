"""Shared test setup for the tensile workflow.

- Puts the demonstrator directory on ``sys.path``, so ``pmd_tensile`` is
  importable also without the editable install of ``environment.yml``.
- Registers the ``network`` marker. Network tests are skipped unless
  ``--run-network`` is given. Outbound socket connections are blocked for the
  whole pytest session, including collection and fixtures of any scope; the
  block is lifted only during a ``network`` test run with ``--run-network``.
- Registers the ``statistical`` marker (Monte Carlo checks such as VP-10).
  Statistical tests are skipped unless ``--run-statistical`` is given.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = DEMO_DIR / "datasets"
DOWNLOADED_DIR = DEMO_DIR / "downloaded"
ORIENTATIONS = ("parallel", "perpendicular", "diagonal")
# The pmdco2 graphs as hosted on Ontodocker (decision D7): downloaded/ holds
# copies isomorphic to the hosted datasets. The older datasets/ files describe
# the same tests (tests/test_pmdco2_sources.py) but write some nodes as
# pseudo-IRIs <_:tNNN>.
PMDCO2_ORIENTATION_FILES = {
    orientation: DOWNLOADED_DIR / f"pmdco2_tto_example_{orientation}.ttl" for orientation in ORIENTATIONS
}
PMDCO2_DATASETS_FILES = {
    orientation: DATASETS_DIR / f"pmdco2_tto_example_{orientation}.ttl" for orientation in ORIENTATIONS
}
# The PMDco3 graphs (decision D8): the slices and the full file carry one CSVW
# curve table per test; the pinned original has none.
PMDCO3_SLICE_FILES = {
    orientation: DATASETS_DIR / f"pmdco3_tto_example_{orientation}.ttl" for orientation in ORIENTATIONS
}
PMDCO3_FULL_FILE = DATASETS_DIR / "pmdco3_tto_example.ttl"
PMDCO3_ORIGINAL_FILE = DATASETS_DIR / "S355_data_tto_pmdco3.ttl"

if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))
# Test helper modules (synthetic, workflow_helpers), also under --import-mode=importlib.
if str(DEMO_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(DEMO_DIR / "tests"))

_REAL_CONNECT = socket.socket.connect


def _refuse_connection(self, address):
    raise OSError(f"offline test attempted a connection to {address!r}")


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked 'network' (only the check that the connection block is lifted, loopback only)",
    )
    parser.addoption(
        "--run-statistical",
        action="store_true",
        default=False,
        help="run tests marked 'statistical' (Monte Carlo tolerance feasibility, VP-10)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: needs read-only network access; skipped unless --run-network is given",
    )
    config.addinivalue_line(
        "markers",
        "statistical: Monte Carlo check; skipped unless --run-statistical is given",
    )
    socket.socket.connect = _refuse_connection


def pytest_unconfigure(config):
    socket.socket.connect = _REAL_CONNECT


def pytest_collection_modifyitems(config, items):
    skips = {}
    if not config.getoption("--run-network"):
        skips["network"] = pytest.mark.skip(reason="network test; use --run-network to run it")
    if not config.getoption("--run-statistical"):
        skips["statistical"] = pytest.mark.skip(
            reason="statistical test; use --run-statistical to run it"
        )
    for item in items:
        for keyword, skip in skips.items():
            if keyword in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def _lift_network_block(request, monkeypatch):
    """Allow connections during a ``network`` test run with ``--run-network``, and only then."""
    if request.node.get_closest_marker("network") and request.config.getoption("--run-network"):
        monkeypatch.setattr(socket.socket, "connect", _REAL_CONNECT)


@pytest.fixture(autouse=True)
def _fresh_graph_cache():
    """Tests must not see graphs parsed by other tests."""
    from pmd_tensile import _sources as sources

    sources.clear_graph_cache()
    yield
    sources.clear_graph_cache()


@pytest.fixture
def pmdco2_source() -> dict:
    """Turtle source over the three local pmdco2 files (read-only assets)."""
    return {
        "kind": "turtle",
        "locations": [str(path) for path in PMDCO2_ORIENTATION_FILES.values()],
    }
