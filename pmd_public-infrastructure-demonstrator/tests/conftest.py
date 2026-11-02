"""Shared test setup for the tensile workflow.

- Puts the demonstrator directory on ``sys.path`` (as the notebooks do), so
  ``tensile_workflow`` is importable.
- Registers the ``network`` marker. Network tests are skipped unless
  ``--run-network`` is given; every other test runs with outbound socket
  connections blocked.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = DEMO_DIR / "datasets"
PMDCO2_ORIENTATION_FILES = {
    orientation: DATASETS_DIR / f"pmdco2_tto_example_{orientation}.ttl"
    for orientation in ("parallel", "perpendicular", "diagonal")
}

if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked 'network' (read-only fetches of pinned public files)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: needs read-only network access; skipped unless --run-network is given",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="network test; use --run-network to run it")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _block_network(request, monkeypatch):
    """Refuse outbound connections in all tests not marked ``network``."""
    if request.node.get_closest_marker("network"):
        return

    def refuse(self, address):
        raise OSError(f"offline test attempted a connection to {address!r}")

    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture(autouse=True)
def _fresh_graph_cache():
    """Tests must not see graphs parsed by other tests."""
    from tensile_workflow import sources

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
