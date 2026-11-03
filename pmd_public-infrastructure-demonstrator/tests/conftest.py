"""Shared test setup for the tensile workflow.

- Puts the demonstrator directory on ``sys.path`` (as the notebooks do), so
  ``tensile_workflow`` is importable.
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
PMDCO2_ORIENTATION_FILES = {
    orientation: DATASETS_DIR / f"pmdco2_tto_example_{orientation}.ttl"
    for orientation in ("parallel", "perpendicular", "diagonal")
}

if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

_REAL_CONNECT = socket.socket.connect


def _refuse_connection(self, address):
    raise OSError(f"offline test attempted a connection to {address!r}")


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked 'network' (read-only fetches of pinned public files)",
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
