"""The offline guard of ``conftest.py``.

Each attempt connects to a live listener on the loopback interface, so only
the guard can make it fail, and no traffic leaves the machine.
"""

from __future__ import annotations

import socket

import pytest

GUARD_MESSAGE = "offline test attempted a connection"


def _attempt_connection() -> str | None:
    """The error text of a connection to a local listener; ``None`` if it succeeded."""
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        with socket.socket() as client:
            try:
                client.connect(server.getsockname())
            except OSError as error:
                return str(error)
    return None


AT_COLLECTION = _attempt_connection()


@pytest.fixture(scope="session")
def at_session_fixture():
    return _attempt_connection()


@pytest.fixture(scope="module")
def at_module_fixture():
    return _attempt_connection()


@pytest.fixture(scope="class")
def at_class_fixture():
    return _attempt_connection()


def test_collection_is_guarded():
    assert GUARD_MESSAGE in AT_COLLECTION


def test_session_scoped_fixtures_are_guarded(at_session_fixture):
    assert GUARD_MESSAGE in at_session_fixture


def test_module_scoped_fixtures_are_guarded(at_module_fixture):
    assert GUARD_MESSAGE in at_module_fixture


class TestClassScope:
    def test_class_scoped_fixtures_are_guarded(self, at_class_fixture):
        assert GUARD_MESSAGE in at_class_fixture


def test_tests_are_guarded():
    assert GUARD_MESSAGE in _attempt_connection()


def test_create_connection_is_guarded():
    with pytest.raises(OSError, match=GUARD_MESSAGE):
        socket.create_connection(("127.0.0.1", 9), timeout=1)


@pytest.mark.network
def test_the_block_is_lifted_only_during_a_network_test_with_run_network():
    # Loopback only, so this needs no external access despite its marker.
    assert _attempt_connection() is None
