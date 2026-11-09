"""SPARQL queries on the Turtle path must not overlap (T9-F1).

pyiron_workflow runs nodes in a thread pool. rdflib's SPARQL grammar is
module-level pyparsing state, and one overlap of ``Graph.query`` calls can break
a grammar element for the life of the process, so ``pmd_tensile.sources``
serializes them with a lock.

- The first test checks the lock directly: at most one ``Graph.query`` at a time.
  Its spy hands prepared queries to rdflib, so that no SPARQL text is parsed in a
  thread and a failure cannot damage the parser of this pytest process.
- The second test checks the behaviour. The damage happens while the grammar is
  used for the first time, so it needs an interpreter that has parsed no query yet.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent import futures

import rdflib
from conftest import DEMO_DIR, PMDCO2_ORIENTATION_FILES
from rdflib.plugins.sparql import prepareQuery

from pmd_tensile import sources

N_THREADS = 10

_TURTLE = """\
@prefix ex: <https://example.org/> .
ex:a ex:value "1" .
"""
_SELECT = "SELECT ?value WHERE { <https://example.org/a> <https://example.org/value> ?value }"
_ASK = "ASK { <https://example.org/a> <https://example.org/value> ?value }"


def test_graph_query_calls_do_not_overlap(tmp_path, monkeypatch):
    path = tmp_path / "small.ttl"
    path.write_text(_TURTLE, encoding="utf-8")
    source = {"kind": "turtle", "locations": [str(path)]}

    prepared = {query: prepareQuery(query) for query in (_SELECT, _ASK)}
    real_query = rdflib.Graph.query
    counter_lock = threading.Lock()
    running = 0
    peak = 0
    calls = 0

    def spy(graph, query):
        nonlocal running, peak, calls
        with counter_lock:
            calls += 1
            running += 1
            peak = max(peak, running)
        time.sleep(0.01)  # Long enough for unserialized threads to meet in here.
        try:
            return real_query(graph, prepared[query])
        finally:
            with counter_lock:
                running -= 1

    monkeypatch.setattr(rdflib.Graph, "query", spy)
    barrier = threading.Barrier(N_THREADS)

    def work(_):
        barrier.wait(timeout=60)
        selected = sources.select(source, _SELECT, ["value"])
        answers = sources.ask(source, _ASK)
        return selected["value"].tolist(), answers[str(path)]

    with futures.ThreadPoolExecutor(N_THREADS) as pool:
        results = list(pool.map(work, range(N_THREADS)))

    assert results == [(["1"], True)] * N_THREADS
    assert calls == 2 * N_THREADS
    assert peak == 1


# Runs in a fresh interpreter: argv is the demonstrator directory and the Turtle
# files. Prints one JSON object. Failures are collected, not raised, so that the
# parent can report all of them.
_FRESH_INTERPRETER_CHECK = """\
import json
import socket
import sys
import threading
from concurrent import futures


def refuse_connection(self, address):
    raise OSError(f"this check must stay offline, tried {address!r}")


socket.socket.connect = refuse_connection
demo_dir, *locations = sys.argv[1:]
sys.path.insert(0, demo_dir)

from pmd_tensile import sources
from pmd_tensile.profiles import get_profile
from pmd_tensile.profiles.base import SelectQuery

N_THREADS = 10
source = {"kind": "turtle", "locations": locations}
profile = get_profile("pmdco2-tto")
selects = {
    name: value for name, value in vars(profile).items() if isinstance(value, SelectQuery)
}
asks = dict(profile.conformance_asks)


def answer(name):
    if name in selects:
        rows = sources.select(source, selects[name].query, selects[name].columns)
        return sorted(repr(row) for row in rows.to_numpy(dtype=object).tolist())
    return sorted(sources.ask(source, asks[name]).items())


names = [*selects, *asks]
for location in locations:  # Only Turtle parsing: leaves the SPARQL grammar unused.
    sources._load_graph(location)
barrier = threading.Barrier(N_THREADS)
sys.setswitchinterval(1e-5)  # Frequent thread switches make an unserialized overlap likely.


def work(_):
    barrier.wait(timeout=60)
    answers, errors = {}, {}
    for name in names:
        try:
            answers[name] = answer(name)
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"[:200]
    return answers, errors


with futures.ThreadPoolExecutor(N_THREADS) as pool:
    threaded = list(pool.map(work, range(N_THREADS)))

thread_errors = sorted({f"{name}: {text}" for _, errors in threaded for name, text in errors.items()})
serial, serial_errors = {}, []
for name in names:
    try:
        serial[name] = answer(name)
    except Exception as error:
        serial_errors.append(f"{name}: {type(error).__name__}: {error}"[:200])
wrong_answers = sorted(
    {
        name
        for answers, _ in threaded
        for name, value in answers.items()
        if name in serial and value != serial[name]
    }
)
print(
    json.dumps(
        {
            "n_selects": len(selects),
            "n_asks": len(asks),
            "n_rows": {name: len(serial[name]) for name in selects if name in serial},
            "all_asks_true": all(
                answer_ for name in asks if name in serial for _, answer_ in serial[name]
            ),
            "thread_errors": thread_errors,
            "serial_errors": serial_errors,
            "wrong_answers": wrong_answers,
        }
    )
)
"""


def test_profile_queries_from_threads_in_a_fresh_interpreter():
    locations = [str(path) for path in PMDCO2_ORIENTATION_FILES.values()]
    done = subprocess.run(
        [sys.executable, "-c", _FRESH_INTERPRETER_CHECK, str(DEMO_DIR), *locations],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=DEMO_DIR,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout.splitlines()[-1])

    assert report["thread_errors"] == []
    assert report["serial_errors"] == []
    assert report["wrong_answers"] == []
    # The check is vacuous unless it really ran the profile's queries on the data.
    assert report["n_selects"] == 4
    assert report["n_asks"] == 4
    assert report["n_rows"] == {"specimens": 10, "quantities": 150, "notes": 10, "curve_columns": 40}
    assert report["all_asks_true"] is True
