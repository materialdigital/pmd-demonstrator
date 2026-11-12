"""sim_atlas module and filesystem mode collect only nodes, macros and dataclass types.

Checked: module mode on every node module and on the packages
``pmd_tensile.nodes`` and ``pmd_tensile.nodes.analysis``, and filesystem mode
on ``pmd_tensile.nodes``, the harvest target, collect exactly the listed
nodes, macros and dataclass types, and no helper. Import mode
(``recursive="import"``) is not imitated: it walks every imported submodule,
including the helper modules outside the harvest target.

sim_atlas is not installed. The test imitates the collection code of the
sim_atlas toolkit at commit 753205bd61ef821517f5ea26bd5c564730e6a114
(``toolkit/src/sim_atlas_toolkit/collector.py``, ``collect_objects``):

- A module is imported by name (collector.py:19-24). sim_atlas logs an import
  error and collects nothing; here it fails the test instead.
- Module mode (``recursive="no"``) walks ``module.__dict__`` (collector.py:52)
  and skips submodules (53-56), names starting with ``_`` (58-59) and objects
  without ``__module__`` (61-62). It keeps an object only if its
  ``__module__`` *starts with* the module name (64-67). Re-exports from other
  modules are therefore not collected.
- Filesystem mode globs ``**/*.py`` below the package directory
  (collector.py:27-28), skips files whose *name* starts with ``_`` (29-31;
  directories are not checked), and collects every remaining module in module
  mode (32-48).

What sim_atlas then makes of a collected object (first parser that returns
metadata wins, parsers/parser.py:26-30, which raises ``ValueError`` if none
does):

- ``flowrep_recipe`` is an ``AtomicRecipe`` or ``WorkflowRecipe``: a FUNCTION
  or WORKFLOW node (parsers/flowrep_parser.py:363-368); a workflow uploads its
  children by module and qualname (flowrep_parser.py:268-285).
- any other plain function: ``fr.parse_atomic`` (flowrep_parser.py:373-376) or
  the plain-function parser (parsers/python_function.py:50) turns it into a
  FUNCTION node. This is how a helper would be mistaken for a node.
- a dataclass type: virtual PACK and UNPACK nodes
  (parsers/dataclass_node.py:239, 279-314).
- legacy pyiron_workflow node classes only for pyiron_workflow < 0.18
  (parsers/pyiron_workflow.py:113-136); this package uses 0.20.3.
- anything else, e.g. an instance such as a default-parameter object whose
  class is defined in the module (so its ``__module__`` matches): no parser,
  so sim_atlas raises.

So every object collected from the node modules must be a node, a macro or
a dataclass type. Constants that sim_atlas skips (strings, tuples, dicts:
no ``__module__``) may stay; helper functions inside a node module are
``_``-prefixed, and helper modules live outside ``pmd_tensile.nodes``.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from flowrep.api.schemas import AtomicRecipe, WorkflowRecipe

import pmd_tensile

PACKAGE = "pmd_tensile.nodes"
"""The harvest target."""
PACKAGE_DIR = Path(pmd_tensile.__path__[0])
"""The whole package, for the offline import check."""

ATOMIC = "atomic node"
MACRO = "workflow macro"
DATACLASS = "dataclass type"
INSTANCE = "instance"
FUNCTION = "plain function"
CLASS = "other class"
PARSABLE = (ATOMIC, MACRO, DATACLASS)
"""The kinds a sim_atlas parser accepts (FUNCTION would be mistaken for a node)."""

NODE_MODULES: dict[str, dict[str, str]] = {
    "pmd_tensile.nodes.exploration": {
        "check_conformance": ATOMIC,
        "list_specimens": ATOMIC,
    },
    "pmd_tensile.nodes.acquisition": {
        "get_specimen_record": ATOMIC,
        "get_reported_properties": ATOMIC,
        "fetch_curve": ATOMIC,
    },
    "pmd_tensile.nodes.analysis.extensometer": {
        "check_time_order": ATOMIC,
        "count_dropped_samples": ATOMIC,
        "extensometer_validity": ATOMIC,
        "DroppedSamples": DATACLASS,
        "ExtensometerDiagnostics": DATACLASS,
    },
    "pmd_tensile.nodes.analysis.elastic_line": {
        "engineering_stress": ATOMIC,
        "provisional_elastic_line": ATOMIC,
        "elastic_line": ATOMIC,
        "proof_strength": ATOMIC,
        "window_sensitivity": ATOMIC,
        "ProvisionalLine": DATACLASS,
        "ElasticLineDiagnostics": DATACLASS,
        "ProofStrengthDiagnostics": DATACLASS,
        "WindowSensitivityDiagnostics": DATACLASS,
    },
    "pmd_tensile.nodes.analysis.yielding": {
        "yield_points": ATOMIC,
        "YieldDiagnostics": DATACLASS,
    },
    "pmd_tensile.nodes.analysis.maximum_force": {
        "maximum_force": ATOMIC,
        "elongation_at_maximum_force": ATOMIC,
        "ElongationDiagnostics": DATACLASS,
    },
    "pmd_tensile.nodes.analysis.geometry": {
        "s0_scatter": ATOMIC,
        "elongation_after_fracture": ATOMIC,
        "reduction_of_area": ATOMIC,
    },
    "pmd_tensile.nodes.analysis.comparison": {
        "compare_with_reported": ATOMIC,
        "summarize_by_group": ATOMIC,
    },
    "pmd_tensile.nodes.publishing": {
        "write_publication_files": ATOMIC,
        "publish_to_ontodocker": ATOMIC,
        "publish_to_dataportal": ATOMIC,
        "publishing_workflow": MACRO,
    },
    "pmd_tensile.nodes.results": {
        "results_graph": ATOMIC,
    },
    "pmd_tensile.nodes.workflows": {
        "analysis_parameters": ATOMIC,
        "curve_channels": ATOMIC,
        "specimen_geometry": ATOMIC,
        "specimen_row": ATOMIC,
        "summary_quantities": ATOMIC,
        "results_table": ATOMIC,
        "comparison_table": ATOMIC,
        "exploration_workflow": MACRO,
        "specimen_workflow": MACRO,
    },
}
"""What module mode collects from each node module: nodes, macros and the dataclass types they need."""

OTHER_PUBLIC_MODULES: dict[str, dict[str, str]] = {}
"""Modules below the harvest target other than the node modules (none)."""

PACKAGES: dict[str, dict[str, str]] = {
    PACKAGE: {},
    "pmd_tensile.nodes.analysis": {},
}
"""Packages, whose ``__init__.py`` filesystem mode skips, and what module mode on them collects."""


# ---------------------------------------------------------------------------
# Imitation of sim_atlas_toolkit.collector.collect_objects
# ---------------------------------------------------------------------------


def _collect_module(module: str | ModuleType) -> list:
    """Module mode (``recursive="no"``), collector.py:19-24 and 51-71."""
    if isinstance(module, str):
        module = importlib.import_module(module)  # errors fail the test
    collected = []
    for name, value in module.__dict__.items():
        if inspect.ismodule(value):
            continue
        if name.startswith("_"):
            continue
        if not hasattr(value, "__module__"):
            continue
        if not value.__module__.startswith(module.__name__):
            continue
        collected.append((name, value))
    return collected


def _filesystem_modules(package: str) -> list[str]:
    """Module names that filesystem mode visits, collector.py:26-38."""
    package_dir = Path(importlib.import_module(package).__path__[0])
    paths = (path for path in package_dir.glob("**/*.py") if not path.name.startswith("_"))
    return [
        ".".join([package] + str(path.relative_to(package_dir).with_suffix("")).split("/"))
        for path in paths
    ]


def _kind(value) -> str:
    """How sim_atlas would treat a collected object (see the module docstring)."""
    recipe = getattr(value, "flowrep_recipe", None)
    if isinstance(recipe, AtomicRecipe):
        return ATOMIC
    if isinstance(recipe, WorkflowRecipe):
        return MACRO
    if inspect.isfunction(value) or inspect.isbuiltin(value):
        return FUNCTION
    if isinstance(value, type):
        return DATACLASS if dataclasses.is_dataclass(value) else CLASS
    return INSTANCE


def _kinds(module: str) -> dict[str, str]:
    return {name: _kind(value) for name, value in _collect_module(module)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_module_mode_collects_exactly_the_expected_nodes_and_macros():
    for module, expected in NODE_MODULES.items():
        collected = _kinds(module)
        helpers = {name: kind for name, kind in collected.items() if kind in (FUNCTION, CLASS)}
        assert not helpers, f"{module} would catalogue helpers as nodes: {helpers}"
        assert collected == expected, module


def test_module_mode_on_the_packages_collects_no_helper():
    for package, expected in PACKAGES.items():
        collected = _kinds(package)
        helpers = {name: kind for name, kind in collected.items() if kind in (FUNCTION, CLASS)}
        assert not helpers, f"{package} would catalogue helpers as nodes: {helpers}"
        assert collected == expected, package


def test_every_public_function_in_a_node_module_is_a_node_or_macro():
    for module in NODE_MODULES:
        for name, value in _collect_module(module):
            if inspect.isfunction(value):
                assert _kind(value) in (ATOMIC, MACRO), f"{module}.{name}"
                assert value.__module__ == module and value.__qualname__ == name


def test_filesystem_mode_visits_only_the_public_modules():
    assert sorted(_filesystem_modules(PACKAGE)) == sorted({*NODE_MODULES, *OTHER_PUBLIC_MODULES})


def test_filesystem_mode_collects_no_helper():
    collected = {
        f"{module}.{name}": _kind(value)
        for module in _filesystem_modules(PACKAGE)
        for name, value in _collect_module(module)
    }
    helpers = {name: kind for name, kind in collected.items() if kind in (FUNCTION, CLASS)}
    assert not helpers, f"sim_atlas would catalogue helpers as nodes: {helpers}"
    expected = {
        f"{module}.{name}": kind
        for modules in (NODE_MODULES, OTHER_PUBLIC_MODULES)
        for module, names in modules.items()
        for name, kind in names.items()
    }
    assert collected == expected


def test_every_collected_object_has_a_sim_atlas_parser():
    """Module and filesystem mode collect only atomic recipes, workflow recipes and dataclass types.

    Anything else would make sim_atlas raise (parser.py:26-30), e.g. a
    default-parameter instance, or be catalogued as a node (a plain function).
    """
    modules = sorted({*NODE_MODULES, *PACKAGES, *_filesystem_modules(PACKAGE)})
    collected = {
        f"{module}.{name}": _kind(value) for module in modules for name, value in _collect_module(module)
    }
    assert collected  # not vacuous
    unparsable = {name: kind for name, kind in collected.items() if kind not in PARSABLE}
    assert not unparsable, f"sim_atlas has no parser for: {unparsable}"
    for expected in (*NODE_MODULES.values(), *OTHER_PUBLIC_MODULES.values(), *PACKAGES.values()):
        assert set(expected.values()) <= set(PARSABLE)


# Runs in a fresh interpreter, so that every module is really imported, with its
# own guard that records connection and name-resolution attempts instead of
# relying on the pytest-wide block in conftest.py. Prints one JSON object.
_IMPORT_CHECK = """\
import importlib
import json
import pkgutil
import socket
import sys

attempts = []


def record_connect(self, address):
    attempts.append(f"connect {address!r}")
    raise OSError("import must stay offline")


def record_getaddrinfo(host, *args, **kwargs):
    attempts.append(f"getaddrinfo {host!r}")
    raise OSError("import must stay offline")


socket.socket.connect = record_connect
socket.getaddrinfo = record_getaddrinfo
sys.path.insert(0, sys.argv[1])

import pmd_tensile

imported = ["pmd_tensile"]
for info in pkgutil.walk_packages(pmd_tensile.__path__, "pmd_tensile."):
    importlib.import_module(info.name)
    imported.append(info.name)
print(json.dumps({"imported": sorted(imported), "attempts": attempts}))
"""


def test_importing_every_module_touches_no_network():
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_CHECK, str(PACKAGE_DIR.parent)],
        capture_output=True,
        text=True,
        timeout=300,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    every_module = set()
    for path in PACKAGE_DIR.glob("**/*.py"):
        parts = path.relative_to(PACKAGE_DIR).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        every_module.add(".".join(("pmd_tensile", *parts)))
    assert set(report["imported"]) == every_module
    assert report["attempts"] == []
