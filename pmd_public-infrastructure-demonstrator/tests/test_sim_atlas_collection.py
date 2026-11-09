"""sim_atlas would catalogue exactly the nodes and macros of pmd_tensile (decisions D5.1, D6).

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

So the public modules may contain nodes, macros, dataclass types and the
constants listed below, and no plain function or other class. Helpers live in
``_``-prefixed modules and reach notebooks through ``pmd_tensile.demo``.
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

PACKAGE = "pmd_tensile"
PACKAGE_DIR = Path(pmd_tensile.__path__[0])

ATOMIC = "atomic node"
MACRO = "workflow macro"
DATACLASS = "dataclass type"
INSTANCE = "instance"
FUNCTION = "plain function"
CLASS = "other class"

ANALYSIS_DATACLASSES = (
    "AnalysisParameters",
    "Bootstrap",
    "CurveAnalysis",
    "ElasticFit",
    "ElasticResult",
    "Elongation",
    "ExtensometerValidity",
    "MaximumForce",
    "ProofStrength",
    "S0Scatter",
    "ScalarResult",
    "WindowSensitivity",
    "Yielding",
)

NODE_MODULES: dict[str, dict[str, str]] = {
    "pmd_tensile.exploration": {
        "check_conformance": ATOMIC,
        "list_specimens": ATOMIC,
    },
    "pmd_tensile.acquisition": {
        "get_specimen_record": ATOMIC,
        "get_reported_properties": ATOMIC,
        "fetch_curve": ATOMIC,
    },
    "pmd_tensile.analysis": {
        "check_time_order": ATOMIC,
        "engineering_stress": ATOMIC,
        "count_dropped_samples": ATOMIC,
        "extensometer_validity": ATOMIC,
        "bootstrap_fit": ATOMIC,
        "yield_analysis": ATOMIC,
        "final_elastic_fit": ATOMIC,
        "proof_strength": ATOMIC,
        "maximum_force": ATOMIC,
        "elongation_at_maximum_force": ATOMIC,
        "window_sensitivity": ATOMIC,
        "assemble_curve_analysis": ATOMIC,
        "elastic_fit": ATOMIC,
        "s0_scatter": ATOMIC,
        "elongation_after_fracture": ATOMIC,
        "reduction_of_area": ATOMIC,
        "compare_with_reported": ATOMIC,
        "summarize_by_group": ATOMIC,
        **dict.fromkeys(ANALYSIS_DATACLASSES, DATACLASS),
        # The method defaults used in node signatures. An instance of a class
        # defined in this module, so its __module__ matches; no sim_atlas parser
        # accepts an instance (parser.py:26-30).
        "DEFAULT_PARAMETERS": INSTANCE,
    },
    "pmd_tensile.workflows": {
        "curve_channels": ATOMIC,
        "specimen_geometry": ATOMIC,
        "percent_value": ATOMIC,
        "computed_properties": ATOMIC,
        "specimen_summary": ATOMIC,
        "summary_quantities": ATOMIC,
        "results_table": ATOMIC,
        "comparison_table": ATOMIC,
        "exploration_workflow": MACRO,
        "specimen_workflow": MACRO,
    },
}
"""What module mode collects from each node module: nodes, macros and the types they need."""

OTHER_PUBLIC_MODULES: dict[str, dict[str, str]] = {
    "pmd_tensile.contract": dict.fromkeys(
        ("ColumnSpec", "Curve", "ReportedProperties", "SpecimenRecord"), DATACLASS
    ),
    "pmd_tensile.demo": {},
    "pmd_tensile.profiles.base": dict.fromkeys(("CurveLocator", "OntologyProfile", "SelectQuery"), DATACLASS),
    "pmd_tensile.profiles.pmdco2_tto": {},
    "pmd_tensile.profiles.pmdco3_tto": {},
}
"""The other modules that filesystem mode reaches, and what it collects from them."""


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


def test_demo_facade_defines_nothing_and_resolves_every_name():
    from pmd_tensile import demo

    assert _collect_module(demo) == []
    for name in demo.__all__:
        value = getattr(demo, name)
        module = getattr(value, "__module__", None)
        assert module != demo.__name__, name
        if module is not None and module.startswith(PACKAGE + "."):
            assert module.split(".")[-1].startswith("_"), f"{name} comes from public {module}"
    public = {name for name in vars(demo) if not name.startswith("_") and name != "annotations"}
    assert public == set(demo.__all__)


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
        every_module.add(".".join((PACKAGE, *parts)))
    assert set(report["imported"]) == every_module
    assert report["attempts"] == []
