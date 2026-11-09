"""Ontology-portable tensile-test workflow for the public demonstrator.

The package separates *data description* from *data analysis*. Its public
modules hold the workflow nodes and macros, the data contract and the
ontology profiles; the helpers live in private ``_``-prefixed modules, so that
sim_atlas catalogues only nodes and macros (see
``tests/test_sim_atlas_collection.py``).

Nodes and macros (flowrep ``@fr.atomic`` and ``@fr.workflow``):

- ``exploration`` and ``acquisition``: the only ontology-aware workflow nodes.
- ``analysis``: ontology-agnostic analysis nodes (method-defined,
  ISO 6892-1-inspired), with their result types and annotations.
- ``workflows``: contract adapter nodes and the exploration and per-specimen
  workflows (macros).

Data description:

- ``contract``: neutral, ontology-free data contract (specimen records, curves,
  lab-reported properties) in fixed units.
- ``profiles``: declarative ontology profiles (SPARQL, unit vocabulary, naming
  conventions), registered under a profile ID.

Helpers for notebooks, re-exported by ``demo``:

- ``_sources``: graph sources (local/remote Turtle via rdflib, Ontodocker via
  praeco) and the token-free curve download.
- ``_exploration``, ``_acquisition``, ``_analysis``: the plain-function forms
  behind the nodes (conformance results, specimen table, specimen record,
  curve download, ``analyse_curve``).
- ``_orchestration``: the batch workflow and
  ``run_tensile_analysis`` (both stages).
- ``_knowledge_graph``: semantikon knowledge-graph export, validation, Turtle
  helpers and example SPARQL queries.

This module deliberately imports nothing, so that importing an
ontology-agnostic submodule (e.g. ``pmd_tensile.contract``) does not pull
in rdflib, praeco or the profiles.
"""
