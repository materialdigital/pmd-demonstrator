"""Ontology-portable tensile-test workflow for the public demonstrator.

The package separates *data description* from *data analysis*:

- ``contract``: neutral, ontology-free data contract (specimen records, curves,
  lab-reported properties) in fixed units.
- ``profiles``: declarative ontology profiles (SPARQL, unit vocabulary, naming
  conventions), registered under a profile ID.
- ``sources``: graph sources (local/remote Turtle via rdflib, Ontodocker via
  praeco) and the token-free curve download.
- ``exploration`` and ``acquisition``: the only ontology-aware workflow nodes.
- ``analysis``: ontology-agnostic analysis nodes (method-defined,
  ISO 6892-1-inspired).
- ``workflows``: the two-stage assembly (exploration workflow, per-specimen
  workflow and batch workflow) and :func:`~tensile_workflow.workflows.run_tensile_analysis`.
- ``knowledge_graph``: semantikon knowledge-graph export, validation, Turtle
  helpers and example SPARQL queries.

This module deliberately imports nothing, so that importing an
ontology-agnostic submodule (e.g. ``tensile_workflow.contract``) does not pull
in rdflib, praeco or the profiles.
"""
