"""Ontology-portable tensile-test workflow of the PMD demonstrator.

The package separates the description of the data from their analysis.

- ``nodes``: the workflow nodes and macros, and the only part that sim_atlas
  harvests. ``nodes.exploration`` and ``nodes.acquisition`` are the only
  ontology-aware nodes; ``nodes.analysis`` holds the analysis of one curve,
  split by topic; ``nodes.workflows`` holds the adapters and the two macros.
- ``contract``: neutral, ontology-free records (specimen, curve, lab values)
  in fixed units.
- ``profiles``: declarative ontology profiles (SPARQL, unit vocabulary),
  registered under a profile ID, and the queries every profile answers.
- ``sources``: graph sources (Turtle files, Ontodocker) and the token-free
  curve download.
- ``orchestration``: the batch workflow, the two-stage run and
  ``analyse_curve``; ``knowledge_graph``: export, validation and example
  queries with semantikon; ``plotting``: the figures of the notebooks;
  ``parameters``, ``units``, ``curve_math``: method parameters, annotated
  types and numerical helpers.

This module imports nothing, so that importing an ontology-agnostic part
(e.g. ``pmd_tensile.contract``) does not pull in rdflib, praeco or the
profiles.
"""
