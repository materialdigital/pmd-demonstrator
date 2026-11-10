"""Helpers for the demonstrator notebooks, in one place.

This module only re-exports plain functions, classes and constants from the
private helper modules; it defines nothing itself. sim_atlas collects only
objects whose ``__module__`` is the collected module, so these re-exports are
not catalogued as nodes. The nodes and macros themselves are imported from
their public modules (``pmd_tensile.exploration``, ``.acquisition``,
``.analysis``, ``.workflows``), the profiles from ``pmd_tensile.profiles``.
"""

from __future__ import annotations

from pmd_tensile._acquisition import build_reported_properties, build_specimen_record, download_curve
from pmd_tensile._analysis import analyse_curve
from pmd_tensile._exploration import ConformanceError, conformance_results, is_conformant, specimen_table
from pmd_tensile._knowledge_graph import (
    EXAMPLE_QUERIES,
    LOSSLESS_FORMAT,
    MODULE_ROLES,
    SEMANTIKON_PMD_NAMESPACE,
    TURTLE_PRECISION_NOTE,
    parse_turtle,
    pmd_iris,
    read_graph,
    read_turtle,
    recipe_knowledge_graph,
    run_example_query,
    run_knowledge_graph,
    to_turtle,
    validate_knowledge_graph,
    write_graph,
    write_turtle,
)
from pmd_tensile._orchestration import (
    TensileAnalysis,
    batch_inputs,
    build_batch_workflow,
    build_exploration_workflow,
    build_specimen_node,
    run_tensile_analysis,
    specimen_curves,
    specimen_input_label,
)
from pmd_tensile._parameters import DEFAULT_PARAMETERS, parameter_values, parameters_from_values
from pmd_tensile._plotting import (
    curves_overview_figure,
    elastic_step_figure,
    format_value,
    lab_comparison_step_figure,
    maximum_force_step_figure,
    orientation_styles,
    results_overview_figure,
    specimen_colors,
    specimen_label,
    specimen_order,
    yield_step_figure,
)
from pmd_tensile._sources import (
    ONTODOCKER_ADDRESS_ENV,
    ONTODOCKER_TOKEN_ENV,
    SourceError,
    ask,
    clear_graph_cache,
    ontodocker_client,
    select,
    validate_source,
)

__all__ = [
    # Sources (_sources)
    "ONTODOCKER_ADDRESS_ENV",
    "ONTODOCKER_TOKEN_ENV",
    "SourceError",
    "ask",
    "clear_graph_cache",
    "ontodocker_client",
    "select",
    "validate_source",
    # Exploration and acquisition without a workflow (_exploration, _acquisition)
    "ConformanceError",
    "conformance_results",
    "is_conformant",
    "specimen_table",
    "build_reported_properties",
    "build_specimen_record",
    "download_curve",
    # Analysis of one curve without a workflow (_analysis)
    "analyse_curve",
    # Analysis parameters and their JSON-able dictionary form (_parameters)
    "DEFAULT_PARAMETERS",
    "parameter_values",
    "parameters_from_values",
    # Building and running the workflows (_orchestration)
    "TensileAnalysis",
    "batch_inputs",
    "build_batch_workflow",
    "build_exploration_workflow",
    "build_specimen_node",
    "run_tensile_analysis",
    "specimen_curves",
    "specimen_input_label",
    # Knowledge graphs (_knowledge_graph)
    "EXAMPLE_QUERIES",
    "LOSSLESS_FORMAT",
    "MODULE_ROLES",
    "SEMANTIKON_PMD_NAMESPACE",
    "TURTLE_PRECISION_NOTE",
    "parse_turtle",
    "pmd_iris",
    "read_graph",
    "read_turtle",
    "recipe_knowledge_graph",
    "run_example_query",
    "run_knowledge_graph",
    "to_turtle",
    "validate_knowledge_graph",
    "write_graph",
    "write_turtle",
    # Figures in the style of the mesh demonstrator (_plotting)
    "curves_overview_figure",
    "elastic_step_figure",
    "format_value",
    "lab_comparison_step_figure",
    "maximum_force_step_figure",
    "orientation_styles",
    "results_overview_figure",
    "specimen_colors",
    "specimen_label",
    "specimen_order",
    "yield_step_figure",
]
