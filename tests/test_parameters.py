"""Analysis parameters as a workflow input (decision D4, follow-up TW-F5), offline (C5).

The runs use the local pmdco2 Turtle files and stubbed curve downloads with
synthetic CSV bytes (see ``workflow_helpers``). The changed parameter is the
proof-strength offset ``offset_plastic_strain`` (M10): for these
discontinuously yielding curves it enters only Rp0.2 and its window range,
so every other output must stay as it is.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math

import numpy as np
import pytest
from workflow_helpers import (
    PMDCO2,
    PMDCO2_SOURCE,
    curve_bytes_for,
    run_with_stubbed_curves,
    stub_fetch,
)

from pmd_tensile import knowledge_graph as kg
from pmd_tensile import orchestration, sources
from pmd_tensile.nodes import acquisition, workflows
from pmd_tensile.nodes.analysis.geometry import elongation_after_fracture
from pmd_tensile.orchestration import analyse_curve
from pmd_tensile.parameters import (
    DEFAULT_PARAMETERS,
    AnalysisParameters,
    parameter_values,
    parameters_from_values,
)

SPECIMENS = ("Zx1", "Zd2")
OFFSET = 0.003
CHANGED = {"offset_plastic_strain": OFFSET}
RP02_COLUMNS = {"rp02_mpa", "rp02_window_min_mpa", "rp02_window_max_mpa"}

# semantikon's data hashing calls networkx, which warns that its graph hashes
# changed in networkx 3.5; an upstream notice, not a test concern.
pytestmark = pytest.mark.filterwarnings("ignore:The hashes produced for directed graphs:UserWarning")


def cell_texts(table) -> dict[str, list[str]]:
    """Every cell as its repr, per column (exact for floats, NaN-safe inside tuples)."""
    return {column: [repr(value) for value in table[column]] for column in table.columns}


def parse_parameters_repr(text: str) -> AnalysisParameters:
    """``AnalysisParameters(...)`` as written into the run graph -> the dataclass."""
    call = ast.parse(text, mode="eval").body
    assert isinstance(call, ast.Call) and call.func.id == "AnalysisParameters" and not call.args
    return AnalysisParameters(**{kw.arg: ast.literal_eval(kw.value) for kw in call.keywords})


@pytest.fixture(scope="module")
def curves() -> dict:
    return curve_bytes_for(SPECIMENS)


@pytest.fixture(scope="module")
def default_run(curves):
    result, _ = run_with_stubbed_curves(curves, PMDCO2, PMDCO2_SOURCE, specimen_ids=SPECIMENS)
    return result


@pytest.fixture(scope="module")
def changed_run(curves):
    result, _ = run_with_stubbed_curves(curves, PMDCO2, PMDCO2_SOURCE, specimen_ids=SPECIMENS, parameters=CHANGED)
    return result


# ---------------------------------------------------------------------------
# The dictionary form
# ---------------------------------------------------------------------------


def test_defaults_are_the_method_defaults():
    assert DEFAULT_PARAMETERS == AnalysisParameters()
    assert parameters_from_values(None) == DEFAULT_PARAMETERS
    assert parameters_from_values({}) == DEFAULT_PARAMETERS
    assert workflows.analysis_parameters() == DEFAULT_PARAMETERS  # the node, called directly


def test_dictionary_form_is_json_and_round_trips():
    values = parameter_values()
    assert list(values) == [field.name for field in dataclasses.fields(AnalysisParameters)]
    assert values["window_grid_lo"] == [0.05, 0.10, 0.15]
    assert values["linearity_windows"] == [[0.05, 0.25], [0.25, 0.50]]
    assert json.loads(json.dumps(values)) == values
    assert parameters_from_values(json.loads(json.dumps(values))) == DEFAULT_PARAMETERS
    changed = AnalysisParameters(offset_plastic_strain=OFFSET, window_grid_hi=(0.3, 0.45))
    assert parameters_from_values(parameter_values(changed)) == changed


def test_missing_keys_keep_the_defaults():
    params = parameters_from_values(CHANGED)
    assert params == dataclasses.replace(DEFAULT_PARAMETERS, offset_plastic_strain=OFFSET)
    assert parameters_from_values({"min_elastic_points": 12}).min_elastic_points == 12
    assert parameters_from_values({"linearity_windows": [[0.1, 0.2]]}).linearity_windows == ((0.1, 0.2),)


# ---------------------------------------------------------------------------
# (a) The default call gives the results of before
# ---------------------------------------------------------------------------


def test_default_and_explicit_default_runs_are_identical(curves, default_run):
    explicit, _ = run_with_stubbed_curves(
        curves, PMDCO2, PMDCO2_SOURCE, specimen_ids=SPECIMENS, parameters=parameter_values()
    )
    for name in ("results", "orientation_summary", "comparison"):
        assert cell_texts(getattr(explicit, name)) == cell_texts(getattr(default_run, name)), name


def test_default_run_equals_analyse_curve_with_default_parameters(curves, default_run):
    """The workflow default is analyse_curve's default (params=None), field by field."""
    rows = default_run.results.set_index("specimen_id")
    for sid in SPECIMENS:
        record = acquisition.get_specimen_record(PMDCO2, PMDCO2_SOURCE, sid)
        curve = sources.parse_curve_csv(curves[sid], record.curve_columns, source_url=record.curve_url)
        a_percent, _ = elongation_after_fracture(record.l0_mm, record.lu_mm)
        args = (curve.force_n, curve.strain, curve.crosshead_mm, record.s0_mm2, record.l0_mm)
        kwargs = dict(a_percent=a_percent, rate_switch_strain=record.rate_switch_strain)
        implicit = analyse_curve(*args, **kwargs)
        explicit = analyse_curve(*args, **kwargs, params=DEFAULT_PARAMETERS)
        assert repr(implicit) == repr(explicit)
        for column in ("e_gpa", "reh_mpa", "rel_mpa", "rp02_mpa", "fm_n", "rm_mpa", "strain_origin"):
            assert repr(float(rows.loc[sid, column])) == repr(float(getattr(implicit, column))), (sid, column)


# ---------------------------------------------------------------------------
# (b) One changed parameter changes the affected outputs only
# ---------------------------------------------------------------------------


def test_changed_offset_changes_only_rp02(default_run, changed_run):
    before, after = cell_texts(default_run.results), cell_texts(changed_run.results)
    assert before.keys() == after.keys()
    changed = {column for column in before if before[column] != after[column]}
    assert changed == RP02_COLUMNS
    for column in RP02_COLUMNS:  # every specimen moves
        assert all(old != new for old, new in zip(before[column], after[column])), column

    # The comparison with lab values changes only in its Rp0.2 rows.
    old, new = default_run.comparison, changed_run.comparison
    unchanged = old["quantity"] != "rp02_mpa"
    assert cell_texts(old[unchanged]) == cell_texts(new[unchanged])
    assert cell_texts(old[~unchanged]) != cell_texts(new[~unchanged])


def test_changed_offset_equals_analyse_curve_with_the_same_parameters(curves, changed_run):
    params = AnalysisParameters(offset_plastic_strain=OFFSET)
    rows = changed_run.results.set_index("specimen_id")
    for sid in SPECIMENS:
        record = acquisition.get_specimen_record(PMDCO2, PMDCO2_SOURCE, sid)
        curve = sources.parse_curve_csv(curves[sid], record.curve_columns, source_url=record.curve_url)
        expected = analyse_curve(
            curve.force_n,
            curve.strain,
            curve.crosshead_mm,
            record.s0_mm2,
            record.l0_mm,
            a_percent=elongation_after_fracture(record.l0_mm, record.lu_mm)[0],
            rate_switch_strain=record.rate_switch_strain,
            params=params,
        )
        assert rows.loc[sid, "rp02_mpa"] == expected.rp02_mpa
        assert (rows.loc[sid, "rp02_window_min_mpa"], rows.loc[sid, "rp02_window_max_mpa"]) == (
            expected.rp02_min_mpa, expected.rp02_max_mpa
        )
        assert math.isfinite(expected.rp02_mpa)


def test_specimen_workflow_takes_the_parameters(curves):
    fetch = stub_fetch(curves)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sources, "fetch_bytes", fetch)
        node = orchestration.build_specimen_node()
        default = node.run(profile_id=PMDCO2, source=PMDCO2_SOURCE, specimen_id="Zx1")
        changed = node.run(profile_id=PMDCO2, source=PMDCO2_SOURCE, specimen_id="Zx1", parameters=CHANGED)
    assert default.outputs.rp02_mpa != changed.outputs.rp02_mpa
    for port in ("e_gpa", "reh_mpa", "rel_mpa", "fm_n", "rm_mpa"):
        assert getattr(default.outputs, port) == getattr(changed.outputs, port), port
    assert np.isnan(default.outputs.ag_percent) == np.isnan(changed.outputs.ag_percent)


# ---------------------------------------------------------------------------
# (c) The run knowledge graph records the values used
# ---------------------------------------------------------------------------


def test_run_graph_records_the_parameters_used(default_run, changed_run):
    for result, expected in ((default_run, DEFAULT_PARAMETERS), (changed_run, parameters_from_values(CHANGED))):
        graph = kg.run_knowledge_graph(result.batch_run)
        inputs = dict(kg.run_example_query(graph, "run_inputs").itertuples(index=False))
        assert ast.literal_eval(inputs["parameters"]) == parameter_values(expected)
        used = kg.run_example_query(graph, "analysis_parameters")
        assert list(used["specimen_id"]) == sorted(SPECIMENS)
        for text in used["parameters"]:
            assert parse_parameters_repr(text) == expected
    recipe = kg.recipe_knowledge_graph(changed_run.batch_workflow)
    assert kg.run_example_query(recipe, "analysis_parameters").empty  # a recipe has no values


def test_single_specimen_run_graph_records_the_parameters(curves):
    fetch = stub_fetch(curves)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sources, "fetch_bytes", fetch)
        run = orchestration.build_specimen_node().run(
            profile_id=PMDCO2, source=PMDCO2_SOURCE, specimen_id="Zd2", parameters=CHANGED
        )
    graph = kg.run_knowledge_graph(run)
    used = kg.run_example_query(graph, "analysis_parameters")
    assert list(used["specimen_id"]) == ["Zd2"]
    assert parse_parameters_repr(used["parameters"].item()) == parameters_from_values(CHANGED)
    inputs = dict(kg.run_example_query(graph, "run_inputs").itertuples(index=False))
    assert ast.literal_eval(inputs["parameters"]) == CHANGED  # a standalone run records the input as given


# ---------------------------------------------------------------------------
# (d) Invalid input fails with a clear error
# ---------------------------------------------------------------------------

INVALID = [
    (["offset_plastic_strain", 0.003], TypeError, "dictionary"),
    (AnalysisParameters(), TypeError, "dictionary"),
    ({"offset": 0.003}, ValueError, r"unknown analysis parameters \['offset'\]"),
    ({"offset_plastic_strain": "0.003"}, TypeError, "'offset_plastic_strain' must be a number"),
    ({"offset_plastic_strain": True}, TypeError, "'offset_plastic_strain' must be a number"),
    ({"offset_plastic_strain": float("nan")}, ValueError, "'offset_plastic_strain' must be finite"),
    ({"offset_plastic_strain": -0.002}, ValueError, "non-negative"),
    ({"min_elastic_points": 10.0}, TypeError, "'min_elastic_points' must be an integer"),
    ({"min_elastic_points": 0}, ValueError, "'min_elastic_points' must be at least 1"),
    ({"f_lo": 0.5}, ValueError, "f_lo < f_hi"),
    ({"window_grid_lo": []}, TypeError, "'window_grid_lo' must be a non-empty list"),
    ({"window_grid_lo": "0.1"}, TypeError, "'window_grid_lo' must be a non-empty list"),
    ({"linearity_windows": [[0.25, 0.05]]}, ValueError, r"\[lo, hi\] pairs with lo < hi"),
    ({"linearity_windows": [[0.05, 0.25, 0.5]]}, ValueError, r"\[lo, hi\] pairs"),
]


@pytest.mark.parametrize(("values", "error", "message"), INVALID)
def test_invalid_parameters_raise_a_clear_error(values, error, message):
    with pytest.raises(error, match=message):
        parameters_from_values(values)
    with pytest.raises(error, match=message):
        orchestration.batch_inputs(PMDCO2, PMDCO2_SOURCE, SPECIMENS, parameters=values)


def test_invalid_parameters_fail_before_any_work(curves):
    """run_tensile_analysis validates first: no graph query and no curve download happen."""
    fetch = stub_fetch(curves)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sources, "fetch_bytes", fetch)
        mp.setattr(sources, "select", lambda *args, **kwargs: pytest.fail("graph queried"))
        mp.setattr(sources, "ask", lambda *args, **kwargs: pytest.fail("graph queried"))
        with pytest.raises(ValueError, match="unknown analysis parameters"):
            orchestration.run_tensile_analysis(PMDCO2, PMDCO2_SOURCE, parameters={"offset": OFFSET})
    assert fetch.fetched == []


def test_invalid_parameters_fail_inside_the_specimen_workflow(curves):
    """A standalone per-specimen run: the analysis_parameters node itself raises."""
    fetch = stub_fetch(curves)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sources, "fetch_bytes", fetch)
        with pytest.raises(ValueError, match="unknown analysis parameters"):
            orchestration.build_specimen_node().run(
                profile_id=PMDCO2, source=PMDCO2_SOURCE, specimen_id="Zx1", parameters={"offset": OFFSET}
            )
    with pytest.raises(ValueError, match="unknown analysis parameters"):
        workflows.analysis_parameters({"offset": OFFSET})  # the node, called directly
