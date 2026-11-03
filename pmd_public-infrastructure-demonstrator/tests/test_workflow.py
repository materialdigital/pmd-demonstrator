"""Two-stage workflow (decision D6): exploration, per-specimen workflow, batch (C9).

Offline: the graphs are the local pmdco2 Turtle files (and, for the pmdco3
parity, the structural pmdco3 fixture); curve downloads are stubbed with
synthetic CSV bytes (see ``workflow_helpers``).

Parity: the per-specimen workflow must give exactly the numbers of
``analyse_curve`` on the same parsed arrays (analysis-api.md), compared
NaN-aware.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re

import numpy as np
import pandas as pd
import pytest
from workflow_helpers import (
    PMDCO2,
    PMDCO2_SOURCE,
    PMDCO3,
    curve_bytes_for,
    pmdco3_twin_of_pmdco2,
    run_with_stubbed_curves,
    stub_fetch,
)

from tensile_workflow import acquisition, exploration, sources, workflows
from tensile_workflow.analysis import (
    CurveAnalysis,
    analyse_curve,
    elongation_after_fracture,
    reduction_of_area,
    s0_scatter,
)

SPECIMENS = ("Zx1", "Zy1", "Zd2")  # one per orientation

# ---------------------------------------------------------------------------
# NaN-aware equality
# ---------------------------------------------------------------------------


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


def differing_fields(a: CurveAnalysis, b: CurveAnalysis) -> list[str]:
    return [f.name for f in dataclasses.fields(CurveAnalysis) if not same(getattr(a, f.name), getattr(b, f.name))]


def reference_analysis(profile_id: str, source: dict, specimen_id: str, data: bytes):
    """``analyse_curve`` and the metadata results on the parsed arrays, outside any workflow."""
    record = acquisition.build_specimen_record(profile_id, source, specimen_id)
    curve = sources.parse_curve_csv(data, record.curve_columns, source_url=record.curve_url)
    a = elongation_after_fracture(record.l0_mm, record.lu_mm)
    z = reduction_of_area(record.s0_mm2, record.su_mm2)
    scatter = s0_scatter(record.s0_mm2, record.cross_section_measurements_mm2)
    analysis = analyse_curve(
        curve.force_n,
        curve.strain,
        curve.crosshead_mm,
        record.s0_mm2,
        record.l0_mm,
        a_percent=a.value,
        rate_switch_strain=record.rate_switch_strain,
    )
    return record, curve, analysis, a, z, scatter


SUMMARY_FIELDS = {
    "yield_type": "yield_type",
    "e_gpa": "e_gpa",
    "reh_mpa": "reh_mpa",
    "rel_mpa": "rel_mpa",
    "rp02_mpa": "rp02_mpa",
    "fm_n": "fm_n",
    "rm_mpa": "rm_mpa",
    "ag_percent": "ag_percent",
    "agt_percent": "agt_percent",
    "e_stderr_gpa": "e_stderr_gpa",
    # R8-3: the spec's report items (analysis-api.md amendment).
    "ag_location_halfwidth_percent": "ag_location_halfwidth_percent",
    "ag_window_range_percent": "ag_window_range_percent",
    "agt_window_range_percent": "agt_window_range_percent",
    "e_window_range_gpa": "e_window_range_gpa",
    "e_linearity_rel_diff": "e_linearity_rel_diff",
    "e_lag1_rho": "e_lag1_rho",
    "e_n_eff": "e_n_eff",
    "e_n": "e_n",
    "e_r2": "e_r2",
    "e_window_mpa": "e_window_mpa",
    "rel_incl_transient_mpa": "rel_incl_transient_mpa",
    "reh_sample_increment_mpa": "reh_sample_increment_mpa",
    "strain_origin": "strain_origin",
    "dropped_samples": "dropped_samples",
}


# ---------------------------------------------------------------------------
# Fixtures: one pmdco2 run over three specimens, shared by the tests below
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pmdco2_curves() -> dict:
    return curve_bytes_for(SPECIMENS)


@pytest.fixture(scope="module")
def pmdco2_run(pmdco2_curves):
    result, fetched = run_with_stubbed_curves(pmdco2_curves, PMDCO2, PMDCO2_SOURCE, specimen_ids=SPECIMENS)
    return result, fetched


# ---------------------------------------------------------------------------
# Stage 1: exploration
# ---------------------------------------------------------------------------


def test_exploration_lists_the_ten_specimens():
    run = workflows.build_exploration_workflow().run(profile_id=PMDCO2, source=PMDCO2_SOURCE)
    assert run.outputs.checked_profile_id == PMDCO2
    ids = run.outputs.specimen_ids
    assert len(ids) == 10
    prefixes = [re.match(r"Z[xyd]", sid).group(0) for sid in ids]
    assert {p: prefixes.count(p) for p in set(prefixes)} == {"Zx": 4, "Zy": 4, "Zd": 2}


def test_exploration_fails_early_on_a_profile_mismatch():
    with pytest.raises(exploration.ConformanceError, match=PMDCO3):
        workflows.build_exploration_workflow().run(profile_id=PMDCO3, source=PMDCO2_SOURCE)


# ---------------------------------------------------------------------------
# Stage 2: batch, end to end
# ---------------------------------------------------------------------------


def test_batch_runs_end_to_end(pmdco2_run, pmdco2_curves):
    result, fetched = pmdco2_run
    assert result.batch_run.status == "finished"
    assert result.specimen_ids == SPECIMENS
    assert sorted(url.rsplit("/", 1)[-1] for url in fetched) == sorted(f"{s}.csv" for s in SPECIMENS)

    table = result.results
    assert list(table["specimen_id"]) == list(SPECIMENS)
    assert table["orientation"].nunique() == 3
    for sid in SPECIMENS:
        row = table.set_index("specimen_id").loc[sid]
        record = acquisition.build_specimen_record(PMDCO2, PMDCO2_SOURCE, sid)
        assert row["curve_sha256"] == hashlib.sha256(pmdco2_curves[sid]).hexdigest()
        assert isinstance(row["curve_sha256"], str) and isinstance(row["curve_url"], str)
        assert row["curve_url"] == record.curve_url
        assert row["orientation"] == record.orientation
        assert row["yield_type"] == "discontinuous"


def test_orientation_summary_and_comparison_tables(pmdco2_run):
    result, _ = pmdco2_run
    summary = result.orientation_summary
    assert list(summary.columns) == [workflows.GROUP_COLUMN, "quantity", "n", "mean", "sd", "min", "max"]
    assert len(summary) == 3 * len(workflows.SUMMARY_QUANTITIES)
    reh = summary[summary["quantity"] == "reh_mpa"].set_index(workflows.GROUP_COLUMN)
    table = result.results.set_index(workflows.GROUP_COLUMN)
    for orientation, row in reh.iterrows():
        assert row["n"] == 1 and math.isnan(row["sd"])
        assert row["mean"] == table.loc[orientation, "reh_mpa"]

    comparison = result.comparison
    assert list(comparison.columns[:2]) == ["specimen_id", "orientation"]
    assert len(comparison) == 7 * len(SPECIMENS)
    assert set(comparison["outcome"]) <= {"within", "outside", "not_comparable"}
    assert set(comparison["quantity"]) == {"fm_n", "rm_mpa", "reh_mpa", "rp02_mpa", "e_gpa", "a_percent", "z_percent"}


def test_batch_results_equal_analyse_curve(pmdco2_run, pmdco2_curves):
    result, _ = pmdco2_run
    rows = result.results.set_index("specimen_id")
    for sid in SPECIMENS:
        record, _, expected, a, z, scatter = reference_analysis(PMDCO2, PMDCO2_SOURCE, sid, pmdco2_curves[sid])
        row = rows.loc[sid]
        for column, field in SUMMARY_FIELDS.items():
            assert same(row[column], getattr(expected, field)), (sid, column)
        assert same(row["a_percent"], a.value) and same(row["z_percent"], z.value)
        assert same(row["s0_u_rel"], scatter.u_rel)
        assert same(row["e_window_min_gpa"], expected.e_window_range_gpa[0])
        assert same(row["rp02_window_max_mpa"], expected.rp02_window_range_mpa[1])
        flags = set(row["flags"].split("; ")) - {""}
        assert set(expected.flags) <= flags


def test_per_specimen_workflow_equals_analyse_curve(pmdco2_curves):
    """Every CurveAnalysis field of the step-node composition equals analyse_curve (NaN-aware)."""
    sid = "Zx1"
    fetch = stub_fetch(pmdco2_curves)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sources, "fetch_bytes", fetch)
        run = workflows.build_specimen_node().run(
            profile_id=PMDCO2, source=PMDCO2_SOURCE, specimen_id=sid, curve_locator=None
        )
    record, curve, expected, _, _, _ = reference_analysis(PMDCO2, PMDCO2_SOURCE, sid, pmdco2_curves[sid])
    assert record.rate_switch_strain is not None  # the flag input is exercised
    assert differing_fields(run.outputs.analysis, expected) == []
    for port in ("e_gpa", "reh_mpa", "rel_mpa", "rp02_mpa", "fm_n", "rm_mpa", "ag_percent", "agt_percent"):
        assert same(getattr(run.outputs, port), getattr(expected, port)), port
    assert run.outputs.curve_sha256 == curve.sha256 == hashlib.sha256(pmdco2_curves[sid]).hexdigest()
    assert run.outputs.curve_url == record.curve_url
    assert run.outputs.n_samples == len(curve.force_n)
    assert isinstance(run.outputs.comparison, pd.DataFrame)


# ---------------------------------------------------------------------------
# Structure: flat recipes, JSON-able inputs
# ---------------------------------------------------------------------------

FLAT_RECIPE_TYPES = {"AtomicRecipe", "WorkflowRecipe", "ConstantRecipe"}


def _recipe_types(recipe) -> list[str]:
    types = []
    for node in recipe.nodes.values():
        types.append(type(node).__name__)
        if hasattr(node, "nodes"):
            types += _recipe_types(node)
    return types


def test_recipes_contain_no_flow_controllers():
    batch = workflows.build_batch_workflow(SPECIMENS)
    for recipe in (batch.recipe, workflows.exploration_workflow.flowrep_recipe):
        assert set(_recipe_types(recipe)) <= FLAT_RECIPE_TYPES


def test_batch_structure_and_inputs():
    batch = workflows.build_batch_workflow(SPECIMENS)
    specimen_nodes = [label for label in batch.nodes if re.fullmatch(r"specimen_\d+", label)]
    assert specimen_nodes == [f"specimen_{i}" for i in range(len(SPECIMENS))]
    assert {"gather_summaries", "gather_comparisons", "results", "orientation_summary", "comparison"} <= set(
        batch.nodes
    )
    inputs = workflows.batch_inputs(PMDCO2, PMDCO2_SOURCE, SPECIMENS)
    assert set(inputs) == set(batch.inputs.keys())
    assert inputs == {
        "profile_id": PMDCO2,
        "source": PMDCO2_SOURCE,
        "curve_locator": None,
        "specimen_id_0": "Zx1",
        "specimen_id_1": "Zy1",
        "specimen_id_2": "Zd2",
    }
    assert set(batch.outputs.keys()) == {"results", "orientation_summary", "comparison", "summaries"}


def test_builders_reject_bad_specimen_lists(pmdco2_curves):
    with pytest.raises(ValueError, match="at least one"):
        workflows.build_batch_workflow([])
    with pytest.raises(LookupError, match="Zq9"):
        run_with_stubbed_curves(pmdco2_curves, PMDCO2, PMDCO2_SOURCE, specimen_ids=["Zx1", "Zq9"])


# ---------------------------------------------------------------------------
# pmdco2 versus pmdco3 (acceptance criterion 4, offline)
# ---------------------------------------------------------------------------
#
# The structural pmdco3 fixture (workflow_helpers.py) has the PMDco3/TTO
# modelling pattern but made-up numbers. Here its values are set to the pmdco2
# values of the same specimens, so both profiles describe the same specimens;
# the curve bytes are identical by construction.

PARITY_SPECIMENS = ("Zx1", "Zd2")


def test_pmdco2_and_pmdco3_give_identical_analysis_for_identical_curve_bytes(tmp_path):
    curves = curve_bytes_for(PARITY_SPECIMENS)
    pmdco3_source = pmdco3_twin_of_pmdco2(PARITY_SPECIMENS, tmp_path)
    for sid in PARITY_SPECIMENS:  # the neutral records agree on what the analysis sees
        r2 = acquisition.build_specimen_record(PMDCO2, PMDCO2_SOURCE, sid)
        r3 = acquisition.build_specimen_record(PMDCO3, pmdco3_source, sid)
        for field in ("s0_mm2", "cross_section_measurements_mm2", "l0_mm", "lu_mm", "su_mm2", "orientation"):
            assert getattr(r2, field) == getattr(r3, field), (sid, field)
        assert r2.rate_switch_strain == pytest.approx(r3.rate_switch_strain, rel=1e-12)

    run2, fetched2 = run_with_stubbed_curves(curves, PMDCO2, PMDCO2_SOURCE, specimen_ids=PARITY_SPECIMENS)
    run3, fetched3 = run_with_stubbed_curves(curves, PMDCO3, pmdco3_source)

    # Only the profile ID and the source differ; the recipe is the same.
    assert run2.batch_workflow.recipe == run3.batch_workflow.recipe
    assert sorted(run3.specimen_ids) == sorted(PARITY_SPECIMENS)
    assert all(url.startswith("https://raw.githubusercontent.com/") for url in fetched3)  # non-semantic locator
    assert all(url.startswith("https://data.pmds-mpcdf-2.pyiron.com/") for url in fetched2)

    t2 = run2.results.set_index("specimen_id")
    t3 = run3.results.set_index("specimen_id")
    compared = [*SUMMARY_FIELDS, "a_percent", "z_percent", "s0_u_rel", "e_window_min_gpa", "e_window_max_gpa",
                "rp02_window_min_mpa", "rp02_window_max_mpa", "curve_sha256"]
    for sid in PARITY_SPECIMENS:
        for column in compared:
            assert same(t2.loc[sid, column], t3.loc[sid, column]), (sid, column)
        assert t2.loc[sid, "curve_url"] != t3.loc[sid, "curve_url"]
