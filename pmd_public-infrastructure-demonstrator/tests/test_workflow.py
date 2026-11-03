"""Two-stage workflow (decision D6): exploration, per-specimen workflow, batch (C9).

Offline: the graphs are the local pmdco2 Turtle files (and, for the pmdco3
parity, T3's structural pmdco3 fixture); curve downloads are stubbed with
synthetic CSV bytes in the layout of the S355 curve files (``;`` delimiter,
``,`` decimal mark, a header row and a unit row). The synthetic stresses are
method-spec §14 (a) with noise, scaled to each specimen's S0; their numbers
are made up, not lab data.

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
from conftest import PMDCO2_ORIENTATION_FILES

import synthetic as sy
from tensile_workflow import acquisition, exploration, sources, workflows
from tensile_workflow.analysis import (
    CurveAnalysis,
    analyse_curve,
    elongation_after_fracture,
    reduction_of_area,
    s0_scatter,
)

PMDCO2 = "pmdco2-tto"
PMDCO3 = "pmdco3-tto"
SPECIMENS = ("Zx1", "Zy1", "Zd2")  # one per orientation
PMDCO2_SOURCE = {"kind": "turtle", "locations": [str(p) for p in PMDCO2_ORIENTATION_FILES.values()]}
NOISE_SEEDS = {"Zx1": 101, "Zy1": 102, "Zd2": 103, "Zx2": 104}

# ---------------------------------------------------------------------------
# Synthetic curve bytes and the stubbed download (shared with test_separation)
# ---------------------------------------------------------------------------


def synthetic_csv(s0_mm2: float, seed: int) -> bytes:
    """§14 (a) with noise, as CSV bytes in the S355 layout, force scaled to ``s0_mm2``."""
    curve = sy.noisy(sy.reference_a(), seed)
    force_kn = curve.force_n / curve.s0_mm2 * s0_mm2 / 1000.0
    time_s = np.arange(len(curve)) * 0.1
    lines = ["Time;Crosshead separation;Load;Extensometer elongation", "(sec);(mm);(kN);(%)"]
    for row in zip(time_s, curve.crosshead_mm, force_kn, curve.strain * 100.0, strict=True):
        lines.append(";".join(f"{value:.10g}".replace(".", ",") for value in row))
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def curve_bytes_for(specimen_ids, profile_id: str = PMDCO2, source: dict = PMDCO2_SOURCE) -> dict:
    """Synthetic CSV bytes per specimen ID, force scaled to the record's S0."""
    return {
        sid: synthetic_csv(acquisition.build_specimen_record(profile_id, source, sid).s0_mm2, NOISE_SEEDS[sid])
        for sid in specimen_ids
    }


def stub_fetch(curves: dict):
    """A ``sources.fetch_bytes`` stand-in: serves the bytes by the URL's file name ``<ID>.csv``."""
    fetched = []

    def fetch(url: str) -> tuple[bytes, str]:
        fetched.append(url)
        data = curves[url.rsplit("/", 1)[-1].removesuffix(".csv")]
        return data, hashlib.sha256(data).hexdigest()

    fetch.fetched = fetched
    return fetch


def run_with_stubbed_curves(curves: dict, profile_id: str, source: dict, **kwargs):
    """``run_tensile_analysis`` with the curve download stubbed; returns (analysis, fetched URLs)."""
    fetch = stub_fetch(curves)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sources, "fetch_bytes", fetch)
        result = workflows.run_tensile_analysis(profile_id, source, **kwargs)
    return result, list(fetch.fetched)


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
# T3's structural pmdco3 fixture (tests/test_acquisition_pmdco3.py) has the
# PMDco3/TTO modelling pattern but made-up numbers. Here its values are set
# to the pmdco2 values of the same specimens, so both profiles describe the
# same specimens; the curve bytes are identical by construction.

PARITY_SPECIMENS = ("Zx1", "Zd2")


def _set_fixture_value(text: str, test: str, name: str, value) -> str:
    """Replace one value literal of T3's fixture, ``ex:<test>_<name>_value``."""
    literal = "NaN" if value is None else repr(float(value))
    pattern = re.compile(
        rf'(ex:{test}_{name}_value a obo:OBI_0001931 ;.*?pmdco:PMD_0000006 ")[^"]*(")', re.DOTALL
    )
    text, count = pattern.subn(rf"\g<1>{literal}\g<2>", text)
    assert count == 1, (test, name)
    return text


def pmdco3_twin_of_pmdco2(specimen_ids, tmp_path) -> dict:
    """A pmdco3 structural graph describing the given pmdco2 specimens with their values."""
    import test_acquisition_pmdco3 as t3

    text = t3._PREFIXES
    for i, sid in enumerate(specimen_ids, start=1):
        record = acquisition.build_specimen_record(PMDCO2, PMDCO2_SOURCE, sid)
        test = f"test_{i}"
        switch = round(100.0 * record.rate_switch_strain, 10)  # percent, as in the graphs
        block = t3._test(test, sid, record.orientation, switch=repr(switch))
        s1, s2, s3 = record.cross_section_measurements_mm2
        values = {
            "S0": record.s0_mm2, "s1": s1, "s2": s2, "s3": s3,
            "L0": record.l0_mm, "Lu": record.lu_mm, "Su": record.su_mm2,
        }
        for name, value in values.items():
            block = _set_fixture_value(block, test, name, value)
        text += block
    path = tmp_path / "pmdco3_twin.ttl"
    path.write_text(text, encoding="utf-8")
    return {"kind": "turtle", "locations": [str(path)]}


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
