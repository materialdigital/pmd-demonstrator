"""Specification-based tests of the non-curve analysis functions (T5).

elastic_fit (M4, M16, VP-6), s0_scatter (M14), A and Z (M13),
compare_with_reported (§13), summarize_by_group (§12), check_time_order (M1)
and the module boundary of acceptance criterion 2. Expected values are
analytic.
"""

from __future__ import annotations

import ast
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import synthetic as sy
from pmd_tensile import analysis
from pmd_tensile._analysis import analyse_curve
from pmd_tensile.analysis import (
    check_time_order,
    compare_with_reported,
    elastic_fit,
    elongation_after_fracture,
    reduction_of_area,
    s0_scatter,
    summarize_by_group,
)
from pmd_tensile.contract import ReportedProperties

E = sy.E_MPA
E0 = 4.9e-4


def assert_nan(value) -> None:
    assert isinstance(value, float) and math.isnan(value), value


def _ols_reference(stress, strain):
    """Textbook OLS of stress on strain: slope, intercept, residuals."""
    slope, intercept = sy.ols(np.asarray(stress, float), np.asarray(strain, float))
    return slope, intercept, stress - (slope * strain + intercept)


# ---------------------------------------------------------------------------
# elastic_fit


def test_elastic_fit_exact_line():
    s = np.arange(3.0, 200.0, 2.0)
    e = E0 + s / E
    fit = elastic_fit(s, e, 40.0, 160.0)
    assert fit.e_gpa == pytest.approx(205.0, rel=1e-12)
    assert fit.intercept_mpa == pytest.approx(-E * E0, rel=1e-9)
    assert fit.n == 60  # odd stresses 41..159
    assert fit.r2 == pytest.approx(1.0, abs=1e-12)
    assert fit.stderr_gpa == pytest.approx(0.0, abs=1e-9)
    assert (fit.lo_mpa, fit.hi_mpa) == (40.0, 160.0)


def test_elastic_fit_window_bounds_are_inclusive():
    s = np.arange(0.0, 201.0, 1.0)
    fit = elastic_fit(s, E0 + s / E, 40.0, 160.0)
    assert fit.n == 121


def test_elastic_fit_needs_ten_points():
    s = np.arange(0.0, 201.0, 10.0)
    e = E0 + s / E
    fit = elastic_fit(s, e, 41.0, 139.0)  # 50, 60, ..., 130: nine samples
    assert fit.n == 9
    for value in (fit.e_gpa, fit.intercept_mpa, fit.r2):
        assert_nan(value)
    assert elastic_fit(s, e, 40.0, 130.0).n == 10
    assert elastic_fit(s, e, 40.0, 130.0).e_gpa == pytest.approx(205.0, rel=1e-12)


def test_elastic_fit_ignores_non_finite_samples():
    s = np.arange(3.0, 200.0, 2.0)
    e = E0 + s / E
    e[30] = np.nan
    s2 = s.copy()
    s2[31] = np.inf
    fit = elastic_fit(s2, e, 40.0, 160.0)
    assert fit.n == 58
    assert fit.e_gpa == pytest.approx(205.0, rel=1e-12)


def test_elastic_fit_vp6_analytic_slope_of_a_quadratic_law():
    q = 2.44e-10
    s = np.arange(3.0, 400.0, 2.0)
    e = E0 + s / E + q * s**2
    for lo, hi in ((20.0, 120.0), (40.0, 160.0), (60.0, 200.0)):
        keep = (s >= lo) & (s <= hi)
        slope, intercept, _ = _ols_reference(s[keep], e[keep])
        fit = elastic_fit(s, e, lo, hi)
        assert fit.e_gpa == pytest.approx(slope / 1000.0, rel=1e-9)
        assert fit.intercept_mpa == pytest.approx(intercept, rel=1e-9)


def test_elastic_fit_m16_standard_error():
    rng = np.random.default_rng(11)
    s_true = np.arange(41.0, 160.0, 2.0)
    e = E0 + s_true / E
    s = s_true + rng.normal(0.0, 0.5, s_true.size)
    slope, _, resid = _ols_reference(s, e)
    n = s.size
    se = math.sqrt(math.fsum(resid**2) / (n - 2) / math.fsum((e - e.mean()) ** 2))
    fit = elastic_fit(s, e, 0.0, 1e9)
    assert fit.n == n
    assert fit.e_gpa == pytest.approx(slope / 1000.0, rel=1e-12)
    assert fit.stderr_gpa == pytest.approx(se / 1000.0, rel=1e-9)
    ss_res = math.fsum(resid**2)
    ss_tot = math.fsum((s - s.mean()) ** 2)
    assert fit.r2 == pytest.approx(1.0 - ss_res / ss_tot, rel=1e-9)


@pytest.mark.parametrize("pattern", ["alternating", "slow"])
def test_elastic_fit_m16_lag1_autocorrelation_and_n_eff(pattern):
    e = E0 + np.linspace(40.0, 160.0, 60) / E
    i = np.arange(60)
    wiggle = (-1.0) ** i if pattern == "alternating" else np.sin(2 * np.pi * i / 30.0)
    s = E * (e - E0) + 0.3 * wiggle
    _, _, resid = _ols_reference(s, e)
    rho_ref = math.fsum(resid[:-1] * resid[1:]) / math.fsum(resid**2)
    fit = elastic_fit(s, e, 0.0, 1e9)
    # Lag-1 estimators differ by O(1/n); the sign and size must match.
    assert fit.lag1_rho == pytest.approx(rho_ref, abs=0.03)
    assert (fit.lag1_rho < -0.9) if pattern == "alternating" else (fit.lag1_rho > 0.9)
    rho = fit.lag1_rho
    assert fit.n_eff == pytest.approx(fit.n * (1 - rho) / (1 + rho), rel=1e-9)


# ---------------------------------------------------------------------------
# s0_scatter (M14) and its exact propagation (acceptance criterion 5)


def test_s0_scatter_three_measurements():
    result = s0_scatter(100.0, [99.0, 100.0, 101.0])
    assert result.u_rel == pytest.approx(1.0 / math.sqrt(3.0) / 100.0, rel=1e-12)
    assert result.dof == 2
    assert result.flags == ()


def test_s0_scatter_two_measurements():
    result = s0_scatter(100.0, [99.0, 101.0])
    assert result.u_rel == pytest.approx(math.sqrt(2.0) / math.sqrt(2.0) / 100.0, rel=1e-12)
    assert result.dof == 1


@pytest.mark.parametrize("measurements", [[], [80.0]])
def test_s0_scatter_needs_two_measurements(measurements):
    result = s0_scatter(80.0, measurements)
    assert_nan(result.u_rel)
    assert result.flags == ("s0:no_s0_spread",)


def test_s0_spread_propagates_to_the_relative_stress_uncertainty():
    # M14: u_rel(X) = u_rel(S0) for every stress-based X, strains unaffected.
    c = sy.reference_a()
    u = s0_scatter(c.s0_mm2, [79.9, 80.0, 80.1]).u_rel
    base = analyse_curve(*c.args())
    shifted = analyse_curve(c.force_n, c.strain, c.crosshead_mm, c.s0_mm2 * (1.0 + u), c.l0_mm)
    for name in ("rm_mpa", "reh_mpa", "rel_mpa", "rp02_mpa", "e_gpa"):
        rel_change = 1.0 - getattr(shifted, name) / getattr(base, name)
        assert rel_change == pytest.approx(u / (1.0 + u), rel=1e-9), name
    assert shifted.ag_percent == pytest.approx(base.ag_percent, rel=1e-12)
    assert shifted.fm_n == base.fm_n


# ---------------------------------------------------------------------------
# A and Z (M13)


def test_elongation_after_fracture():
    result = elongation_after_fracture(50.0, 62.5)
    assert result.value == pytest.approx(25.0, rel=1e-12)
    assert result.flags == ()


@pytest.mark.parametrize("l0, lu", [(50.0, None), (None, 62.5), (None, None)])
def test_elongation_after_fracture_missing_input(l0, lu):
    result = elongation_after_fracture(l0, lu)
    assert_nan(result.value)
    assert result.flags == ("a:missing_input",)


def test_reduction_of_area():
    result = reduction_of_area(80.0, 30.0)
    assert result.value == pytest.approx(62.5, rel=1e-12)
    assert result.flags == ()


@pytest.mark.parametrize("s0, su", [(80.0, None), (None, 30.0)])
def test_reduction_of_area_missing_input(s0, su):
    result = reduction_of_area(s0, su)
    assert_nan(result.value)
    assert result.flags == ("z:missing_input",)


# ---------------------------------------------------------------------------
# compare_with_reported (§13)

QUANTITIES = ("fm_n", "rm_mpa", "reh_mpa", "rp02_mpa", "e_gpa", "a_percent", "z_percent")
LAB = ReportedProperties(
    "Zx9",
    e_gpa=205.0,
    rp02_mpa=380.0,
    reh_mpa=400.0,
    rel_mpa=None,
    rm_mpa=510.0,
    fm_n=40800.0,
    a_percent=25.0,
    z_percent=70.0,
)


def _outcomes(computed: dict) -> dict:
    table = compare_with_reported(computed, LAB)
    assert list(table.columns) == [
        "quantity", "computed", "reported", "rel_deviation", "outcome", "tolerance"
    ]
    return dict(zip(table["quantity"], table["outcome"]))


def test_compare_has_one_row_per_quantity_and_no_rel_row():
    computed = dict.fromkeys(QUANTITIES, 1.0) | {"rel_mpa": 379.0}
    table = compare_with_reported(computed, LAB)
    assert tuple(table["quantity"]) == QUANTITIES
    assert not any("rel" in q.lower() for q in table["quantity"])
    assert all(isinstance(t, str) and t for t in table["tolerance"])


@pytest.mark.parametrize(
    "quantity, value, outcome",
    [
        ("fm_n", 40830.0, "within"),  # 0.074 %
        ("fm_n", 40770.0, "within"),
        ("fm_n", 40850.0, "outside"),  # 0.12 %
        ("rm_mpa", 510.9, "within"),  # tolerance 0.51 + 0.5 = 1.01 MPa
        ("rm_mpa", 509.1, "within"),
        ("rm_mpa", 511.2, "outside"),
        ("rm_mpa", 508.8, "outside"),
        ("reh_mpa", 407.0, "within"),  # 2 % = 8 MPa
        ("reh_mpa", 393.0, "within"),
        ("reh_mpa", 409.0, "outside"),
        ("reh_mpa", 391.0, "outside"),
        ("rp02_mpa", 386.0, "within"),
        ("rp02_mpa", 388.0, "outside"),
        ("e_gpa", 220.0, "within"),  # 10 %
        ("e_gpa", 190.0, "within"),
        ("e_gpa", 230.0, "outside"),
        ("e_gpa", 180.0, "outside"),
        ("a_percent", 24.6, "within"),  # rounds to 25
        ("a_percent", 25.4, "within"),
        ("a_percent", 24.5, "within"),  # tie: +-0.5 pp
        ("a_percent", 25.5, "within"),  # tie
        ("a_percent", 24.4, "outside"),
        ("a_percent", 25.6, "outside"),
        ("z_percent", 69.5, "within"),
        ("z_percent", 70.49, "within"),
        ("z_percent", 70.51, "outside"),
    ],
)
def test_compare_tolerances(quantity, value, outcome):
    assert _outcomes({quantity: value})[quantity] == outcome


@pytest.mark.parametrize("value", [None, float("nan")])
def test_compare_missing_computed_value_is_not_comparable(value):
    assert _outcomes({"rm_mpa": value})["rm_mpa"] == "not_comparable"


def test_compare_missing_reported_value_is_not_comparable():
    lab = ReportedProperties("Zx9", rm_mpa=None, fm_n=40800.0)
    table = compare_with_reported({"rm_mpa": 510.0, "fm_n": 40800.0}, lab)
    outcomes = dict(zip(table["quantity"], table["outcome"]))
    assert outcomes["rm_mpa"] == "not_comparable"
    assert outcomes["fm_n"] == "within"
    assert outcomes["e_gpa"] == "not_comparable"  # neither value given


def test_compare_relative_deviation():
    table = compare_with_reported({"fm_n": 40830.0}, LAB).set_index("quantity")
    assert abs(table.loc["fm_n", "rel_deviation"]) == pytest.approx(30.0 / 40800.0, rel=1e-12)
    assert table.loc["fm_n", "computed"] == 40830.0
    assert table.loc["fm_n", "reported"] == 40800.0


# ---------------------------------------------------------------------------
# summarize_by_group (§12)


def test_summarize_by_group():
    nan = float("nan")
    table = pd.DataFrame(
        {
            "orientation": ["y", "x", "x", "z", "x", "z"],
            "rm_mpa": [505.0, 500.0, 510.0, nan, 520.0, nan],
            "e_gpa": [nan, 200.0, nan, 205.0, 210.0, 207.0],
        }
    )
    summary = summarize_by_group(table, "orientation", ["rm_mpa", "e_gpa"])
    assert list(summary.columns) == ["orientation", "quantity", "n", "mean", "sd", "min", "max"]
    assert list(zip(summary["orientation"], summary["quantity"])) == [
        ("x", "rm_mpa"),
        ("x", "e_gpa"),
        ("y", "rm_mpa"),
        ("y", "e_gpa"),
        ("z", "rm_mpa"),
        ("z", "e_gpa"),
    ]
    rows = summary.set_index(["orientation", "quantity"])
    expected = {
        ("x", "rm_mpa"): (3, 510.0, 10.0, 500.0, 520.0),
        ("x", "e_gpa"): (2, 205.0, math.sqrt(50.0), 200.0, 210.0),
        ("y", "rm_mpa"): (1, 505.0, nan, 505.0, 505.0),
        ("y", "e_gpa"): (0, nan, nan, nan, nan),
        ("z", "rm_mpa"): (0, nan, nan, nan, nan),
        ("z", "e_gpa"): (2, 206.0, math.sqrt(2.0), 205.0, 207.0),
    }
    for key, (n, mean, sd, lo, hi) in expected.items():
        row = rows.loc[key]
        assert row["n"] == n, key
        np.testing.assert_allclose(
            [row["mean"], row["sd"], row["min"], row["max"]], [mean, sd, lo, hi], rtol=1e-12,
            err_msg=str(key),
        )


# ---------------------------------------------------------------------------
# M1 time order


def test_check_time_order():
    assert check_time_order(np.array([0.0, 0.1, 0.1, np.nan, 0.3])) == 5
    with pytest.raises(ValueError):
        check_time_order(np.array([0.0, 0.2, 0.1]))


# ---------------------------------------------------------------------------
# Module boundary (acceptance criterion 2)

ANALYSIS_PATH = Path(analysis.__file__)
FORBIDDEN = (
    "rdflib",
    "praeco",
    "pmd_tensile._sources",
    "pmd_tensile.profiles",
    "pmd_tensile.exploration",
    "pmd_tensile._exploration",
    "pmd_tensile.acquisition",
    "pmd_tensile._acquisition",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = "pmd_tensile" + ("." + base if base else "")
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in ("__import__", "import_module") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    names.add(arg.value)
    return names


def test_analysis_module_imports_no_ontology_aware_module():
    imported = _imported_modules(ANALYSIS_PATH)
    assert "pmd_tensile.contract" in imported or any(
        n.startswith("pmd_tensile.contract") for n in imported
    )
    for name in imported:
        for forbidden in FORBIDDEN:
            assert name != forbidden and not name.startswith(forbidden + "."), name
        assert "sparql" not in name.lower(), name


def test_analysis_module_source_has_no_pmd_ontology_iris():
    text = ANALYSIS_PATH.read_text(encoding="utf-8")
    assert "w3id.org/pmd" not in text


def test_importing_analysis_loads_no_ontology_aware_module():
    code = (
        "import sys; import pmd_tensile.analysis; "
        "print('\\n'.join(sorted(m for m in sys.modules "
        "if m.startswith('pmd_tensile') or m.split('.')[0] == 'praeco')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ANALYSIS_PATH.parents[1],
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    ).stdout.split()
    assert set(out) <= {
        "pmd_tensile",
        "pmd_tensile.analysis",
        "pmd_tensile.contract",
        "pmd_tensile._contract_units",
    }
