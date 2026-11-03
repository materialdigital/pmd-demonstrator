"""Specification-based tests of ``analyse_curve`` (method-spec v2.3, T5).

Every expected value comes from the synthetic generator's declared truth
(``tests/synthetic.py``) or from the specification text, never from the
implementation's output. Tolerances are those of method-spec §14:

- noise-free: E +-0.1 % (or the analytic slope), ReH the exact sample value,
  ReL +-0.5 MPa, Rp0.2 +-0.5 MPa, Rm/Fm exact, Ag +-0.05 percentage points
  at the declared mid-point, k_ext +-1 crosshead window;
- noisy (sigma 0.5 MPa, e 2e-6): E +-2 %, ReH +-3 MPa, ReL +-2 MPa,
  Rp0.2 +-3 MPa, Rm +-0.5 %, Ag per VP-8 (+-100*sqrt(2 n delta) pp).

"Exact" is checked to a relative 1e-12, the floating-point resolution of
F/S0.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import re

import numpy as np
import pytest

import synthetic as sy
from tensile_workflow.analysis import AnalysisParameters, CurveAnalysis, analyse_curve

EXACT = 1e-12
E_REL_TOL = 1e-3
REL_TOL_MPA = 0.5
RP02_TOL_MPA = 0.5
AG_TOL_PP = 0.05
NOISY = {"e_rel": 0.02, "reh": 3.0, "rel": 2.0, "rp02": 3.0, "rm_rel": 0.005}
NOISY_SEEDS = (0, 1, 2, 3, 4)  # fixed before any run
CROSSHEAD_WINDOW_MM = 0.1

FLAG_PROPERTIES = ("e", "reh", "rel", "rp02", "fm", "ag", "agt", "extensometer", "window")
FLAG_NAMES = (
    "extensometer_invalid",
    "extensometer_ended_early",
    "insufficient_elastic_points",
    "e_window_not_converged",
    "no_upper_yield",
    "single_minimum",
    "in_yield_plateau",
    "no_work_hardening",
    "near_rate_switch",
    "missing_input",
    "no_s0_spread",
    # R8-5, analysis-api.md amendment of 2026-11-03, 22:22 CET.
    "not_reached",
    "no_maximum_force",
    "no_elastic_modulus",
    "no_complete_window",
)
FLAG_PATTERN = re.compile(rf"^({'|'.join(FLAG_PROPERTIES)}):({'|'.join(FLAG_NAMES)})$")

CASES = {
    "a": sy.reference_a,
    "ii": lambda: sy.discontinuous(serration=3.0),
    "iii": lambda: sy.discontinuous(plateau=None),
    "iv": lambda: sy.discontinuous(upper_peak=False, serration=0.0),
    "a-phase": lambda: sy.discontinuous(serration_phase=0.5),
    "ro15": lambda: sy.ramberg_osgood(15),
    "ro20": lambda: sy.ramberg_osgood(20),
    "ro5": lambda: sy.ramberg_osgood(5),
    "too-few-elastic": lambda: sy.ramberg_osgood(15, elastic_step=20.0, elastic_offset=15.0),
    "flat-top": lambda: sy.discontinuous(hardening="flat_top"),
    "rise-below-reh": lambda: sy.discontinuous(rm=395.0, ag_p=0.06, p_end=0.12),
    "rate-step": lambda: sy.discontinuous(rate_step=0.006),
    "quadratic": lambda: sy.discontinuous(elastic_quadratic=2.44e-10),
}


@functools.lru_cache(maxsize=None)
def curve(name: str) -> sy.SyntheticCurve:
    return CASES[name]()


@functools.lru_cache(maxsize=None)
def result(name: str) -> CurveAnalysis:
    return analyse_curve(*curve(name).args())


def run(c: sy.SyntheticCurve, **kwargs) -> CurveAnalysis:
    return analyse_curve(*c.args(), **kwargs)


def assert_nan(value) -> None:
    assert isinstance(value, float) and math.isnan(value), value


def flag_set(r: CurveAnalysis) -> set[str]:
    return set(r.flags)


def ag_tolerance_pp(truth: sy.Truth, stress_sd: float) -> float:
    """VP-8 tolerance, declared before any run: 100*sqrt(2 n delta) pp.

    n is the generator's curvature parameter of ln s at the maximum and
    delta = max(2e-4, 3*sigma/Rm) the M12 band depth for the true noise.
    """
    delta = max(sy.BAND_REL_MIN, 3.0 * stress_sd / truth.rm_mpa)
    return 100.0 * math.sqrt(2.0 * truth.hardening_n * delta)


def assert_k_ext_within_one_window(r: CurveAnalysis, c: sy.SyntheticCurve, expected: int) -> None:
    """k_ext within one crosshead window (0.1 mm; a sampled window ends at the
    first sample at or beyond 0.1 mm, so it spans < 0.1 mm + one sample)."""
    cross = c.crosshead_mm
    lo, hi = max(expected - 3, 1), min(expected + 3, len(cross) - 1)
    sample_travel = float(np.max(np.diff(cross[lo - 1 : hi + 1])))
    k = min(r.k_ext, len(cross) - 1)
    assert abs(cross[k] - cross[expected]) <= CROSSHEAD_WINDOW_MM + sample_travel + 1e-12


# ---------------------------------------------------------------------------
# Reference case (a) = M9 case (i), noise-free


def test_reference_a_values():
    c, r = curve("a"), result("a")
    t = c.truth
    assert r.yield_type == "discontinuous"
    assert r.e_gpa == pytest.approx(t.e_gpa, rel=E_REL_TOL)
    assert r.strain_origin == pytest.approx(t.strain_origin, abs=1e-9)
    assert r.reh_mpa == pytest.approx(t.reh_mpa, rel=EXACT)
    assert r.reh_sample_increment_mpa == pytest.approx(t.reh_sample_increment_mpa, rel=1e-6)
    assert r.rel_mpa == pytest.approx(t.rel_mpa, abs=REL_TOL_MPA)
    assert r.rel_incl_transient_mpa == pytest.approx(t.rel_incl_transient_mpa, rel=EXACT)
    assert r.rp02_mpa == pytest.approx(t.rp02_mpa, abs=RP02_TOL_MPA)
    assert r.fm_n == pytest.approx(t.fm_n, rel=EXACT)
    assert r.rm_mpa == pytest.approx(t.rm_mpa, rel=EXACT)
    assert r.ag_percent == pytest.approx(t.ag_percent, abs=AG_TOL_PP)
    assert r.agt_percent == pytest.approx(t.agt_percent, abs=AG_TOL_PP)
    assert r.ag_location_halfwidth_percent == pytest.approx(t.ag_halfwidth_percent, abs=AG_TOL_PP)
    assert r.k_ext == len(c)


def test_reference_a_flags():
    # M10: the Rp0.2 crossing lies in the yield range [i_p, i_h].
    assert flag_set(result("a")) == {"rp02:in_yield_plateau"}


def test_reference_a_elastic_fit_details():
    r = result("a")
    # M4: window [0.10, 0.40] * ReH = [40, 160] MPa; odd stresses 41..159.
    assert r.e_window_mpa == pytest.approx((40.0, 160.0), rel=EXACT)
    assert r.e_n == 60
    assert r.e_r2 == pytest.approx(1.0, abs=1e-12)
    assert r.noise_mpa == pytest.approx(0.0, abs=1e-9)  # M5: s_n of an exact line
    assert r.e_intercept_mpa == pytest.approx(-sy.E_MPA * curve("a").e0, rel=1e-9)


def test_window_sensitivity_is_zero_on_an_exactly_linear_curve():
    # Acceptance criterion 5 / M15: every window gives the same E.
    r = result("a")
    lo, hi = r.e_window_range_gpa
    assert lo == pytest.approx(205.0, rel=1e-9) and hi == pytest.approx(205.0, rel=1e-9)
    assert hi - lo <= 1e-9 * 205.0
    for rng in (r.rp02_window_range_mpa, r.ag_window_range_percent, r.agt_window_range_percent):
        assert rng[1] - rng[0] == pytest.approx(0.0, abs=1e-9)
    assert r.e_linearity_rel_diff == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# M9 ReL reference cases (VP-3)


@pytest.mark.parametrize(
    "name, expected_rel, single_minimum",
    [
        ("a", 379.0, False),  # (i) serrations +-1 < d
        ("a-phase", 379.05, False),  # (i) with the extremes between samples
        ("ii", 377.0, False),  # (ii) serrations +-3 > d
        ("iii", 372.0, True),  # (iii) monotone rise, no plateau
    ],
)
def test_m9_rel_cases(name, expected_rel, single_minimum):
    r = result(name)
    assert curve(name).truth.rel_mpa == expected_rel
    assert r.yield_type == "discontinuous"
    assert r.rel_mpa == pytest.approx(expected_rel, abs=REL_TOL_MPA)
    assert r.rel_incl_transient_mpa == pytest.approx(372.0, rel=EXACT)
    assert ("rel:single_minimum" in r.flags) is single_minimum


def test_m9_case_iv_plateau_without_upper_peak_is_continuous():
    c, r = curve("iv"), result("iv")
    assert r.yield_type == "continuous"
    assert_nan(r.reh_mpa)
    assert_nan(r.rel_mpa)
    assert {"reh:no_upper_yield", "rel:no_upper_yield"} <= flag_set(r)
    assert not any(f.startswith("rp02:") for f in r.flags)
    assert r.rp02_mpa == pytest.approx(c.truth.rp02_mpa, abs=RP02_TOL_MPA)
    assert r.e_gpa == pytest.approx(205.0, rel=E_REL_TOL)
    assert r.fm_n == pytest.approx(c.truth.fm_n, rel=EXACT)


@pytest.mark.parametrize("seed", NOISY_SEEDS)
def test_m9_case_v_noise_free_and_noisy_rel_agree(seed):
    noisy = run(sy.noisy(curve("a"), seed))
    assert noisy.rel_mpa == pytest.approx(result("a").rel_mpa, abs=NOISY["rel"])


# ---------------------------------------------------------------------------
# (b) continuous yielding, Ramberg-Osgood


@pytest.mark.parametrize("name", ["ro15", "ro20", "ro5"])
def test_ramberg_osgood_e_and_rp02_from_the_analytic_window_slope(name):
    c, r = curve(name), result(name)
    t = c.truth
    assert r.yield_type == "continuous"
    assert r.e_gpa == pytest.approx(t.e_gpa, rel=E_REL_TOL)
    assert r.rp02_mpa == pytest.approx(t.rp02_mpa, abs=RP02_TOL_MPA)
    assert r.fm_n == pytest.approx(t.fm_n, rel=EXACT)
    assert r.rm_mpa == pytest.approx(t.rm_mpa, rel=EXACT)
    assert_nan(r.reh_mpa)
    assert_nan(r.rel_mpa)
    assert flag_set(r) == {"reh:no_upper_yield", "rel:no_upper_yield"}
    assert r.k_ext == len(c)


def test_ramberg_osgood_n5_differs_from_e_and_from_rp():
    # Guards that the n = 5 check is discriminating: the analytic window
    # slope is not E, and Rp0.2 is not the RO parameter Rp.
    t = curve("ro5").truth
    assert abs(t.e_gpa - 205.0) > 205.0 * E_REL_TOL * 10
    assert abs(t.rp02_mpa - 300.0) > RP02_TOL_MPA


@pytest.mark.parametrize("name", ["ro15", "ro20"])
def test_ramberg_osgood_ag_at_the_declared_midpoint(name):
    c, r = curve(name), result(name)
    assert r.ag_percent == pytest.approx(c.truth.ag_percent, abs=AG_TOL_PP)
    assert r.agt_percent == pytest.approx(c.truth.agt_percent, abs=AG_TOL_PP)


# ---------------------------------------------------------------------------
# (d) too few elastic points


def test_too_few_elastic_points():
    c, r = curve("too-few-elastic"), result("too-few-elastic")
    assert_nan(r.e_gpa)
    assert "e:insufficient_elastic_points" in r.flags
    assert r.e_n < 10
    # Everything that needs E or the strain origin is undefined (M6, M10, M12).
    for value in (r.rp02_mpa, r.ag_percent, r.agt_percent, r.strain_origin):
        assert_nan(value)
    # Force-only properties are unaffected (M1).
    assert r.fm_n == pytest.approx(c.truth.fm_n, rel=EXACT)
    assert r.rm_mpa == pytest.approx(c.truth.rm_mpa, rel=EXACT)


# ---------------------------------------------------------------------------
# (e) noise variants, fixed seeds


@pytest.mark.parametrize("seed", NOISY_SEEDS)
def test_noisy_reference_a_within_the_noisy_tolerances(seed):
    c = sy.noisy(curve("a"), seed)
    r, t = run(c), c.truth
    assert r.yield_type == "discontinuous"
    assert r.e_gpa == pytest.approx(t.e_gpa, rel=NOISY["e_rel"])
    assert r.reh_mpa == pytest.approx(t.reh_mpa, abs=NOISY["reh"])
    assert r.rel_mpa == pytest.approx(t.rel_mpa, abs=NOISY["rel"])
    assert r.rp02_mpa == pytest.approx(t.rp02_mpa, abs=NOISY["rp02"])
    assert r.rm_mpa == pytest.approx(t.rm_mpa, rel=NOISY["rm_rel"])
    tol = ag_tolerance_pp(t, sy.STRESS_NOISE_MPA)
    assert r.ag_percent == pytest.approx(t.ag_peak_percent, abs=tol)


@pytest.mark.parametrize("seed", NOISY_SEEDS)
def test_noisy_reh_is_the_m8_sample_value(seed):
    # M7/M8 on raw samples: ReH = s(i_p), i_p = argmax s[0..j*], evaluated
    # independently from the spec text on the same noisy samples.
    c = sy.noisy(curve("a"), seed)
    reh, _ = sy.spec_reh(c)
    assert run(c).reh_mpa == pytest.approx(reh, rel=EXACT)


@pytest.mark.parametrize("seed", NOISY_SEEDS)
def test_noisy_ramberg_osgood_within_the_noisy_tolerances(seed):
    c = sy.noisy(curve("ro15"), seed)
    r, t = run(c), c.truth
    assert r.yield_type == "continuous"
    assert r.e_gpa == pytest.approx(t.e_gpa, rel=NOISY["e_rel"])
    assert r.rp02_mpa == pytest.approx(t.rp02_mpa, abs=NOISY["rp02"])
    assert r.rm_mpa == pytest.approx(t.rm_mpa, rel=NOISY["rm_rel"])
    tol = ag_tolerance_pp(t, sy.STRESS_NOISE_MPA)
    assert r.ag_percent == pytest.approx(t.ag_peak_percent, abs=tol)


# ---------------------------------------------------------------------------
# VP-1 bootstrap independence


def _assert_identical(r1: CurveAnalysis, r2: CurveAnalysis) -> None:
    for field in dataclasses.fields(CurveAnalysis):
        v1, v2 = getattr(r1, field.name), getattr(r2, field.name)
        if isinstance(v1, float):
            assert (math.isnan(v1) and math.isnan(v2)) or v1 == v2, field.name
        elif isinstance(v1, tuple) and v1 and isinstance(v1[0], float):
            np.testing.assert_array_equal(v1, v2, err_msg=field.name)
        else:
            assert v1 == v2, field.name


@pytest.mark.parametrize("name", ["a", "iv"])
@pytest.mark.parametrize("fraction", [0.4, 0.6])
def test_vp1_outputs_do_not_depend_on_the_provisional_window(name, fraction):
    # Exactly linear provisional window, no noise: every output is identical.
    params = AnalysisParameters(provisional_fraction_of_rm=fraction)
    _assert_identical(result(name), run(curve(name), params=params))


@pytest.mark.parametrize("fraction", [0.4, 0.6])
def test_vp1_continuous_fixed_point_is_independent_of_the_start(fraction):
    # M5 step 4 stops at a relative E change < 1e-4, so sigma_ref (and the
    # window bounds) may depend on the start below that; s_n is by M5 step 2
    # a statistic of the provisional window itself. Everything else is
    # identical.
    base = result("ro15")
    r = run(curve("ro15"), params=AnalysisParameters(provisional_fraction_of_rm=fraction))
    np.testing.assert_allclose(r.e_window_mpa, base.e_window_mpa, rtol=1e-6)
    assert r.e_reference_stress_mpa == pytest.approx(base.e_reference_stress_mpa, rel=1e-6)
    tolerant = {"noise_mpa", "e_window_mpa", "e_reference_stress_mpa"}
    stripped = {name: getattr(base, name) for name in tolerant}
    _assert_identical(dataclasses.replace(r, **stripped), base)


@pytest.mark.parametrize("fraction", [0.4, 0.6])
def test_vp1_noisy_property_values_do_not_depend_on_the_provisional_window(fraction):
    # With noise, s_n (M5 step 2) and hence delta_F (M12) change with the
    # provisional window, so Ag is not covered here (see the T5 report).
    c = sy.noisy(curve("a"), 0)
    base = run(c)
    r = run(c, params=AnalysisParameters(provisional_fraction_of_rm=fraction))
    for name in ("yield_type", "e_gpa", "reh_mpa", "rel_mpa", "rp02_mpa", "fm_n", "rm_mpa", "k_ext"):
        assert getattr(r, name) == getattr(base, name), name


# ---------------------------------------------------------------------------
# VP-4 S0 scaling (M14 exact propagation)

STRESS_FIELDS = (
    "noise_mpa",
    "e_gpa",
    "e_intercept_mpa",
    "e_stderr_gpa",
    "reh_mpa",
    "reh_sample_increment_mpa",
    "rel_mpa",
    "rel_incl_transient_mpa",
    "rp02_mpa",
    "rm_mpa",
    "e_reference_stress_mpa",
    "e_residual_noise_mpa",
    "rel_plateau_level_mpa",
)
STRESS_PAIRS = ("e_window_mpa", "e_window_range_gpa", "rp02_window_range_mpa")
INVARIANT_FIELDS = (
    "e_r2",
    "e_lag1_rho",
    "e_n_eff",
    "strain_origin",
    "fm_n",
    "agt_percent",
    "ag_percent",
    "ag_location_halfwidth_percent",
    "extensometer_max_valid_strain",
    "e_linearity_rel_diff",
    "rel_plateau_extent",
)
INVARIANT_PAIRS = ("ag_window_range_percent", "agt_window_range_percent")
IDENTICAL_FIELDS = (
    "yield_type",
    "k_ext",
    "e_n",
    "e_window_iterations",
    "dropped_samples",
    "flags",
    "indices",
)


def test_vp4_field_classification_covers_every_output():
    classified = set(
        STRESS_FIELDS + STRESS_PAIRS + INVARIANT_FIELDS + INVARIANT_PAIRS + IDENTICAL_FIELDS
    )
    assert classified == {f.name for f in dataclasses.fields(CurveAnalysis)}


def _scaled_pair(c: sy.SyntheticCurve, delta: float):
    base = run(c)
    scaled = analyse_curve(
        c.force_n, c.strain, c.crosshead_mm, c.s0_mm2 * (1.0 + delta), c.l0_mm
    )
    return base, scaled


@pytest.mark.parametrize("delta", [1e-3, 0.1])
@pytest.mark.parametrize("which", ["a", "a-noisy", "ro5"])
def test_vp4_stresses_scale_exactly_with_s0(which, delta):
    c = sy.noisy(curve("a"), 0) if which == "a-noisy" else curve(which)
    base, scaled = _scaled_pair(c, delta)
    factor = 1.0 / (1.0 + delta)
    for name in STRESS_FIELDS:
        np.testing.assert_allclose(
            getattr(scaled, name), getattr(base, name) * factor, rtol=EXACT, atol=1e-9, err_msg=name
        )
    for name in STRESS_PAIRS:
        np.testing.assert_allclose(
            getattr(scaled, name), np.asarray(getattr(base, name)) * factor, rtol=EXACT, atol=1e-9,
            err_msg=name,
        )
    for name in INVARIANT_FIELDS + INVARIANT_PAIRS:
        np.testing.assert_allclose(
            getattr(scaled, name), getattr(base, name), rtol=EXACT, atol=1e-14, err_msg=name
        )
    for name in IDENTICAL_FIELDS:
        assert getattr(scaled, name) == getattr(base, name), name


# ---------------------------------------------------------------------------
# VP-5 origin invariance


@pytest.mark.parametrize("name", ["a", "ro15"])
@pytest.mark.parametrize("shift", [0.0, 5e-4, 2e-3, "toe"])
def test_vp5_strain_origin_invariance(name, shift):
    c0 = curve(name)
    c = sy.with_toe(c0) if shift == "toe" else sy.with_strain_shift(c0, shift)
    base, r = result(name), run(c)
    assert r.e_gpa == pytest.approx(base.e_gpa, rel=1e-9)
    assert r.rp02_mpa == pytest.approx(base.rp02_mpa, abs=1e-9)
    assert r.ag_percent == pytest.approx(base.ag_percent, abs=1e-9)
    assert r.agt_percent == pytest.approx(base.agt_percent, abs=1e-9)
    moved = 1e-4 if shift == "toe" else shift
    assert r.strain_origin == pytest.approx(base.strain_origin + moved, abs=1e-12)
    assert r.flags == base.flags


# ---------------------------------------------------------------------------
# VP-6 analytic window slopes for e = s/E + q s^2 (M4, M15)


def _analytic_slope(c: sy.SyntheticCurve, f_lo: float, f_hi: float, ref: float) -> float:
    s, e = sy.window_samples(c, ref, f_lo, f_hi)
    return sy.ols(s, e)[0] / 1000.0


def test_vp6_final_fit_matches_the_analytic_window_slope():
    c, r = curve("quadratic"), result("quadratic")
    assert r.reh_mpa == pytest.approx(400.0, rel=EXACT)  # sigma_ref of the windows
    assert r.e_gpa == pytest.approx(_analytic_slope(c, 0.10, 0.40, 400.0), rel=1e-9)


def test_vp6_window_grid_range_matches_the_analytic_slopes():
    c, r = curve("quadratic"), result("quadratic")
    slopes = [
        _analytic_slope(c, lo, hi, 400.0)
        for lo in (0.05, 0.10, 0.15)
        for hi in (0.30, 0.40, 0.50)
    ]
    assert (max(slopes) - min(slopes)) / min(slopes) > 1e-3  # discriminating
    assert r.e_window_range_gpa[0] == pytest.approx(min(slopes), rel=1e-9)
    assert r.e_window_range_gpa[1] == pytest.approx(max(slopes), rel=1e-9)


def test_vp6_linearity_diagnostic_matches_the_analytic_slopes():
    c, r = curve("quadratic"), result("quadratic")
    e1 = _analytic_slope(c, 0.05, 0.25, 400.0)
    e2 = _analytic_slope(c, 0.25, 0.50, 400.0)
    diff = abs(e1 - e2)
    # The spec does not fix the denominator of the "relative difference";
    # any of E1, E2 or an E between them gives a value in this interval.
    assert diff / max(e1, e2) * (1 - 1e-9) <= abs(r.e_linearity_rel_diff)
    assert abs(r.e_linearity_rel_diff) <= diff / min(e1, e2) * (1 + 1e-9)


# ---------------------------------------------------------------------------
# VP-7 extensometer end modes (M1-M3)

SPEEDS_MM_S = (0.02, 0.5)


@functools.lru_cache(maxsize=None)
def speed_base(speed: float) -> sy.SyntheticCurve:
    dp = speed / (sy.SAMPLING_RATE_HZ * sy.L0_MM)
    return sy.discontinuous(hardening_dp=dp)


@pytest.mark.parametrize("speed", SPEEDS_MM_S)
@pytest.mark.parametrize("mode", ["freeze", "nan", "zero"])
def test_vp7_extensometer_removal_is_detected(mode, speed):
    base_curve = speed_base(speed)
    c = sy.extensometer_end(base_curve, mode)
    base, r = run(base_curve), run(c, a_percent=25.0)
    assert_k_ext_within_one_window(r, c, c.truth.k_ext)
    assert r.k_ext < len(c)
    # Ag/Agt are evaluated beyond k_ext (M2, M12).
    assert_nan(r.ag_percent)
    assert_nan(r.agt_percent)
    assert {"ag:extensometer_invalid", "agt:extensometer_invalid"} <= flag_set(r)
    # Independent end check (M3): 0.039 < 0.5 * 25 %.
    assert "extensometer:extensometer_ended_early" in r.flags
    # Force-only properties and strain properties before removal unchanged.
    assert r.fm_n == base.fm_n
    assert r.rm_mpa == base.rm_mpa
    for name in ("e_gpa", "reh_mpa", "rel_mpa", "rp02_mpa"):
        assert getattr(r, name) == pytest.approx(getattr(base, name), rel=1e-12), name


@pytest.mark.parametrize("speed", SPEEDS_MM_S)
def test_vp7_nan_tail_is_counted_as_dropped_strain(speed):
    c = sy.extensometer_end(speed_base(speed), "nan")
    r = run(c)
    assert r.dropped_samples["strain"] == len(c) - c.truth.k_ext
    assert r.dropped_samples["force"] == 0
    assert r.k_ext <= c.truth.k_ext  # M1: the first non-finite strain ends it


@pytest.mark.parametrize("speed", SPEEDS_MM_S)
def test_vp7_crosshead_substitution_is_not_detected(speed):
    # Documented residual risk (M3, A1): a strain channel continued from the
    # crosshead cannot be told apart from a valid extensometer.
    c = sy.extensometer_end(speed_base(speed), "crosshead")
    r = run(c, a_percent=25.0)
    assert r.k_ext == len(c)
    assert not any(f.startswith("extensometer:") for f in r.flags)
    assert not any(f.endswith(":extensometer_invalid") for f in r.flags)


@pytest.mark.parametrize("speed", SPEEDS_MM_S)
def test_vp7_hold_is_not_detected(speed):
    c = sy.with_hold(speed_base(speed))
    r, t = run(c, a_percent=25.0), c.truth
    assert r.k_ext == len(c)
    assert not any("extensometer" in f for f in r.flags)
    assert r.fm_n == pytest.approx(t.fm_n, rel=EXACT)
    assert r.ag_percent == pytest.approx(t.ag_percent, abs=AG_TOL_PP)
    assert r.rel_mpa == pytest.approx(t.rel_mpa, abs=REL_TOL_MPA)


def test_vp7_temporary_flat_strain_is_not_detected():
    c = sy.with_temporary_flat_strain(curve("a"))
    r, t = run(c), c.truth
    assert r.k_ext == len(c)
    assert not any("extensometer" in f for f in r.flags)
    assert r.e_gpa == pytest.approx(t.e_gpa, rel=E_REL_TOL)
    assert r.rel_mpa == pytest.approx(t.rel_mpa, abs=REL_TOL_MPA)
    assert r.rp02_mpa == pytest.approx(t.rp02_mpa, abs=RP02_TOL_MPA)
    assert r.ag_percent == pytest.approx(t.ag_percent, abs=AG_TOL_PP)


def test_m1_single_nan_strain_ends_the_extensometer():
    # M1: "the first non-finite e after a finite e marks extensometer end".
    c0 = curve("a")
    k = c0.index_at_plastic_strain(0.05)
    strain = c0.strain.copy()
    strain[k] = np.nan
    c = dataclasses.replace(c0, strain=strain)
    r = run(c)
    assert r.k_ext <= k
    assert_k_ext_within_one_window(r, c, k)
    assert_nan(r.ag_percent)
    assert "ag:extensometer_invalid" in r.flags


def test_m2_rp02_beyond_k_ext_is_nan_and_flagged():
    # With L0 = 200 mm one 0.1 mm crosshead window spans 5e-4 strain, so a
    # removal between ReH (p = 2.9e-4) and the Rp0.2 crossing (p = 0.002)
    # leaves ReH and E valid. The crosshead is rescaled consistently (r = 1).
    a = curve("a")
    long_gauge = dataclasses.replace(a, crosshead_mm=a.crosshead_mm * 4.0, l0_mm=200.0)
    c = sy.extensometer_end(long_gauge, "nan", at_plastic_strain=0.0015)
    r = run(c)
    assert_k_ext_within_one_window(r, c, c.truth.k_ext)
    assert_nan(r.rp02_mpa)
    assert "rp02:extensometer_invalid" in r.flags
    assert r.reh_mpa == pytest.approx(400.0, rel=EXACT)
    assert r.e_gpa == pytest.approx(205.0, rel=E_REL_TOL)
    assert r.fm_n == pytest.approx(c.truth.fm_n, rel=EXACT)


def _long_gauge(c: sy.SyntheticCurve) -> sy.SyntheticCurve:
    # L0 = 200 mm: one 0.1 mm crosshead window spans 5e-4 strain (see above).
    return dataclasses.replace(c, crosshead_mm=c.crosshead_mm * 4.0, l0_mm=200.0)


@pytest.mark.parametrize("mode", ["freeze", "nan", "zero"])
@pytest.mark.parametrize(
    "at_plastic_strain",
    [0.003, 0.010],
    ids=["extent-below-plateau-min-extent", "extent-above-plateau-min-extent"],
)
def test_m2_plateau_beyond_k_ext_gives_nan_rel(mode, at_plastic_strain):
    # R8-2: the extensometer ends inside the plateau (it runs to p = 0.015).
    # ReL is neither the transient minimum nor a plateau value.
    c = sy.extensometer_end(_long_gauge(curve("a")), mode, at_plastic_strain=at_plastic_strain)
    r = run(c)
    assert_k_ext_within_one_window(r, c, c.truth.k_ext)
    assert r.reh_mpa == pytest.approx(400.0, rel=EXACT)
    assert_nan(r.rel_mpa)
    assert "rel:extensometer_invalid" in r.flags
    assert "rel:single_minimum" not in r.flags
    assert r.fm_n == pytest.approx(c.truth.fm_n, rel=EXACT)


def test_m2_extensometer_ending_after_the_plateau_keeps_rel():
    # Ends in the hardening rise, above the band: the plateau is complete.
    # k_ext is the start of a crosshead window (2e-3 strain at L0 = 50 mm), so
    # the removal is placed more than one window after the plateau.
    c = sy.extensometer_end(curve("a"), "nan", at_plastic_strain=0.02)
    r = run(c)
    assert c.stress_mpa[r.k_ext - 1] > 381.0
    assert r.rel_mpa == pytest.approx(c.truth.rel_mpa, abs=REL_TOL_MPA)
    assert "rel:extensometer_invalid" not in r.flags


def test_m1_non_finite_forces_do_not_change_fm():
    c0 = curve("a")
    force = c0.force_n.copy()
    idx = [c0.index_at_plastic_strain(p) for p in (0.03, 0.05, 0.2)]
    force[idx] = np.nan
    r = run(dataclasses.replace(c0, force_n=force))
    assert r.dropped_samples["force"] == 3
    assert r.fm_n == pytest.approx(c0.truth.fm_n, rel=EXACT)
    assert r.rel_mpa == pytest.approx(c0.truth.rel_mpa, abs=REL_TOL_MPA)


def test_m3_extensometer_ended_early_is_not_raised_for_a_full_record():
    r = run(curve("a"), a_percent=25.0)
    assert not any(f.startswith("extensometer:") for f in r.flags)


# ---------------------------------------------------------------------------
# VP-8 near the maximum (deterministic parts)


def test_vp8_exact_flat_maximum_gives_its_midpoint():
    c, r = curve("flat-top"), result("flat-top")
    p1, p2 = 0.12, 0.16
    assert r.ag_percent == pytest.approx(100.0 * (p1 + p2) / 2, abs=AG_TOL_PP)
    assert r.ag_location_halfwidth_percent == pytest.approx(100.0 * (p2 - p1) / 2, abs=AG_TOL_PP)
    assert r.fm_n == pytest.approx(c.truth.fm_n, rel=EXACT)


def test_vp8_reh_above_the_post_yield_maximum_gives_fm_per_m11():
    # §8 v2.2/v2.3: the stress never returns to ReH but later rises above
    # the plateau band + d; Fm is the highest sample after that rise.
    c, r = curve("rise-below-reh"), result("rise-below-reh")
    t = c.truth
    assert r.yield_type == "discontinuous"
    assert r.reh_mpa == pytest.approx(400.0, rel=EXACT)
    assert r.fm_n == pytest.approx(t.fm_n, rel=EXACT)
    assert r.rm_mpa == pytest.approx(395.0, rel=EXACT)
    assert r.rel_mpa == pytest.approx(t.rel_mpa, abs=REL_TOL_MPA)
    assert r.ag_percent == pytest.approx(t.ag_percent, abs=AG_TOL_PP)
    assert not any(f.endswith(":no_work_hardening") for f in r.flags)


@pytest.mark.parametrize("phase", [0.0, 0.5])
def test_m11_rise_above_the_band_with_off_vertex_serrations(phase):
    c = sy.discontinuous(rm=395.0, ag_p=0.06, p_end=0.12, serration_phase=phase)
    r = run(c)
    assert r.fm_n == pytest.approx(c.truth.fm_n, rel=EXACT)
    assert r.rel_mpa == pytest.approx(c.truth.rel_mpa, abs=REL_TOL_MPA)


# ---------------------------------------------------------------------------
# §8 edge cases: no work hardening


def _assert_no_work_hardening(c: sy.SyntheticCurve, r: CurveAnalysis) -> None:
    assert r.yield_type == "discontinuous"
    assert_nan(r.fm_n)
    assert_nan(r.rm_mpa)
    assert {"fm:no_work_hardening", "rel:no_work_hardening"} <= flag_set(r)
    assert r.rel_mpa == pytest.approx(c.truth.rel_mpa, abs=REL_TOL_MPA)  # plateau minimum
    assert "rp02:in_yield_plateau" in r.flags


@pytest.mark.parametrize(
    "transient, phase",
    [(None, 0.0), (None, 0.5), (372.0, 0.0)],
    ids=["no-dip", "no-dip-off-vertex", "dip-vertex-sampled"],
)
def test_no_work_hardening_plateau(transient, phase):
    c = sy.discontinuous(
        transient_min=transient, hardening=None, plateau_end_p=0.03, serration_phase=phase
    )
    _assert_no_work_hardening(c, run(c))


def test_no_work_hardening_after_transient_dip_and_flat_serrated_plateau():
    # v2.3: work hardening needs sigma > 2P - ReL + d. The serrated plateau
    # (extremes between samples) never exceeds that, so Fm is NaN and ReL is
    # the plateau minimum, despite the transient dip lowering P.
    c = sy.discontinuous(transient_min=372.0, hardening=None, plateau_end_p=0.03, serration_phase=0.5)
    _assert_no_work_hardening(c, run(c))


# ---------------------------------------------------------------------------
# §8 v2.5: terminal force drop in the no-return branch (R8-1)
#
# Without a return to ReH, the M7 search threshold is 0.5*ReH = 200 MPa. An
# abrupt drop has no sample between that threshold and the plateau.


def _assert_same_values(with_tail: CurveAnalysis, without_tail: CurveAnalysis) -> None:
    names = (
        "yield_type", "noise_mpa", "e_gpa", "reh_mpa", "rel_mpa", "rel_incl_transient_mpa",
        "rel_plateau_level_mpa", "rel_plateau_extent", "rp02_mpa", "fm_n", "rm_mpa",
        "agt_percent", "ag_percent", "ag_location_halfwidth_percent", "flags", "indices",
    )
    for name in names:
        expected = getattr(without_tail, name)
        if isinstance(expected, float) and math.isnan(expected):
            assert_nan(getattr(with_tail, name))
        else:
            assert getattr(with_tail, name) == expected, name


@pytest.mark.parametrize("phase", [0.0, 0.5])
def test_v2_5_abrupt_terminal_drop_leaves_rel_and_fm_unchanged(phase):
    c = sy.discontinuous(rm=395.0, ag_p=0.06, p_end=0.12, serration_phase=phase)
    tailed = sy.with_terminal_drop(c, first_mpa=190.0)
    r = run(tailed)
    assert r.rel_mpa == pytest.approx(c.truth.rel_mpa, abs=REL_TOL_MPA)
    assert r.fm_n == pytest.approx(c.truth.fm_n, rel=EXACT)
    assert not any(f.endswith(":no_work_hardening") for f in r.flags)
    _assert_same_values(r, run(c))


@pytest.mark.parametrize("seed", NOISY_SEEDS)
def test_v2_5_abrupt_terminal_drop_on_a_noisy_curve(seed):
    c = sy.discontinuous(rm=395.0, ag_p=0.06, p_end=0.12)
    noisy = sy.noisy(c, seed)
    # Appended to the noisy samples, so both runs share the same pre-drop noise.
    tail = sy.with_terminal_drop(c, first_mpa=190.0)
    n = len(c)
    force = np.append(noisy.force_n, tail.force_n[n:])
    strain = np.append(noisy.strain, tail.strain[n:])
    r = analyse_curve(force, strain, tail.crosshead_mm, c.s0_mm2, c.l0_mm)
    assert r.rel_mpa == pytest.approx(c.truth.rel_mpa, abs=NOISY["rel"])
    assert r.fm_n == pytest.approx(c.truth.fm_n, rel=NOISY["rm_rel"])
    _assert_same_values(r, run(noisy))


@pytest.mark.parametrize("transient", [None, 372.0], ids=["no-dip", "dip"])
def test_v2_5_no_work_hardening_with_abrupt_terminal_drop(transient):
    c = sy.discontinuous(
        transient_min=transient, hardening=None, plateau_end_p=0.03, serration_phase=0.5
    )
    tailed = sy.with_terminal_drop(c, first_mpa=190.0)
    r = run(tailed)
    _assert_no_work_hardening(tailed, r)
    assert "rel:single_minimum" not in r.flags
    _assert_same_values(r, run(c))


def test_v2_5_remainder_ends_at_the_last_sample_above_the_threshold():
    # The 205 MPa sample stays in the remainder (ReL incl. transient); the
    # 5 MPa sample after it does not.
    c = sy.discontinuous(rm=395.0, ag_p=0.06, p_end=0.12)
    r = run(sy.with_terminal_drop(c, first_mpa=205.0, to_mpa=5.0, n=2))
    assert r.rel_incl_transient_mpa == pytest.approx(205.0, rel=EXACT)


def test_terminal_drop_does_not_change_a_curve_that_returns_to_reh():
    # The regular M11 branch has no truncation: Y ends at the return to ReH.
    c = curve("a")
    _assert_same_values(run(sy.with_terminal_drop(c, first_mpa=250.0, n=40)), result("a"))


@pytest.mark.xfail(
    strict=True,
    reason="v2.5 truncates only below 0.5*Rm_provisional; a gradual drop through the band "
    "keeps samples above it, so ReL and Fm change. Escalated to the human (Run 15, R8-1).",
)
@pytest.mark.parametrize(
    "first_mpa, n",
    [(388.5, 60), (350.0, 3)],
    ids=["linear-drop-of-the-r8-1-repro", "one-sample-below-the-plateau"],
)
def test_v2_5_gradual_terminal_drop_leaves_rel_and_fm_unchanged(first_mpa, n):
    c = sy.discontinuous(rm=395.0, ag_p=0.06, p_end=0.12, serration_phase=0.5)
    r = run(sy.with_terminal_drop(c, first_mpa=first_mpa, n=n))
    assert r.rel_mpa == pytest.approx(c.truth.rel_mpa, abs=REL_TOL_MPA)
    assert r.fm_n == pytest.approx(c.truth.fm_n, rel=EXACT)


# ---------------------------------------------------------------------------
# VP-9 rate step inside the plateau


def _dip_strain(c: sy.SyntheticCurve) -> float:
    seg = c.segment("plateau")
    k = seg.start + int(np.argmin(c.stress_mpa[seg.start : seg.stop]))
    assert c.stress_mpa[k] == 377.0
    return float(c.strain[k])


def test_vp9_rel_is_the_dip_minimum():
    c, r = curve("rate-step"), result("rate-step")
    assert r.rel_mpa == pytest.approx(377.0, abs=REL_TOL_MPA)
    assert r.rel_incl_transient_mpa == pytest.approx(372.0, rel=EXACT)
    assert not any(f.endswith(":near_rate_switch") for f in r.flags)
    assert r.fm_n == pytest.approx(c.truth.fm_n, rel=EXACT)


def test_vp9_near_rate_switch_flag_when_the_switch_strain_is_given():
    c = curve("rate-step")
    r = run(c, rate_switch_strain=_dip_strain(c))
    assert r.rel_mpa == pytest.approx(377.0, abs=REL_TOL_MPA)
    assert "rel:near_rate_switch" in r.flags


def test_vp9_no_near_rate_switch_flag_for_a_distant_switch():
    r = run(curve("rate-step"), rate_switch_strain=0.2)
    assert "rel:near_rate_switch" not in r.flags


def test_rate_switch_strain_does_not_change_values():
    c = curve("rate-step")
    plain, flagged = result("rate-step"), run(c, rate_switch_strain=_dip_strain(c))
    for name in ("e_gpa", "reh_mpa", "rel_mpa", "rp02_mpa", "fm_n", "ag_percent"):
        assert getattr(flagged, name) == getattr(plain, name), name


# ---------------------------------------------------------------------------
# Output conventions: flags and NaN (never None)


def _all_results():
    names = sorted(CASES)
    results = [(n, result(n)) for n in names]
    results.append(("a-noisy", run(sy.noisy(curve("a"), 0))))
    results.append(("nan-end", run(sy.extensometer_end(curve("a"), "nan"), a_percent=25.0)))
    rs = curve("rate-step")
    results.append(("rate-step-flagged", run(rs, rate_switch_strain=_dip_strain(rs))))
    return results


def _truncated(c: sy.SyntheticCurve, stop: int) -> sy.SyntheticCurve:
    return dataclasses.replace(
        c, force_n=c.force_n[:stop], strain=c.strain[:stop], crosshead_mm=c.crosshead_mm[:stop]
    )


def test_rp02_not_reached_in_a_record_with_valid_strain_to_its_end():
    # Discontinuous: the record ends in the plateau, before e_p = 0.002.
    a = curve("a")
    r = run(_truncated(a, a.index_at_plastic_strain(0.0015)))
    assert r.k_ext == a.index_at_plastic_strain(0.0015)
    assert_nan(r.rp02_mpa)
    assert "rp02:not_reached" in r.flags
    assert "rp02:extensometer_invalid" not in r.flags


def test_rp02_not_reached_in_an_elastic_record():
    # Continuous: the E window has no reference without a first crossing.
    c = curve("ro15")
    r = run(_truncated(c, c.index_at_plastic_strain(0.0005)))
    assert r.yield_type == "continuous"
    assert_nan(r.rp02_mpa)
    assert "rp02:not_reached" in r.flags


def test_rp02_not_reached_when_the_first_window_sample_exceeds_the_offset():
    a = curve("a")
    k = int(np.argmax(a.stress_mpa >= 0.10 * a.truth.reh_mpa))
    strain = a.strain.copy()
    strain[k] += 0.003
    r = run(dataclasses.replace(a, strain=strain))
    assert_nan(r.rp02_mpa)
    assert "rp02:not_reached" in r.flags


def test_rp02_extensometer_invalid_is_not_also_not_reached():
    # The removal case of test_m2_rp02_beyond_k_ext_is_nan_and_flagged.
    c = sy.extensometer_end(_long_gauge(curve("a")), "nan", at_plastic_strain=0.0015)
    assert "rp02:not_reached" not in run(c).flags


def test_ag_no_maximum_force():
    c = sy.discontinuous(hardening=None, plateau_end_p=0.03)
    r = run(c)
    assert_nan(r.fm_n)
    assert_nan(r.ag_percent)
    assert "ag:no_maximum_force" in r.flags
    assert "ag:no_elastic_modulus" not in r.flags


def test_ag_no_elastic_modulus():
    r = result("too-few-elastic")
    assert_nan(r.e_gpa)
    assert_nan(r.ag_percent)
    assert "ag:no_elastic_modulus" in r.flags
    assert "ag:no_maximum_force" not in r.flags


def test_extensometer_no_complete_window():
    a = curve("a")
    r = run(dataclasses.replace(a, crosshead_mm=a.crosshead_mm * 0.001))  # travel < 0.1 mm
    assert r.k_ext == len(a)
    assert "extensometer:no_complete_window" in r.flags
    assert "extensometer:missing_input" not in r.flags


def test_missing_crosshead_is_not_no_complete_window():
    a = curve("a")
    r = analyse_curve(a.force_n, a.strain, None, a.s0_mm2, a.l0_mm)
    assert "extensometer:missing_input" in r.flags
    assert "extensometer:no_complete_window" not in r.flags


@pytest.mark.parametrize("name", ["a", "ro15"])
def test_regular_curves_have_none_of_the_undefined_value_flags(name):
    new = ("rp02:not_reached", "ag:no_maximum_force", "ag:no_elastic_modulus", "extensometer:no_complete_window")
    assert not set(new) & flag_set(result(name))


def test_flags_are_property_colon_flag_strings():
    seen = set()
    for name, r in _all_results():
        assert isinstance(r.flags, tuple), name
        for flag in r.flags:
            assert isinstance(flag, str) and FLAG_PATTERN.match(flag), (name, flag)
            seen.add(flag)
    assert len(seen) >= 8  # the cases above exercise most of the vocabulary


def _walk(value, path):
    if value is None:
        raise AssertionError(f"None at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _walk(item, f"{path}[{key!r}]")
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            _walk(item, f"{path}[{i}]")


def test_undefined_values_are_nan_never_none():
    for name, r in _all_results():
        for field in dataclasses.fields(CurveAnalysis):
            if field.name == "indices":  # T4 addition; integer indices, not values
                continue
            _walk(getattr(r, field.name), f"{name}.{field.name}")
    r = result("iv")
    for value in (r.reh_mpa, r.rel_mpa, r.reh_sample_increment_mpa, r.rel_incl_transient_mpa):
        assert_nan(value)
