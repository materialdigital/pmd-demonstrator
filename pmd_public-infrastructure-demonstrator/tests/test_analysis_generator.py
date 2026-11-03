"""Self-checks of the synthetic generator against method-spec §14.

These tests do not call the analysis; they make sure the generated curves and
their declared truths match the specification before any analysis test relies
on them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import synthetic as sy


@pytest.fixture(scope="module")
def a():
    return sy.reference_a()


def _segment_values(curve, name):
    seg = curve.segment(name)
    return curve.stress_mpa[seg.start : seg.stop], curve.plastic_strain[seg.start : seg.stop]


def test_segments_cover_the_record_contiguously(a):
    assert a.segments[0].start == 0
    for before, after in zip(a.segments, a.segments[1:]):
        assert before.stop == after.start
    assert a.segments[-1].stop == len(a)
    assert all(seg.density for seg in a.segments)


def test_contract_units_and_channel_relations(a):
    np.testing.assert_allclose(a.force_n, a.stress_mpa * sy.S0_MM2, rtol=1e-15)
    # r = de/(dc/L0) = 1 everywhere while the extensometer is valid.
    np.testing.assert_allclose(np.diff(a.strain), np.diff(a.crosshead_mm) / sy.L0_MM, atol=1e-15)
    assert np.all(np.diff(a.crosshead_mm) > 0)
    assert np.all(np.isfinite(a.strain)) and np.all(a.strain > 0)


def test_reference_a_start_point_and_elastic_law(a):
    assert a.stress_mpa[0] == 2.0
    assert a.strain[0] == pytest.approx(5e-4, abs=1e-15)
    s, p = _segment_values(a, "elastic")
    assert np.all(p == 0.0)
    assert np.all(np.diff(s) == 2.0)
    assert np.all(s % 2 == 1)  # odd stresses: never on an even window bound
    e = a.strain[a.segment("elastic").start : a.segment("elastic").stop]
    np.testing.assert_allclose(e, a.e0 + s / sy.E_MPA, rtol=1e-14)


def test_reference_a_rounded_peak(a):
    s, p = _segment_values(a, "rounded")
    assert s[-1] == 400.0 == a.stress_mpa.max(where=a.plastic_strain < 0.01, initial=0)
    increments = np.diff(s)
    assert np.all(increments > 0)
    # §14 (a): at most 0.5 MPa per sample near the peak (the last 15 samples,
    # i.e. the last 3.75 MPa below ReH).
    assert increments[-15:].max() <= 0.5
    assert s[-16] == pytest.approx(400.0 - 3.75)
    assert p[-1] <= 0.005  # M7: e_p(i_p) <= 0.005
    assert a.truth.reh_sample_increment_mpa == pytest.approx(s[-1] - s[-2], rel=1e-12)


def test_reference_a_transient_and_plateau(a):
    s_drop, _ = _segment_values(a, "drop")
    assert s_drop.min() == 372.0
    s_pl, p_pl = _segment_values(a, "plateau")
    assert s_pl.min() == 379.0 and s_pl.max() == 381.0
    assert np.median(s_pl) == pytest.approx(380.0, abs=0.05)
    assert p_pl[-1] == pytest.approx(0.015, abs=1e-12)
    assert p_pl[-1] - p_pl[0] >= 0.002  # plateau extent (M9)
    # Serration amplitude is below d = 0.005*ReH = 2 MPa peak-to-peak bound.
    assert s_pl.max() - s_pl.min() <= 2.0


def test_reference_a_hardening_maximum(a):
    s, p = _segment_values(a, "hardening")
    i = int(np.argmax(s))
    assert p[i] == pytest.approx(0.15, abs=1e-12)
    assert s[i] == pytest.approx(510.0, rel=1e-14)
    assert a.truth.fm_n == 510.0 * sy.S0_MM2
    assert np.all(s[: i + 1] < 510.0 + 1e-9)
    # Never returns to the plastic strain range below the maximum after it.
    assert np.all(np.diff(s[i:]) < 0)


def test_voce_band_halfwidth_matches_sqrt_2_n_delta(a):
    # VP-8 basis: half-width of F >= Fm (1 - delta) is sqrt(2 n delta).
    expected = 100.0 * math.sqrt(2.0 * a.truth.hardening_n * sy.BAND_REL_MIN)
    assert a.truth.ag_halfwidth_percent == pytest.approx(expected, rel=0.01)
    assert a.truth.ag_percent == pytest.approx(15.0, abs=0.02)


def test_reference_a_rp02_truth_is_in_the_plateau(a):
    assert a.truth.rp02_mpa == pytest.approx(380.0, abs=1e-9)


@pytest.mark.parametrize("serration, expected", [(1.0, 379.0), (3.0, 377.0)])
def test_m9_plateau_minima(serration, expected):
    curve = sy.discontinuous(serration=serration)
    s_pl, _ = _segment_values(curve, "plateau")
    assert s_pl.min() == expected == curve.truth.rel_mpa


def test_m9_case_iii_rises_monotonically_after_the_dip():
    curve = sy.discontinuous(plateau=None)
    seg = curve.segment("hardening")
    s = curve.stress_mpa[seg.start - 1 : seg.start + 1000]
    assert s[0] == 372.0
    assert np.all(np.diff(s) > 0)


def test_m9_case_iv_has_no_peak_above_the_plateau():
    curve = sy.discontinuous(upper_peak=False, serration=0.0)
    s_pl, _ = _segment_values(curve, "plateau")
    assert np.all(s_pl == 380.0)
    before = curve.stress_mpa[: curve.segment("plateau").start]
    assert before.max() == 380.0 and np.all(np.diff(before) > 0)


def test_rate_step_dip_and_step():
    curve = sy.discontinuous(rate_step=0.006)
    s_pl, p_pl = _segment_values(curve, "plateau")
    k = int(np.argmin(s_pl))
    assert s_pl[k] == 377.0  # 3 MPa below the plateau level
    # The switch snaps to the next serration phase 0 (one period = 4e-4).
    assert 0.006 <= p_pl[k] <= 0.006 + 4e-4 + 3e-5
    assert np.all(s_pl[k + 6 :] >= 394.0)  # +15 MPa step (+-1 serration)
    assert s_pl[k + 6 :].max() < 400.0  # still below ReH: inside the yield range


def test_edge_case_post_yield_maximum_below_reh():
    curve = sy.discontinuous(rm=395.0, ag_p=0.06, p_end=0.12)
    s_hard, _ = _segment_values(curve, "hardening")
    assert s_hard.max() == pytest.approx(395.0, rel=1e-14)
    i_peak = curve.segment("rounded").stop - 1
    assert curve.stress_mpa[i_peak + 1 :].max() < 400.0  # never returns to ReH
    # The rise exceeds the v2.3 band 2P - ReL + d = 2*380 - 379 + 2 = 383.
    assert curve.truth.rm_mpa > 383.0


@pytest.mark.parametrize("transient", [372.0, None])
def test_edge_case_no_work_hardening(transient):
    curve = sy.discontinuous(transient_min=transient, hardening=None, plateau_end_p=0.03)
    i_peak = curve.segment("rounded").stop - 1
    after = curve.stress_mpa[i_peak + 1 :]
    s_pl, _ = _segment_values(curve, "plateau")
    assert s_pl.max() == 381.0 and s_pl.min() == 379.0
    assert after.max() < 400.0
    assert curve.segments[-1].name == "plateau"
    assert math.isnan(curve.truth.fm_n)


@pytest.mark.parametrize("n", [15.0, 20.0, 5.0])
def test_ramberg_osgood_pre_yield_law(n):
    curve = sy.ramberg_osgood(n)
    seg = curve.segment("ro_fine")
    s = curve.stress_mpa[: seg.stop]
    e = curve.strain[: seg.stop]
    np.testing.assert_allclose(
        e - curve.e0 - s / sy.E_MPA, 0.002 * (s / 300.0) ** n, rtol=1e-9, atol=1e-15
    )
    fine = np.diff(curve.stress_mpa[seg.start : seg.stop])
    assert fine.max() <= 0.5 + 1e-9


def test_ramberg_osgood_truth_is_near_rp_for_high_n():
    curve = sy.ramberg_osgood(15)
    assert curve.truth.rp02_mpa == pytest.approx(300.0, abs=0.01)
    assert curve.truth.e_gpa == pytest.approx(205.0, rel=1e-5)


def test_ramberg_osgood_n5_window_slope_differs_from_e():
    # The low-n variant is where the analytic window slope matters.
    curve = sy.ramberg_osgood(5)
    assert abs(curve.truth.e_gpa / 205.0 - 1.0) > 1e-3


def test_too_few_elastic_points_in_the_m4_window():
    curve = sy.ramberg_osgood(15, elastic_step=20.0, elastic_offset=15.0)
    s, _ = sy.window_samples(curve, 300.0, 0.10, 0.40)
    assert 3 <= len(s) < 10


def test_noise_is_seeded_and_has_the_declared_level(a):
    one = sy.noisy(a, 7)
    two = sy.noisy(a, 7)
    other = sy.noisy(a, 8)
    np.testing.assert_array_equal(one.force_n, two.force_n)
    assert not np.array_equal(one.force_n, other.force_n)
    stress_noise = one.force_n / a.s0_mm2 - a.stress_mpa
    strain_noise = one.strain - a.strain
    assert np.std(stress_noise) == pytest.approx(0.5, rel=0.05)
    assert np.std(strain_noise) == pytest.approx(2e-6, rel=0.05)
    np.testing.assert_array_equal(one.crosshead_mm, a.crosshead_mm)


@pytest.mark.parametrize("mode", sy.END_MODES)
def test_extensometer_end_modes_change_only_the_tail(a, mode):
    curve = sy.extensometer_end(a, mode)
    k = a.index_at_plastic_strain(0.039)
    np.testing.assert_array_equal(curve.strain[:k], a.strain[:k])
    np.testing.assert_array_equal(curve.force_n, a.force_n)
    tail = curve.strain[k:]
    if mode == "freeze":
        assert np.all(tail == a.strain[k - 1])
    elif mode == "nan":
        assert np.all(np.isnan(tail))
    elif mode == "zero":
        assert np.all(tail == 0.0)
    else:
        assert np.all(np.diff(tail) > 0)
    assert curve.truth.k_ext == (len(a) if mode == "crosshead" else k)


def test_hold_keeps_crosshead_and_strain_constant(a):
    curve = sy.with_hold(a)
    seg = curve.segment("hold")
    hold = slice(seg.start, seg.start + 50)
    assert np.ptp(curve.crosshead_mm[seg.start - 1 : seg.start + 50]) == 0.0
    assert np.ptp(curve.strain[hold]) < 1e-15
    assert curve.stress_mpa[hold].min() == pytest.approx(curve.stress_mpa[seg.start - 1] - 3.0)
    assert curve.stress_mpa.max() == a.stress_mpa.max()


def test_temporary_flat_strain_recovers(a):
    curve = sy.with_temporary_flat_strain(a)
    i1 = a.index_at_plastic_strain(0.005)
    i2 = a.index_at_plastic_strain(0.008)
    i3 = a.index_at_plastic_strain(0.011)
    assert np.ptp(curve.strain[i1:i2]) == 0.0
    np.testing.assert_array_equal(curve.strain[i3:], a.strain[i3:])
    np.testing.assert_array_equal(curve.strain[:i1], a.strain[:i1])


def test_toe_is_below_every_window(a):
    curve = sy.with_toe(a)
    above = a.stress_mpa >= 12.0
    np.testing.assert_allclose(curve.strain[above] - a.strain[above], 1e-4, rtol=1e-9)
    below = ~above
    assert np.all(curve.strain[below] - a.strain[below] < 1e-4)
