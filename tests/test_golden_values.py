"""Golden values: the curve analysis on the synthetic cases must not change unnoticed.

The fixture ``golden_curve_values.json`` holds the results of the analysis on
every synthetic case of the curve tests, on noisy variants of the reference
case and on a record whose extensometer ends early. Restructuring the code
must reproduce it: floating-point values to a relative tolerance of 1e-9,
everything else exactly. A deliberate change of the method updates the
fixture, and the diff of the fixture shows what changed.

To rewrite the fixture from the current code, run this module with the
environment variable ``GOLDEN_WRITE=1``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import synthetic as sy

from pmd_tensile.orchestration import analyse_curve

FIXTURE = Path(__file__).with_name("golden_curve_values.json")
REL_TOL = 1e-9


def _cases() -> dict:
    """Every case as (curve, keyword arguments of the analysis)."""
    a = sy.reference_a()
    rate_step = sy.discontinuous(rate_step=0.006)
    plateau = rate_step.segment("plateau")
    dip = plateau.start + int(np.argmin(rate_step.stress_mpa[plateau.start : plateau.stop]))
    cases = {
        "a": (a, {}),
        "ii": (sy.discontinuous(serration=3.0), {}),
        "iii": (sy.discontinuous(plateau=None), {}),
        "iv": (sy.discontinuous(upper_peak=False, serration=0.0), {}),
        "a-phase": (sy.discontinuous(serration_phase=0.5), {}),
        "ro15": (sy.ramberg_osgood(15), {}),
        "ro20": (sy.ramberg_osgood(20), {}),
        "ro5": (sy.ramberg_osgood(5), {}),
        "too-few-elastic": (sy.ramberg_osgood(15, elastic_step=20.0, elastic_offset=15.0), {}),
        "flat-top": (sy.discontinuous(hardening="flat_top"), {}),
        "rise-below-reh": (sy.discontinuous(rm=395.0, ag_p=0.06, p_end=0.12), {}),
        "rate-step": (rate_step, {}),
        "rate-step-switch": (rate_step, {"rate_switch_strain": float(rate_step.strain[dip])}),
        "quadratic": (sy.discontinuous(elastic_quadratic=2.44e-10), {}),
        "a-extensometer-nan": (sy.extensometer_end(a, "nan"), {"a_percent": 25.0}),
        "ro15-noisy": (sy.noisy(sy.ramberg_osgood(15), 0), {}),
    }
    for seed in range(5):
        cases[f"a-noisy-{seed}"] = (sy.noisy(a, seed), {})
    return cases


def _values(c: sy.SyntheticCurve, kwargs: dict) -> dict:
    r = analyse_curve(*c.args(), **kwargs)
    return {
        "yield_type": r.yield_type,
        "noise_mpa": r.noise_mpa,
        "k_ext": r.k_ext,
        "e_gpa": r.e_gpa,
        "strain_origin": r.strain_origin,
        "e_window_mpa": list(r.elastic.window_mpa),
        "e_n": r.elastic.n,
        "e_stderr_gpa": r.elastic.stderr_gpa,
        "reh_mpa": r.reh_mpa,
        "rel_mpa": r.rel_mpa,
        "rel_incl_transient_mpa": r.yielding.rel_incl_transient_mpa,
        "rp02_mpa": r.rp02_mpa,
        "fm_n": r.fm_n,
        "rm_mpa": r.rm_mpa,
        "agt_percent": r.agt_percent,
        "ag_percent": r.ag_percent,
        "e_window_range_gpa": [r.e_min_gpa, r.e_max_gpa],
        "rp02_window_range_mpa": [r.rp02_min_mpa, r.rp02_max_mpa],
        "e_linearity_rel_diff": r.sensitivity.linearity_rel_diff,
        "flags": sorted(r.flags),
    }


def _same(expected, actual) -> bool:
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(map(_same, expected, actual))
    if isinstance(expected, float) or isinstance(actual, float):
        expected, actual = float(expected), float(actual)
        if math.isnan(expected) or math.isnan(actual):
            return math.isnan(expected) and math.isnan(actual)
        return math.isclose(expected, actual, rel_tol=REL_TOL, abs_tol=0.0)
    return expected == actual


def test_golden_values():
    actual = {name: _values(c, kwargs) for name, (c, kwargs) in _cases().items()}
    if os.environ.get("GOLDEN_WRITE") == "1":
        FIXTURE.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"fixture written: {FIXTURE.name}")
    expected = json.loads(FIXTURE.read_text())
    assert sorted(actual) == sorted(expected)
    differences = [
        f"{case}.{key}: expected {expected[case][key]!r}, got {actual[case][key]!r}"
        for case in expected
        for key in expected[case]
        if not _same(expected[case][key], actual[case].get(key))
    ]
    assert not differences, "\n".join(differences)
