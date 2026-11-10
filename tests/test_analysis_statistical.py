"""Tolerances on noisy curves, ReH on noisy curves and the accuracy of Ag: Monte Carlo checks.

Opt-in: ``pytest --run-statistical -m statistical``. Each check runs the
implementation over seeded noisy curves and counts how often the result is
within the noisy tolerance of the **analytic** truth of the generator. The
required pass rate is at least 99 %. A NaN result counts as a failure.

Truths: (a) E = 205 GPa, ReH = 400 MPa, ReL = 379 MPa (plateau minimum),
Rp0.2 = 380 MPa (plateau stress at e_p = 0.002), Rm = 510 MPa, Ag = 15 %;
RO n = 15: E and Rp0.2 from the analytic window-slope fixed point,
Rm = 480 MPa, Ag = 12 %.

Tolerances (declared in advance): E +-2 %, ReH +-3 MPa, ReL +-2 MPa,
Rp0.2 +-3 MPa, Rm +-0.5 %; ReH on noisy curves: ReH in [true - 1.5, true + 1.5] MPa
at sd = 0.5 MPa, and ReH equal to the independent upper yield point evaluation
``synthetic.spec_reh`` in every seed; Ag: |Ag - Ag_true| <= 100 sqrt(2 n delta) pp with
delta = max(2e-4, 3 sigma/Rm) and n the generator's curvature parameter.

Seeds: ``SEEDS`` per configuration (at least 100 for ReH on noisy curves and
200 for Ag; their pass rates on the first 100/200 seeds are reported too).
Rates come with a Clopper-Pearson 95 % interval. A rate below 99 % is an
finding to be explained; tolerances are never loosened here.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import time

import numpy as np
import pytest
import synthetic as sy
from scipy.stats import beta

from pmd_tensile.orchestration import analyse_curve

pytestmark = pytest.mark.statistical

SEEDS = 1000
REQUIRED = 0.99
NOISE_LEVELS_MPA = (0.25, 0.5, 1.0)  # Ag accuracy: three noise levels
SEED_BASE = {("a", 0.25): 20_000, ("a", 0.5): 10_000, ("a", 1.0): 30_000, ("ro15", 0.5): 40_000}


@functools.lru_cache(maxsize=None)
def base_curve(name: str) -> sy.SyntheticCurve:
    return sy.reference_a() if name == "a" else sy.ramberg_osgood(15)


@functools.lru_cache(maxsize=None)
def runs(name: str, stress_sd: float):
    """Analyse SEEDS noisy copies; returns (results, seconds per call)."""
    base = base_curve(name)
    first = SEED_BASE[(name, stress_sd)]
    results = []
    start = time.perf_counter()
    for seed in range(first, first + SEEDS):
        c = sy.noisy(base, seed, stress_sd=stress_sd)
        results.append(analyse_curve(*c.args()))
    return results, (time.perf_counter() - start) / SEEDS


def ag_tolerance_pp(truth: sy.Truth, stress_sd: float) -> float:
    delta = max(sy.BAND_REL_MIN, 3.0 * stress_sd / truth.rm_mpa)
    return 100.0 * math.sqrt(2.0 * truth.hardening_n * delta)


@dataclasses.dataclass(frozen=True)
class Check:
    id: str
    case: str
    stress_sd: float
    field: str
    truth: float
    lo: float  # allowed error interval [lo, hi] around the truth
    hi: float
    spec_seeds: int = 100


def _checks() -> list[Check]:
    ta, tr = base_curve("a").truth, base_curve("ro15").truth
    checks = [
        Check("E (a)", "a", 0.5, "e_gpa", ta.e_gpa, -0.02 * ta.e_gpa, 0.02 * ta.e_gpa),
        Check("E (RO n=15)", "ro15", 0.5, "e_gpa", tr.e_gpa, -0.02 * tr.e_gpa, 0.02 * tr.e_gpa),
        Check("ReH", "a", 0.5, "reh_mpa", ta.reh_mpa, -3.0, 3.0),
        Check("ReH in [true-1.5, true+1.5]", "a", 0.5, "reh_mpa", ta.reh_mpa, -1.5, 1.5),
        Check("ReL", "a", 0.5, "rel_mpa", ta.rel_mpa, -2.0, 2.0),
        Check("Rp0.2 (a, in plateau)", "a", 0.5, "rp02_mpa", ta.rp02_mpa, -3.0, 3.0),
        Check("Rp0.2 (RO n=15)", "ro15", 0.5, "rp02_mpa", tr.rp02_mpa, -3.0, 3.0),
        Check("Rm (a)", "a", 0.5, "rm_mpa", ta.rm_mpa, -0.005 * ta.rm_mpa, 0.005 * ta.rm_mpa),
        Check("Rm (RO n=15)", "ro15", 0.5, "rm_mpa", tr.rm_mpa, -0.005 * tr.rm_mpa, 0.005 * tr.rm_mpa),
    ]
    for sd in NOISE_LEVELS_MPA:
        tol = ag_tolerance_pp(ta, sd)
        checks.append(
            Check(f"Ag (a, sigma={sd:g})", "a", sd, "ag_percent", ta.ag_peak_percent, -tol, tol, 200)
        )
    tol = ag_tolerance_pp(tr, 0.5)
    checks.append(
        Check("Ag (RO n=15, sigma=0.5)", "ro15", 0.5, "ag_percent", tr.ag_peak_percent, -tol, tol, 200)
    )
    return checks


CHECKS = _checks()


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    alpha = 1.0 - level
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


@functools.lru_cache(maxsize=None)
def evaluate(check: Check) -> dict:
    results, per_call = runs(check.case, check.stress_sd)
    values = np.array([getattr(r, check.field) for r in results], dtype=float)
    errors = values - check.truth
    ok = np.isfinite(errors) & (errors >= check.lo) & (errors <= check.hi)
    k, n = int(ok.sum()), len(ok)
    m = check.spec_seeds
    return {
        "check": check,
        "k": k,
        "n": n,
        "rate": k / n,
        "ci": clopper_pearson(k, n),
        "spec_rate": float(ok[:m].mean()),
        "mean_error": float(np.nanmean(errors)),
        "worst_low": float(np.nanmin(errors)),
        "worst_high": float(np.nanmax(errors)),
        "nan": int((~np.isfinite(values)).sum()),
        "per_call_s": per_call,
    }


# Checks whose rate is below 99 %, by name, with the reason; a tolerance is never
# loosened here. None at present.
ESCALATED: dict[str, str] = {}


def _param(check: Check):
    marks = []
    if check.id in ESCALATED:
        marks.append(pytest.mark.xfail(strict=True, reason=ESCALATED[check.id]))
    return pytest.param(check, id=check.id, marks=marks)


@pytest.mark.parametrize("check", [_param(c) for c in CHECKS])
def test_noisy_tolerance_is_met_in_99_percent_of_seeds(check):
    outcome = evaluate(check)
    assert outcome["rate"] >= REQUIRED, _row(outcome)


@functools.lru_cache(maxsize=None)
def m8_agreement() -> dict:
    """Implementation ReH versus the independent upper yield point evaluation, (a) sigma=0.5."""
    base = base_curve("a")
    first = SEED_BASE[("a", 0.5)]
    results, _ = runs("a", 0.5)
    equal = below = above = 0
    for seed, r in zip(range(first, first + SEEDS), results):
        reh, _ = sy.spec_reh(sy.noisy(base, seed))
        equal += (math.isnan(r.reh_mpa) and math.isnan(reh)) or math.isclose(r.reh_mpa, reh, rel_tol=1e-12)
        below += r.reh_mpa < base.truth.reh_mpa
        above += r.reh_mpa > base.truth.reh_mpa + 1.5
    return {"equal": equal, "below": below, "above": above}


def test_reh_equals_the_independent_evaluation_in_every_seed():
    # Separates the accuracy of ReH on noisy curves from the correctness of the implementation.
    assert m8_agreement()["equal"] == SEEDS


# ---------------------------------------------------------------------------
# A noisy plateau without upper peak: a known limitation of the method, measured and recorded
# ---------------------------------------------------------------------------

NO_PEAK_SEEDS = 1000
NO_PEAK_SEED_BASE = {0.25: 60_000, 0.5: 50_000, 1.0: 70_000}


def test_record_the_misclassification_of_a_noisy_plateau_without_upper_peak(capsys):
    """Case (iv) with noise: how often does the drop rule report a spurious discontinuous yield?

    This documents the rate and asserts no pass threshold: the limitation is
    accepted and stated in the method notebook; a prominence criterion would
    remove it. The expected classification is continuous, as in the noise-free case.
    """
    base = sy.discontinuous(upper_peak=False, serration=0.0)
    lines = []
    for stress_sd, first in NO_PEAK_SEED_BASE.items():
        results = [
            analyse_curve(*sy.noisy(base, seed, stress_sd=stress_sd).args())
            for seed in range(first, first + NO_PEAK_SEEDS)
        ]
        spurious = [r.reh_mpa for r in results if r.yield_type == "discontinuous"]
        k = len(spurious)
        lo, hi = clopper_pearson(k, NO_PEAK_SEEDS)
        rate = f"{k}/{NO_PEAK_SEEDS} = {100 * k / NO_PEAK_SEEDS:.1f} % (95 % CI [{100 * lo:.1f}, {100 * hi:.1f}] %)"
        reh = f"; spurious ReH {min(spurious):.2f} to {max(spurious):.2f} MPa (plateau 380 MPa)" if k else ""
        lines.append(f"- case (iv), sigma={stress_sd:g} MPa: classified discontinuous in {rate}{reh}")
    with capsys.disabled():
        print()
        print("Noisy plateau without upper peak (no pass threshold; serration 0):")
        print("\n".join(lines))


def _row(o: dict) -> str:
    c = o["check"]
    lo, hi = o["ci"]
    verdict = "pass" if o["rate"] >= REQUIRED else "FAIL (<99 %)"
    return (
        f"| {c.id} | {o['n']} | {o['k']}/{o['n']} = {100 * o['rate']:.1f} % "
        f"| [{100 * lo:.1f}, {100 * hi:.1f}] % | {100 * o['spec_rate']:.0f} % ({c.spec_seeds}) "
        f"| [{c.lo:+.3g}, {c.hi:+.3g}] | {o['mean_error']:+.3g} "
        f"| [{o['worst_low']:+.3g}, {o['worst_high']:+.3g}] | {o['nan']} | {verdict} |"
    )


def test_summary_table(capsys):
    header = (
        "| check | seeds | pass rate | 95 % CI (Clopper-Pearson) | rate on first spec seeds "
        "| error interval | mean error | error range | NaN | verdict |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = [_row(evaluate(c)) for c in CHECKS]
    timing = {key: runs(*key)[1] for key in SEED_BASE}
    classification = {
        key: sum(r.yield_type == ("discontinuous" if key[0] == "a" else "continuous") for r in runs(*key)[0])
        for key in SEED_BASE
    }
    with capsys.disabled():
        print()
        print(header)
        print("\n".join(rows))
        print()
        for key in SEED_BASE:
            print(
                f"- {key[0]} sigma={key[1]:g}: {1000 * timing[key]:.2f} ms per analyse_curve call; "
                f"expected yield type in {classification[key]}/{SEEDS} seeds"
            )
        m8 = m8_agreement()
        print(
            f"- ReH detail (a, sigma=0.5): implementation ReH equals the independent upper yield point "
            f"evaluation in {m8['equal']}/{SEEDS} seeds; ReH < true in {m8['below']}, "
            f"ReH > true + 3 sd in {m8['above']}"
        )
