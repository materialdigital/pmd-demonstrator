"""Synthetic tensile curves with known truth (method-spec v2.3, §14), for T5.

The generator is written from the method specification, not from the
implementation. Each curve is built from a noise-free engineering stress
``s`` (MPa) and a generator plastic strain ``p`` per sample. The channels in
contract units (method-spec §2) follow from them:

- extensometer strain (ratio): ``e = e0 + s/E + q*s**2 + p``;
- force (N): ``F = s * S0``;
- crosshead (mm): ``c = L0 * (e_clean - e0)``. The crosshead follows the
  specimen elongation without machine compliance, so the M3 ratio
  ``r = de / (dc / L0)`` is exactly 1 while the extensometer is valid.

``e0`` is fixed by the reference start point (§14 (a)): the first sample is at
``s = 2 MPa`` with ``e = 5e-4``. With ``q = 0`` the elastic part is exactly
linear, so the M4 fit gives ``E`` and ``e0 = -b/E`` exactly, and the method's
plastic strain ``e - e0 - s/E`` equals ``p``. ``q != 0`` gives the VP-6 curve
``e = s/E + q*s**2`` (plus ``e0``).

Default sampling densities (stated per segment, as §14 requires; every curve
records them in :attr:`SyntheticCurve.segments`):

=================  ==========================================================
segment            density
=================  ==========================================================
start              one sample at 2 MPa
elastic            2 MPa per sample at odd stresses 3, 5, 7, ... MPa, so that
                   no sample lies on a window bound f*sigma_ref (even values)
rounded pre-yield  C1 parabola from 340 MPa to ReH in 60 equal strain steps
                   (9.8e-6 per step); 2 MPa per sample at its start, at most
                   0.5 MPa per sample over the last 15 samples (the last
                   3.75 MPa below the peak); the peak is a sample
drop               2 MPa per sample, crosshead advancing 2/E per sample
recovery           1 MPa per sample (transient minimum to plateau level)
plateau            1e-5 plastic strain per sample; triangular serrations with
                   a period of 40 samples, so minima and maxima are samples
hardening          1e-4 plastic strain per sample (default) on the absolute
                   grid k*dp, which contains the maximum-force strain; for the
                   end-mode cases dp = v/(f_s*L0) with f_s = 10 Hz, i.e.
                   4e-5 at 0.02 mm/s and 1e-3 at 0.5 mm/s
RO pre-yield       2 MPa per sample (odd stresses) below 0.8*Rp, then
                   0.5 MPa per sample up to the switch to Voce hardening
=================  ==========================================================

Hardening is Voce in true stress over the plastic strain, converted to
engineering stress ``s = sigma_V(p) / (1 + p)``; its parameters are solved so
that the maximum is ``Rm`` at ``p = Ag`` exactly. Near the maximum,
``ln s = ln Rm - (p - Ag)**2 / (2 n)`` with ``n = (1 + Ag) * tau``, so the M12
band ``F >= Fm (1 - delta)`` has the half-width ``sqrt(2 n delta)`` (VP-8).

Truth values are analytic or read off the noise-free construction (for
example the plateau minimum is the lowest plateau sample by construction).
They are never taken from the implementation. Undefined truths are NaN.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy.optimize import brentq

NAN = float("nan")

E_MPA = 205_000.0
"""Young's modulus of the reference curves (205 GPa)."""
S0_MM2 = 80.0
L0_MM = 50.0
START_STRESS_MPA = 2.0
START_STRAIN = 5e-4
SAMPLING_RATE_HZ = 10.0
"""Assumed sampling rate for the crosshead-speed cases (c)."""

STRESS_NOISE_MPA = 0.5
STRAIN_NOISE = 2e-6
"""§14 (e) noise levels (standard deviations)."""

BAND_REL_MIN = 2e-4
"""M12 delta_F floor; the noise-free band uses it (s_n = 0)."""
OFFSET_STRAIN = 0.002
"""M10 offset plastic strain."""
SERRATION_PERIOD_SAMPLES = 40


@dataclasses.dataclass(frozen=True)
class Segment:
    """One generator segment: samples ``[start, stop)`` and their density."""

    name: str
    start: int
    stop: int
    density: str


@dataclasses.dataclass(frozen=True)
class Truth:
    """Expected values derived from the specification (NaN: undefined/unchecked)."""

    yield_type: str
    k_ext: int
    e_gpa: float = NAN
    strain_origin: float = NAN
    reh_mpa: float = NAN
    reh_sample_increment_mpa: float = NAN
    rel_mpa: float = NAN
    rel_incl_transient_mpa: float = NAN
    rp02_mpa: float = NAN
    fm_n: float = NAN
    rm_mpa: float = NAN
    ag_percent: float = NAN
    """Ag at the declared mid-point of the noise-free band (delta_F = 2e-4)."""
    agt_percent: float = NAN
    ag_halfwidth_percent: float = NAN
    ag_peak_percent: float = NAN
    """100 * plastic strain at the analytic force maximum."""
    hardening_n: float = NAN
    """n in sqrt(2 n delta) (VP-8): (1 + Ag) * tau of the Voce hardening."""


@dataclasses.dataclass(frozen=True, eq=False)
class SyntheticCurve:
    """A synthetic curve in contract units, with its noise-free construction."""

    force_n: np.ndarray
    strain: np.ndarray
    crosshead_mm: np.ndarray
    s0_mm2: float
    l0_mm: float
    stress_mpa: np.ndarray
    """Noise-free engineering stress per sample."""
    plastic_strain: np.ndarray
    """Generator plastic strain p per sample (noise-free)."""
    clean_strain: np.ndarray
    """Noise-free, unmodified extensometer strain."""
    e_mpa: float
    e0: float
    quadratic: float
    truth: Truth
    segments: tuple[Segment, ...]
    description: str

    def __len__(self) -> int:
        return len(self.force_n)

    def args(self) -> tuple:
        """Positional arguments of ``analyse_curve``."""
        return (self.force_n, self.strain, self.crosshead_mm, self.s0_mm2, self.l0_mm)

    def index_at_plastic_strain(self, p: float) -> int:
        """First sample with generator plastic strain >= ``p``."""
        return int(np.argmax(self.plastic_strain >= p))

    def segment(self, name: str) -> Segment:
        for segment in self.segments:
            if segment.name == name:
                return segment
        raise KeyError(name)


# ---------------------------------------------------------------------------
# Building blocks


class _Builder:
    def __init__(self, e_mpa: float, quadratic: float):
        self.e_mpa = e_mpa
        self.q = quadratic
        self.s: list[float] = []
        self.p: list[float] = []
        self.segments: list[Segment] = []

    def add(self, name: str, s, p, density: str) -> None:
        start = len(self.s)
        self.s.extend(float(v) for v in s)
        self.p.extend(float(v) for v in p)
        self.segments.append(Segment(name, start, len(self.s), density))

    def mech(self, s: float, p: float) -> float:
        """Mechanical strain e - e0 of a state."""
        return s / self.e_mpa + self.q * s * s + p

    @property
    def last(self) -> tuple[float, float]:
        return self.s[-1], self.p[-1]


def _e0(e_mpa: float, quadratic: float) -> float:
    return START_STRAIN - START_STRESS_MPA / e_mpa - quadratic * START_STRESS_MPA**2


def _assemble(
    b: _Builder,
    truth_kwargs: dict,
    description: str,
    *,
    s0_mm2: float = S0_MM2,
    l0_mm: float = L0_MM,
) -> SyntheticCurve:
    s = np.asarray(b.s, dtype=float)
    p = np.asarray(b.p, dtype=float)
    e0 = _e0(b.e_mpa, b.q)
    clean = e0 + s / b.e_mpa + b.q * s * s + p
    crosshead = l0_mm * (clean - e0)
    _check_construction(s, p, crosshead)
    truth = Truth(k_ext=len(s), **truth_kwargs)
    return SyntheticCurve(
        force_n=s * s0_mm2,
        strain=clean.copy(),
        crosshead_mm=crosshead,
        s0_mm2=s0_mm2,
        l0_mm=l0_mm,
        stress_mpa=s,
        plastic_strain=p,
        clean_strain=clean,
        e_mpa=b.e_mpa,
        e0=e0,
        quadratic=b.q,
        truth=truth,
        segments=tuple(b.segments),
        description=description,
    )


def _check_construction(s: np.ndarray, p: np.ndarray, crosshead: np.ndarray) -> None:
    """Physical sanity of the noise-free construction (not of the analysis)."""
    if np.any(np.diff(p) < -1e-15):
        raise AssertionError("generator plastic strain decreases")
    if np.any(np.diff(crosshead) < -1e-12):
        raise AssertionError("generator crosshead decreases")
    if np.any(s <= 0):
        raise AssertionError("generator stress must stay positive")


def _tri(j: int, period: int = SERRATION_PERIOD_SAMPLES, phase: float = 0.0) -> float:
    """Triangle wave at sample position (j + phase)/period.

    0 -> 0, 1/4 -> 1, 1/2 -> 0, 3/4 -> -1. With ``phase = 0`` the extremes
    are samples; with ``phase = 0.5`` no sample lies on an extreme or on the
    mean level.
    """
    x = (j + phase) % period
    quarter = period / 4
    if x <= quarter:
        return x / quarter
    if x <= 3 * quarter:
        return 2.0 - x / quarter
    return x / quarter - 4.0


@dataclasses.dataclass(frozen=True)
class Voce:
    """Engineering stress ``s(p) = sigma_V(p)/(1 + p)`` with Voce ``sigma_V``.

    ``sigma_V(p) = sat - (sat - a) exp(-(p - p_start)/tau)``; solved so that
    ``s(p_start) = s_start`` and the maximum of ``s`` is ``rm`` at ``p = p_max``.
    """

    p_start: float
    s_start: float
    rm: float
    p_max: float
    tau: float
    sat: float
    a: float

    @classmethod
    def solve(cls, p_start: float, s_start: float, rm: float, p_max: float) -> "Voce":
        if not (rm > s_start and p_max > p_start):
            raise ValueError("Voce hardening needs rm > s_start and p_max > p_start")
        a = s_start * (1.0 + p_start)
        top = rm * (1.0 + p_max)  # sigma_V at the maximum
        gap = top - a
        span = p_max - p_start

        # Maximum of s: sigma_V' = sigma_V/(1+p)  =>  sat = top + tau*rm.
        def f(tau: float) -> float:
            return tau * rm - (gap + tau * rm) * math.exp(-span / tau)

        tau = brentq(f, 1e-6, 50.0, xtol=1e-15, rtol=1e-15, maxiter=500)
        voce = cls(p_start, s_start, rm, p_max, tau, top + tau * rm, a)
        voce._check()
        return voce

    def sigma_v(self, p):
        return self.sat - (self.sat - self.a) * np.exp(-(np.asarray(p) - self.p_start) / self.tau)

    def __call__(self, p):
        p = np.asarray(p, dtype=float)
        return self.sigma_v(p) / (1.0 + p)

    @property
    def n(self) -> float:
        """Curvature parameter: d2 ln s / dp2 = -1/n at the maximum."""
        return (1.0 + self.p_max) * self.tau

    def _check(self) -> None:
        if not math.isclose(float(self(self.p_start)), self.s_start, rel_tol=1e-12):
            raise AssertionError("Voce start value")
        if not math.isclose(float(self(self.p_max)), self.rm, rel_tol=1e-12):
            raise AssertionError("Voce maximum value")
        grid = np.linspace(self.p_start, self.p_max + 0.5, 20001)
        values = self(grid)
        if abs(grid[np.argmax(values)] - self.p_max) > 1e-4:
            raise AssertionError("Voce maximum is not at p_max")


@dataclasses.dataclass(frozen=True)
class FlatTop:
    """Linear rise to ``rm`` at ``p1``, exactly flat to ``p2``, symmetric fall."""

    p_start: float
    s_start: float
    rm: float
    p1: float
    p2: float

    @property
    def slope(self) -> float:
        return (self.rm - self.s_start) / (self.p1 - self.p_start)

    @property
    def p_max(self) -> float:
        return 0.5 * (self.p1 + self.p2)

    def __call__(self, p):
        p = np.asarray(p, dtype=float)
        rise = self.s_start + self.slope * (p - self.p_start)
        fall = self.rm - self.slope * (p - self.p2)
        return np.where(p < self.p1, rise, np.where(p <= self.p2, self.rm, fall))


def _grid_after(p_from: float, dp: float, p_end: float) -> np.ndarray:
    """Absolute grid k*dp with p_from < k*dp <= p_end."""
    k0 = math.floor(p_from / dp + 1e-9) + 1
    k1 = math.floor(p_end / dp + 1e-9)
    return np.arange(k0, k1 + 1) * dp


def _band_truth(curve_fn, p_lo: float, p_max: float, p_hi: float, rm: float, e_mpa: float) -> dict:
    """Declared M12 mid-point of the noise-free band (delta_F = 2e-4), analytic."""
    threshold = rm * (1.0 - BAND_REL_MIN)

    def g(p: float) -> float:
        return float(curve_fn(p)) - threshold

    if g(p_hi) >= 0 or g(p_lo) >= 0:
        return {}
    left = brentq(g, p_lo, min(p_max, getattr(curve_fn, "p1", p_max)), xtol=1e-14)
    right = brentq(g, max(p_max, getattr(curve_fn, "p2", p_max)), p_hi, xtol=1e-14)
    mid = 0.5 * (left + right)
    agt = 100.0 * (threshold / e_mpa + mid)
    return {
        "agt_percent": agt,
        "ag_percent": agt - 100.0 * rm / e_mpa,
        "ag_halfwidth_percent": 100.0 * 0.5 * (right - left),
        "ag_peak_percent": 100.0 * p_max,
    }


def _add_elastic(b: _Builder, top: float, step: float = 2.0, offset: float = 1.0) -> None:
    """Start sample at 2 MPa, then ``offset + k*step`` below ``top``."""
    b.add("start", [START_STRESS_MPA], [0.0], "one sample at 2 MPa")
    k = np.arange(0, int(math.ceil((top - offset) / step)) + 1)
    s = offset + k * step
    s = s[(s > START_STRESS_MPA) & (s < top)]
    b.add("elastic", s, np.zeros_like(s), f"{step:g} MPa per sample")


def _add_rounded(b: _Builder, s_from: float, s_to: float, n: int = 60) -> None:
    """C1 parabola s(u) = s_to - (s_to - s_from)(1-u)^2 over mechanical strain."""
    depth = s_to - s_from
    tangent_compliance = 1.0 / b.e_mpa + 2.0 * b.q * s_from
    width = 2.0 * depth * tangent_compliance
    m0 = b.mech(s_from, 0.0)
    u = np.arange(0, n + 1) / n
    s = s_to - depth * (1.0 - u) ** 2
    s[-1] = s_to  # the peak is an exact sample
    mech = m0 + u * width
    p = mech - s / b.e_mpa - b.q * s * s
    p[0] = 0.0
    p = np.maximum.accumulate(np.maximum(p, 0.0))
    b.add(
        "rounded",
        s,
        p,
        f"{n} equal strain steps of {width / n:.3g}; {2 * depth / n:.3g} MPa/sample at start, "
        f"falling linearly to {depth / n**2:.3g} at the peak",
    )


def _add_drop(b: _Builder, s_from: float, s_to: float, step: float = 2.0) -> None:
    """Load drop in ``step`` MPa decrements to exactly ``s_to``; crosshead advances 2/E."""
    s_list = []
    s = s_from
    while s - step > s_to + 1e-9:
        s -= step
        s_list.append(s)
    s_list.append(s_to)
    mech = b.mech(*b.last)
    p_list = []
    for value in s_list:
        mech += 2.0 / b.e_mpa
        p_list.append(mech - value / b.e_mpa - b.q * value * value)
    b.add("drop", s_list, p_list, f"{step:g} MPa per sample")


# ---------------------------------------------------------------------------
# Public generators


def discontinuous(
    *,
    reh: float = 400.0,
    upper_peak: bool = True,
    transient_min: float | None = 372.0,
    plateau: float | None = 380.0,
    serration: float = 1.0,
    serration_phase: float = 0.0,
    plateau_end_p: float = 0.015,
    hardening: str | None = "voce",
    rm: float = 510.0,
    ag_p: float = 0.15,
    flat_top: tuple[float, float] = (0.12, 0.16),
    p_end: float = 0.25,
    plateau_dp: float = 1e-5,
    hardening_dp: float = 1e-4,
    elastic_quadratic: float = 0.0,
    rate_step: float | None = None,
    round_start: float = 340.0,
    e_mpa: float = E_MPA,
) -> SyntheticCurve:
    """Discontinuous-yielding family: reference case (a) and its variants.

    Defaults give §14 (a) = M9 case (i): E = 205 GPa, start at 2 MPa with
    e = 5e-4, rounded pre-yield to ReH = 400 MPa, drop to the transient
    minimum 372 MPa, recovery to the plateau 380 +- 1 MPa up to p = 0.015,
    Voce hardening to Rm = 510 MPa at Ag = 15 %.

    - ``serration``: plateau half-amplitude (M9 (ii): 3).
    - ``serration_phase``: sample offset of the triangle wave; 0.5 puts no
      sample on an extreme or on the plateau level.
    - ``plateau=None``: no plateau; hardening starts at the transient minimum
      (M9 (iii)).
    - ``upper_peak=False``: the rounded part ends tangentially on the plateau
      (M9 (iv)); use ``serration=0``.
    - ``transient_min=None``: the drop goes straight to the plateau.
    - ``hardening``: ``"voce"``, ``"flat_top"`` (exact flat maximum between
      ``flat_top`` plastic strains, VP-8) or ``None`` (the plateau runs to
      ``p_end``: no work hardening, §8).
    - ``rate_step``: plastic strain of a rate switch inside the plateau (VP-9):
      a 3 MPa relaxation dip (1 MPa per sample) below the plateau level, then
      a +15 MPa step (3 MPa per sample) to a plateau 15 MPa higher.
    - ``elastic_quadratic``: q in e = s/E + q s**2 (VP-6).
    """
    b = _Builder(e_mpa, elastic_quadratic)
    _add_elastic(b, round_start)
    peak = reh if upper_peak else plateau
    _add_rounded(b, round_start, peak)
    i_peak = len(b.s) - 1
    reh_increment = (peak - round_start) / 60**2

    truth: dict = {"yield_type": "discontinuous" if upper_peak else "continuous"}
    if elastic_quadratic == 0.0:
        truth["e_gpa"] = e_mpa / 1000.0
        truth["strain_origin"] = _e0(e_mpa, 0.0)

    period = SERRATION_PERIOD_SAMPLES * plateau_dp
    if upper_peak and transient_min is not None:
        _add_drop(b, reh, transient_min)
    elif upper_peak:
        # Drop straight to 2 MPa above the plateau level; the plateau then
        # starts at phase 0 (its level) with no transient below it.
        _add_drop(b, reh, plateau + 2.0)
    p_after = b.last[1]

    plateau_min = NAN
    s_hard_start = None
    if plateau is not None:
        if upper_peak and transient_min is not None:
            # Plateau grid starts at phase 0 (value = plateau level), leaving
            # room for a 1 MPa-per-sample recovery.
            n_rec = int(round(plateau - transient_min))
            m = math.floor((plateau_end_p - p_after) / period - 1e-9)
            while plateau_end_p - m * period - p_after < n_rec * plateau_dp:
                m -= 1
            p_start = plateau_end_p - m * period
            k_back = np.arange(m * SERRATION_PERIOD_SAMPLES, -1, -1)
            j = np.arange(1, n_rec)
            b.add(
                "recovery",
                transient_min + (plateau - transient_min) * j / n_rec,
                p_after + (p_start - p_after) * j / n_rec,
                "1 MPa per sample",
            )
        elif upper_peak:
            m = math.floor((plateau_end_p - p_after) / period - 1e-9)
            k_back = np.arange(m * SERRATION_PERIOD_SAMPLES, -1, -1)
        else:
            k_back = np.arange(
                math.floor((plateau_end_p - p_after) / plateau_dp - 1e-9), -1, -1
            )
            k_back = k_back[plateau_end_p - k_back * plateau_dp > p_after + 1e-12]
        p_plateau = plateau_end_p - k_back * plateau_dp
        values = np.array([plateau + serration * _tri(-int(k), phase=serration_phase) for k in k_back])
        density = f"{plateau_dp:g} plastic strain per sample; serrations +-{serration:g} MPa"
        if rate_step is not None:
            # Snap the switch to a phase-0 grid point.
            at_switch = (p_plateau >= rate_step) & (k_back % SERRATION_PERIOD_SAMPLES == 0)
            if not np.any(at_switch[:-10]):
                raise ValueError("rate step must lie inside the plateau")
            i_sw = int(np.argmax(at_switch))
            for i in range(i_sw + 1, len(values)):
                offset = i - i_sw
                if offset <= 3:
                    values[i] = plateau - offset
                elif offset <= 8:
                    values[i] = plateau + 3.0 * (offset - 3)
                else:
                    values[i] = plateau + 15.0 + serration * _tri(-int(k_back[i]), phase=serration_phase)
            density += "; rate step: 3 MPa dip at 1 MPa/sample, +15 MPa at 3 MPa/sample"
        b.add("plateau", values, p_plateau, density)
        plateau_min = float(values.min())
        s_hard_start = float(values[-1])
        p_hard_start = float(p_plateau[-1])
    else:
        s_hard_start, p_hard_start = b.last

    rel_incl = NAN
    if upper_peak:
        rel_incl = float(min(b.s[i_peak + 1 :]))
        truth["reh_mpa"] = reh
        truth["reh_sample_increment_mpa"] = reh_increment
        truth["rel_incl_transient_mpa"] = rel_incl
        truth["rel_mpa"] = plateau_min if plateau is not None else float(transient_min)

    if hardening == "voce":
        curve_fn = Voce.solve(p_hard_start, s_hard_start, rm, ag_p)
        p_grid = _grid_after(p_hard_start, hardening_dp, p_end)
        if not np.any(np.isclose(p_grid, ag_p, rtol=0, atol=1e-12)):
            raise ValueError("Ag must lie on the hardening grid")
        b.add("hardening", curve_fn(p_grid), p_grid, f"{hardening_dp:g} plastic strain per sample")
        truth["hardening_n"] = curve_fn.n
    elif hardening == "flat_top":
        curve_fn = FlatTop(p_hard_start, s_hard_start, rm, *flat_top)
        p_grid = _grid_after(p_hard_start, hardening_dp, p_end)
        b.add("hardening", curve_fn(p_grid), p_grid, f"{hardening_dp:g} plastic strain per sample")
    elif hardening is None:
        curve_fn = None
    else:
        raise ValueError(f"unknown hardening {hardening!r}")

    if curve_fn is not None:
        truth["fm_n"] = rm * S0_MM2
        truth["rm_mpa"] = rm
        if elastic_quadratic == 0.0:  # the band truth assumes the linear elastic law
            truth.update(_band_truth(curve_fn, p_hard_start, curve_fn.p_max, p_end, rm, e_mpa))

    if elastic_quadratic == 0.0:
        p_arr = np.asarray(b.p)
        s_arr = np.asarray(b.s)
        k = int(np.argmax(p_arr >= OFFSET_STRAIN))
        if k > 0 and p_arr[k] >= OFFSET_STRAIN:
            w = (OFFSET_STRAIN - p_arr[k - 1]) / (p_arr[k] - p_arr[k - 1])
            truth["rp02_mpa"] = float(s_arr[k - 1] + w * (s_arr[k] - s_arr[k - 1]))

    description = (
        f"discontinuous family: ReH={reh if upper_peak else None}, transient={transient_min}, "
        f"plateau={plateau}+-{serration}, hardening={hardening}"
    )
    return _assemble(b, truth, description)


def reference_a() -> SyntheticCurve:
    """§14 (a), identical to M9 case (i)."""
    return discontinuous()


@dataclasses.dataclass(frozen=True)
class RambergOsgood:
    """``e = e0 + s/E + 0.002 (s/rp)**n``."""

    n: float
    rp: float
    e_mpa: float

    def plastic(self, s):
        return OFFSET_STRAIN * (np.asarray(s, dtype=float) / self.rp) ** self.n

    def stress_at_plastic(self, p: float) -> float:
        return self.rp * (p / OFFSET_STRAIN) ** (1.0 / self.n)


def ramberg_osgood(
    n: float = 15.0,
    *,
    rp: float = 300.0,
    p_switch: float = 0.005,
    rm: float = 480.0,
    ag_p: float = 0.12,
    p_end: float = 0.22,
    elastic_step: float = 2.0,
    elastic_offset: float = 1.0,
    hardening_dp: float = 1e-4,
    e_mpa: float = E_MPA,
) -> SyntheticCurve:
    """§14 (b): continuous yielding, Ramberg-Osgood pre-yield, Voce hardening.

    Expected E and Rp0.2 are the M4/M5/M10 fixed point evaluated with exact
    OLS sums over the generated window samples and an exact (root-found)
    crossing on the continuous model ("analytic window slope").
    """
    ro = RambergOsgood(n, rp, e_mpa)
    b = _Builder(e_mpa, 0.0)
    fine_from = 0.8 * rp
    _add_elastic(b, fine_from, step=elastic_step, offset=elastic_offset)
    # The elastic segment is RO too (its plastic part is tiny but not zero).
    seg = b.segments[-1]
    for i in range(seg.start, seg.stop):
        b.p[i] = float(ro.plastic(b.s[i]))
    b.p[0] = float(ro.plastic(b.s[0]))
    s_switch = ro.stress_at_plastic(p_switch)
    s_fine = np.arange(fine_from + 0.25, s_switch, 0.5)
    s_fine = np.append(s_fine, s_switch)
    b.add("ro_fine", s_fine, ro.plastic(s_fine), "0.5 MPa per sample")
    voce = Voce.solve(p_switch, s_switch, rm, ag_p)
    p_grid = _grid_after(p_switch, hardening_dp, p_end)
    b.add("hardening", voce(p_grid), p_grid, f"{hardening_dp:g} plastic strain per sample")

    # e0 is fixed by the start point including its (tiny) RO plastic strain.
    truth = {"yield_type": "continuous", "fm_n": rm * S0_MM2, "rm_mpa": rm}
    curve = _assemble(b, truth, f"Ramberg-Osgood n={n:g}, Rp={rp:g} MPa")
    e_fit, b_fit, rp02 = spec_fixed_point(curve, ro)
    truth.update(
        {
            "e_gpa": e_fit / 1000.0,
            "strain_origin": -b_fit / e_fit,
            "rp02_mpa": rp02,
            "hardening_n": voce.n,
        }
    )
    truth.update(_band_truth(voce, p_switch, ag_p, p_end, rm, e_mpa))
    # The band truth assumes the exact E; with n >= 15 the fitted E equals it
    # to ~1e-9, so this is only declared for those curves.
    if n < 15:
        for key in ("ag_percent", "agt_percent", "ag_halfwidth_percent"):
            truth.pop(key, None)
    return dataclasses.replace(curve, truth=Truth(k_ext=len(curve), **truth))


def ols(stress: np.ndarray, strain: np.ndarray) -> tuple[float, float]:
    """Exact OLS sigma = E e + b with centred sums (math.fsum)."""
    em = math.fsum(strain) / len(strain)
    sm = math.fsum(stress) / len(stress)
    de = strain - em
    ds = stress - sm
    slope = math.fsum(de * ds) / math.fsum(de * de)
    return slope, sm - slope * em


def window_samples(curve: SyntheticCurve, ref: float, f_lo: float, f_hi: float):
    """Noise-free first-loading-branch samples with f_lo*ref <= s <= f_hi*ref."""
    s = curve.stress_mpa
    first = int(np.argmax(s >= ref)) if np.any(s >= ref) else len(s)
    s_b = s[:first]
    e_b = curve.clean_strain[:first]
    keep = (s_b >= f_lo * ref) & (s_b <= f_hi * ref)
    return s_b[keep], e_b[keep]


def spec_fixed_point(curve: SyntheticCurve, ro: RambergOsgood, tol: float = 1e-13):
    """M5 steps 1, 2, 4 and M10 for a noise-free RO curve (my own evaluation).

    Returns E (MPa), b (MPa) and Rp0.2 (MPa). The crossing of
    e - e0_fit - s/E_fit = 0.002 is solved on the continuous RO model.
    """
    rm = float(curve.stress_mpa.max())
    e0 = curve.e0

    def model_e(s: float) -> float:
        return e0 + s / ro.e_mpa + float(ro.plastic(s))

    def crossing(e_fit: float, b_fit: float) -> float:
        def g(s: float) -> float:
            return model_e(s) + b_fit / e_fit - s / e_fit - OFFSET_STRAIN

        s_hi = ro.stress_at_plastic(0.005)
        return brentq(g, 0.5 * ro.rp, s_hi, xtol=1e-13)

    e_fit, b_fit = ols(*window_samples(curve, 0.5 * rm, 0.10, 0.40))
    rp02 = crossing(e_fit, b_fit)
    for _ in range(200):
        e_new, b_new = ols(*window_samples(curve, rp02, 0.10, 0.40))
        converged = abs(e_new - e_fit) / e_fit < tol
        e_fit, b_fit = e_new, b_new
        rp02 = crossing(e_fit, b_fit)
        if converged:
            break
    return e_fit, b_fit, rp02


def spec_reh(curve: SyntheticCurve) -> tuple[float, int]:
    """ReH and i_p by the M5 step 2 and M7/M8 text, on the (noisy) samples.

    My own evaluation of the definitions, for curves with finite channels:
    "Rm" is the maximum stress; the provisional OLS uses the samples before
    the stress first reaches 0.5*Rm within [0.10, 0.40]*0.5*Rm;
    s_n = 1.4826 * MAD (about the median) of its residuals;
    d(j) = max(0.005*M(j), 5*s_n); j* is the first j with s(j) > 0.5*Rm and
    s(j) <= M(j) - d(j); i_p = argmax s[0..j*]. Raises if there is no j*.
    """
    s = curve.force_n / curve.s0_mm2
    e = curve.strain
    ref = 0.5 * float(np.max(s))
    first = int(np.argmax(s >= ref))
    s_b, e_b = s[:first], e[:first]
    keep = (s_b >= 0.10 * ref) & (s_b <= 0.40 * ref)
    slope, intercept = ols(s_b[keep], e_b[keep])
    resid = s_b[keep] - (slope * e_b[keep] + intercept)
    s_n = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    running_max = np.maximum.accumulate(s)
    d = np.maximum(0.005 * running_max, 5.0 * s_n)
    j_star = int(np.nonzero((s > ref) & (s <= running_max - d))[0][0])
    i_p = int(np.argmax(s[: j_star + 1]))
    return float(s[i_p]), i_p


# ---------------------------------------------------------------------------
# Transformations of a curve


def _with_strain(curve: SyntheticCurve, strain: np.ndarray, **truth_changes) -> SyntheticCurve:
    truth = dataclasses.replace(curve.truth, **truth_changes) if truth_changes else curve.truth
    return dataclasses.replace(curve, strain=strain, truth=truth)


def noisy(
    curve: SyntheticCurve,
    seed: int,
    stress_sd: float = STRESS_NOISE_MPA,
    strain_sd: float = STRAIN_NOISE,
) -> SyntheticCurve:
    """§14 (e): Gaussian noise on force (stress_sd*S0) and strain; fixed seed."""
    rng = np.random.default_rng(seed)
    force = curve.force_n + rng.normal(0.0, stress_sd * curve.s0_mm2, len(curve))
    strain = curve.strain + rng.normal(0.0, strain_sd, len(curve))
    return dataclasses.replace(
        curve, force_n=force, strain=strain, description=curve.description + f"; noise seed {seed}"
    )


END_MODES = ("freeze", "nan", "zero", "crosshead")


def extensometer_end(
    curve: SyntheticCurve,
    mode: str,
    at_plastic_strain: float = 0.039,
    substitution_gauge_ratio: float = 1.2,
) -> SyntheticCurve:
    """§14 (c): the extensometer is removed at the first sample with p >= at.

    - ``freeze``: the strain keeps its last value;
    - ``nan``: the strain becomes NaN;
    - ``zero``: the strain drops to 0;
    - ``crosshead``: vendor substitution, e continues as
      ``e_last + (c - c_last) / (ratio * L0)``; undetectable by design (M3, A1),
      so the expected ``k_ext`` stays the record length.
    """
    k = curve.index_at_plastic_strain(at_plastic_strain)
    strain = curve.strain.copy()
    if mode == "freeze":
        strain[k:] = strain[k - 1]
    elif mode == "nan":
        strain[k:] = np.nan
    elif mode == "zero":
        strain[k:] = 0.0
    elif mode == "crosshead":
        c = curve.crosshead_mm
        strain[k:] = strain[k - 1] + (c[k:] - c[k - 1]) / (substitution_gauge_ratio * curve.l0_mm)
        return _with_strain(curve, strain)
    else:
        raise ValueError(mode)
    return _with_strain(curve, strain, k_ext=k)


def with_temporary_flat_strain(
    curve: SyntheticCurve, p_flat: float = 0.005, p_resume: float = 0.008, p_rejoin: float = 0.011
) -> SyntheticCurve:
    """§14 (c): Lüders band outside the gauge; e flat, then catches up (r > 1)."""
    i1 = curve.index_at_plastic_strain(p_flat)
    i2 = curve.index_at_plastic_strain(p_resume)
    i3 = curve.index_at_plastic_strain(p_rejoin)
    strain = curve.strain.copy()
    strain[i1:i2] = strain[i1]
    w = (np.arange(i2, i3) - (i2 - 1)) / (i3 - (i2 - 1))
    strain[i2:i3] = strain[i1] + w * (strain[i3] - strain[i1])
    return _with_strain(curve, strain)


def with_hold(
    curve: SyntheticCurve, at_plastic_strain: float = 0.05, n_hold: int = 50, relax_mpa: float = 3.0
) -> SyntheticCurve:
    """§14 (c): a crosshead hold with stress relaxation, then elastic reloading.

    During the hold the crosshead and the strain are constant; the stress
    relaxes by ``relax_mpa`` (1/n_hold of it per sample) as plastic strain
    replaces elastic strain. Reloading takes 3 samples at constant p.
    """
    if curve.quadratic != 0.0:
        raise ValueError("hold needs a linear elastic law")
    k = curve.index_at_plastic_strain(at_plastic_strain)
    s0, p0 = curve.stress_mpa[k], curve.plastic_strain[k]
    if curve.plastic_strain[k + 1] - p0 <= relax_mpa / curve.e_mpa:
        raise ValueError("hardening grid too fine for the hold relaxation")
    j = np.arange(1, n_hold + 1)
    s_hold = s0 - relax_mpa * j / n_hold
    p_hold = p0 + (s0 - s_hold) / curve.e_mpa
    s_reload = s_hold[-1] + (s0 - s_hold[-1]) * np.arange(1, 4) / 3
    p_reload = np.full(3, p_hold[-1])
    s = np.concatenate([curve.stress_mpa[: k + 1], s_hold, s_reload, curve.stress_mpa[k + 1 :]])
    p = np.concatenate([curve.plastic_strain[: k + 1], p_hold, p_reload, curve.plastic_strain[k + 1 :]])
    clean = curve.e0 + s / curve.e_mpa + p
    crosshead = curve.l0_mm * (clean - curve.e0)
    crosshead[k + 1 : k + 1 + n_hold] = crosshead[k]  # exactly constant during the hold
    _check_construction(s, p, crosshead)
    segments = curve.segments + (Segment("hold", k + 1, k + 1 + n_hold + 3, "hold: 50 samples, c constant"),)
    return dataclasses.replace(
        curve,
        force_n=s * curve.s0_mm2,
        strain=clean.copy(),
        crosshead_mm=crosshead,
        stress_mpa=s,
        plastic_strain=p,
        clean_strain=clean,
        truth=dataclasses.replace(curve.truth, k_ext=len(s)),
        segments=segments,
        description=curve.description + "; hold",
    )


def with_terminal_drop(
    curve: SyntheticCurve, *, first_mpa: float, to_mpa: float = 5.0, n: int = 5
) -> SyntheticCurve:
    """§8 v2.5: append a terminal force drop (fracture or unloading) of ``n`` samples.

    The stress falls linearly from ``first_mpa`` (first appended sample) to
    ``to_mpa`` (last sample). The crosshead keeps its last step per sample and
    the strain follows it (r = 1), so the extensometer stays valid. Truth
    values describe the curve before the drop and stay as they are, except
    the record length ``k_ext``.
    """
    if curve.quadratic != 0.0:
        raise ValueError("terminal drop needs a linear elastic law")
    step = curve.crosshead_mm[-1] - curve.crosshead_mm[-2]
    crosshead = np.append(curve.crosshead_mm, curve.crosshead_mm[-1] + step * np.arange(1, n + 1))
    s = np.append(curve.stress_mpa, np.linspace(first_mpa, to_mpa, n))
    clean = curve.e0 + crosshead / curve.l0_mm
    p = clean - curve.e0 - s / curve.e_mpa
    p[: len(curve)] = curve.plastic_strain
    _check_construction(s, p, crosshead)
    start = len(curve)
    segments = curve.segments + (
        Segment("terminal_drop", start, len(s), f"linear from {first_mpa:g} to {to_mpa:g} MPa"),
    )
    return dataclasses.replace(
        curve,
        force_n=s * curve.s0_mm2,
        strain=clean.copy(),
        crosshead_mm=crosshead,
        stress_mpa=s,
        plastic_strain=p,
        clean_strain=clean,
        truth=dataclasses.replace(curve.truth, k_ext=len(s)),
        segments=segments,
        description=curve.description + "; terminal drop",
    )


def with_strain_shift(curve: SyntheticCurve, shift: float) -> SyntheticCurve:
    """VP-5: e -> e + shift (moves the strain origin by ``shift``)."""
    return _with_strain(
        curve, curve.strain + shift, strain_origin=curve.truth.strain_origin + shift
    )


def with_toe(curve: SyntheticCurve, amount: float = 1e-4, below_mpa: float = 12.0) -> SyntheticCurve:
    """VP-5: nonlinear seating toe below ``below_mpa`` (C1), constant shift above.

    ``toe(s) = amount * x (2 - x)`` with ``x = min(s / below_mpa, 1)``.
    """
    x = np.minimum(curve.stress_mpa / below_mpa, 1.0)
    toe = amount * x * (2.0 - x)
    return _with_strain(curve, curve.strain + toe, strain_origin=curve.truth.strain_origin + amount)
