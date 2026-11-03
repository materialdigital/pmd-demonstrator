"""Ontology-agnostic tensile-test analysis (method-defined, ISO 6892-1-inspired).

Implements ``.notes/tensile-workflow/method-spec.md`` (APPROVED v2) behind the
public interface fixed in ``.notes/tensile-workflow/analysis-api.md``. Every
property is **"method-defined, ISO 6892-1-inspired"**; nothing here claims
conformity with ISO 6892-1 or its Annex G. E is the **"regression slope of the
elastic part (mE-type), not Annex-G-conformant"**, not a material modulus.

Units (fixed): force N, strain as a ratio (not percent), crosshead and lengths
mm, areas mm**2, stresses MPa, E GPa, Ag/Agt/A/Z percent. Values that are not
defined are ``float("nan")``, never ``None``. Flags are strings
``"<property>:<flag>"`` (e.g. ``"a:missing_input"``).

Module boundary (acceptance criterion 2): this module imports no rdflib,
praeco, SPARQL helper or profile, and no ``tensile_workflow`` module other
than ``contract``. It sees no ontology IRIs; its only IRIs are the generic
QUDT quantity kinds below.

Annotations (decision D4, conventions IC1/IC2)
----------------------------------------------
Step nodes carry ``semantikon.u`` annotations with units (``"N"``,
``"Unitless"`` for ratio strain, ``"percent"``, ``"MPa"``, ``"GPa"``,
``"mm"``, ``"mm**2"``, ``"s"``; never ``"dimensionless"``) and QUDT quantity
kinds. semantikon needs ``uri=`` to be an ``rdflib.URIRef``; because this
module may not import rdflib, the IRIs are built with ``semantikon.SemantikonURI``,
semantikon's public ``URIRef`` subclass. The quantity-kind IRIs are hard-coded
(semantikon's bundled QUDT file has no quantity kinds, so they cannot be
validated there) and are listed for review:

- ``http://qudt.org/vocab/quantitykind/Force``
- ``http://qudt.org/vocab/quantitykind/Stress``
- ``http://qudt.org/vocab/quantitykind/Strain`` (ratio and percent strains)
- ``http://qudt.org/vocab/quantitykind/ModulusOfElasticity``
- ``http://qudt.org/vocab/quantitykind/Area``
- ``http://qudt.org/vocab/quantitykind/Length``
- ``http://qudt.org/vocab/quantitykind/Time``

Z (reduction of area), relative uncertainties, counts, indices, flags and
result dataclasses carry no quantity kind.

Step nodes (flowrep ``@fr.atomic``; each is also a plain, directly callable function)
-------------------------------------------------------------------------------------
Curve analysis, in the M5 bootstrap order. :func:`analyse_curve` is exactly
this composition, so a workflow wiring the same nodes gives the same numbers.

1. :func:`count_dropped_samples` (M1): checks lengths; non-finite counts per channel.
2. :func:`extensometer_validity` (M1, M2, M3): ``k_ext`` and the end checks.
3. :func:`bootstrap_fit` (M5 steps 1-2): Fm from force alone; provisional
   E0, b0; the noise level s_n, fixed from here on.
4. :func:`yield_analysis` (M5 step 3; M7, M8, M9): yield type, ReH, ReL.
5. :func:`final_elastic_fit` (M4, M5 step 4-5, M6, M16): E, strain origin.
6. :func:`proof_strength` (M10): Rp0.2.
7. :func:`maximum_force` (M11): Fm, Rm.
8. :func:`elongation_at_maximum_force` (M12): Agt, Ag.
9. :func:`window_sensitivity` (M15): E-window ranges, linearity.
10. :func:`assemble_curve_analysis`: collects 1-9 into :class:`CurveAnalysis`.

Other nodes: :func:`check_time_order` (M1 time order; the curve analysis has
no time input by the fixed API), :func:`engineering_stress` (display),
:func:`elastic_fit`, :func:`s0_scatter`, :func:`elongation_after_fracture`,
:func:`reduction_of_area`, :func:`compare_with_reported`,
:func:`summarize_by_group`.

Implementation notes (no change of the method)
----------------------------------------------
- **Force space.** All curve decisions (windows, drops, bands, crossings)
  are made on the force array; S0 only converts results into stresses. With
  every threshold relative, this makes S0 scaling (VP-4) exact: no sample
  selection can flip through rounding of F/S0. The fit slope dF/de is E*S0.
- **Rm in M5/M7.** Before the yield type is known, "Rm" (provisional
  reference 0.5*Rm, and the M7 search condition sigma > 0.5*Rm) is the
  maximum over all finite-force samples. M11's final Fm may differ only
  under discontinuous yielding without work hardening.
- **First loading branch.** For a reference stress sigma_ref, the samples
  before sigma first reaches sigma_ref (with valid strain, M2).
- **Provisional fit** needs at least 3 points (for a residual MAD); the M4
  minimum of 10 applies to the final fit and the M15 fits.
- **Yield section (M7-M9)** uses the bootstrap E0, b0 and s_n throughout, so
  the final fit cannot feed back into the yield type, ReH or ReL.
- **Rp0.2 search** starts at the first valid sample at or above the lower
  bound of the E window it uses, so a nonlinear toe below the window cannot
  produce a spurious crossing.
- **Ag band mid-point** is (min e + max e)/2 over the band; the half-width is
  (max e - min e)/2.
- **near_rate_switch** compares the raw strain channel at the evaluation
  point of ReH, ReL, Rp0.2 and the extensometer end with ``rate_switch_strain``
  (distance <= ``rate_switch_proximity``).

M3 extensometer end (concrete rule)
-----------------------------------
Windows are consecutive runs of samples over which the crosshead travels
``ext_window_mm`` (|c[end] - c[start]| >= 0.1 mm; consecutive windows share
their boundary sample; samples with non-finite crosshead never close a
window, so holds do not advance it). A trailing remainder of less than one
window is merged into the last window; with no complete window, M3 is not
applied. A window is *invalid* if its strain has a non-finite value, an exact
zero, a single-step drop larger than ``ext_jump``, or
r = (e[end] - e[start]) / ((c[end] - c[start]) / L0) < ``ext_ratio_min``.
``k_ext`` is the start of the earliest window from which every window to the
end of the record is invalid; per M1 it is also at most the index of the
first non-finite strain after a finite one. Missing crosshead or L0 skips the
window rule with flag ``extensometer:missing_input``.

M9 ReL (concrete rule; option (A))
----------------------------------
Y is the yield range (i_p, i_h) on finite-force samples, where i_h is the
first sample after i_p with sigma >= ReH.

**Interim, decision pending:** if the stress never returns to ReH, the
method's fallback i_h = i_m is circular (M11 leaves Fm, hence i_m, undefined
without work hardening). Then ReL, ReL incl. transient, Fm and Rm are NaN with
flags ``rel:no_work_hardening`` and ``fm:no_work_hardening``, and the yield
range runs to the end of the record for ``rp02:in_yield_plateau``.

1. ``rel_incl_transient`` = min sigma over Y.
2. Plateau level P = median of the samples in Y with
   sigma <= (rel_incl_transient + ReH)/2. The bound keeps the work-hardening
   rise towards ReH out of the estimate.
3. Transient: starting at i_p, the first sample with sigma < P - d begins it
   (d = max(drop_rel*ReH, drop_noise_factor*s_n), the M7 drop threshold at
   ReH; this hysteresis stops noise on the falling flank from ending the
   transient early). The first later sample with sigma >= P ends it
   (completed recovery to the plateau level) at i_r. If no sample is below
   P - d, there is no transient (i_r = first sample of Y). If the curve never
   recovers to P, there is no plateau (go to 6).
4. Candidate ReL = min sigma over [i_r, i_h).
5. Plateau extent: the band [candidate, P + (P - candidate)] is the plateau
   band (it widens with the observed serration depth). The extent is
   e_p(i_last) - e_p(i_r), with i_last the last sample in [i_r, i_h) inside
   the band (strain-valid samples only). If it is >= ``plateau_min_extent``,
   ReL = candidate.
6. Otherwise ReL = the transient minimum (min sigma over (i_p, i_r), or over
   Y without a transient or recovery) with flag ``rel:single_minimum``.

Reference cases (method-spec M9): (i) plateau 380 +- 1 after a 372 dip: P is
about 380, recovery ends on the rising flank, ReL = 379. (ii) serrations
+- 3: P about 380, band [377, 383], ReL = 377. (iii) monotone rise after the
dip: the candidate is the recovery sample itself, the band collapses to
[P, P], the extent is about 0, so ReL = 372 with ``single_minimum``.
(iv) no upper peak: M7 gives continuous yielding, ReL is NaN with
``no_upper_yield``. (v) noise: P is a median, so recovery is found at the
same place; the noisy ReL is low by the noise extreme over the plateau (the
min/max noise bias, A5). Known limitation: a smoothly *rising* plateau looks
like case (iii) when noise-free; with noise its band may widen enough to
count as a plateau. This is reported for review (T8.1).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence

import flowrep as fr
import numpy as np
import pandas as pd
from semantikon import SemantikonURI, u

from tensile_workflow.contract import ReportedProperties

# ---------------------------------------------------------------------------
# Labels for outputs and reports (method-spec §1, §11)
# ---------------------------------------------------------------------------

METHOD_LABEL = "method-defined, ISO 6892-1-inspired"
E_LABEL = "E: regression slope of the elastic part (mE-type), not Annex-G-conformant"
E_STDERR_LABEL = "OLS standard error (conditional on window and independence; not an uncertainty)"
S0_SCATTER_LABEL = (
    "within-specimen dimensional scatter of the mean (nu = 2); excludes instrument bias"
)
OMITTED_UNCERTAINTY: tuple[str, ...] = (
    "force class (ISO 7500-1) and extensometer class (ISO 9513, unknown here)",
    "instrument calibration",
    "uncertainty in Lu and Su",
    "ReH sampling resolution (reported as a diagnostic instead)",
    "min/max noise bias of ReL, ReH and Fm",
    "rate effects",
    "material variability (shown only through the group statistics)",
)
RESIDUAL_RISKS: tuple[str, ...] = (
    "A strain channel that vendor software continues from the crosshead after "
    "extensometer removal cannot be detected (M3).",
)

# ---------------------------------------------------------------------------
# Annotated types (generic QUDT quantity kinds only)
# ---------------------------------------------------------------------------

QUDT_QUANTITY_KIND = "http://qudt.org/vocab/quantitykind/"

QK_FORCE = SemantikonURI(QUDT_QUANTITY_KIND + "Force")
QK_STRESS = SemantikonURI(QUDT_QUANTITY_KIND + "Stress")
QK_STRAIN = SemantikonURI(QUDT_QUANTITY_KIND + "Strain")
QK_MODULUS = SemantikonURI(QUDT_QUANTITY_KIND + "ModulusOfElasticity")
QK_AREA = SemantikonURI(QUDT_QUANTITY_KIND + "Area")
QK_LENGTH = SemantikonURI(QUDT_QUANTITY_KIND + "Length")
QK_TIME = SemantikonURI(QUDT_QUANTITY_KIND + "Time")

QUANTITY_KIND_IRIS: tuple[str, ...] = tuple(
    str(iri) for iri in (QK_FORCE, QK_STRESS, QK_STRAIN, QK_MODULUS, QK_AREA, QK_LENGTH, QK_TIME)
)
"""Every quantity-kind IRI used in annotations (for review and tests)."""

ForceArray = u(np.ndarray, units="N", uri=QK_FORCE)
StressArray = u(np.ndarray, units="MPa", uri=QK_STRESS)
StrainArray = u(np.ndarray, units="Unitless", uri=QK_STRAIN)
CrossheadArray = u(np.ndarray, units="mm", uri=QK_LENGTH)
TimeArray = u(np.ndarray, units="s", uri=QK_TIME)

ForceN = u(float, units="N", uri=QK_FORCE)
StressMPa = u(float, units="MPa", uri=QK_STRESS)
ModulusGPa = u(float, units="GPa", uri=QK_MODULUS)
StrainRatio = u(float, units="Unitless", uri=QK_STRAIN)
OptionalStrainRatio = u(float | None, units="Unitless", uri=QK_STRAIN)
StrainPercent = u(float, units="percent", uri=QK_STRAIN)
OptionalStrainPercent = u(float | None, units="percent", uri=QK_STRAIN)
AreaMM2 = u(float, units="mm**2", uri=QK_AREA)
OptionalAreaMM2 = u(float | None, units="mm**2", uri=QK_AREA)
LengthMM = u(float, units="mm", uri=QK_LENGTH)
OptionalLengthMM = u(float | None, units="mm", uri=QK_LENGTH)
OptionalCrossheadArray = u(np.ndarray | None, units="mm", uri=QK_LENGTH)

NAN = float("nan")

# ---------------------------------------------------------------------------
# Parameters and result types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AnalysisParameters:
    """Method parameters (method-spec v2 defaults). All thresholds are relative."""

    f_lo: float = 0.10  # M4 window fractions of sigma_ref
    f_hi: float = 0.40
    provisional_fraction_of_rm: float = 0.5  # M5 provisional reference = 0.5*Rm
    min_elastic_points: int = 10  # M4
    drop_rel: float = 0.005  # M7 d = max(drop_rel*M, k*s_n)
    drop_noise_factor: float = 5.0
    peak_max_plastic_strain: float = 0.005  # M7
    min_flow_plastic_strain: float = 0.0005  # M7 (0.05 %)
    plateau_min_extent: float = 0.002  # M9 fallback to single_minimum
    offset_plastic_strain: float = 0.002  # M10
    ext_window_mm: float = 0.1  # M3
    ext_ratio_min: float = 0.05
    ext_jump: float = 1e-3
    ext_end_fraction_of_a: float = 0.5  # M3 end check
    band_rel_min: float = 2e-4  # M12
    band_noise_factor: float = 3.0
    window_grid_lo: tuple = (0.05, 0.10, 0.15)  # M15
    window_grid_hi: tuple = (0.30, 0.40, 0.50)
    linearity_windows: tuple = ((0.05, 0.25), (0.25, 0.50))  # M15
    rate_switch_proximity: float = 0.002  # strain distance for flag near_rate_switch
    max_window_iterations: int = 20  # M5
    window_rel_tol: float = 1e-4


DEFAULT_PARAMETERS = AnalysisParameters()


@dataclasses.dataclass(frozen=True)
class ElasticFit:
    """OLS fit sigma = E*e + b in a stress window (M4, M16).

    ``e_gpa`` is NaN if fewer than the minimum number of points are in the
    window. ``stderr_gpa`` is the OLS standard error of the slope
    (see :data:`E_STDERR_LABEL`), ``lag1_rho`` the lag-1 autocorrelation of
    the residuals in sample order and ``n_eff = n(1 - rho)/(1 + rho)``.
    """

    e_gpa: float
    intercept_mpa: float
    n: int
    r2: float
    stderr_gpa: float
    lag1_rho: float
    n_eff: float
    lo_mpa: float
    hi_mpa: float


@dataclasses.dataclass(frozen=True)
class S0Scatter:
    """Relative standard uncertainty of S0 from the scatter of s_i (M14)."""

    u_rel: float
    dof: int
    flags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ScalarResult:
    """A single value with its flags (M13)."""

    value: float
    flags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------

_MAD_TO_SD = 1.4826


def _is_number(value) -> bool:
    return value is not None and isinstance(value, (int, float, np.floating, np.integer)) and math.isfinite(value)


@dataclasses.dataclass(frozen=True)
class _Ols:
    """OLS of y on x, in the units of the inputs."""

    slope: float
    intercept: float
    n: int
    r2: float
    stderr: float
    lag1_rho: float
    n_eff: float
    residual_noise: float  # 1.4826 * MAD of the residuals


_OLS_UNDEFINED = dict(slope=NAN, intercept=NAN, r2=NAN, stderr=NAN, lag1_rho=NAN, n_eff=NAN, residual_noise=NAN)


def _ols(x: np.ndarray, y: np.ndarray, min_points: int) -> _Ols:
    """Ordinary least squares y = slope*x + intercept on the given samples (in order)."""
    n = int(len(x))
    if n < max(min_points, 2):
        return _Ols(n=n, **_OLS_UNDEFINED)
    x_mean, y_mean = x.mean(), y.mean()
    dx, dy = x - x_mean, y - y_mean
    sxx = float(np.dot(dx, dx))
    if sxx == 0.0:
        return _Ols(n=n, **_OLS_UNDEFINED)
    slope = float(np.dot(dx, dy)) / sxx
    intercept = float(y_mean - slope * x_mean)
    residuals = y - (slope * x + intercept)
    ssr = float(np.dot(residuals, residuals))
    sst = float(np.dot(dy, dy))
    r2 = 1.0 - ssr / sst if sst > 0.0 else NAN
    stderr = math.sqrt(ssr / (n - 2) / sxx) if n > 2 else NAN
    if n > 2 and ssr > 0.0:
        rho = float(np.dot(residuals[:-1], residuals[1:])) / ssr
        n_eff = n * (1.0 - rho) / (1.0 + rho) if rho > -1.0 else NAN
    else:
        rho, n_eff = NAN, NAN
    noise = _MAD_TO_SD * float(np.median(np.abs(residuals - np.median(residuals))))
    return _Ols(slope, intercept, n, r2, stderr, rho, n_eff, noise)


# ---------------------------------------------------------------------------
# M4: elastic fit (public helper)
# ---------------------------------------------------------------------------


@fr.atomic
def elastic_fit(
    stress_mpa: StressArray,
    strain: StrainArray,
    lo_mpa: StressMPa,
    hi_mpa: StressMPa,
    *,
    min_points: int = DEFAULT_PARAMETERS.min_elastic_points,
) -> ElasticFit:
    """OLS sigma = E*e + b on the samples with ``lo_mpa <= sigma <= hi_mpa`` (M4).

    The caller passes the first-loading-branch arrays. Samples where either
    channel is non-finite are ignored. With fewer than ``min_points`` samples
    in the window (M4: 10), all fit values are NaN and ``n`` is still given.
    """
    stress = np.asarray(stress_mpa, dtype=float)
    eps = np.asarray(strain, dtype=float)
    mask = np.isfinite(stress) & np.isfinite(eps) & (stress >= lo_mpa) & (stress <= hi_mpa)
    ols = _ols(eps[mask], stress[mask], min_points)
    fit = ElasticFit(
        e_gpa=ols.slope / 1000.0,
        intercept_mpa=ols.intercept,
        n=ols.n,
        r2=ols.r2,
        stderr_gpa=ols.stderr / 1000.0,
        lag1_rho=ols.lag1_rho,
        n_eff=ols.n_eff,
        lo_mpa=float(lo_mpa),
        hi_mpa=float(hi_mpa),
    )
    return fit


# ---------------------------------------------------------------------------
# M14: S0 scatter; M13: A and Z
# ---------------------------------------------------------------------------


@fr.atomic
def s0_scatter(s0_mm2: AreaMM2, measurements_mm2: Sequence[float]) -> S0Scatter:
    """u_rel(S0) = sd(s_1..s_n)/sqrt(n)/S0 with nu = n - 1 (M14).

    Labelled :data:`S0_SCATTER_LABEL`. Propagation is exact: every
    stress-based property (including E) has the same relative uncertainty,
    and strain outputs are unaffected. Fewer than 2 finite measurements give
    NaN with flag ``"s0:no_s0_spread"``.
    """
    values = np.asarray([m for m in measurements_mm2 if _is_number(m)], dtype=float)
    if len(values) < 2 or not _is_number(s0_mm2) or s0_mm2 <= 0:
        scatter = S0Scatter(u_rel=NAN, dof=max(len(values) - 1, 0), flags=("s0:no_s0_spread",))
    else:
        u_abs = float(np.std(values, ddof=1)) / math.sqrt(len(values))
        scatter = S0Scatter(u_rel=u_abs / float(s0_mm2), dof=len(values) - 1)
    return scatter


@fr.atomic
def elongation_after_fracture(l0_mm: OptionalLengthMM, lu_mm: OptionalLengthMM) -> ScalarResult:
    """A = (Lu - L0)/L0 * 100 % from metadata (M13); NaN, ``"a:missing_input"`` if missing."""
    if _is_number(l0_mm) and _is_number(lu_mm) and l0_mm > 0:
        result = ScalarResult(value=(float(lu_mm) - float(l0_mm)) / float(l0_mm) * 100.0)
    else:
        result = ScalarResult(value=NAN, flags=("a:missing_input",))
    return result


@fr.atomic
def reduction_of_area(s0_mm2: OptionalAreaMM2, su_mm2: OptionalAreaMM2) -> ScalarResult:
    """Z = (S0 - Su)/S0 * 100 % from metadata (M13); NaN, ``"z:missing_input"`` if missing."""
    if _is_number(s0_mm2) and _is_number(su_mm2) and s0_mm2 > 0:
        result = ScalarResult(value=(float(s0_mm2) - float(su_mm2)) / float(s0_mm2) * 100.0)
    else:
        result = ScalarResult(value=NAN, flags=("z:missing_input",))
    return result


# ---------------------------------------------------------------------------
# §13: comparison with lab-reported values; §12: aggregation
# ---------------------------------------------------------------------------

# (computed key, ReportedProperties field, tolerance kind, relative part, absolute part, text)
_COMPARISONS: tuple[tuple[str, str, str, float, float, str], ...] = (
    ("fm_n", "fm_n", "relative", 0.001, 0.0, "±0.1 %"),
    ("rm_mpa", "rm_mpa", "relative", 0.001, 0.5, "±(0.1 % + 0.5 MPa)"),
    ("reh_mpa", "reh_mpa", "relative", 0.02, 0.0, "±2 %"),
    ("rp02_mpa", "rp02_mpa", "relative", 0.02, 0.0, "±2 %"),
    ("e_gpa", "e_gpa", "relative", 0.10, 0.0, "±10 %"),
    ("a_percent", "a_percent", "rounding", 0.0, 0.5, "rounding-aware: rounds to the lab integer (±0.5 pp at a tie)"),
    ("z_percent", "z_percent", "rounding", 0.0, 0.5, "rounding-aware: rounds to the lab integer (±0.5 pp at a tie)"),
)
"""Tolerances fixed at G1 (method-spec §13). ReL has no lab values and no row."""


@fr.atomic
def compare_with_reported(
    computed: Mapping[str, float | None], reported: ReportedProperties
) -> pd.DataFrame:
    """Consistency check of computed against lab-reported values (method-spec §13).

    One row per quantity (``fm_n``, ``rm_mpa``, ``reh_mpa``, ``rp02_mpa``,
    ``e_gpa``, ``a_percent``, ``z_percent``) with columns ``quantity``,
    ``computed``, ``reported``, ``rel_deviation``, ``outcome`` and
    ``tolerance``. ``outcome`` is ``"not_comparable"`` if either value is
    missing or NaN. A and Z are ``"within"`` if the computed value rounds to
    the lab's integer value, which with ties counted as within is
    ``|computed - reported| <= 0.5``. This is a consistency check with the
    lab's software, not a validation.
    """
    rows = []
    for key, field, kind, rel, abs_, text in _COMPARISONS:
        value = computed.get(key)
        lab = getattr(reported, field)
        value = float(value) if _is_number(value) else NAN
        lab = float(lab) if _is_number(lab) else NAN
        if math.isnan(value) or math.isnan(lab):
            outcome, deviation = "not_comparable", NAN
        else:
            deviation = (value - lab) / lab if lab != 0.0 else NAN
            if kind == "relative":
                allowed = rel * abs(lab) + abs_
            else:
                allowed = abs_
            outcome = "within" if abs(value - lab) <= allowed else "outside"
        rows.append(
            {
                "quantity": key,
                "computed": value,
                "reported": lab,
                "rel_deviation": deviation,
                "outcome": outcome,
                "tolerance": text,
            }
        )
    comparison = pd.DataFrame(
        rows, columns=["quantity", "computed", "reported", "rel_deviation", "outcome", "tolerance"]
    )
    return comparison


@fr.atomic
def summarize_by_group(
    table: pd.DataFrame, group_column: str, quantities: Sequence[str]
) -> pd.DataFrame:
    """Descriptive statistics per group and quantity (method-spec §12).

    Long format with columns ``group_column``, ``quantity``, ``n`` (non-NaN
    count), ``mean``, ``sd`` (ddof = 1; NaN for n < 2), ``min`` and ``max``.
    Groups appear in sorted order, quantities in the given order. No
    hypothesis tests and no exclusions.
    """
    rows = []
    for group, frame in table.groupby(group_column, sort=True):
        for quantity in quantities:
            values = pd.to_numeric(frame[quantity], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            n = int(len(values))
            rows.append(
                {
                    group_column: group,
                    "quantity": quantity,
                    "n": n,
                    "mean": float(values.mean()) if n else NAN,
                    "sd": float(values.std(ddof=1)) if n >= 2 else NAN,
                    "min": float(values.min()) if n else NAN,
                    "max": float(values.max()) if n else NAN,
                }
            )
    summary = pd.DataFrame(rows, columns=[group_column, "quantity", "n", "mean", "sd", "min", "max"])
    return summary


# ---------------------------------------------------------------------------
# Curve analysis: result types
# ---------------------------------------------------------------------------

CONTINUOUS = "continuous"
DISCONTINUOUS = "discontinuous"

_YIELD_SEARCH_FRACTION_OF_RM = 0.5
"""M7: the drop is searched only where sigma > 0.5*Rm (fixed by the method)."""

_PROVISIONAL_MIN_POINTS = 3
"""M5 step 2: the provisional fit needs residuals for the noise level."""


@dataclasses.dataclass(frozen=True)
class CurveAnalysis:
    """Per-specimen curve analysis (fields per analysis-api.md, then extras).

    Undefined values are NaN. ``k_ext`` equals the number of samples if the
    extensometer is valid to the end. ``flags`` entries are
    ``"<property>:<flag>"``. Fields after ``flags`` are additional
    diagnostics: the reference stress of the final E window, the final
    residual noise (1.4826*MAD, M5 step 5), the number of E-window
    iterations, the ReL plateau level and extent (M9), and sample indices
    (-1 if undefined) for plotting.
    """

    yield_type: str
    noise_mpa: float
    e_gpa: float
    e_intercept_mpa: float
    e_n: int
    e_r2: float
    e_stderr_gpa: float
    e_lag1_rho: float
    e_n_eff: float
    e_window_mpa: tuple[float, float]
    strain_origin: float
    reh_mpa: float
    reh_sample_increment_mpa: float
    rel_mpa: float
    rel_incl_transient_mpa: float
    rp02_mpa: float
    fm_n: float
    rm_mpa: float
    agt_percent: float
    ag_percent: float
    ag_location_halfwidth_percent: float
    k_ext: int
    extensometer_max_valid_strain: float
    e_window_range_gpa: tuple[float, float]
    rp02_window_range_mpa: tuple[float, float]
    ag_window_range_percent: tuple[float, float]
    agt_window_range_percent: tuple[float, float]
    e_linearity_rel_diff: float
    dropped_samples: dict
    flags: tuple[str, ...]
    e_reference_stress_mpa: float = NAN
    e_residual_noise_mpa: float = NAN
    e_window_iterations: int = 0
    rel_plateau_level_mpa: float = NAN
    rel_plateau_extent: float = NAN
    indices: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ExtensometerValidity:
    """Result of :func:`extensometer_validity` (M1-M3)."""

    k_ext: int
    max_valid_strain: float
    n_windows: int
    flags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Bootstrap:
    """Result of :func:`bootstrap_fit` (M5 steps 1-2), in force units.

    ``slope_n`` is the provisional dF/de (E0*S0, N per unit strain);
    ``noise_n`` is the force noise s_F = s_n*S0 (1.4826*MAD of the
    provisional residuals), fixed for M7, M9 and M12.
    """

    fm_n: float
    slope_n: float
    intercept_n: float
    noise_n: float
    n: int
    noise_mpa: float


@dataclasses.dataclass(frozen=True)
class Yielding:
    """Result of :func:`yield_analysis` (M7-M9). Indices are -1 if undefined."""

    yield_type: str
    i_p: int
    i_h: int
    i_rel: int
    work_hardening: bool
    reh_mpa: float
    reh_sample_increment_mpa: float
    rel_mpa: float
    rel_incl_transient_mpa: float
    plateau_level_mpa: float
    plateau_extent: float
    flags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ElasticResult:
    """Result of :func:`final_elastic_fit` (M4-M6, M16).

    ``fit`` is in MPa/GPa; ``slope_n``, ``intercept_n``, ``reference_n``,
    ``lo_n`` and ``hi_n`` are the same quantities in force units.
    """

    fit: ElasticFit
    slope_n: float
    intercept_n: float
    strain_origin: float
    reference_n: float
    reference_mpa: float
    lo_n: float
    hi_n: float
    residual_noise_mpa: float
    iterations: int
    flags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ProofStrength:
    """Result of :func:`proof_strength` (M10); ``index`` is the sample after the crossing."""

    rp02_mpa: float
    index: int
    flags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class MaximumForce:
    """Result of :func:`maximum_force` (M11)."""

    fm_n: float
    rm_mpa: float
    i_m: int
    flags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Elongation:
    """Result of :func:`elongation_at_maximum_force` (M12).

    ``band`` holds the first and last sample index of the near-maximum band;
    ``strain_mid`` is its strain mid-point (raw strain, ratio).
    """

    agt_percent: float
    ag_percent: float
    ag_location_halfwidth_percent: float
    band: tuple[int, int]
    strain_mid: float
    flags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class WindowSensitivity:
    """Result of :func:`window_sensitivity` (M15): ranges (min, max), never sd."""

    e_range_gpa: tuple[float, float]
    rp02_range_mpa: tuple[float, float]
    ag_range_percent: tuple[float, float]
    agt_range_percent: tuple[float, float]
    linearity_rel_diff: float
    flags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Curve analysis: helpers (force space)
# ---------------------------------------------------------------------------


def _params(params: AnalysisParameters | None) -> AnalysisParameters:
    return DEFAULT_PARAMETERS if params is None else params


def _channels(force_n, strain, crosshead_mm=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Float arrays of equal length; a missing crosshead becomes all-NaN."""
    force = np.asarray(force_n, dtype=float)
    eps = np.asarray(strain, dtype=float)
    if force.ndim != 1 or eps.shape != force.shape:
        raise ValueError(f"force and strain must be 1-D of equal length, got {force.shape}, {eps.shape}")
    if crosshead_mm is None:
        cross = np.full(force.shape, NAN)
    else:
        cross = np.asarray(crosshead_mm, dtype=float)
        if cross.shape != force.shape:
            raise ValueError(f"crosshead must have the shape of force, got {cross.shape}")
    return force, eps, cross


def _check_s0(s0_mm2) -> float:
    if not (_is_number(s0_mm2) and s0_mm2 > 0):
        raise ValueError(f"S0 must be a positive finite number, got {s0_mm2!r}")
    return float(s0_mm2)


def _first(mask: np.ndarray) -> int:
    """Index of the first True entry, or -1."""
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else -1


def _strain_valid(force: np.ndarray, eps: np.ndarray, k_ext: int) -> np.ndarray:
    """M1/M2: samples with finite force and strain below ``k_ext``."""
    valid = np.isfinite(force) & np.isfinite(eps)
    valid[max(int(k_ext), 0):] = False
    return valid


def _near_switch(strain_value: float, rate_switch_strain, p: AnalysisParameters) -> bool:
    return (
        _is_number(rate_switch_strain)
        and _is_number(strain_value)
        and abs(strain_value - rate_switch_strain) <= p.rate_switch_proximity
    )


def _strain_at(eps: np.ndarray, index: int) -> float:
    return float(eps[index]) if 0 <= index < len(eps) else NAN


def _window_ols(
    force: np.ndarray,
    eps: np.ndarray,
    valid: np.ndarray,
    reference_n: float,
    f_lo: float,
    f_hi: float,
    min_points: int,
) -> tuple[_Ols, float, float]:
    """OLS F = slope*e + b on the first loading branch below ``reference_n``.

    The branch is every sample before the force first reaches the reference;
    the window is ``f_lo*ref <= F <= f_hi*ref`` on strain-valid samples.
    """
    lo, hi = f_lo * reference_n, f_hi * reference_n
    if not math.isfinite(reference_n):
        return _Ols(n=0, **_OLS_UNDEFINED), lo, hi
    reached = _first(np.isfinite(force) & (force >= reference_n))
    branch = np.zeros(len(force), dtype=bool)
    branch[: len(force) if reached < 0 else reached] = True
    mask = valid & branch & (force >= lo) & (force <= hi)
    return _ols(eps[mask], force[mask], min_points), lo, hi


def _proof_crossing(
    force: np.ndarray,
    eps: np.ndarray,
    valid: np.ndarray,
    slope: float,
    intercept: float,
    start_n: float,
    offset: float,
) -> tuple[float, int, bool]:
    """First crossing of e_p = ``offset`` (M10), linearly interpolated in force.

    The search runs over consecutive strain-valid samples from the first one
    with F >= ``start_n``. Returns (force at the crossing, index of the sample
    after it, reached); ``reached`` is False only if e_p stays below the
    offset to the end of the valid strain.
    """
    if not (math.isfinite(slope) and slope != 0.0 and math.isfinite(intercept) and math.isfinite(start_n)):
        return NAN, -1, True
    indices = np.flatnonzero(valid)
    f_valid = force[indices]
    plastic = eps[indices] - (f_valid - intercept) / slope
    start = _first(f_valid >= start_n)
    if start < 0:
        return NAN, -1, True
    beyond = _first(plastic[start:] >= offset)
    if beyond < 0:
        return NAN, -1, False
    if beyond == 0:
        return NAN, -1, True
    a, b = start + beyond - 1, start + beyond
    fraction = (offset - plastic[a]) / (plastic[b] - plastic[a])
    crossing = f_valid[a] + fraction * (f_valid[b] - f_valid[a])
    return float(crossing), int(indices[b]), True


def _value_range(values: list[float]) -> tuple[float, float]:
    finite = [v for v in values if math.isfinite(v)]
    return (min(finite), max(finite)) if finite else (NAN, NAN)


def _unique(flags) -> tuple[str, ...]:
    return tuple(dict.fromkeys(flags))


# ---------------------------------------------------------------------------
# Step nodes
# ---------------------------------------------------------------------------


@fr.atomic
def check_time_order(time_s: TimeArray) -> int:
    """M1: the sample order must be the time order; a decrease raises ValueError.

    Non-finite times are ignored; equal consecutive times are allowed.
    Returns the number of samples.
    """
    times = np.asarray(time_s, dtype=float)
    finite = times[np.isfinite(times)]
    if np.any(np.diff(finite) < 0.0):
        first = int(np.flatnonzero(np.diff(finite) < 0.0)[0])
        raise ValueError(f"time is not monotonic (decreases after finite sample {first})")
    n_samples = int(len(times))
    return n_samples


@fr.atomic
def engineering_stress(force_n: ForceArray, s0_mm2: AreaMM2) -> StressArray:
    """sigma = F/S0 in MPa (for display; the analysis decides in force space)."""
    stress_mpa = np.asarray(force_n, dtype=float) / _check_s0(s0_mm2)
    return stress_mpa


@fr.atomic
def count_dropped_samples(
    force_n: ForceArray, strain: StrainArray, crosshead_mm: OptionalCrossheadArray = None
) -> dict:
    """M1: number of non-finite samples per channel (``force``, ``strain``, ``crosshead``).

    Also checks that the channels have equal length (ValueError otherwise).
    """
    force, eps, cross = _channels(force_n, strain, crosshead_mm)
    dropped_samples = {
        "force": int(np.sum(~np.isfinite(force))),
        "strain": int(np.sum(~np.isfinite(eps))),
        "crosshead": int(np.sum(~np.isfinite(cross))),
    }
    return dropped_samples


def _window_invalid(eps: np.ndarray, cross: np.ndarray, a: int, b: int, l0: float, p: AnalysisParameters) -> bool:
    segment = eps[a : b + 1]
    if not np.all(np.isfinite(segment)) or np.any(segment == 0.0):
        return True
    if np.any(np.diff(segment) < -p.ext_jump):
        return True
    ratio = (eps[b] - eps[a]) / ((cross[b] - cross[a]) / l0)
    return bool(ratio < p.ext_ratio_min)


def _extensometer_end(eps: np.ndarray, cross: np.ndarray, l0_mm, p: AnalysisParameters) -> tuple[int, int, bool]:
    """(k_ext, number of crosshead windows, whether the window rule could run)."""
    n = len(eps)
    finite = np.isfinite(eps)
    first_finite = _first(finite)
    if first_finite < 0:
        k_nan = 0
    else:
        after = _first(~finite[first_finite:])
        k_nan = n if after < 0 else first_finite + after

    crosshead_samples = np.flatnonzero(np.isfinite(cross))
    usable = _is_number(l0_mm) and l0_mm > 0 and len(crosshead_samples) >= 2
    windows: list[tuple[int, int]] = []
    if usable:
        start = int(crosshead_samples[0])
        for i in crosshead_samples[1:]:
            if abs(cross[i] - cross[start]) >= p.ext_window_mm:
                windows.append((start, int(i)))
                start = int(i)
        last = int(crosshead_samples[-1])
        if windows and windows[-1][1] != last:
            windows[-1] = (windows[-1][0], last)

    k_window = n
    for a, b in reversed(windows):
        if not _window_invalid(eps, cross, a, b, float(l0_mm), p):
            break
        k_window = a
    return min(k_nan, k_window), len(windows), usable


@fr.atomic
def extensometer_validity(
    strain: StrainArray,
    crosshead_mm: OptionalCrossheadArray,
    l0_mm: OptionalLengthMM,
    a_percent: OptionalStrainPercent = None,
    rate_switch_strain: OptionalStrainRatio = None,
    params: AnalysisParameters | None = None,
) -> tuple[ExtensometerValidity, int, StrainRatio]:
    """M1-M3: extensometer end ``k_ext`` (rule in the module docstring) and checks.

    Flags: ``extensometer:missing_input`` (no crosshead or L0, window rule
    skipped), ``extensometer:extensometer_ended_early`` (largest valid strain
    below 0.5*A/100 when A is given) and ``extensometer:near_rate_switch``.
    A strain channel continued from the crosshead cannot be detected
    (:data:`RESIDUAL_RISKS`).
    """
    p = _params(params)
    eps = np.asarray(strain, dtype=float)
    _, _, cross = _channels(eps, eps, crosshead_mm)
    k_end, n_windows, usable = _extensometer_end(eps, cross, l0_mm, p)
    flags = []
    if not usable:
        flags.append("extensometer:missing_input")
    head = eps[:k_end]
    head = head[np.isfinite(head)]
    max_valid = float(head.max()) if len(head) else NAN
    if _is_number(a_percent) and not max_valid >= p.ext_end_fraction_of_a * a_percent / 100.0:
        flags.append("extensometer:extensometer_ended_early")
    if k_end < len(eps) and _near_switch(max_valid, rate_switch_strain, p):
        flags.append("extensometer:near_rate_switch")
    extensometer = ExtensometerValidity(
        k_ext=int(k_end), max_valid_strain=max_valid, n_windows=n_windows, flags=tuple(flags)
    )
    k_ext = extensometer.k_ext
    extensometer_max_valid_strain = extensometer.max_valid_strain
    return extensometer, k_ext, extensometer_max_valid_strain


@fr.atomic
def bootstrap_fit(
    force_n: ForceArray,
    strain: StrainArray,
    s0_mm2: AreaMM2,
    k_ext: int,
    params: AnalysisParameters | None = None,
) -> tuple[Bootstrap, StressMPa]:
    """M5 steps 1-2: Fm from force alone, provisional E0/b0 and the noise level s_n.

    The provisional window is [f_lo, f_hi]*(0.5*Rm) on the samples before
    sigma first reaches 0.5*Rm, with Rm = max over all finite forces / S0.
    s_n = 1.4826*MAD of its residuals; it is fixed here for M7, M9 and M12.
    """
    p = _params(params)
    s0 = _check_s0(s0_mm2)
    force, eps, _ = _channels(force_n, strain)
    finite_force = force[np.isfinite(force)]
    fm = float(finite_force.max()) if len(finite_force) else NAN
    valid = _strain_valid(force, eps, k_ext)
    reference = p.provisional_fraction_of_rm * fm
    ols, _, _ = _window_ols(force, eps, valid, reference, p.f_lo, p.f_hi, _PROVISIONAL_MIN_POINTS)
    bootstrap = Bootstrap(
        fm_n=fm,
        slope_n=ols.slope,
        intercept_n=ols.intercept,
        noise_n=ols.residual_noise,
        n=ols.n,
        noise_mpa=ols.residual_noise / s0,
    )
    noise_mpa = bootstrap.noise_mpa
    return bootstrap, noise_mpa


def _yield_points(
    force: np.ndarray,
    eps: np.ndarray,
    valid: np.ndarray,
    boot: Bootstrap,
    rate_switch_strain,
    p: AnalysisParameters,
) -> dict:
    """M7-M9 in force units; see the module docstring for the M9 rule."""
    result = dict(
        yield_type=CONTINUOUS, i_p=-1, i_h=-1, i_rel=-1, work_hardening=False,
        reh=NAN, increment=NAN, rel=NAN, rel_incl=NAN, level=NAN, extent=NAN, flags=[],
    )
    no_upper = ["reh:no_upper_yield", "rel:no_upper_yield"]

    finite_idx = np.flatnonzero(np.isfinite(force))
    f = force[finite_idx]
    if len(f) < 2 or not math.isfinite(boot.fm_n):
        result["flags"] = no_upper
        return result

    # M7: running-maximum drawdown.
    noise = boot.noise_n if math.isfinite(boot.noise_n) else 0.0
    running = np.maximum.accumulate(f)
    drop = np.maximum(p.drop_rel * running, p.drop_noise_factor * noise)
    j_star = _first((f > _YIELD_SEARCH_FRACTION_OF_RM * boot.fm_n) & (f <= running - drop))
    if j_star < 0:
        result["flags"] = no_upper
        return result
    peak = int(np.argmax(f[: j_star + 1]))
    i_p = int(finite_idx[peak])

    slope, intercept = boot.slope_n, boot.intercept_n
    if not (math.isfinite(slope) and slope > 0 and math.isfinite(intercept)):
        result["flags"] = ["reh:insufficient_elastic_points", *no_upper]
        return result
    if not valid[i_p]:
        result["flags"] = ["reh:extensometer_invalid", *no_upper]
        return result
    plastic = eps - (force - intercept) / slope
    if plastic[i_p] > p.peak_max_plastic_strain:
        result["flags"] = no_upper
        return result
    recovered = _first(f[peak + 1 :] > f[peak])
    stop = len(force) if recovered < 0 else int(finite_idx[peak + 1 + recovered])
    flow = np.arange(i_p + 1, stop)
    flow = flow[valid[flow]]
    if len(flow) == 0:
        result["flags"] = ["reh:extensometer_invalid", *no_upper]
        return result
    if plastic[flow].max() - plastic[i_p] < p.min_flow_plastic_strain:
        result["flags"] = no_upper
        return result

    # Discontinuous yielding: ReH (M8) and the end of yielding i_h.
    flags = []
    reh = float(f[peak])
    increment = float(f[peak] - f[peak - 1]) if peak > 0 else NAN
    hardening = _first(f[peak + 1 :] >= reh)
    work_hardening = hardening >= 0
    if work_hardening:
        h = peak + 1 + hardening
        i_h = int(finite_idx[h])
    else:
        # INTERIM (decision pending): the stress never returns to ReH. The method
        # sets i_h = i_m, but M11 leaves Fm (hence i_m) undefined here, so the
        # end of the yield range is undefined: ReL is not evaluated (flag below)
        # and the yield range is taken to run to the end of the record for the
        # in_yield_plateau check.
        h = len(f)
        i_h = len(force)
        flags.append("rel:no_work_hardening")

    # M9: ReL, option (A).
    y_force = f[peak + 1 : h] if work_hardening else f[:0]
    y_index = finite_idx[peak + 1 : h] if work_hardening else finite_idx[:0]
    rel = rel_incl = level = extent = NAN
    i_rel = -1
    if len(y_force):
        rel_incl = float(y_force.min())
        level = float(np.median(y_force[y_force <= 0.5 * (rel_incl + reh)]))
        # Hysteresis against noise: the transient begins below level - d (the
        # M7 drop threshold at ReH); recovery is complete at sigma >= level.
        below = _first(y_force < level - max(p.drop_rel * reh, p.drop_noise_factor * noise))
        if below < 0:
            start = 0
        else:
            back = _first(y_force[below + 1 :] >= level)
            start = -1 if back < 0 else below + 1 + back
        single = start < 0
        if not single:
            if not valid[y_index[start]]:
                flags.append("rel:extensometer_invalid")
            else:
                candidate_pos = start + int(np.argmin(y_force[start:]))
                upper = 2.0 * level - y_force[candidate_pos]
                inside = np.flatnonzero((y_force[start:] <= upper) & valid[y_index[start:]])
                last = start + int(inside[-1]) if len(inside) else start
                extent = float(plastic[y_index[last]] - plastic[y_index[start]])
                if extent >= p.plateau_min_extent:
                    i_rel = int(y_index[candidate_pos])
                else:
                    single = True
        if single:
            transient = y_force[:start] if start > 0 else y_force
            i_rel = int(y_index[int(np.argmin(transient))])
            flags.append("rel:single_minimum")
        if i_rel >= 0:
            rel = float(force[i_rel])

    if _near_switch(_strain_at(eps, i_p), rate_switch_strain, p):
        flags.append("reh:near_rate_switch")
    if i_rel >= 0 and valid[i_rel] and _near_switch(_strain_at(eps, i_rel), rate_switch_strain, p):
        flags.append("rel:near_rate_switch")
    result.update(
        yield_type=DISCONTINUOUS, i_p=i_p, i_h=i_h, i_rel=i_rel, work_hardening=work_hardening,
        reh=reh, increment=increment, rel=rel, rel_incl=rel_incl, level=level, extent=extent,
        flags=flags,
    )
    return result


@fr.atomic
def yield_analysis(
    force_n: ForceArray,
    strain: StrainArray,
    s0_mm2: AreaMM2,
    k_ext: int,
    bootstrap: Bootstrap,
    rate_switch_strain: OptionalStrainRatio = None,
    params: AnalysisParameters | None = None,
) -> tuple[Yielding, str, StressMPa, StressMPa]:
    """M5 step 3, M7-M9: yield type, ReH (raw sample) and ReL (option A).

    Uses only the bootstrap E0, b0 and s_n. Continuous yielding gives NaN
    ReH/ReL with flags ``reh:no_upper_yield`` and ``rel:no_upper_yield``.
    """
    p = _params(params)
    s0 = _check_s0(s0_mm2)
    force, eps, _ = _channels(force_n, strain)
    valid = _strain_valid(force, eps, k_ext)
    points = _yield_points(force, eps, valid, bootstrap, rate_switch_strain, p)
    yielding = Yielding(
        yield_type=points["yield_type"],
        i_p=points["i_p"],
        i_h=points["i_h"],
        i_rel=points["i_rel"],
        work_hardening=points["work_hardening"],
        reh_mpa=points["reh"] / s0,
        reh_sample_increment_mpa=points["increment"] / s0,
        rel_mpa=points["rel"] / s0,
        rel_incl_transient_mpa=points["rel_incl"] / s0,
        plateau_level_mpa=points["level"] / s0,
        plateau_extent=points["extent"],
        flags=_unique(points["flags"]),
    )
    yield_type = yielding.yield_type
    reh_mpa = yielding.reh_mpa
    rel_mpa = yielding.rel_mpa
    return yielding, yield_type, reh_mpa, rel_mpa


def _continuous_window(
    force: np.ndarray, eps: np.ndarray, valid: np.ndarray, boot: Bootstrap, p: AnalysisParameters
) -> tuple[_Ols, float, float, float, int, str]:
    """M5 step 4 for continuous yielding: fixed point sigma_ref = Rp0.2(E).

    Returns (fit, lo, hi, reference, iterations, status) in force units;
    status is ``converged``, ``not_converged``, ``insufficient`` or
    ``no_reference``.
    """
    undefined = _Ols(n=0, **_OLS_UNDEFINED)
    slope, intercept = boot.slope_n, boot.intercept_n
    if not math.isfinite(slope):
        return undefined, NAN, NAN, NAN, 0, "insufficient"
    start = p.f_lo * p.provisional_fraction_of_rm * boot.fm_n
    rp, _, _ = _proof_crossing(force, eps, valid, slope, intercept, start, p.offset_plastic_strain)
    ols, lo, hi, reference = undefined, NAN, NAN, NAN
    iterations, status = 0, "no_reference"
    for _ in range(p.max_window_iterations):
        if not math.isfinite(rp):
            break
        reference = rp
        ols, lo, hi = _window_ols(force, eps, valid, reference, p.f_lo, p.f_hi, p.min_elastic_points)
        iterations += 1
        if not math.isfinite(ols.slope):
            status = "insufficient"
            break
        change = abs(ols.slope - slope) / abs(slope)
        slope = ols.slope
        rp, _, _ = _proof_crossing(force, eps, valid, ols.slope, ols.intercept, lo, p.offset_plastic_strain)
        if change < p.window_rel_tol:
            status = "converged"
            break
        status = "not_converged"
    return ols, lo, hi, reference, iterations, status


@fr.atomic
def final_elastic_fit(
    force_n: ForceArray,
    strain: StrainArray,
    s0_mm2: AreaMM2,
    k_ext: int,
    bootstrap: Bootstrap,
    yielding: Yielding,
    params: AnalysisParameters | None = None,
) -> tuple[ElasticResult, ModulusGPa, StrainRatio]:
    """M4, M5 steps 4-5, M6, M16: final E window fit and strain origin e0 = -b/E.

    sigma_ref = ReH for discontinuous yielding; for continuous yielding the
    fixed point sigma_ref = Rp0.2(E) from the Rp0.2 with E0 (relative change
    of E < ``window_rel_tol``, at most ``max_window_iterations``). Flags:
    ``e:insufficient_elastic_points`` (fewer than ``min_elastic_points``)
    and ``e:e_window_not_converged``. E is labelled :data:`E_LABEL`.
    """
    p = _params(params)
    s0 = _check_s0(s0_mm2)
    force, eps, _ = _channels(force_n, strain)
    valid = _strain_valid(force, eps, k_ext)
    flags = []
    if yielding.yield_type == DISCONTINUOUS:
        reference = float(force[yielding.i_p])
        ols, lo, hi = _window_ols(force, eps, valid, reference, p.f_lo, p.f_hi, p.min_elastic_points)
        iterations = 1
        status = "converged" if math.isfinite(ols.slope) else "insufficient"
    else:
        ols, lo, hi, reference, iterations, status = _continuous_window(force, eps, valid, bootstrap, p)
    if status == "insufficient":
        flags.append("e:insufficient_elastic_points")
    elif status in ("not_converged", "no_reference"):
        flags.append("e:e_window_not_converged")
    fit = ElasticFit(
        e_gpa=ols.slope / s0 / 1000.0,
        intercept_mpa=ols.intercept / s0,
        n=ols.n,
        r2=ols.r2,
        stderr_gpa=ols.stderr / s0 / 1000.0,
        lag1_rho=ols.lag1_rho,
        n_eff=ols.n_eff,
        lo_mpa=lo / s0,
        hi_mpa=hi / s0,
    )
    elastic = ElasticResult(
        fit=fit,
        slope_n=ols.slope,
        intercept_n=ols.intercept,
        strain_origin=-ols.intercept / ols.slope if math.isfinite(ols.slope) else NAN,
        reference_n=reference,
        reference_mpa=reference / s0,
        lo_n=lo,
        hi_n=hi,
        residual_noise_mpa=ols.residual_noise / s0,
        iterations=iterations,
        flags=tuple(flags),
    )
    e_gpa = elastic.fit.e_gpa
    strain_origin = elastic.strain_origin
    return elastic, e_gpa, strain_origin


@fr.atomic
def proof_strength(
    force_n: ForceArray,
    strain: StrainArray,
    s0_mm2: AreaMM2,
    k_ext: int,
    elastic: ElasticResult,
    yielding: Yielding,
    rate_switch_strain: OptionalStrainRatio = None,
    params: AnalysisParameters | None = None,
) -> tuple[ProofStrength, StressMPa]:
    """M10: Rp0.2 at the first crossing of e_p = 0.002 (linear interpolation).

    Flags: ``rp02:extensometer_invalid`` (crossing not reached before
    ``k_ext``), ``rp02:in_yield_plateau`` (discontinuous yielding, crossing in
    [i_p, i_h]) and ``rp02:near_rate_switch``.
    """
    p = _params(params)
    s0 = _check_s0(s0_mm2)
    force, eps, _ = _channels(force_n, strain)
    valid = _strain_valid(force, eps, k_ext)
    crossing, index, reached = _proof_crossing(
        force, eps, valid, elastic.slope_n, elastic.intercept_n, elastic.lo_n, p.offset_plastic_strain
    )
    flags = []
    if not reached and k_ext < len(force):
        flags.append("rp02:extensometer_invalid")
    if index >= 0:
        if yielding.yield_type == DISCONTINUOUS and yielding.i_p <= index <= yielding.i_h:
            flags.append("rp02:in_yield_plateau")
        if _near_switch(_strain_at(eps, index), rate_switch_strain, p):
            flags.append("rp02:near_rate_switch")
    proof = ProofStrength(rp02_mpa=crossing / s0, index=index, flags=tuple(flags))
    rp02_mpa = proof.rp02_mpa
    return proof, rp02_mpa


@fr.atomic
def maximum_force(
    force_n: ForceArray, s0_mm2: AreaMM2, yielding: Yielding
) -> tuple[MaximumForce, ForceN, StressMPa]:
    """M11: Fm and Rm = Fm/S0.

    Continuous yielding: maximum over all finite forces. Discontinuous: the
    maximum from i_h on (work hardening); without work hardening, NaN with
    flag ``fm:no_work_hardening``.
    """
    s0 = _check_s0(s0_mm2)
    force = np.asarray(force_n, dtype=float)
    candidates = np.flatnonzero(np.isfinite(force))
    flags = []
    if yielding.yield_type == DISCONTINUOUS:
        if yielding.work_hardening:
            candidates = candidates[candidates >= yielding.i_h]
        else:
            candidates = candidates[:0]
            flags.append("fm:no_work_hardening")
    if len(candidates):
        i_m = int(candidates[int(np.argmax(force[candidates]))])
        fm = float(force[i_m])
    else:
        i_m, fm = -1, NAN
    maximum = MaximumForce(fm_n=fm, rm_mpa=fm / s0, i_m=i_m, flags=tuple(flags))
    fm_n = maximum.fm_n
    rm_mpa = maximum.rm_mpa
    return maximum, fm_n, rm_mpa


@fr.atomic
def elongation_at_maximum_force(
    force_n: ForceArray,
    strain: StrainArray,
    k_ext: int,
    bootstrap: Bootstrap,
    elastic: ElasticResult,
    maximum: MaximumForce,
    params: AnalysisParameters | None = None,
) -> tuple[Elongation, StrainPercent, StrainPercent]:
    """M12: Agt and Ag at the strain mid-point of the near-maximum band.

    The band is the contiguous run of finite-force samples around i_m with
    F >= Fm*(1 - delta_F), delta_F = max(band_rel_min, 3*s_F/Fm). Agt =
    (e_mid - e0)*100 %, Ag = Agt - Rm/E*100 %. If any band sample is at or
    beyond ``k_ext`` (or has no finite strain), both are NaN with flags
    ``ag:extensometer_invalid`` and ``agt:extensometer_invalid``.
    """
    p = _params(params)
    force, eps, _ = _channels(force_n, strain)
    flags = []
    agt = ag = halfwidth = strain_mid = NAN
    band = (-1, -1)
    if maximum.i_m >= 0:
        noise = bootstrap.noise_n if math.isfinite(bootstrap.noise_n) else 0.0
        delta = max(p.band_rel_min, p.band_noise_factor * noise / maximum.fm_n)
        threshold = maximum.fm_n * (1.0 - delta)
        finite_idx = np.flatnonzero(np.isfinite(force))
        centre = int(np.searchsorted(finite_idx, maximum.i_m))
        lo = centre
        while lo > 0 and force[finite_idx[lo - 1]] >= threshold:
            lo -= 1
        hi = centre
        while hi + 1 < len(finite_idx) and force[finite_idx[hi + 1]] >= threshold:
            hi += 1
        members = finite_idx[lo : hi + 1]
        band = (int(members[0]), int(members[-1]))
        band_strain = eps[members]
        if members[-1] >= k_ext or not np.all(np.isfinite(band_strain)):
            flags += ["ag:extensometer_invalid", "agt:extensometer_invalid"]
        else:
            strain_mid = 0.5 * (float(band_strain.min()) + float(band_strain.max()))
            halfwidth = 0.5 * (float(band_strain.max()) - float(band_strain.min())) * 100.0
            if math.isfinite(elastic.slope_n):
                agt = (strain_mid - elastic.strain_origin) * 100.0
                ag = agt - maximum.fm_n / elastic.slope_n * 100.0
    elongation = Elongation(
        agt_percent=agt,
        ag_percent=ag,
        ag_location_halfwidth_percent=halfwidth,
        band=band,
        strain_mid=strain_mid,
        flags=tuple(flags),
    )
    agt_percent = elongation.agt_percent
    ag_percent = elongation.ag_percent
    return elongation, agt_percent, ag_percent


@fr.atomic
def window_sensitivity(
    force_n: ForceArray,
    strain: StrainArray,
    s0_mm2: AreaMM2,
    k_ext: int,
    elastic: ElasticResult,
    maximum: MaximumForce,
    elongation: Elongation,
    params: AnalysisParameters | None = None,
) -> WindowSensitivity:
    """M15: E-window sensitivity with sigma_ref fixed at its nominal value.

    Ranges (min, max) of E, Rp0.2, Ag and Agt over the grid f_lo x f_hi, and
    the linearity diagnostic (E_b - E_a)/E_a for the windows 0.05-0.25 (a)
    and 0.25-0.50 (b). Describes sensitivity only; never selects a window.
    Flag ``window:insufficient_elastic_points`` if a grid or linearity window
    has too few points.
    """
    p = _params(params)
    s0 = _check_s0(s0_mm2)
    force, eps, _ = _channels(force_n, strain)
    valid = _strain_valid(force, eps, k_ext)
    reference = elastic.reference_n
    e_values, rp_values, ag_values, agt_values = [], [], [], []
    short = False
    for f_lo in p.window_grid_lo:
        for f_hi in p.window_grid_hi:
            ols, lo, _ = _window_ols(force, eps, valid, reference, f_lo, f_hi, p.min_elastic_points)
            if not math.isfinite(ols.slope):
                short = short or math.isfinite(reference)
                continue
            e_values.append(ols.slope / s0 / 1000.0)
            crossing, _, _ = _proof_crossing(
                force, eps, valid, ols.slope, ols.intercept, lo, p.offset_plastic_strain
            )
            rp_values.append(crossing / s0)
            if math.isfinite(elongation.strain_mid) and math.isfinite(maximum.fm_n):
                agt = (elongation.strain_mid + ols.intercept / ols.slope) * 100.0
                agt_values.append(agt)
                ag_values.append(agt - maximum.fm_n / ols.slope * 100.0)
    moduli = []
    for f_lo, f_hi in p.linearity_windows:
        ols, _, _ = _window_ols(force, eps, valid, reference, f_lo, f_hi, p.min_elastic_points)
        short = short or (math.isfinite(reference) and not math.isfinite(ols.slope))
        moduli.append(ols.slope)
    linearity = (moduli[1] - moduli[0]) / moduli[0] if len(moduli) == 2 else NAN
    sensitivity = WindowSensitivity(
        e_range_gpa=_value_range(e_values),
        rp02_range_mpa=_value_range(rp_values),
        ag_range_percent=_value_range(ag_values),
        agt_range_percent=_value_range(agt_values),
        linearity_rel_diff=float(linearity),
        flags=("window:insufficient_elastic_points",) if short else (),
    )
    return sensitivity


@fr.atomic
def assemble_curve_analysis(
    dropped_samples: dict,
    extensometer: ExtensometerValidity,
    bootstrap: Bootstrap,
    yielding: Yielding,
    elastic: ElasticResult,
    proof: ProofStrength,
    maximum: MaximumForce,
    elongation: Elongation,
    sensitivity: WindowSensitivity,
) -> CurveAnalysis:
    """Collect the step results into one :class:`CurveAnalysis`."""
    fit = elastic.fit
    flags = _unique(
        extensometer.flags
        + elastic.flags
        + yielding.flags
        + proof.flags
        + maximum.flags
        + elongation.flags
        + sensitivity.flags
    )
    analysis = CurveAnalysis(
        yield_type=yielding.yield_type,
        noise_mpa=bootstrap.noise_mpa,
        e_gpa=fit.e_gpa,
        e_intercept_mpa=fit.intercept_mpa,
        e_n=fit.n,
        e_r2=fit.r2,
        e_stderr_gpa=fit.stderr_gpa,
        e_lag1_rho=fit.lag1_rho,
        e_n_eff=fit.n_eff,
        e_window_mpa=(fit.lo_mpa, fit.hi_mpa),
        strain_origin=elastic.strain_origin,
        reh_mpa=yielding.reh_mpa,
        reh_sample_increment_mpa=yielding.reh_sample_increment_mpa,
        rel_mpa=yielding.rel_mpa,
        rel_incl_transient_mpa=yielding.rel_incl_transient_mpa,
        rp02_mpa=proof.rp02_mpa,
        fm_n=maximum.fm_n,
        rm_mpa=maximum.rm_mpa,
        agt_percent=elongation.agt_percent,
        ag_percent=elongation.ag_percent,
        ag_location_halfwidth_percent=elongation.ag_location_halfwidth_percent,
        k_ext=extensometer.k_ext,
        extensometer_max_valid_strain=extensometer.max_valid_strain,
        e_window_range_gpa=sensitivity.e_range_gpa,
        rp02_window_range_mpa=sensitivity.rp02_range_mpa,
        ag_window_range_percent=sensitivity.ag_range_percent,
        agt_window_range_percent=sensitivity.agt_range_percent,
        e_linearity_rel_diff=sensitivity.linearity_rel_diff,
        dropped_samples=dict(dropped_samples),
        flags=flags,
        e_reference_stress_mpa=elastic.reference_mpa,
        e_residual_noise_mpa=elastic.residual_noise_mpa,
        e_window_iterations=elastic.iterations,
        rel_plateau_level_mpa=yielding.plateau_level_mpa,
        rel_plateau_extent=yielding.plateau_extent,
        indices={
            "reh": yielding.i_p,
            "rel": yielding.i_rel,
            "work_hardening_start": yielding.i_h,
            "rp02": proof.index,
            "fm": maximum.i_m,
            "ag_band": elongation.band,
        },
    )
    return analysis


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def analyse_curve(
    force_n,
    strain,
    crosshead_mm,
    s0_mm2: float,
    l0_mm: float,
    *,
    a_percent: float | None = None,
    rate_switch_strain: float | None = None,
    params: AnalysisParameters | None = None,
) -> CurveAnalysis:
    """Analyse one curve (M1-M12, M15-M16) in the M5 bootstrap order.

    Arrays may contain NaN (per-channel handling, M1). ``a_percent`` (A from
    metadata) enables the M3 end check; ``rate_switch_strain`` (ratio) only
    sets ``near_rate_switch`` flags. This is a plain composition of the step
    nodes listed in the module docstring.
    """
    _check_s0(s0_mm2)
    dropped_samples = count_dropped_samples(force_n, strain, crosshead_mm)
    extensometer, k_ext, _ = extensometer_validity(
        strain, crosshead_mm, l0_mm, a_percent, rate_switch_strain, params
    )
    bootstrap, _ = bootstrap_fit(force_n, strain, s0_mm2, k_ext, params)
    yielding, _, _, _ = yield_analysis(force_n, strain, s0_mm2, k_ext, bootstrap, rate_switch_strain, params)
    elastic, _, _ = final_elastic_fit(force_n, strain, s0_mm2, k_ext, bootstrap, yielding, params)
    proof, _ = proof_strength(force_n, strain, s0_mm2, k_ext, elastic, yielding, rate_switch_strain, params)
    maximum, _, _ = maximum_force(force_n, s0_mm2, yielding)
    elongation, _, _ = elongation_at_maximum_force(force_n, strain, k_ext, bootstrap, elastic, maximum, params)
    sensitivity = window_sensitivity(force_n, strain, s0_mm2, k_ext, elastic, maximum, elongation, params)
    analysis = assemble_curve_analysis(
        dropped_samples, extensometer, bootstrap, yielding, elastic, proof, maximum, elongation, sensitivity
    )
    return analysis
