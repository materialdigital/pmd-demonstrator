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
