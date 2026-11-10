"""Figures for the tensile notebooks, in the style of the mesh demonstrator (private helpers).

The style follows ``archive/mesh-demonstrator/notebooks/04-stress_strain_analysis.ipynb``
(decision D12): large multi-panel figures, one shared legend on top in three
columns with per-specimen values in the labels, one colour per specimen
(tab10), one line style and marker per orientation, and a "data retrieved"
entry. Undefined values are shown as "–", and a specimen without a value is
named in the panel instead of being dropped.

Everything drawn comes from the analysis itself: the neutral curves and the
node outputs (or the :class:`~pmd_tensile.analysis.CurveAnalysis` that collects
them) and the result tables. Nothing is downloaded or smoothed here. Every
transformation (strain-origin shift, zoom limits, normalisation) is stated in
the figure. Lines that the method does not output directly (fit residuals,
the yield-drop threshold) are recomputed here from the node outputs with the
rules of method-spec v2.6, for display only.

- Style: :func:`specimen_colors`, :func:`orientation_styles`,
  :func:`format_value`, :func:`specimen_label`.
- Overview figures of a :class:`~pmd_tensile._orchestration.TensileAnalysis`
  (notebook 05): :func:`curves_overview_figure` (panels a-d) and
  :func:`results_overview_figure` (panels e-h).
- Step figures for the node tour (notebook 02), from the outputs of single
  nodes: :func:`yield_step_figure`, :func:`elastic_step_figure`,
  :func:`maximum_force_step_figure`, :func:`lab_comparison_step_figure`.

The module is private so that sim_atlas does not catalogue its functions as
nodes; notebooks reach them through :mod:`pmd_tensile.demo`.
"""

from __future__ import annotations

import dataclasses
import datetime
import math
import re
from collections.abc import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from pmd_tensile.contract import Curve, ReportedProperties, SpecimenRecord
from pmd_tensile.nodes.analysis.comparison import COMPARISON_RULES
from pmd_tensile.orchestration import SpecimenAnalysis
from pmd_tensile.parameters import (
    DEFAULT_PARAMETERS,
    AnalysisParameters,
    parameters_from_values,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

FIGURE_SIZE: tuple[float, float] = (12.0, 12.0)
"""Size of the 2x2 figures in inches, as in the mesh demonstrator."""

LEGEND_COLUMNS = 3
LEGEND_FONT_SIZE = 10.0
MISSING = "–"
"""How an undefined (NaN or missing) value is written in labels."""

ORIENTATION_ORDER: tuple[str, ...] = (
    "in rolling direction",
    "perpendicular to rolling direction",
    "diagonal to rolling direction",
)
"""Orientation labels of the S355 tests, in legend order; others follow sorted."""

_STYLE_CYCLE: tuple[tuple[object, str], ...] = (
    ("solid", "o"),
    ("dotted", "x"),
    ("dashed", "^"),
    ("dashdot", "s"),
    ((0, (5, 1, 1, 1, 1, 1)), "D"),
    ((0, (8, 3)), "v"),
)
"""(line style, marker) per orientation, in orientation order (mesh-04: solid/o, dotted/x, dashed/^)."""

PROPERTY_MARKERS: dict[str, str] = {"ReH": "D", "ReL": "s", "Rp0.2": "P", "Fm": "*", "extensometer end": "|"}
"""Markers of evaluation points on curves (different from the orientation markers)."""

_SHORT_ORIENTATION = {
    "in rolling direction": "in RD",
    "perpendicular to rolling direction": "⊥ RD",
    "diagonal to rolling direction": "diagonal",
}

_QUANTITY_LABELS = {
    "fm_n": "Fm",
    "rm_mpa": "Rm",
    "reh_mpa": "ReH",
    "rp02_mpa": "Rp0.2",
    "e_gpa": "E",
    "a_percent": "A",
    "z_percent": "Z",
}


def _natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(text))]


def _ordered_orientations(orientations: Iterable[str]) -> list[str]:
    present = list(dict.fromkeys(orientations))
    known = [o for o in ORIENTATION_ORDER if o in present]
    return known + sorted(o for o in present if o not in ORIENTATION_ORDER)


def orientation_styles(orientations: Iterable[str]) -> dict[str, tuple[object, str]]:
    """(line style, marker) per orientation label.

    The S355 orientations get the mesh-04 styles (in rolling direction: solid,
    o; perpendicular: dotted, x; diagonal: dashed, ^); other labels follow in
    sorted order.
    """
    ordered = _ordered_orientations(orientations)
    return {o: _STYLE_CYCLE[i % len(_STYLE_CYCLE)] for i, o in enumerate(ordered)}


def specimen_order(orientations: Mapping[str, str]) -> list[str]:
    """Specimen IDs grouped by orientation (:data:`ORIENTATION_ORDER`), naturally sorted within."""
    rank = {o: i for i, o in enumerate(_ordered_orientations(orientations.values()))}
    return sorted(orientations, key=lambda sid: (rank[orientations[sid]], _natural_key(sid)))


def specimen_colors(orientations: Mapping[str, str]) -> dict[str, str]:
    """One tab10 colour per specimen ID, given ``{specimen ID: orientation}``.

    Colours follow :func:`specimen_order`, so the same set of specimens gets
    the same colours in every figure and notebook (for the ten S355 tests: Zx1
    blue ... Zx4 red, Zy1 purple ... Zy4 grey, Zd2 olive, Zd3 cyan).
    """
    palette = plt.get_cmap("tab10").colors
    return {
        sid: _hex(palette[i % len(palette)]) for i, sid in enumerate(specimen_order(orientations))
    }


def _hex(rgb) -> str:
    return "#" + "".join(f"{round(255 * c):02x}" for c in rgb[:3])


def _is_value(value) -> bool:
    return value is not None and isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(value)


def format_value(value, digits: int = 1, unit: str = "") -> str:
    """``value`` with ``digits`` decimals and an optional unit; NaN or ``None`` gives "–"."""
    if not _is_value(value):
        return MISSING
    text = f"{float(value):.{digits}f}"
    return f"{text} {unit}" if unit else text


def specimen_label(specimen_id: str, orientation: str, values: Sequence[tuple[str, object, str, int]]) -> str:
    """Legend label: ``"<ID> (<orientation>)"`` and one line of ``name = value unit`` items.

    ``values`` holds ``(name, value, unit, digits)``; an undefined value is
    written "–" without its unit.
    """
    items = "; ".join(f"{name} = {format_value(value, digits, unit)}" for name, value, unit, digits in values)
    return f"{specimen_id} ({orientation})\n{items}"


def _retrieved_text(retrieved) -> str:
    if retrieved is None:
        return "not recorded"
    if isinstance(retrieved, (datetime.datetime, pd.Timestamp)):
        return pd.Timestamp(retrieved).isoformat(timespec="seconds")
    return str(retrieved)


def _blank() -> Line2D:
    return Line2D([], [], linestyle="none", marker="none", color="none")


# ---------------------------------------------------------------------------
# One specimen as the panels see it
# ---------------------------------------------------------------------------

NAN = float("nan")


@dataclasses.dataclass(frozen=True)
class _View:
    """What the panels need of one specimen, from a CurveAnalysis or from node outputs.

    Stresses in MPa, strains as ratios, E in GPa; NaN or -1 where unknown.
    """

    specimen_id: str
    orientation: str
    s0_mm2: float
    curve: Curve
    color: str
    linestyle: object
    marker: str
    k_ext: int
    noise_mpa: float = NAN
    e_gpa: float = NAN
    intercept_mpa: float = NAN
    strain_origin: float = NAN
    e_window_mpa: tuple[float, float] = (NAN, NAN)
    e_reference_mpa: float = NAN
    reh_mpa: float = NAN
    rel_mpa: float = NAN
    rp02_mpa: float = NAN
    fm_n: float = NAN
    rm_mpa: float = NAN
    i_reh: int = -1
    i_rel: int = -1
    i_fm: int = -1

    @property
    def stress_mpa(self) -> np.ndarray:
        return np.asarray(self.curve.force_n, dtype=float) / self.s0_mm2

    @property
    def strain(self) -> np.ndarray:
        return np.asarray(self.curve.strain, dtype=float)

    @property
    def crosshead_travel_mm(self) -> np.ndarray:
        crosshead = np.asarray(self.curve.crosshead_mm, dtype=float)
        finite = crosshead[np.isfinite(crosshead)]
        return crosshead - (finite[0] if len(finite) else 0.0)

    @property
    def valid(self) -> np.ndarray:
        """Samples the method evaluates with strain: finite force and strain, below k_ext (M1/M2)."""
        valid = np.isfinite(self.stress_mpa) & np.isfinite(self.strain)
        valid[max(int(self.k_ext), 0):] = False
        return valid

    @property
    def end_index(self) -> int:
        """Last strain-valid sample (the extensometer end as drawn), or -1."""
        indices = np.flatnonzero(self.valid)
        return int(indices[-1]) if len(indices) else -1

    @property
    def shifted_strain(self) -> np.ndarray:
        """Strain minus the fitted strain origin e0; unshifted if e0 is NaN."""
        return self.strain - (self.strain_origin if _is_value(self.strain_origin) else 0.0)


def _styles(orientations: Mapping[str, str], colors: Mapping[str, str] | None):
    colors = dict(colors) if colors is not None else specimen_colors(orientations)
    missing = sorted(set(orientations) - set(colors))
    if missing:
        raise ValueError(f"no colour given for specimens {missing}")
    return colors, orientation_styles(orientations.values())


def _views(
    orientations: Mapping[str, str],
    s0: Mapping[str, float],
    curves: Mapping[str, Curve],
    analyses: Mapping[str, SpecimenAnalysis],
    colors: Mapping[str, str] | None,
) -> list[_View]:
    """One view per specimen of ``orientations``, in :func:`specimen_order`."""
    colors, styles = _styles(orientations, colors)
    views = []
    for sid in specimen_order(orientations):
        a = analyses[sid]
        linestyle, marker = styles[orientations[sid]]
        views.append(
            _View(
                specimen_id=sid,
                orientation=orientations[sid],
                s0_mm2=float(s0[sid]),
                curve=curves[sid],
                color=colors[sid],
                linestyle=linestyle,
                marker=marker,
                k_ext=int(a.k_ext),
                noise_mpa=a.noise_mpa,
                e_gpa=a.e_gpa,
                intercept_mpa=-a.strain_origin * 1000.0 * a.e_gpa,
                strain_origin=a.strain_origin,
                e_window_mpa=a.elastic.window_mpa,
                e_reference_mpa=a.reference_force_n / float(s0[sid]),
                reh_mpa=a.reh_mpa,
                rel_mpa=a.rel_mpa,
                rp02_mpa=a.rp02_mpa,
                fm_n=a.fm_n,
                rm_mpa=a.rm_mpa,
                i_reh=a.upper_yield_index,
                i_rel=a.yielding.rel_index,
                i_fm=a.max_force_index,
            )
        )
    return views


def _record_views(records: Mapping[str, SpecimenRecord], curves, analyses, colors) -> list[_View]:
    orientations = {sid: record.orientation for sid, record in records.items()}
    s0 = {sid: float(record.s0_mm2) for sid, record in records.items()}
    return _views(orientations, s0, curves, analyses, colors)


def _result_views(result, colors: Mapping[str, str] | None) -> list[_View]:
    table = result.results.set_index("specimen_id")
    ids = list(result.specimen_ids)
    orientations = {sid: str(table.loc[sid, "orientation"]) for sid in ids}
    s0 = {sid: float(table.loc[sid, "s0_mm2"]) for sid in ids}
    return _views(orientations, s0, result.curves, result.specimen_analyses, colors)


def _result_parameters(result) -> AnalysisParameters:
    """The analysis parameters the batch run used: its ``parameters`` input (``None``: the defaults).

    No fallback: if the run data have no such input, the figure is not drawn,
    rather than drawn with parameters the run did not use.
    """
    return parameters_from_values(result.batch_run.result.input_ports["parameters"].value)


# ---------------------------------------------------------------------------
# Shared legend
# ---------------------------------------------------------------------------


def _shared_legend(fig, views: Sequence[_View], labels: Mapping[str, str], retrieved, *, lines: bool) -> None:
    """One legend above the panels: one column per orientation, then the retrieval time."""
    groups: dict[str, list] = {}
    for view in views:
        if lines:
            handle = Line2D([], [], color=view.color, linestyle=view.linestyle, linewidth=1.5)
        else:
            handle = Line2D([], [], color=view.color, marker=view.marker, linestyle="none", markersize=8)
        groups.setdefault(view.orientation, []).append((handle, labels[view.specimen_id]))
    stamp = (_blank(), r"$\bf{data\ retrieved:}$" + f"\n{_retrieved_text(retrieved)}")
    columns = list(groups.values())
    if len(columns) <= LEGEND_COLUMNS:
        columns += [[] for _ in range(LEGEND_COLUMNS - len(columns))]
        rows = max(len(column) for column in columns)
        if len(columns[-1]) >= rows:
            rows += 1
        columns[-1] = columns[-1] + [(_blank(), " ")] * (rows - 1 - len(columns[-1])) + [stamp]
        entries = [item for column in columns for item in column + [(_blank(), " ")] * (rows - len(column))]
    else:
        entries = [item for column in columns for item in column] + [stamp]
    handles, texts = zip(*entries)
    fig.legend(
        handles,
        texts,
        loc="outside upper center",
        ncol=LEGEND_COLUMNS,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.5,
    )


def _overview_label(view: _View) -> str:
    return specimen_label(
        view.specimen_id,
        view.orientation,
        [("E", view.e_gpa, "GPa", 1), ("ReH", view.reh_mpa, "MPa", 1), ("Rm", view.rm_mpa, "MPa", 1)],
    )


def _note(ax, text: str, loc: str | tuple[float, float, str, str] = "lower right") -> None:
    """A text box inside a panel (for transformations and unplotted specimens).

    ``loc`` is a named corner or ``(x, y, ha, va)`` in axes coordinates.
    """
    positions = {
        "lower right": (0.98, 0.02, "right", "bottom"),
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "center right": (0.98, 0.5, "right", "center"),
    }
    x, y, ha, va = positions[loc] if isinstance(loc, str) else loc
    ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )


def _names(views: Iterable[_View]) -> str:
    return ", ".join(view.specimen_id for view in views)


# ---------------------------------------------------------------------------
# Curve panels
# ---------------------------------------------------------------------------


def _plot_shifted_curves(ax, views: Sequence[_View]) -> None:
    for view in views:
        valid = view.valid
        ax.plot(
            100 * view.shifted_strain[valid], view.stress_mpa[valid],
            color=view.color, linestyle=view.linestyle, linewidth=1.0,
        )


def _shift_note(views: Sequence[_View]) -> str:
    origins = [view.strain_origin for view in views if _is_value(view.strain_origin)]
    text = "x: strain − fitted strain origin e₀ of each curve"
    if origins:
        text += f"\n(e₀ from {100 * min(origins):+.3f} % to {100 * max(origins):+.3f} %)"
    unshifted = [view for view in views if not _is_value(view.strain_origin)]
    if unshifted:
        text += f"\nnot shifted (e₀ {MISSING}): {_names(unshifted)}"
    return text + "\nshown up to the extensometer end; raw samples, no smoothing"


def _rp02_point(view: _View, offset: float) -> tuple[float, float]:
    """(shifted strain, stress) of the Rp0.2 crossing: e − e0 = Rp0.2/E + offset (M6, M10)."""
    if not (_is_value(view.rp02_mpa) and _is_value(view.e_gpa)):
        return NAN, NAN
    return view.rp02_mpa / (1000.0 * view.e_gpa) + offset, view.rp02_mpa


def _mark_yield_points(ax, views: Sequence[_View], offset: float, *, shifted: bool) -> None:
    strain_of = (lambda v: v.shifted_strain) if shifted else (lambda v: v.strain)
    for view in views:
        strain = strain_of(view)
        if view.i_reh >= 0 and _is_value(view.reh_mpa):
            ax.plot(100 * strain[view.i_reh], view.reh_mpa, PROPERTY_MARKERS["ReH"], color=view.color,
                    markeredgecolor="k", markersize=7, zorder=5)
        if view.i_rel >= 0 and _is_value(view.rel_mpa):
            ax.plot(100 * strain[view.i_rel], view.rel_mpa, PROPERTY_MARKERS["ReL"], color=view.color,
                    markeredgecolor="k", markersize=6, zorder=5)
        x, y = _rp02_point(view, offset)
        if shifted and _is_value(x):
            ax.plot(100 * x, y, PROPERTY_MARKERS["Rp0.2"], color=view.color, markeredgecolor="k",
                    markersize=8, zorder=5)


def _property_legend(ax, names: Sequence[str], loc: str = "lower right") -> None:
    handles = [
        Line2D([], [], marker=PROPERTY_MARKERS[name], color="0.6", markeredgecolor="k", linestyle="none",
               markersize=8 if name != "extensometer end" else 12)
        for name in names
    ]
    ax.legend(handles, names, loc=loc, fontsize=9, framealpha=0.9)


def _missing_note(views: Sequence[_View], names: Sequence[str]) -> str:
    lines = []
    for name in names:
        attribute = {"ReH": "reh_mpa", "ReL": "rel_mpa", "Rp0.2": "rp02_mpa", "Fm": "fm_n", "E": "e_gpa"}[name]
        absent = [view for view in views if not _is_value(getattr(view, attribute))]
        if absent:
            lines.append(f"{name} {MISSING} (no marker): {_names(absent)}")
    return "\n".join(lines)


def _yield_zoom_limits(views: Sequence[_View], offset: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Strain (%) and stress (MPa) limits around ReH, ReL and Rp0.2 of all specimens."""
    stresses = [v for view in views for v in (view.reh_mpa, view.rel_mpa, view.rp02_mpa) if _is_value(v)]
    if not stresses:
        return (0.0, 2.0), (0.0, 500.0)
    low = 5.0 * math.floor((min(stresses) - 15.0) / 5.0)
    high = 5.0 * math.ceil((max(stresses) + 15.0) / 5.0)
    return (0.0, 2.0), (low, high)


def _method_panel(ax, view: _View, params: AnalysisParameters, *, inset: bool = True) -> None:
    """One specimen: E window, elastic and offset lines, Rp0.2, ReH/ReL, extensometer end (raw strain)."""
    strain, stress, valid = view.strain, view.stress_mpa, view.valid
    ax.plot(100 * strain[valid], stress[valid], color=view.color, linewidth=1.2,
            label=f"{view.specimen_id}: measured (raw samples)")
    lo, hi = view.e_window_mpa
    offset = params.offset_plastic_strain
    top = np.nanmax(stress[valid]) + 30.0 if valid.any() else 500.0
    if _is_value(lo) and _is_value(hi):
        ax.axhspan(lo, hi, color=view.color, alpha=0.15,
                   label=f"E window {format_value(lo)}–{format_value(hi)} MPa "
                         f"({100 * params.f_lo:.0f}–{100 * params.f_hi:.0f} % of σ_ref = "
                         f"{format_value(view.e_reference_mpa)} MPa)")
    if _is_value(view.e_gpa) and _is_value(view.strain_origin):
        e_mpa, e0 = 1000.0 * view.e_gpa, view.strain_origin
        sigma = np.array([0.0, top])
        ax.plot(100 * (e0 + sigma / e_mpa), sigma, color="k", dashes=(6.0, 2.0),
                label=f"elastic line σ = E·(e − e₀), E = {format_value(view.e_gpa)} GPa, "
                      f"e₀ = {100 * e0:+.3f} %")
        ax.plot(100 * (e0 + offset + sigma / e_mpa), sigma, color="g", dashes=(6.0, 2.0),
                label=f"offset line, shifted by {100 * offset:g} % plastic strain")
        x, y = _rp02_point(view, offset)
        if _is_value(x):
            ax.plot(100 * (x + e0), y, "o", color="r", zorder=6, label=f"Rp0.2 = {format_value(y)} MPa")
    if view.i_reh >= 0 and _is_value(view.reh_mpa):
        ax.plot(100 * strain[view.i_reh], view.reh_mpa, PROPERTY_MARKERS["ReH"], color=view.color,
                markeredgecolor="k", zorder=6, label=f"ReH = {format_value(view.reh_mpa)} MPa")
    if view.i_rel >= 0 and _is_value(view.rel_mpa):
        ax.plot(100 * strain[view.i_rel], view.rel_mpa, PROPERTY_MARKERS["ReL"], color=view.color,
                markeredgecolor="k", zorder=6, label=f"ReL = {format_value(view.rel_mpa)} MPa")
    end = view.end_index
    if end >= 0:
        ax.axvline(100 * strain[end], color="0.4", linestyle=":",
                   label=f"extensometer end (e = {100 * strain[end]:.2f} %)")
    ax.set_ylim(0.0, top)
    if end >= 0:
        ax.set_xlim(-0.1, 100 * strain[end] + 0.4)
    ax.set_xlabel("strain (extensometer, not shifted) / %")
    ax.set_ylabel("engineering stress / MPa")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    top_value = view.reh_mpa if _is_value(view.reh_mpa) else view.rp02_mpa
    if inset and _is_value(view.e_gpa) and _is_value(view.strain_origin) and _is_value(top_value):
        e_mpa, e0 = 1000.0 * view.e_gpa, view.strain_origin
        x_lo, x_hi = 100 * e0 - 0.02, 100 * (e0 + 2.0 * offset + top_value / e_mpa)
        y_hi = top_value + 15.0
        axins = ax.inset_axes([0.40, 0.46, 0.57, 0.28])  # above the legend's top edge, tick labels included
        axins.plot(100 * strain[valid], stress[valid], color=view.color, linewidth=1.2)
        if _is_value(lo) and _is_value(hi):
            axins.axhspan(lo, hi, color=view.color, alpha=0.15)
        sigma = np.array([0.0, y_hi])
        axins.plot(100 * (e0 + sigma / e_mpa), sigma, color="k", dashes=(6.0, 2.0))
        axins.plot(100 * (e0 + offset + sigma / e_mpa), sigma, color="g", dashes=(6.0, 2.0))
        x, y = _rp02_point(view, offset)
        if _is_value(x):
            axins.plot(100 * (x + e0), y, "o", color="r", zorder=6)
        for index, value, name in ((view.i_reh, view.reh_mpa, "ReH"), (view.i_rel, view.rel_mpa, "ReL")):
            if index >= 0 and _is_value(value):
                axins.plot(100 * strain[index], value, PROPERTY_MARKERS[name], color=view.color,
                           markeredgecolor="k", zorder=6)
        axins.set_xlim(x_lo, x_hi)
        axins.set_ylim(0.0, y_hi)
        axins.tick_params(labelsize=8)
        axins.set_title(f"inset: strain {x_lo:.2f} to {x_hi:.2f} %, stress 0–{y_hi:.0f} MPa", fontsize=8)
        ax.indicate_inset_zoom(axins, edgecolor="black")


def _crosshead_panel(ax, views: Sequence[_View]) -> None:
    """Stress against crosshead travel over the whole record, Fm and the extensometer end marked."""
    for view in views:
        travel, stress = view.crosshead_travel_mm, view.stress_mpa
        finite = np.isfinite(travel) & np.isfinite(stress)
        ax.plot(travel[finite], stress[finite], color=view.color, linestyle=view.linestyle, linewidth=1.0)
        if view.i_fm >= 0:
            ax.plot(travel[view.i_fm], stress[view.i_fm], PROPERTY_MARKERS["Fm"], color=view.color,
                    markeredgecolor="k", markersize=12, zorder=5)
        end = view.end_index
        if end >= 0:
            ax.plot(travel[end], stress[end], PROPERTY_MARKERS["extensometer end"], color="k", markersize=14,
                    markeredgewidth=2, zorder=5)
    ax.set_xlabel("crosshead travel from the first sample / mm")
    ax.set_ylabel("engineering stress (force / S0) / MPa")
    handles = [
        Line2D([], [], marker=PROPERTY_MARKERS["Fm"], color="0.6", markeredgecolor="k", linestyle="none",
               markersize=12),
        Line2D([], [], marker=PROPERTY_MARKERS["extensometer end"], color="k", linestyle="none", markersize=14,
               markeredgewidth=2),
    ]
    ax.legend(handles, ["Fm (Rm = Fm / S0, in the labels)", "extensometer end"], loc="center",
              fontsize=9, framealpha=0.9)
    missing = _missing_note(views, ["Fm"])
    _note(ax, "force channel only,\nall finite samples to the end of the record"
          + (f"\n{missing}" if missing else ""), loc="lower right")


# ---------------------------------------------------------------------------
# Overview figure of the curves (notebook 05, panels a-d)
# ---------------------------------------------------------------------------


def curves_overview_figure(
    result,
    *,
    retrieved=None,
    method_specimen: str = "Zx3",
    yield_zoom: tuple[tuple[float, float], tuple[float, float]] | None = None,
    colors: Mapping[str, str] | None = None,
):
    """Stress-strain overview of a :class:`~pmd_tensile._orchestration.TensileAnalysis` (D12, a-d).

    - (a) every curve up to its extensometer end, shifted by its fitted strain
      origin e0 (the range of e0 is stated in the panel);
    - (b) the yield region of (a) with ReH, ReL and Rp0.2 markers (zoom limits
      in the title; ``yield_zoom`` = ((strain lo, hi) in %, (stress lo, hi) in
      MPa) overrides the automatic choice);
    - (c) the method on ``method_specimen``: E window, elastic and 0.2 %
      offset lines, Rp0.2, ReH, ReL, extensometer end, with an inset;
    - (d) stress against crosshead travel to fracture with Fm marked.

    The curves and analyses are ``result.curves`` and
    ``result.specimen_analyses`` (the arrays the run used), S0 and the
    orientation come from ``result.results``. ``retrieved`` (a timestamp or
    text) is shown as the "data retrieved" legend entry. ``colors`` maps
    specimen IDs to colours (default: :func:`specimen_colors`). Returns the
    matplotlib Figure.
    """
    views = _result_views(result, colors)
    params = _result_parameters(result)
    offset = params.offset_plastic_strain
    by_id = {view.specimen_id: view for view in views}
    if method_specimen not in by_id:
        raise KeyError(f"method_specimen {method_specimen!r} is not in the result: {sorted(by_id)}")

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, layout="constrained")
    ax_a, ax_b, ax_c, ax_d = axes.flat

    _plot_shifted_curves(ax_a, views)
    ax_a.set_title("(a) all curves, shifted by their strain origin")
    ax_a.set_xlabel("strain − e₀ / %")
    ax_a.set_ylabel("engineering stress / MPa")
    _note(ax_a, _shift_note(views))

    _plot_shifted_curves(ax_b, views)
    _mark_yield_points(ax_b, views, offset, shifted=True)
    (x_lo, x_hi), (y_lo, y_hi) = yield_zoom if yield_zoom is not None else _yield_zoom_limits(views, offset)
    ax_b.set_xlim(x_lo, x_hi)
    ax_b.set_ylim(y_lo, y_hi)
    ax_b.set_title(f"(b) yield region: zoom of (a) to {x_lo:g}–{x_hi:g} %, {y_lo:g}–{y_hi:g} MPa")
    ax_b.set_xlabel("strain − e₀ / %")
    ax_b.set_ylabel("engineering stress / MPa")
    _property_legend(ax_b, ["ReH", "ReL", "Rp0.2"])
    missing = _missing_note(views, ["ReH", "ReL", "Rp0.2"])
    if missing:
        _note(ax_b, missing, loc="upper right")

    _method_panel(ax_c, by_id[method_specimen], params)
    ax_c.set_title(f"(c) the method on {method_specimen}")

    _crosshead_panel(ax_d, views)
    ax_d.set_title("(d) to fracture: stress against crosshead travel")

    _shared_legend(fig, views, {view.specimen_id: _overview_label(view) for view in views}, retrieved, lines=True)
    return fig


# ---------------------------------------------------------------------------
# Lab comparison panel (shared by notebook 02 and 05)
# ---------------------------------------------------------------------------


def _tolerance_text(quantity: str) -> str:
    for key, _, _, kind, _, _, text in COMPARISON_RULES:
        if key == quantity:
            return text if kind == "relative" else "±0.5 pp, rounding"
    raise KeyError(quantity)


def _lab_deviation_panel(ax, comparison: pd.DataFrame, views: Sequence[_View]) -> None:
    """(computed - lab) / tolerance per specimen and quantity; the band [-1, 1] is 'within'."""
    quantities = [key for key, *_ in COMPARISON_RULES if key in set(comparison["quantity"])]
    order = [view.specimen_id for view in views if view.specimen_id in set(comparison["specimen_id"])]
    style = {view.specimen_id: view for view in views}
    spread = np.linspace(-0.3, 0.3, len(order)) if len(order) > 1 else np.zeros(1)
    not_comparable: dict[str, list[str]] = {}
    largest = 1.0
    for row, quantity in enumerate(quantities):
        ax.axhspan(row - 0.42, row + 0.42, xmin=0, xmax=1, color="0.96", zorder=0)
        for k, sid in enumerate(order):
            match = comparison[(comparison["specimen_id"] == sid) & (comparison["quantity"] == quantity)]
            if match.empty:
                continue
            entry = match.iloc[0]
            x = entry["deviation_in_tolerance"]
            if not _is_value(x):
                not_comparable.setdefault(_QUANTITY_LABELS[quantity], []).append(sid)
                continue
            largest = max(largest, abs(x))
            view = style[sid]
            ax.plot(x, row + spread[k], view.marker, color=view.color, markersize=8,
                    markeredgecolor=view.color if view.marker == "x" else "k", markeredgewidth=1.2)
    ax.axvspan(-1.0, 1.0, color="tab:green", alpha=0.12, zorder=0)
    ax.axvline(0.0, color="0.3", linewidth=0.8)
    limit = 1.15 * largest
    ax.set_xlim(-limit, limit)
    ax.set_ylim(len(quantities) + 0.7, -0.5)  # the free row at the bottom holds the note
    ax.set_yticks(range(len(quantities)))
    ax.set_yticklabels([f"{_QUANTITY_LABELS[q]}\n({_tolerance_text(q)})" for q in quantities], fontsize=9)
    ax.set_xlabel("(computed − lab) / tolerance   (green band: within tolerance)")
    note = "consistency check with the lab's software, not a validation"
    if not_comparable:
        note += "\nnot comparable (a value missing): " + "; ".join(
            f"{name} {', '.join(ids)}" for name, ids in not_comparable.items()
        )
    _note(ax, note, loc="lower left")


# ---------------------------------------------------------------------------
# Overview figure of the results (notebook 05, panels e-h)
# ---------------------------------------------------------------------------


def _scatter_panel(ax, views, table, x_column, y_column, x_label, y_label, x_range=None, y_range=None):
    absent = []
    for view in views:
        row = table.loc[view.specimen_id]
        x, y = row[x_column], row[y_column]
        if not (_is_value(x) and _is_value(y)):
            absent.append(view.specimen_id)
            continue
        xerr = yerr = None
        if x_range is not None:
            lo, hi = row[x_range[0]], row[x_range[1]]
            if _is_value(lo) and _is_value(hi):
                xerr = [[max(x - lo, 0.0)], [max(hi - x, 0.0)]]
        if y_range is not None:
            lo, hi = row[y_range[0]], row[y_range[1]]
            if _is_value(lo) and _is_value(hi):
                yerr = [[max(y - lo, 0.0)], [max(hi - y, 0.0)]]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=view.marker, color=view.color, markersize=10,
                    elinewidth=1.0, capsize=3, markeredgecolor=view.color if view.marker == "x" else "k")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return absent


def _group_panels(subfig, views, table, summary, group_column: str) -> None:
    """(h) mean per orientation (line) with sd (band) and the specimens with their S0-scatter bars."""
    quantities = (("e_gpa", "E / GPa"), ("reh_mpa", "ReH / MPa"), ("rp02_mpa", "Rp0.2 / MPa"),
                  ("rm_mpa", "Rm / MPa"))
    orientations = _ordered_orientations(view.orientation for view in views)
    axes = subfig.subplots(2, 2)
    largest_u = 0.0
    for ax, (quantity, label) in zip(axes.flat, quantities):
        for position, orientation in enumerate(orientations):
            members = [view for view in views if view.orientation == orientation]
            stats = summary[(summary[group_column] == orientation) & (summary["quantity"] == quantity)]
            if not stats.empty:
                mean, sd, n = stats.iloc[0]["mean"], stats.iloc[0]["sd"], int(stats.iloc[0]["n"])
                if _is_value(mean):
                    ax.hlines(mean, position - 0.32, position + 0.32, color="k", linewidth=2)
                    if _is_value(sd):
                        ax.add_patch(plt.Rectangle((position - 0.32, mean - sd), 0.64, 2 * sd, color="0.6",
                                                   alpha=0.35, linewidth=0))
                ax.annotate(f"n={n}", (position, 0.0), xycoords=("data", "axes fraction"), ha="center",
                            va="bottom", fontsize=7.5, color="0.3")
            offsets = np.linspace(-0.2, 0.2, len(members)) if len(members) > 1 else np.zeros(1)
            for offset, view in zip(offsets, members):
                row = table.loc[view.specimen_id]
                value, u_rel = row[quantity], row["s0_u_rel"]
                if not _is_value(value):
                    continue
                u = abs(value * u_rel) if _is_value(u_rel) else None
                if u is not None and quantity != "e_gpa":
                    largest_u = max(largest_u, u)
                ax.errorbar(position + offset, value, yerr=u, fmt=view.marker, color=view.color, markersize=6,
                            elinewidth=1.5, capsize=2, markeredgecolor=view.color if view.marker == "x" else "k")
        low, high = ax.get_ylim()
        ax.set_ylim(low - 0.12 * (high - low), high)  # room for the n labels
        ax.set_xticks(range(len(orientations)))
        ax.set_xticklabels([_SHORT_ORIENTATION.get(o, o[:10]) for o in orientations], fontsize=8.5)
        ax.set_xlim(-0.6, len(orientations) - 0.4)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8.5)
        absent = [view.specimen_id for view in views if not _is_value(table.loc[view.specimen_id, quantity])]
        if absent:
            ax.text(0.02, 0.98, f"{MISSING}: {', '.join(absent)}", transform=ax.transAxes, ha="left", va="top",
                    fontsize=7.5, color="0.3")
    subfig.suptitle(
        "(h) mean per orientation (black line) ± sd (grey band);\n"
        f"points: specimens with ± u from S0 scatter (k = 1, ≤ {largest_u:.2f} MPa for the stresses)",
        fontsize=10,
    )
    subfig.supxlabel("RD: rolling direction", fontsize=8.5)


def results_overview_figure(result, *, retrieved=None, colors: Mapping[str, str] | None = None):
    """Results overview of a :class:`~pmd_tensile._orchestration.TensileAnalysis` (D12, e-h).

    - (e) computed minus lab value per specimen and quantity, divided by the
      tolerance of method-spec §13, so the band from -1 to 1 is "within";
      quantities without a lab or computed value are listed in the panel;
    - (f) ReH against Rm, marker per orientation;
    - (g) E against Rp0.2, with the E-window sensitivity ranges (M15) as bars;
    - (h) per orientation the mean and sd of E, ReH, Rp0.2 and Rm from
      ``result.orientation_summary``, with each specimen's value and its
      S0-scatter uncertainty (M14), never combined.

    Values come from ``result.results``, ``result.comparison`` and
    ``result.orientation_summary``; nothing is recomputed. ``retrieved`` and
    ``colors`` as for :func:`curves_overview_figure`. Returns the Figure.
    """
    from pmd_tensile.nodes.workflows import GROUP_COLUMN

    views = _result_views(result, colors)
    table = result.results.set_index("specimen_id")
    fig = plt.figure(figsize=FIGURE_SIZE, layout="constrained")
    top, bottom = fig.subfigures(2, 1)
    ax_e, ax_f = top.subplots(1, 2)
    left, right = bottom.subfigures(1, 2)
    ax_g = left.subplots()

    _lab_deviation_panel(ax_e, result.comparison, views)
    ax_e.set_title("(e) deviation from the lab value, in units of the tolerance")

    absent = _scatter_panel(ax_f, views, table, "rm_mpa", "reh_mpa", "Rm / MPa", "ReH / MPa")
    ax_f.set_title("(f) ReH against Rm")
    if absent:
        _note(ax_f, f"not shown (ReH or Rm {MISSING}): {', '.join(absent)}", loc="lower right")

    absent = _scatter_panel(
        ax_g, views, table, "e_gpa", "rp02_mpa", "E (regression slope, mE-type) / GPa", "Rp0.2 / MPa",
        x_range=("e_window_min_gpa", "e_window_max_gpa"), y_range=("rp02_window_min_mpa", "rp02_window_max_mpa"),
    )
    ax_g.set_title("(g) E against Rp0.2")
    note = "bars: range over the E-window grid (M15),\na sensitivity, not an uncertainty"
    if absent:
        note += f"\nnot shown (E or Rp0.2 {MISSING}): {', '.join(absent)}"
    _note(ax_g, note, loc="lower left")

    _group_panels(right, views, table, result.orientation_summary, GROUP_COLUMN)

    _shared_legend(fig, views, {view.specimen_id: _overview_label(view) for view in views}, retrieved, lines=False)
    return fig


# ---------------------------------------------------------------------------
# Step figures for the node tour (notebook 02)
# ---------------------------------------------------------------------------


def _first_branch(stress: np.ndarray, reference: float) -> np.ndarray:
    """Samples before the stress first reaches ``reference`` (the first loading branch)."""
    branch = np.zeros(len(stress), dtype=bool)
    if not _is_value(reference):
        return branch
    reached = np.flatnonzero(np.isfinite(stress) & (stress >= reference))
    branch[: reached[0] if len(reached) else len(stress)] = True
    return branch


def _provisional_window(view: _View, analysis: SpecimenAnalysis, params: AnalysisParameters):
    """Mask, line and window of the provisional fit (M5 step 2), in stress units."""
    reference = params.provisional_fraction_of_rm * analysis.highest_force_n / view.s0_mm2
    lo, hi = params.f_lo * reference, params.f_hi * reference
    stress = view.stress_mpa
    with np.errstate(invalid="ignore"):
        mask = view.valid & _first_branch(stress, reference) & (stress >= lo) & (stress <= hi)
    slope = 1000.0 * analysis.e0_gpa
    intercept = analysis.intercept0_mpa
    return mask, slope, intercept, (lo, hi)


def _drawdown(view: _View, analysis: SpecimenAnalysis, params: AnalysisParameters):
    """M7 on the finite-force samples: running maximum M and threshold M - d, in MPa.

    Also the largest drop below M before the extensometer end where the search
    applies (sigma > 0.5*Rm, Rm from force alone), and the required d there.
    """
    stress = view.stress_mpa
    finite = np.flatnonzero(np.isfinite(stress))
    running = np.full(len(stress), NAN)
    running[finite] = np.maximum.accumulate(stress[finite])
    noise = analysis.noise_mpa if _is_value(analysis.noise_mpa) else 0.0
    required = np.maximum(params.drop_rel * running, params.drop_noise_factor * noise)
    rm = analysis.highest_force_n / view.s0_mm2
    searched = np.zeros(len(stress), dtype=bool)
    searched[finite] = stress[finite] > 0.5 * rm
    before_end = np.zeros(len(stress), dtype=bool)
    before_end[: max(view.end_index + 1, 0)] = True
    candidates = np.flatnonzero(searched & before_end)
    if len(candidates):
        drops = running[candidates] - stress[candidates]
        k = int(np.argmax(drops))
        largest, needed = float(drops[k]), float(required[candidates[k]])
    else:
        largest, needed = NAN, NAN
    return running, running - required, largest, needed


def yield_step_figure(
    records: Mapping[str, SpecimenRecord],
    curves: Mapping[str, Curve],
    analyses: Mapping[str, SpecimenAnalysis],
    *,
    params: AnalysisParameters | None = None,
    colors: Mapping[str, str] | None = None,
    retrieved=None,
):
    """Node tour, extensometer validity and yield (M1-M3, M5 steps 1-3, M7-M9), 2x2.

    - (a) strain against crosshead travel with the extensometer end
      (``extensometer_validity``);
    - (b) the low-stress part with the provisional window and line
      (``bootstrap_fit``);
    - (c) residuals of the provisional fit with the noise level ± s_n;
    - (d) the yield search: stress, running maximum M and the threshold
      M − d with d = max(0.005·M, 5·s_n), up to the extensometer end, with
      ReH and ReL (``yield_analysis``).

    All inputs map specimen IDs to the node outputs of that specimen.
    ``params`` are the parameters the nodes ran with (default: the method
    defaults). Returns the Figure.
    """
    p = DEFAULT_PARAMETERS if params is None else params
    views = _record_views(records, curves, analyses, colors)
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, layout="constrained")
    ax_a, ax_b, ax_c, ax_d = axes.flat
    labels = {}
    for view in views:
        analysis = analyses[view.specimen_id]
        travel, strain = view.crosshead_travel_mm, view.strain
        finite = np.isfinite(travel) & np.isfinite(strain)
        ax_a.plot(travel[finite], 100 * strain[finite], color=view.color, linestyle=view.linestyle)
        end = view.end_index
        if end >= 0:
            ax_a.plot(travel[end], 100 * strain[end], PROPERTY_MARKERS["extensometer end"], color="k",
                      markersize=16, markeredgewidth=2)

        mask, slope, intercept, (lo, hi) = _provisional_window(view, analysis, p)
        stress = view.stress_mpa
        low = view.valid & _first_branch(stress, 1.2 * hi)
        ax_b.plot(100 * strain[low], stress[low], color=view.color, linestyle=view.linestyle, linewidth=1.0)
        if _is_value(slope):
            sigma = np.array([0.0, 1.2 * hi])
            ax_b.plot(100 * (sigma - intercept) / slope, sigma, color=view.color, linewidth=0.8, dashes=(4, 2))
        ax_b.axhspan(lo, hi, color=view.color, alpha=0.08)

        residual = stress[mask] - (slope * strain[mask] + intercept)
        ax_c.plot(stress[mask], residual, view.marker, color=view.color, markersize=3, alpha=0.6,
                  linestyle="none")
        if _is_value(analysis.noise_mpa):
            for sign in (-1, 1):
                ax_c.axhline(sign * analysis.noise_mpa, color=view.color, linestyle=view.linestyle, linewidth=1.2)

        running, threshold, largest, needed = _drawdown(view, analysis, p)
        valid = view.valid
        ax_d.plot(100 * strain[valid], stress[valid], color=view.color, linestyle=view.linestyle, linewidth=1.0)
        ax_d.plot(100 * strain[valid], threshold[valid], color=view.color, linewidth=0.9, dashes=(2, 2),
                  alpha=0.9)
        labels[view.specimen_id] = specimen_label(
            view.specimen_id,
            view.orientation,
            [("ext. end at e", 100 * strain[end] if end >= 0 else NAN, "%", 2),
             ("s_n", analysis.noise_mpa, "MPa", 2), ("ReH", view.reh_mpa, "MPa", 1),
             ("ReL", view.rel_mpa, "MPa", 1)],
        ) + f"\nlargest drop before ext. end {format_value(largest, 1, 'MPa')}, required d {format_value(needed, 1, 'MPa')}"
    _mark_yield_points(ax_d, views, p.offset_plastic_strain, shifted=False)

    ax_a.set_title("(a) extensometer validity: strain against crosshead travel")
    ax_a.set_xlabel("crosshead travel from the first sample / mm")
    ax_a.set_ylabel("strain (extensometer) / %")
    ax_a.legend([Line2D([], [], marker="|", color="k", linestyle="none", markersize=16, markeredgewidth=2)],
                ["extensometer end found by the method (k_ext)"], loc="lower right", fontsize=9)
    _note(ax_a, "strain is evaluated only before the end found by the method;\n"
          "the recorded strain may continue a little beyond it", loc="upper left")

    ax_b.set_title("(b) provisional fit (bootstrap): window and line")
    ax_b.set_xlabel("strain (extensometer, not shifted) / %")
    ax_b.set_ylabel("engineering stress / MPa")
    _note(ax_b, f"shaded: provisional window {100 * p.f_lo:.0f}–{100 * p.f_hi:.0f} % of "
                f"{p.provisional_fraction_of_rm:g}·Rm (Rm from force alone)\ndashed: provisional line E₀, b₀",
          loc="lower right")

    ax_c.set_title("(c) residuals of the provisional fit and noise level s_n")
    ax_c.set_xlabel("engineering stress / MPa")
    ax_c.set_ylabel("stress − provisional line / MPa")
    _note(ax_c, "lines: ± s_n = 1.4826 · MAD of the residuals\n(fixed here; used by the yield search)",
          loc="lower right")

    ax_d.set_title("(d) yield search: drop below the running maximum")
    ax_d.set_xlabel("strain (extensometer, not shifted) / %")
    ax_d.set_ylabel("engineering stress / MPa")
    (_, _), (y_lo, y_hi) = _yield_zoom_limits(views, p.offset_plastic_strain)
    ends = [100 * view.strain[view.end_index] for view in views if view.end_index >= 0]
    ax_d.set_ylim(y_lo, y_hi)
    ax_d.set_xlim(0.0, max(ends) + 0.2 if ends else 4.5)
    handles = [
        Line2D([], [], color="0.4", dashes=(2, 2)),
        Line2D([], [], marker=PROPERTY_MARKERS["ReH"], color="0.6", markeredgecolor="k", linestyle="none"),
        Line2D([], [], marker=PROPERTY_MARKERS["ReL"], color="0.6", markeredgecolor="k", linestyle="none"),
    ]
    ax_d.legend(handles, ["threshold M − d (a drop below it marks yielding)", "ReH", "ReL"],
                loc="lower right", fontsize=9)
    _note(ax_d, f"zoom: stress {y_lo:g}–{y_hi:g} MPa, up to the extensometer end"
          + (f"\n{_missing_note(views, ['ReH', 'ReL'])}" if _missing_note(views, ["ReH", "ReL"]) else ""),
          loc=(0.98, 0.2, "right", "bottom"))

    _shared_legend(fig, views, labels, retrieved, lines=True)
    return fig


def elastic_step_figure(
    records: Mapping[str, SpecimenRecord],
    curves: Mapping[str, Curve],
    analyses: Mapping[str, SpecimenAnalysis],
    reported: Mapping[str, ReportedProperties],
    *,
    method_specimen: str = "Zx3",
    params: AnalysisParameters | None = None,
    colors: Mapping[str, str] | None = None,
    retrieved=None,
):
    """Node tour, elastic modulus and Rp0.2 (M4-M6, M10, M15), 2x2.

    - (a) the method on ``method_specimen`` (``final_elastic_fit``,
      ``proof_strength``), with an inset;
    - (b) residuals of the final E fit in its window (is the segment straight?);
    - (c) E for each window of the sensitivity grid, with the lab E;
    - (d) the 0.2 % offset construction of all specimens: strain − e0 with the
      offset lines, Rp0.2 and ReH.

    ``reported`` maps specimen IDs to the lab values (for the lab E only).
    Returns the Figure.
    """
    p = DEFAULT_PARAMETERS if params is None else params
    views = _record_views(records, curves, analyses, colors)
    by_id = {view.specimen_id: view for view in views}
    if method_specimen not in by_id:
        raise KeyError(f"method_specimen {method_specimen!r} is not among {sorted(by_id)}")
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, layout="constrained")
    ax_a, ax_b, ax_c, ax_d = axes.flat

    _method_panel(ax_a, by_id[method_specimen], p)
    ax_a.set_title(f"(a) E window, elastic and offset lines on {method_specimen}")

    labels = {}
    grid_labels = []
    for view in views:
        analysis = analyses[view.specimen_id]
        lo, hi = view.e_window_mpa
        stress, strain = view.stress_mpa, view.strain
        with np.errstate(invalid="ignore"):
            mask = (view.valid & _first_branch(stress, view.e_reference_mpa)
                    & (stress >= lo) & (stress <= hi))
        if _is_value(view.e_gpa):
            residual = stress[mask] - (1000.0 * view.e_gpa * strain[mask] + view.intercept_mpa)
            ax_b.plot(stress[mask], residual, view.marker, color=view.color, markersize=3.5, linestyle="none",
                      alpha=0.8)

        windows = [(f_lo, f_hi) for f_lo in p.window_grid_lo for f_hi in p.window_grid_hi]
        moduli = [e for row in analysis.sensitivity.grid_e_gpa for e in row]
        for k, ((f_lo, f_hi), e_gpa) in enumerate(zip(windows, moduli, strict=True)):
            ax_c.plot(k, e_gpa, view.marker, color=view.color, markersize=8,
                      markeredgecolor=view.color if view.marker == "x" else "k")
            if view is views[0]:
                grid_labels.append(f"{100 * f_lo:.0f}–{100 * f_hi:.0f}")
        lab = reported[view.specimen_id].e_gpa
        if _is_value(lab):
            ax_c.axhline(lab, color=view.color, linestyle=view.linestyle, linewidth=1.2)

        valid = view.valid
        x = 100 * view.shifted_strain[valid]
        ax_d.plot(x, stress[valid], color=view.color, linestyle=view.linestyle, linewidth=1.0)
        if _is_value(view.e_gpa):
            sigma = np.array([0.0, 600.0])
            ax_d.plot(100 * (p.offset_plastic_strain + sigma / (1000.0 * view.e_gpa)), sigma, color=view.color,
                      linewidth=0.8, dashes=(5, 2))
        labels[view.specimen_id] = specimen_label(
            view.specimen_id,
            view.orientation,
            [("E", view.e_gpa, "GPa", 1), ("lab", lab, "GPa", 0), ("R²", analysis.elastic.r2, "", 4),
             ("Rp0.2", view.rp02_mpa, "MPa", 1)],
        )
    _mark_yield_points(ax_d, views, p.offset_plastic_strain, shifted=True)

    ax_b.axhline(0.0, color="0.3", linewidth=0.8)
    ax_b.set_title("(b) residuals of the final E fit in its window")
    ax_b.set_xlabel("engineering stress / MPa")
    ax_b.set_ylabel("stress − fitted line / MPa")
    _note(ax_b, "a straight elastic segment scatters around 0;\na trend means the segment is curved",
          loc="lower right")

    ax_c.set_xticks(range(len(grid_labels)))
    ax_c.set_xticklabels(grid_labels, rotation=45, fontsize=8.5)
    ax_c.set_title("(c) E in each window of the sensitivity grid")
    ax_c.set_xlabel("window / % of the reference stress (nominal: "
                    f"{100 * p.f_lo:.0f}–{100 * p.f_hi:.0f})")
    ax_c.set_ylabel("E / GPa (horizontal lines: lab E)")  # a note box inside the panel could hide a point

    reference = [v for view in views for v in (view.reh_mpa, view.rp02_mpa) if _is_value(v)]
    y_hi = 5.0 * math.ceil((max(reference) + 20.0) / 5.0) if reference else 500.0
    ax_d.set_xlim(0.0, 1.0)
    ax_d.set_ylim(250.0, y_hi)
    ax_d.set_title("(d) 0.2 % offset construction")
    ax_d.set_xlabel("strain − e₀ / %")
    ax_d.set_ylabel("engineering stress / MPa")
    handles = [
        Line2D([], [], color="0.4", dashes=(5, 2)),
        Line2D([], [], marker=PROPERTY_MARKERS["Rp0.2"], color="0.6", markeredgecolor="k", linestyle="none"),
        Line2D([], [], marker=PROPERTY_MARKERS["ReH"], color="0.6", markeredgecolor="k", linestyle="none"),
        Line2D([], [], marker=PROPERTY_MARKERS["ReL"], color="0.6", markeredgecolor="k", linestyle="none"),
    ]
    ax_d.legend(handles, [f"offset line σ = E·(e − e₀ − {p.offset_plastic_strain:g})", "Rp0.2", "ReH", "ReL"],
                loc="lower right", fontsize=9)
    missing = _missing_note(views, ["ReH", "ReL", "Rp0.2"])
    _note(ax_d, f"zoom: strain − e₀ 0–1 %, stress 250–{y_hi:g} MPa" + (f"\n{missing}" if missing else ""),
          loc="center right")

    _shared_legend(fig, views, labels, retrieved, lines=True)
    return fig


def maximum_force_step_figure(
    records: Mapping[str, SpecimenRecord],
    curves: Mapping[str, Curve],
    analyses: Mapping[str, SpecimenAnalysis],
    *,
    colors: Mapping[str, str] | None = None,
    retrieved=None,
):
    """Node tour, maximum force (M11) and why Ag/Agt need the strain at Fm (M12), 1x2.

    - (a) stress against crosshead travel to fracture, Fm and the extensometer
      end marked (``maximum_force`` uses the force channel only);
    - (b) strain against crosshead travel with the crosshead position of Fm.

    Returns the Figure.
    """
    views = _record_views(records, curves, analyses, colors)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIGURE_SIZE[0], 6.0), layout="constrained")
    _crosshead_panel(ax_a, views)
    ax_a.set_title("(a) maximum force from the force channel")
    labels = {}
    for view in views:
        travel, strain = view.crosshead_travel_mm, view.strain
        finite = np.isfinite(travel) & np.isfinite(strain)
        ax_b.plot(travel[finite], 100 * strain[finite], color=view.color, linestyle=view.linestyle)
        end = view.end_index
        if end >= 0:
            ax_b.plot(travel[end], 100 * strain[end], PROPERTY_MARKERS["extensometer end"], color="k",
                      markersize=16, markeredgewidth=2)
        if view.i_fm >= 0:
            ax_b.axvline(travel[view.i_fm], color=view.color, linestyle=view.linestyle, linewidth=1.5)
        labels[view.specimen_id] = specimen_label(
            view.specimen_id,
            view.orientation,
            [("Fm", view.fm_n / 1000.0 if _is_value(view.fm_n) else NAN, "kN", 2), ("Rm", view.rm_mpa, "MPa", 1),
             ("ext. end at e", 100 * strain[end] if end >= 0 else NAN, "%", 2)],
        )
    ax_b.set_title("(b) the strain record ends long before Fm")
    ax_b.set_xlabel("crosshead travel from the first sample / mm")
    ax_b.set_ylabel("strain (extensometer) / %")
    ax_b.set_ylim(bottom=0.0)
    ax_b.legend(
        [Line2D([], [], color="0.4"),
         Line2D([], [], marker="|", color="k", linestyle="none", markersize=16, markeredgewidth=2)],
        ["crosshead position at Fm", "extensometer end"], loc="center left", bbox_to_anchor=(0.42, 0.5), fontsize=9,
    )
    _shared_legend(fig, views, labels, retrieved, lines=True)
    return fig


def lab_comparison_step_figure(
    records: Mapping[str, SpecimenRecord],
    s0_u_rel: Mapping[str, float],
    comparison: pd.DataFrame,
    *,
    colors: Mapping[str, str] | None = None,
    retrieved=None,
):
    """Node tour, S0 scatter (M14) and the comparison with lab values (§13), 1x2.

    - (a) the cross-section measurements s1..sn and S0 with ± u(S0)
      (``s0_scatter``), as differences from S0;
    - (b) (computed − lab) / tolerance per quantity (``compare_with_reported``;
      ``comparison`` needs the column ``specimen_id``).

    Returns the Figure.
    """
    orientations = {sid: record.orientation for sid, record in records.items()}
    colors, styles = _styles(orientations, colors)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIGURE_SIZE[0], 6.0), layout="constrained")
    views = []
    labels = {}
    for position, sid in enumerate(specimen_order(orientations)):
        record = records[sid]
        linestyle, marker = styles[record.orientation]
        views.append(_View(sid, record.orientation, float(record.s0_mm2), None, colors[sid], linestyle, marker, -1))
        measurements = np.asarray(record.cross_section_measurements_mm2, dtype=float)
        u_rel = s0_u_rel[sid]
        u = abs(u_rel * record.s0_mm2) if _is_value(u_rel) else NAN
        ax_a.plot(np.full(len(measurements), position), measurements - record.s0_mm2, marker, color=colors[sid],
                  markersize=8, linestyle="none", markeredgecolor=colors[sid] if marker == "x" else "k")
        ax_a.hlines(0.0, position - 0.25, position + 0.25, color="k", linewidth=2)
        if _is_value(u):
            ax_a.add_patch(plt.Rectangle((position - 0.25, -u), 0.5, 2 * u, color=colors[sid], alpha=0.25,
                                         linewidth=0))
        labels[sid] = specimen_label(
            sid, record.orientation,
            [("S0", record.s0_mm2, "mm²", 3), ("u(S0)/S0", 1e4 * u_rel if _is_value(u_rel) else NAN,
                                               "·10⁻⁴", 1)],
        )
    order = specimen_order(orientations)
    ax_a.set_xticks(range(len(order)))
    ax_a.set_xticklabels(order)
    ax_a.set_xlim(-0.6, len(order) - 0.4)
    ax_a.set_ylabel("measurement − S0 / mm²")
    ax_a.set_xlabel("points: s1..sn; black line: S0 (their mean); band: ± u(S0) = sd/√n (k = 1, ν = n − 1)")
    ax_a.set_title("(a) cross-section measurements around S0")
    _lab_deviation_panel(ax_b, comparison, views)
    ax_b.set_title("(b) deviation from the lab value, in units of the tolerance")
    _shared_legend(fig, views, labels, retrieved, lines=False)
    return fig
