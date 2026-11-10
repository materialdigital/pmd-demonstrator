"""Figures of the tensile notebooks (decision D12; ``pmd_tensile._plotting``).

Offline: the graphs are the local pmdco2 Turtle files and the curve downloads
are stubbed with synthetic CSV bytes (``workflow_helpers``). The tests check
the structure of the figures (panels, shared legend, labels, notes), that
undefined values are shown as "–" and that no specimen is dropped silently,
and that the plotting helpers stay out of sim_atlas collection. They do not
judge how the figures look.
"""

from __future__ import annotations

import dataclasses
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from workflow_helpers import (  # noqa: E402
    PMDCO2,
    PMDCO2_SOURCE,
    curve_bytes_for,
    run_with_stubbed_curves,
)

from pmd_tensile import plotting
from pmd_tensile.nodes import acquisition
from pmd_tensile.nodes.analysis.elastic_line import (
    bootstrap_fit,
    final_elastic_fit,
    proof_strength,
    window_sensitivity,
)
from pmd_tensile.nodes.analysis.extensometer import extensometer_validity
from pmd_tensile.nodes.analysis.geometry import elongation_after_fracture, s0_scatter
from pmd_tensile.nodes.analysis.maximum_force import (
    elongation_at_maximum_force,
    maximum_force,
)
from pmd_tensile.nodes.analysis.yielding import yield_analysis
from pmd_tensile.parameters import DEFAULT_PARAMETERS

SPECIMENS = ("Zx1", "Zx2", "Zy1", "Zd2")  # the IDs with synthetic curves; all three orientations
STAMP = pd.Timestamp("2026-11-09 12:34:56.789", tz="Europe/Berlin")
NAN = float("nan")


@pytest.fixture(scope="module")
def result():
    analysis_result, _ = run_with_stubbed_curves(
        curve_bytes_for(SPECIMENS), PMDCO2, PMDCO2_SOURCE, specimen_ids=SPECIMENS
    )
    return analysis_result


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _titles(fig) -> list[str]:
    return [ax.get_title() for ax in fig.axes if ax.get_title().startswith("(")]


def _legend_texts(fig) -> list[str]:
    assert len(fig.legends) == 1, "one shared legend"
    return [text.get_text() for text in fig.legends[0].get_texts()]


def _panel(fig, prefix: str):
    (ax,) = [ax for ax in fig.axes if ax.get_title().startswith(prefix)]
    return ax


def _texts(ax) -> str:
    return "\n".join(text.get_text() for text in ax.texts)


def _with_undefined_values(result):
    """A copy of the result in which Zy1 has no ReH/ReL and Zd2 no E (hence no strain origin)."""
    analyses = dict(result.curve_analyses)
    indices = dict(analyses["Zy1"].indices, reh=-1, rel=-1)
    analyses["Zy1"] = dataclasses.replace(analyses["Zy1"], reh_mpa=NAN, rel_mpa=NAN, indices=indices)
    analyses["Zd2"] = dataclasses.replace(analyses["Zd2"], e_gpa=NAN, strain_origin=NAN, rp02_mpa=NAN)
    results = result.results.copy()
    results.loc[results["specimen_id"] == "Zy1", ["reh_mpa", "rel_mpa"]] = NAN
    results.loc[results["specimen_id"] == "Zd2", ["e_gpa", "rp02_mpa", "strain_origin"]] = NAN
    comparison = result.comparison.copy()
    undefined = (comparison["specimen_id"] == "Zy1") & (comparison["quantity"] == "reh_mpa")
    comparison.loc[undefined, ["computed", "rel_deviation"]] = NAN
    comparison.loc[undefined, "outcome"] = "not_comparable"
    return dataclasses.replace(result, results=results, comparison=comparison, curve_analyses=analyses)


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------


def test_specimen_colors_follow_orientation_then_natural_order():
    orientations = {
        "Zd3": "diagonal to rolling direction",
        "Zx10": "in rolling direction",
        "Zy1": "perpendicular to rolling direction",
        "Zx2": "in rolling direction",
        "Zd2": "diagonal to rolling direction",
    }
    assert plotting.specimen_order(orientations) == ["Zx2", "Zx10", "Zy1", "Zd2", "Zd3"]
    tab10 = [plotting._hex(rgb) for rgb in plt.get_cmap("tab10").colors]
    assert plotting.specimen_colors(orientations) == {
        "Zx2": tab10[0], "Zx10": tab10[1], "Zy1": tab10[2], "Zd2": tab10[3], "Zd3": tab10[4]
    }


def test_orientation_styles_use_the_mesh_04_styles_and_cover_unknown_labels():
    styles = plotting.orientation_styles(["diagonal to rolling direction", "other", "in rolling direction",
                                      "perpendicular to rolling direction"])
    assert styles["in rolling direction"] == ("solid", "o")
    assert styles["perpendicular to rolling direction"] == ("dotted", "x")
    assert styles["diagonal to rolling direction"] == ("dashed", "^")
    assert styles["other"] not in (styles["in rolling direction"], styles["diagonal to rolling direction"])


@pytest.mark.parametrize("value", [NAN, None, float("inf"), np.float64("nan")])
def test_undefined_values_are_written_as_a_dash(value):
    assert plotting.format_value(value, 1, "MPa") == "–"


def test_specimen_label_shows_values_with_units_and_dashes():
    label = plotting.specimen_label("Zy3", "perpendicular to rolling direction",
                                [("E", 205.0917, "GPa", 1), ("ReH", NAN, "MPa", 1)])
    assert label == "Zy3 (perpendicular to rolling direction)\nE = 205.1 GPa; ReH = –"


# ---------------------------------------------------------------------------
# Overview figures (notebook 05)
# ---------------------------------------------------------------------------


def test_curves_overview_has_the_four_panels_and_one_shared_legend(result):
    fig = plotting.curves_overview_figure(result, retrieved=STAMP, method_specimen="Zx1")
    assert tuple(fig.get_size_inches()) == (12.0, 12.0)
    assert [title[:3] for title in _titles(fig)] == ["(a)", "(b)", "(c)", "(d)"]
    assert fig.legends[0]._ncols == 3
    texts = _legend_texts(fig)
    table = result.results.set_index("specimen_id")
    for sid in SPECIMENS:
        row = table.loc[sid]
        expected = (f"E = {row['e_gpa']:.1f} GPa; ReH = {row['reh_mpa']:.1f} MPa; "
                    f"Rm = {row['rm_mpa']:.1f} MPa")
        assert sum(text.startswith(f"{sid} (") and text.endswith(expected) for text in texts) == 1, sid
    assert any("data\\ retrieved" in text and "2026-11-09T12:34:56+01:00" in text for text in texts)
    # every specimen is drawn in (a), shifted by its strain origin as stated in the panel
    ax_a = _panel(fig, "(a)")
    assert len(ax_a.get_lines()) == len(SPECIMENS)
    assert "strain origin e₀" in _texts(ax_a)
    # the zoom limits of (b) are in its title
    ax_b = _panel(fig, "(b)")
    low, high = ax_b.get_ylim()
    assert f"{low:g}–{high:g} MPa" in ax_b.get_title()
    # (c) has the inset and the construction lines of the method specimen
    ax_c = _panel(fig, "(c)")
    assert ax_c.child_axes, "inset"
    labels = [line.get_label() for line in ax_c.get_lines()]
    assert any(label.startswith("elastic line") for label in labels)
    assert any(label.startswith("offset line") for label in labels)
    assert any(label.startswith("Rp0.2 =") for label in labels)


def test_curves_overview_draws_exactly_the_arrays_the_run_used(result):
    fig = plotting.curves_overview_figure(result, method_specimen="Zx1")
    ax_a = _panel(fig, "(a)")
    colors = plotting.specimen_colors(dict(zip(result.results["specimen_id"], result.results["orientation"])))
    table = result.results.set_index("specimen_id")
    for sid in SPECIMENS:
        (line,) = [line for line in ax_a.get_lines() if line.get_color() == colors[sid]]
        curve, curve_analysis = result.curves[sid], result.curve_analyses[sid]
        k = curve_analysis.k_ext
        valid = np.isfinite(curve.force_n[:k]) & np.isfinite(curve.strain[:k])
        x, y = line.get_data()
        np.testing.assert_array_equal(x, 100 * (curve.strain[:k][valid] - curve_analysis.strain_origin))
        np.testing.assert_array_equal(y, curve.force_n[:k][valid] / table.loc[sid, "s0_mm2"])


def test_curves_overview_uses_the_offset_of_the_run(result):
    """A run with offset 0.003 is drawn with its own offset line and Rp markers (T9.1 NR1-7)."""
    offset_result, _ = run_with_stubbed_curves(
        curve_bytes_for(SPECIMENS), PMDCO2, PMDCO2_SOURCE, specimen_ids=SPECIMENS,
        parameters={"offset_plastic_strain": 0.003},
    )
    fig = plotting.curves_overview_figure(offset_result, method_specimen="Zx1")
    table = offset_result.results.set_index("specimen_id")

    ax_c = _panel(fig, "(c)")
    labels = [line.get_label() for line in ax_c.get_lines()]
    assert "offset line, shifted by 0.3 % plastic strain" in labels
    (rp_line,) = [line for line in ax_c.get_lines() if line.get_label().startswith("Rp0.2 =")]
    zx1 = table.loc["Zx1"]
    x, y = rp_line.get_data()
    np.testing.assert_allclose(x, [100 * (zx1["rp02_mpa"] / (1000 * zx1["e_gpa"]) + 0.003 + zx1["strain_origin"])])
    np.testing.assert_allclose(y, [zx1["rp02_mpa"]])

    # (b): one Rp marker per specimen, at Rp/E + 0.003 of shifted strain
    ax_b = _panel(fig, "(b)")
    markers = {(round(line.get_xdata()[0], 9), round(line.get_ydata()[0], 9))
               for line in ax_b.get_lines() if line.get_marker() == plotting.PROPERTY_MARKERS["Rp0.2"]}
    expected = {(round(100 * (row["rp02_mpa"] / (1000 * row["e_gpa"]) + 0.003), 9), round(row["rp02_mpa"], 9))
                for _, row in table.loc[list(SPECIMENS)].iterrows()}
    assert markers == expected
    # the default run differs, so the test is not vacuous
    assert not table["rp02_mpa"].equals(result.results.set_index("specimen_id")["rp02_mpa"])


def test_curves_overview_without_a_timestamp_says_so(result):
    fig = plotting.curves_overview_figure(result, method_specimen="Zx1")
    assert any("not recorded" in text for text in _legend_texts(fig))


def test_curves_overview_rejects_an_unknown_method_specimen(result):
    with pytest.raises(KeyError, match="Zx3"):
        plotting.curves_overview_figure(result, method_specimen="Zx3")


def test_results_overview_has_the_four_panels(result):
    fig = plotting.results_overview_figure(result, retrieved=STAMP)
    assert tuple(fig.get_size_inches()) == (12.0, 12.0)
    assert [title[:3] for title in _titles(fig)] == ["(e)", "(f)", "(g)"]
    (subfigure,) = [sub for sub in fig.subfigs[1].subfigs if sub._suptitle is not None]
    assert subfigure._suptitle.get_text().startswith("(h)")
    assert [ax.get_title() for ax in subfigure.axes] == ["E / GPa", "ReH / MPa", "Rp0.2 / MPa", "Rm / MPa"]
    texts = _legend_texts(fig)
    assert sum(text.startswith(tuple(f"{sid} (" for sid in SPECIMENS)) for text in texts) == len(SPECIMENS)
    # (e): one row per compared quantity, the tolerance in the tick label
    ticks = [tick.get_text() for tick in _panel(fig, "(e)").get_yticklabels()]
    assert ticks[0] == "Fm\n(±0.1 %)" and ticks[4] == "E\n(±10 %)" and len(ticks) == 7


def test_lab_deviation_is_scaled_by_the_tolerance(result):
    fig = plotting.results_overview_figure(result)
    ax_e = _panel(fig, "(e)")
    comparison = result.comparison
    row = comparison[(comparison["specimen_id"] == "Zx1") & (comparison["quantity"] == "e_gpa")].iloc[0]
    expected = (row["computed"] - row["reported"]) / (0.10 * row["reported"])
    plotted = [line.get_xdata()[0] for line in ax_e.get_lines() if len(line.get_xdata()) == 1]
    assert any(math.isclose(x, expected, rel_tol=1e-12) for x in plotted)


def test_undefined_values_are_labelled_and_no_specimen_is_dropped(result):
    changed = _with_undefined_values(result)
    fig = plotting.curves_overview_figure(changed, method_specimen="Zx1")
    texts = _legend_texts(fig)
    assert any(text.startswith("Zy1 (") and "ReH = –;" in text for text in texts)
    assert any(text.startswith("Zd2 (") and "E = –;" in text for text in texts)
    assert len(_panel(fig, "(a)").get_lines()) == len(SPECIMENS)
    assert "not shifted (e₀ –): Zd2" in _texts(_panel(fig, "(a)"))
    notes = _texts(_panel(fig, "(b)"))
    assert "ReH – (no marker): Zy1" in notes and "Rp0.2 – (no marker): Zd2" in notes

    fig = plotting.results_overview_figure(changed)
    assert "not comparable (a value missing): ReH Zy1" in _texts(_panel(fig, "(e)"))
    assert "not shown (ReH or Rm –): Zy1" in _texts(_panel(fig, "(f)"))
    assert "not shown (E or Rp0.2 –): Zd2" in _texts(_panel(fig, "(g)"))


# ---------------------------------------------------------------------------
# Step figures (notebook 02)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def steps(result):
    """The node outputs of notebook 02 for two specimens, on the curves of the stubbed run."""
    ids = ("Zx1", "Zy1")
    out = {name: {} for name in ("records", "reported", "extensometer", "bootstrap", "yielding", "elastic",
                                 "proof", "maximum", "elongation", "scatter")}
    windows, comparisons = [], []
    for sid in ids:
        record = acquisition.get_specimen_record(PMDCO2, PMDCO2_SOURCE, sid)
        curve = result.curves[sid]
        out["records"][sid] = record
        out["reported"][sid] = acquisition.get_reported_properties(PMDCO2, PMDCO2_SOURCE, sid)
        a = elongation_after_fracture(record.l0_mm, record.lu_mm)
        ext, k, _ = extensometer_validity(curve.strain, curve.crosshead_mm, record.l0_mm, a.value,
                                                   record.rate_switch_strain)
        boot, _ = bootstrap_fit(curve.force_n, curve.strain, record.s0_mm2, k)
        yielding, *_ = yield_analysis(curve.force_n, curve.strain, record.s0_mm2, k, boot,
                                               record.rate_switch_strain)
        elastic, *_ = final_elastic_fit(curve.force_n, curve.strain, record.s0_mm2, k, boot, yielding)
        proof, _ = proof_strength(curve.force_n, curve.strain, record.s0_mm2, k, elastic, yielding)
        maximum, *_ = maximum_force(curve.force_n, record.s0_mm2, yielding)
        elongation, *_ = elongation_at_maximum_force(curve.force_n, curve.strain, k, boot, elastic,
                                                              maximum)
        for name, value in (("extensometer", ext), ("bootstrap", boot), ("yielding", yielding),
                            ("elastic", elastic), ("proof", proof), ("maximum", maximum),
                            ("elongation", elongation)):
            out[name][sid] = value
        out["scatter"][sid] = s0_scatter(record.s0_mm2, record.cross_section_measurements_mm2)
        for f_lo in (0.05, 0.10):
            one = dataclasses.replace(DEFAULT_PARAMETERS, window_grid_lo=(f_lo,), window_grid_hi=(0.4,))
            sens = window_sensitivity(curve.force_n, curve.strain, record.s0_mm2, k, elastic, maximum,
                                               elongation, one)
            windows.append({"specimen_id": sid, "f_lo": f_lo, "f_hi": 0.4, "e_gpa": sens.e_range_gpa[0]})
        comparisons.append(
            result.comparison[result.comparison["specimen_id"] == sid].drop(columns="orientation")
        )
    out["window_moduli"] = pd.DataFrame(windows)
    out["comparison"] = pd.concat(comparisons, ignore_index=True)
    out["curves"] = {sid: result.curves[sid] for sid in ids}
    return out


def test_yield_step_figure(steps):
    fig = plotting.yield_step_figure(steps["records"], steps["curves"], steps["extensometer"], steps["bootstrap"],
                                 steps["yielding"], retrieved=STAMP)
    assert [title[:3] for title in _titles(fig)] == ["(a)", "(b)", "(c)", "(d)"]
    texts = _legend_texts(fig)
    for sid in ("Zx1", "Zy1"):
        noise = steps["bootstrap"][sid].noise_mpa
        (label,) = [text for text in texts if text.startswith(f"{sid} (")]
        assert f"s_n = {noise:.2f} MPa" in label and "required d" in label


def test_elastic_step_figure(steps):
    fig = plotting.elastic_step_figure(
        steps["records"], steps["curves"], steps["extensometer"], steps["yielding"], steps["elastic"],
        steps["proof"], steps["window_moduli"], steps["reported"], method_specimen="Zy1", retrieved=STAMP,
    )
    assert [title[:3] for title in _titles(fig)] == ["(a)", "(b)", "(c)", "(d)"]
    assert [tick.get_text() for tick in _panel(fig, "(c)").get_xticklabels()] == ["5–40", "10–40"]
    with pytest.raises(KeyError, match="Zx3"):
        plotting.elastic_step_figure(
            steps["records"], steps["curves"], steps["extensometer"], steps["yielding"], steps["elastic"],
            steps["proof"], steps["window_moduli"], steps["reported"],
        )


def test_maximum_force_and_lab_comparison_step_figures(steps):
    fig = plotting.maximum_force_step_figure(steps["records"], steps["curves"], steps["extensometer"],
                                         steps["maximum"], retrieved=STAMP)
    assert [title[:3] for title in _titles(fig)] == ["(a)", "(b)"]
    assert sum(text.startswith(("Zx1 (", "Zy1 (")) for text in _legend_texts(fig)) == 2

    fig = plotting.lab_comparison_step_figure(steps["records"], steps["scatter"], steps["comparison"],
                                          retrieved=STAMP)
    assert [title[:3] for title in _titles(fig)] == ["(a)", "(b)"]
    assert [tick.get_text() for tick in _panel(fig, "(a)").get_xticklabels()] == ["Zx1", "Zy1"]


def test_step_figures_need_a_colour_for_every_specimen(steps):
    with pytest.raises(ValueError, match="no colour"):
        plotting.maximum_force_step_figure(steps["records"], steps["curves"], steps["extensometer"],
                                       steps["maximum"], colors={"Zx1": "#000000"})
