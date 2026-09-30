"""Figure presentation contracts.

Three things used to go wrong and are easy to reintroduce:

* Headers overlapped. Titles and horizontal legends were both anchored above
  the plotting area, so any legend long enough to wrap printed through the
  title. These tests assert the legend is anchored below the plot and that the
  reserved margins can actually hold a title.
* The PSD view was cramped and its band captions landed in the title band.
* The short codes (``qSQI`` and friends) appeared in figure text. The codes
  remain valid *data* column names; what must never appear is a code in a
  label a reader sees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecg_eval.models.result import SQI_KEYS, SQI_LABELS
from ecg_eval.visualization import (
    ecg_figure,
    fuzzy_matrix_heatmap,
    heatmap_figure,
    membership_figure,
    multilead_figure,
    psd_figure,
    quality_distribution_figure,
    sqi_box_figure,
    sqi_label,
    sqi_trend_figure,
)

CODES = ("qSQI", "pSQI", "kSQI", "basSQI")

FIGURE_NAMES = [
    "ecg_figure", "multilead_figure", "psd_figure", "membership_figure",
    "quality_distribution_figure", "sqi_box_figure", "heatmap_figure",
    "sqi_trend_figure", "fuzzy_matrix_heatmap",
]


def visible_text(figure) -> str:
    """Every string a reader can see: title, legend, axes, traces and annotations."""
    parts: list[str] = []
    layout = figure.layout
    if layout.title and layout.title.text:
        parts.append(str(layout.title.text))
    if layout.xaxis.title and layout.xaxis.title.text:
        parts.append(str(layout.xaxis.title.text))
    if layout.yaxis.title and layout.yaxis.title.text:
        parts.append(str(layout.yaxis.title.text))
    for trace in figure.data:
        parts.append(str(getattr(trace, "name", "") or ""))
        parts.append(str(getattr(trace, "hovertemplate", "") or ""))
        legend_title = getattr(trace, "legendgrouptitle", None)
        if legend_title is not None and legend_title.text:
            parts.append(str(legend_title.text))
    for annotation in layout.annotations or ():
        parts.append(str(annotation.text or ""))
    return "\n".join(parts)


@pytest.fixture
def results_df() -> pd.DataFrame:
    rows = []
    for subject in range(6):
        for position in ("SUPINE", "SITTING", "STANDING"):
            rows.append(
                {
                    "subject_id": f"S{subject:02d}",
                    "session_id": f"s{subject}",
                    "position": position,
                    "frame_id": f"{subject:06d}",
                    "qSQI": 0.9 - 0.05 * subject,
                    "pSQI": 0.4 + 0.02 * subject,
                    "kSQI": 6.0 - 0.2 * subject,
                    "basSQI": 0.95 - 0.01 * subject,
                    "fuzzy_excellent": 0.7,
                    "fuzzy_barely_acceptable": 0.2,
                    "fuzzy_unacceptable": 0.1,
                    "quality_class": "Excellent",
                    "valid": True,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def signal() -> np.ndarray:
    rng = np.random.default_rng(0)
    time = np.arange(2500) / 250.0
    base = np.sin(2 * np.pi * 1.2 * time)[:, None]
    qrs = np.zeros_like(base)
    for beat in np.arange(0.35, 10.0, 0.8):
        qrs += np.exp(-((time - beat) ** 2) / (2 * 0.01 ** 2))[:, None]
    noise = rng.normal(0, 0.02, base.shape)
    return np.column_stack([base + qrs + noise] * 3)


def all_figures(signal, results_df):
    frequencies = np.linspace(0.5, 125, 300)
    matrix = results_df.pivot_table(
        index="subject_id", columns="position", values="qSQI", aggfunc="mean"
    )
    return {
        "ecg_figure": ecg_figure(
            signal, 250.0, peaks_a=[400, 600, 800], peaks_b=[402, 598, 805],
            label_a="Detector A · zhao_wavelet",
            label_b="Detector B · pan_tompkins",
            title="ses000000000005 · SUPINE · frame 000000",
        ),
        "multilead_figure": multilead_figure(signal, 250.0),
        "psd_figure": psd_figure(
            frequencies, frequencies ** -1.0, qrs_band=(5, 15), baseline_band=(0, 1)
        ),
        "membership_figure": membership_figure(
            {"Excellent": 0.6, "Barely Acceptable": 0.3, "Unacceptable": 0.1}
        ),
        "quality_distribution_figure": quality_distribution_figure(results_df),
        "sqi_box_figure": sqi_box_figure(results_df),
        "heatmap_figure": heatmap_figure(matrix, value_label=f"mean {sqi_label('qSQI')}"),
        "sqi_trend_figure": sqi_trend_figure(results_df),
        "fuzzy_matrix_heatmap": fuzzy_matrix_heatmap(
            {key: {"Excellent": 0.7, "Barely Acceptable": 0.2, "Unacceptable": 0.1} for key in SQI_KEYS}
        ),
    }


@pytest.fixture
def figures(signal, results_df):
    return all_figures(signal, results_df)


@pytest.mark.parametrize("name", FIGURE_NAMES)
def test_no_short_code_reaches_the_reader(figures, name):
    """Codes stay in the data; readers get the spelled-out name."""
    text = visible_text(figures[name])
    leaked = [code for code in CODES if code in text]
    assert not leaked, f"{name} still displays {leaked}"


@pytest.mark.parametrize("name", FIGURE_NAMES)
def test_figures_are_light_mode(figures, name):
    """Report figures must not inherit a dark theme from the host app."""
    layout = figures[name].layout
    assert _is_white(layout.paper_bgcolor), f"{name} paper is {layout.paper_bgcolor}"
    assert _is_white(layout.plot_bgcolor), f"{name} plot area is {layout.plot_bgcolor}"
    # The template must not reintroduce a dark background either.
    assert _is_white(layout.template.layout.paper_bgcolor)
    assert _is_white(layout.template.layout.plot_bgcolor)
    assert layout.font.color and _luminance(layout.font.color) < 0.5, (
        "text must be dark on white"
    )


def _is_white(colour: str | None) -> bool:
    """Accept both '#ffffff' and plotly's named 'white'."""
    return colour is None or str(colour).lower() in ("white", "#ffffff", "rgb(255, 255, 255)")


def _luminance(colour: str) -> float:
    value = colour.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@pytest.mark.parametrize("name", FIGURE_NAMES)
def test_legend_and_title_occupy_separate_pixel_bands(figures, name):
    """Stronger than the y-sign check: the bands must not physically overlap.

    The title is drawn in the top margin and the legend below the plot area, so
    this also pins the arithmetic that connects the two -- a bottom margin that
    is too small for a legend that wraps onto several rows would push the legend
    off the figure even though its anchor is nominally below the plot.
    """
    figure = figures[name]
    layout = figure.layout
    if layout.showlegend is False:
        return

    height = float(layout.height)
    top = float(layout.margin.t)
    bottom = float(layout.margin.b)
    plot_height = height - top - bottom

    legend = layout.legend
    legend_top_px = top + plot_height * (1 - float(legend.y))

    # The title occupies roughly the top 26px; the legend starts below the plot.
    assert legend_top_px > top + 26, (
        f"{name}: title band and legend band overlap (legend starts at "
        f"{legend_top_px:.0f}px, plot starts at {top:.0f}px)"
    )
    assert legend_top_px < height, f"{name}: legend is pushed off the figure"

    # A horizontal legend wraps; the reserved margin has to hold every row.
    labels = [
        str(t.name) for t in figure.data
        if t.name and not getattr(t, "showlegend", True) is False
    ]
    labels += [
        str(t.legendgrouptitle.text) for t in figure.data
        if getattr(t, "legendgrouptitle", None) is not None and t.legendgrouptitle.text
    ]
    if not labels:
        return
    # ~6.5px per character at 11px in a proportional face.
    per_row = max(1, int(1100 / 6.5))
    rows = max(1, -(-sum(len(label) for label in labels) // per_row))
    assert bottom >= rows * 16 + 16, (
        f"{name}: {rows} legend row(s) need about {rows * 16 + 16}px of bottom "
        f"margin but only {bottom:.0f}px is reserved"
    )


@pytest.mark.parametrize("name", FIGURE_NAMES)
def test_legend_is_anchored_below_the_plot(figures, name):
    """A legend above the plot is what made titles collide with headers."""
    figure = figures[name]
    assert figure.data, f"{name} produced no traces"
    if figure.layout.showlegend is False:
        return
    legend = figure.layout.legend
    assert legend is not None, f"{name} has traces but no legend configuration"
    assert legend.y is not None and legend.y < 0, (
        f"{name} anchors its legend at y={legend.y}; it must sit below the plot area"
    )


@pytest.mark.parametrize("name", FIGURE_NAMES)
def test_titled_figures_reserve_room_above_the_plot(figures, name):
    """A title is drawn in the top margin, so the margin has to be big enough."""
    layout = figures[name].layout
    if not (layout.title and layout.title.text):
        return
    assert layout.margin.t >= 60, f"{name} leaves no room for its title"


def test_psd_is_stretched_and_keeps_band_labels_inside_the_plot():
    """The PSD view was cramped, and its captions sat in the title band."""
    frequencies = np.linspace(0.5, 125, 300)
    figure = psd_figure(frequencies, frequencies ** -1.0, qrs_band=(5, 15), baseline_band=(0, 1))

    assert figure.layout.height >= 520, "PSD figure is still cramped"
    annotations = [a for a in figure.layout.annotations if a.text]
    assert annotations, "band captions disappeared"
    for annotation in annotations:
        # Plotly materialises annotation_position into y/yanchor/yref rather
        # than keeping the keyword: "inside top"/"inside bottom" resolve to the
        # y DOMAIN, while "top"/"bottom" resolve to paper coordinates, which is
        # where the figure title is drawn. Assert on the reference.
        assert str(annotation.yref) == "y domain", (
            f"band caption {annotation.text!r} is anchored to {annotation.yref}, "
            "so it can collide with the header"
        )
        assert 0 <= float(annotation.y) <= 1
    labels = {a.text for a in annotations}
    assert labels == {SQI_LABELS["pSQI"], SQI_LABELS["basSQI"]}


def test_subplot_headers_are_spaced_enough_not_to_collide():
    """multilead_figure draws a header per lead; they must not overlap."""
    signal = np.random.default_rng(1).normal(0, 1, (2500, 3))
    figure = multilead_figure(signal, 250.0)

    titles = [a for a in figure.layout.annotations if a.text]
    assert len(titles) == 3
    # 11% of the plot height per gap is what keeps a 12px header clear of the
    # trace underneath it; assert the spacing is not the old cramped value.
    assert figure.layout.yaxis.domain[1] - figure.layout.yaxis.domain[0] < 0.95
    assert figure.layout.height >= 560


def test_sqi_labels_are_used_for_every_index():
    assert set(SQI_LABELS) == set(CODES)
    assert [sqi_label(key) for key in SQI_KEYS] == [
        "Deteksi R-peak",
        "Distribusi Daya Spektral QRS",
        "Kurtosis Sinyal",
        "Daya Relatif Baseline",
    ]
    assert sqi_label("something_else") == "something_else"


def test_sqi_labels_never_collide_with_each_other():
    """Legend entries are laid out from these strings, so they must be distinct."""
    names = list(SQI_LABELS.values())
    assert len(set(names)) == len(names)


def test_machine_column_names_are_untouched():
    """Renaming labels must not leak into the data contract."""
    assert SQI_KEYS == CODES