"""``ytitle_position``：y 轴标题的丁字（top）与侧边（side）两种排布。

- ``"top"``（默认）：标题横排在轴的上端点，轴线从标题下方垂下（丁字）。
- ``"side"``：传统竖排在轴侧（matplotlib 默认行为）。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from Ts.TsPlots import plot_acf, plot_pacf, plot_series


def _ylabel_state(ax) -> tuple[str, float, tuple[float, float]]:
    label = ax.yaxis.get_label()
    return label.get_text(), label.get_rotation(), label.get_position()


def test_plot_series_default_is_top_end_horizontal() -> None:
    fig, ax = plot_series({"a": [1, 2, 3], "b": [3, 2, 1]}, facet=False)
    text, rotation, (x, y) = _ylabel_state(ax)
    assert text == "Value"
    assert rotation == 0
    assert x == 0  # 正对 y 轴轴心
    assert y > 1


def test_plot_series_side_keeps_traditional_layout() -> None:
    fig, ax = plot_series(
        {"a": [1, 2, 3]}, facet=False, ytitle_position="side"
    )
    text, rotation, (x, y) = _ylabel_state(ax)
    assert text == "Value"
    assert rotation == 90
    assert y == 0.5


def test_plot_series_facet_panels_use_top_layout() -> None:
    fig, axes = plot_series({"a": [1, 2, 3], "b": [3, 2, 1]})
    for panel in np.asarray(axes).ravel():
        _, rotation, (x, y) = _ylabel_state(panel)
        assert rotation == 0
        assert y > 1


def test_plot_series_invalid_position_raises() -> None:
    with pytest.raises(ValueError, match="ytitle_position"):
        plot_series({"a": [1, 2, 3]}, ytitle_position="left")


def test_plot_acf_top_end_horizontal() -> None:
    fig, ax = plot_acf(np.arange(20.0), nlags=8)
    text, rotation, (x, y) = _ylabel_state(ax)
    assert text == "ACF值"
    assert rotation == 0
    assert y > 1


def test_plot_pacf_side_keeps_traditional_layout() -> None:
    fig, ax = plot_pacf(np.arange(20.0), nlags=8, ytitle_position="side")
    text, rotation, (x, y) = _ylabel_state(ax)
    assert text == "PACF值"
    assert rotation == 90
    assert y == 0.5


def test_plot_acf_invalid_position_raises() -> None:
    with pytest.raises(ValueError, match="ytitle_position"):
        plot_acf(np.arange(20.0), nlags=8, ytitle_position="left")
