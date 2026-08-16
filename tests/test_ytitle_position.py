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


def test_plot_series_left_title_and_top_ylabel_do_not_overlap() -> None:
    """图标题靠左 + y 标题置顶：同一高度、左右留间隙、互不重叠。"""
    fig, ax = plot_series(
        {"a": [1, 2, 3]},
        title="左侧标题",
        title_loc="left",
        ytitle_position="top",
        facet=False,
    )
    label = ax.yaxis.get_label()
    text, rotation, (x, y) = _ylabel_state(ax)
    assert text == "Value"
    assert rotation == 0
    assert label.get_ha() == "right"
    assert x < 0
    # 左侧图标题仍存在（matplotlib 3.11 写入 _left_title 槽位）
    assert ax._left_title.get_text() == "左侧标题"
    # 同一高度：y 轴标题底边与图标题底边接近（同一水平线）
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    label_bottom = inv.transform(
        (0, label.get_window_extent(renderer).y0)
    )[1]
    title_bottom = inv.transform(
        (0, ax._left_title.get_window_extent(renderer).y0)
    )[1]
    assert abs(label_bottom - title_bottom) < 0.05


def test_plot_series_facet_panel_titles_clear_top_ylabel() -> None:
    """分面面板标题恒为左上角，y 标题置顶时同样让位。"""
    fig, axes = plot_series({"a": [1, 2, 3], "b": [3, 2, 1]})
    for panel in np.asarray(axes).ravel():
        label = panel.yaxis.get_label()
        assert label.get_rotation() == 0
        assert label.get_ha() == "right"
        assert label.get_position()[0] < 0


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
