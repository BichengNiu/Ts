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


def test_plot_series_left_title_moves_right_of_top_ylabel() -> None:
    """图标题靠左 + y 标题置顶：y 标题位置不变，图标题移到其右侧。"""
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
    assert label.get_ha() == "center"
    assert x == 0  # y 轴标题仍正对轴心，位置不变
    title = ax._left_title
    assert title.get_text() == "左侧标题"
    # 图标题左缘位于 y 标题右缘右侧（互不重叠）
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_x1 = label.get_window_extent(renderer).x1
    title_x0 = title.get_window_extent(renderer).x0
    assert title_x0 > label_x1


def test_plot_series_facet_panel_titles_move_right_of_top_ylabel() -> None:
    """分面面板标题恒为左上角，y 标题置顶时标题让位、y 标题位置不变。"""
    fig, axes = plot_series({"a": [1, 2, 3], "b": [3, 2, 1]})
    for panel in np.asarray(axes).ravel():
        label = panel.yaxis.get_label()
        assert label.get_rotation() == 0
        assert label.get_ha() == "center"
        assert label.get_position()[0] == 0
        assert panel._left_title.get_text()  # 面板标题仍在


def test_plot_series_vlines_accepts_date_strings() -> None:
    """参考线接受日期字符串（matplotlib 3.11 date2num 回归修复）。"""
    import pandas as pd

    index = pd.date_range("2020-01-01", periods=12, freq="MS")
    frame = pd.DataFrame({"v": range(12)}, index=index)
    fig, ax = plot_series(
        frame, vlines=["2020-07-01"], xtitle="", ytitle="V", facet=False
    )
    vertical = [
        line for line in ax.lines
        if len(line.get_xdata()) == 2
        and line.get_xdata()[0] == line.get_xdata()[1]
    ]
    assert vertical


def test_plot_series_xtitle_loc() -> None:
    """X 轴标题位置：left/center/right 三档生效。"""
    fig, ax = plot_series(
        {"a": [1, 2, 3]}, xtitle="X 标签", xtitle_loc="left", ytitle="V", facet=False
    )
    label = ax.xaxis.get_label()
    assert label.get_text() == "X 标签"
    assert label.get_ha() == "left"
    assert label.get_position()[0] == 0

    fig, ax = plot_series(
        {"a": [1, 2, 3]}, xtitle="X 标签", xtitle_loc="right", ytitle="V", facet=False
    )
    label = ax.xaxis.get_label()
    assert label.get_ha() == "right"
    assert label.get_position()[0] == 1


def test_plot_series_xtitle_loc_invalid_raises() -> None:
    with pytest.raises(ValueError, match="xtitle_loc"):
        plot_series({"a": [1, 2, 3]}, xtitle="X", xtitle_loc="top")


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
