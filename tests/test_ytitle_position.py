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
    fig, ax = plot_series(
        {"a": [1, 2, 3], "b": [3, 2, 1]}, facet=False, ytitle="产出"
    )
    text, rotation, (x, y) = _ylabel_state(ax)
    assert text == "产出"
    assert rotation == 0
    assert x == 0  # 正对 y 轴轴心
    assert y > 1

    # 设计契约：不传 ytitle / unit 时默认不显示 y 轴标题。
    fig2, ax2 = plot_series({"a": [1, 2, 3]}, facet=False)
    assert _ylabel_state(ax2)[0] == ""


def test_plot_series_side_keeps_traditional_layout() -> None:
    fig, ax = plot_series(
        {"a": [1, 2, 3]}, facet=False, ytitle_position="side", ytitle="产出"
    )
    text, rotation, (x, y) = _ylabel_state(ax)
    assert text == "产出"
    assert rotation == 90
    assert y == 0.5


def test_plot_series_facet_panels_use_top_layout() -> None:
    fig, axes = plot_series({"a": [1, 2, 3], "b": [3, 2, 1]}, ytitle="产出")
    for panel in np.asarray(axes).ravel():
        text, rotation, (x, y) = _ylabel_state(panel)
        assert text == "产出"
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
        ytitle="产出",
        facet=False,
    )
    label = ax.yaxis.get_label()
    text, rotation, (x, y) = _ylabel_state(ax)
    assert text == "产出"
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


def test_plot_series_facet_panel_titles_centered() -> None:
    """分面面板标题默认上居中，y 标题置顶时互不重叠。"""
    fig, axes = plot_series({"a": [1, 2, 3], "b": [3, 2, 1]}, ytitle="产出")
    for panel in np.asarray(axes).ravel():
        label = panel.yaxis.get_label()
        assert label.get_rotation() == 0
        assert label.get_ha() == "center"
        assert label.get_position()[0] == 0
        assert panel.get_title()  # 面板标题居中槽
        assert panel.title.get_horizontalalignment() == "center"


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


def test_plot_series_ytick_count_minor_ticks() -> None:
    """Y 轴刻度数 = 两个主刻度（标签）之间的子刻度线数。"""
    import numpy as np

    fig, ax = plot_series(
        {"a": range(50)},
        ytitle="V",
        xtitle="",
        ytick_count=2,
        facet=False,
        show_legend=False,
    )
    fig.canvas.draw()
    lo, hi = ax.get_ylim()
    majors = [m for m in ax.get_yticks() if lo <= m <= hi]
    minors = ax.get_yticks(minor=True)
    assert len(minors) > 0
    per_interval = len([m for m in minors if majors[0] < m < majors[1]])
    assert per_interval == 2


def test_plot_series_xtick_count_minor_ticks() -> None:
    """X 轴刻度数 = 两个主刻度（标签）之间的子刻度线数。"""
    import pandas as pd

    index = pd.date_range("2020-01-01", periods=60, freq="MS")
    frame = pd.DataFrame({"v": range(60)}, index=index)
    fig, ax = plot_series(
        frame, xtitle="", ytitle="V", xtick_count=3, facet=False, show_legend=False
    )
    fig.canvas.draw()
    lo, hi = ax.get_xlim()
    majors = [m for m in ax.get_xticks() if lo <= m <= hi]
    minors = ax.get_xticks(minor=True)
    assert len(minors) > 0
    # 日期主刻度间隔 365/366 天波动，允许 ±1
    per_interval = len([m for m in minors if majors[0] < m < majors[1]])
    assert 2 <= per_interval <= 4


def test_plot_series_no_minor_ticks_by_default() -> None:
    """默认不画子刻度。"""
    fig, ax = plot_series({"a": range(10)}, ytitle="V", xtitle="", facet=False)
    fig.canvas.draw()
    assert len(ax.get_yticks(minor=True)) == 0


def test_plot_series_ylabel_count_thins_labels() -> None:
    """Y 轴标签数：刻度全保留、标签抽稀。"""
    fig, ax = plot_series(
        {"a": range(50)},
        ytitle="V",
        xtitle="",
        ylabel_count=3,
        facet=False,
        show_legend=False,
    )
    fig.canvas.draw()
    ticks = ax.get_yticks()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    non_empty = [label for label in labels if label]
    assert len(ticks) >= 4
    assert 1 <= len(non_empty) <= 3


def test_plot_series_xlabel_count_thins_labels() -> None:
    """X 轴标签数：日期轴标签抽稀，刻度保留。"""
    import pandas as pd

    index = pd.date_range("2020-01-01", periods=60, freq="MS")
    frame = pd.DataFrame({"v": range(60)}, index=index)
    fig, ax = plot_series(
        frame, xtitle="", ytitle="V", xlabel_count=3, facet=False, show_legend=False
    )
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels()]
    non_empty = [label for label in labels if label]
    assert 1 <= len(non_empty) <= 3


def test_plot_series_xmin_sets_left_limit() -> None:
    """X 轴起点：时间轴从指定日期开始。"""
    import pandas as pd

    index = pd.date_range("2020-01-01", periods=24, freq="MS")
    frame = pd.DataFrame({"v": range(24)}, index=index)
    start = pd.Timestamp("2020-06-01")
    fig, ax = plot_series(
        frame, xtitle="", ytitle="V", xmin=start, facet=False, show_legend=False
    )
    fig.canvas.draw()
    import matplotlib.dates as mdates

    left = ax.get_xlim()[0]
    assert abs(left - mdates.date2num(start)) < 1.0


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
