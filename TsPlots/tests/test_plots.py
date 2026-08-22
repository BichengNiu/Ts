import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from Ts.TsPlots import (
    plot_acf,
    plot_correlation_matrix,
    plot_correlogram,
    plot_pacf,
    plot_scatter,
    plot_series,
)
from Ts.TsPlots.style import (
    BLACK,
    DARK_BLUE,
    DARK_RED,
    GRAY,
    DEFAULT_LINESTYLES,
    DEFAULT_MARKERS,
    DEFAULT_PALETTE,
    _body_font_family,
    format_compact_y_axis,
    _title_font_family,
)


class TestPlotSeries:
    def test_returns_fig_ax(self):
        data = np.random.randn(50)
        fig, ax = plot_series(data)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_dataframe_input(self):
        df = pd.DataFrame({"A": np.random.randn(30), "B": np.random.randn(30)})
        fig, axes = plot_series(df)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (2,)
        assert all(isinstance(axis, plt.Axes) for axis in axes)
        plt.close(fig)

    def test_multiple_series_are_faceted_and_share_only_x_by_default(self):
        data = {"small": [1, 2, 3], "large": [100, 200, 300]}

        fig, axes = plot_series(data)

        assert axes[0].get_shared_x_axes().joined(axes[0], axes[1])
        assert not axes[0].get_shared_y_axes().joined(axes[0], axes[1])
        assert [axis.lines[0].get_label() for axis in axes] == ["small", "large"]
        assert [axis.get_title() for axis in axes] == ["small", "large"]
        plt.close(fig)

    def test_facet_can_share_y_without_sharing_x(self):
        data = {"a": [1, 2, 3], "b": [3, 2, 1]}

        fig, axes = plot_series(data, sharex=False, sharey=True)

        assert not axes[0].get_shared_x_axes().joined(axes[0], axes[1])
        assert axes[0].get_shared_y_axes().joined(axes[0], axes[1])
        plt.close(fig)

    def test_single_series_ignores_facet_and_returns_axes(self):
        fig, ax = plot_series({"only": [1, 2, 3]}, facet=True)

        assert isinstance(ax, plt.Axes)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_overlay_automatically_uses_dual_y_axis_for_scale_gap(self):
        data = {"small": [1, 2, 3], "large": [1000, 2000, 3000]}

        fig, ax = plot_series(data, facet=False)

        assert len(fig.axes) == 2
        assert ax.right_ax is fig.axes[1]
        assert [line.get_label() for line in ax.lines] == ["small"]
        assert [line.get_label() for line in ax.right_ax.lines] == ["large"]
        plt.close(fig)

    def test_overlay_can_disable_automatic_dual_y_axis(self):
        data = {"small": [1, 2, 3], "large": [1000, 2000, 3000]}

        fig, ax = plot_series(data, facet=False, auto_dual_y=False)

        assert len(fig.axes) == 1
        assert [line.get_label() for line in ax.lines] == ["small", "large"]
        plt.close(fig)

    def test_scale_ratio_threshold_controls_dual_axis(self):
        data = {"small": [1, 2, 3], "large": [1000, 2000, 3000]}

        fig, ax = plot_series(data, facet=False, scale_ratio_threshold=2000)

        assert len(fig.axes) == 1
        assert not hasattr(ax, "right_ax")
        plt.close(fig)

    def test_three_distinct_scales_create_three_axes(self):
        data = {
            "middle": [100, 200, 300],
            "small": [1, 2, 3],
            "large": [1_000_000, 2_000_000, 3_000_000],
        }

        fig, ax = plot_series(data, facet=False)

        assert len(fig.axes) == 3
        assert [line.get_label() for line in ax.lines] == ["middle"]
        assert ax.right_ax is ax.extra_y_axes[0]
        assert [line.get_label() for line in ax.extra_y_axes[0].lines] == ["small"]
        assert [line.get_label() for line in ax.extra_y_axes[1].lines] == ["large"]
        position, value = ax.extra_y_axes[1].spines["right"].get_position()
        assert position == "axes"
        assert value == pytest.approx(1.12)
        plt.close(fig)

    def test_similar_scales_remain_in_one_automatic_group(self):
        data = {
            "small": [1, 2, 3],
            "similar": [5, 10, 15],
            "large": [1000, 2000, 3000],
        }

        fig, ax = plot_series(data, facet=False)

        assert len(fig.axes) == 2
        assert [line.get_label() for line in ax.lines] == ["small", "similar"]
        assert [line.get_label() for line in ax.right_ax.lines] == ["large"]
        plt.close(fig)

    def test_max_y_axes_merges_the_closest_automatic_groups(self):
        data = {
            "small": [1, 2, 3],
            "middle": [100, 200, 300],
            "large": [1_000_000, 2_000_000, 3_000_000],
        }

        fig, ax = plot_series(data, facet=False, max_y_axes=2)

        assert len(fig.axes) == 2
        assert [line.get_label() for line in ax.lines] == ["small", "middle"]
        assert [line.get_label() for line in ax.right_ax.lines] == ["large"]
        plt.close(fig)

    def test_manual_axis_groups_override_automatic_scale_groups(self):
        data = {
            "level": [100, 200, 300],
            "rate_a": [1, 2, 3],
            "rate_b": [4, 5, 6],
        }
        groups = {"level": "level", "rate_a": "rates", "rate_b": "rates"}

        fig, ax = plot_series(
            data,
            facet=False,
            auto_dual_y=False,
            axis_groups=groups,
        )

        assert [line.get_label() for line in ax.lines] == ["level"]
        assert [line.get_label() for line in ax.right_ax.lines] == [
            "rate_a",
            "rate_b",
        ]
        assert ax.extra_y_axes == [ax.right_ax]
        plt.close(fig)

    def test_second_and_third_axis_vars(self):
        data = {"a": [1, 2, 3], "b": [100, 200, 300], "c": [5, 6, 7]}

        fig, ax = plot_series(
            data,
            facet=False,
            auto_dual_y=False,
            second_axis_vars=["b"],
            third_axis_vars=["c"],
        )

        assert [line.get_label() for line in ax.lines] == ["a"]
        assert [line.get_label() for line in ax.extra_y_axes[0].lines] == ["b"]
        assert [line.get_label() for line in ax.extra_y_axes[1].lines] == ["c"]
        plt.close(fig)

    def test_second_third_axis_titles(self):
        data = {"a": [1, 2, 3], "b": [100, 200, 300], "c": [5, 6, 7]}

        fig, ax = plot_series(
            data,
            facet=False,
            auto_dual_y=False,
            second_axis_vars=["b"],
            third_axis_vars=["c"],
            second_axis_title="第二轴",
            third_axis_title="第三轴",
        )

        assert ax.extra_y_axes[0].get_ylabel() == "第二轴"
        assert ax.extra_y_axes[1].get_ylabel() == "第三轴"
        plt.close(fig)

    def test_log_vars_applies_log_scale(self):
        data = {"a": [1, 2, 3], "b": [100, 200, 300]}

        fig, ax = plot_series(
            data,
            facet=False,
            auto_dual_y=False,
            log_vars=["a"],
        )

        assert ax.get_yscale() == "log"
        plt.close(fig)

        fig, ax = plot_series(
            data,
            facet=False,
            auto_dual_y=False,
            second_axis_vars=["b"],
            log_vars=["b"],
        )
        assert ax.extra_y_axes[0].get_yscale() == "log"
        assert ax.get_yscale() == "linear"
        plt.close(fig)

    def test_rejects_unknown_axis_and_log_vars(self):
        with pytest.raises(ValueError, match="second_axis_vars contains unknown"):
            plot_series(
                {"a": [1, 2]}, facet=False, second_axis_vars=["nope"]
            )
        with pytest.raises(ValueError, match="log_vars contains unknown"):
            plot_series({"a": [1, 2]}, facet=False, log_vars=["nope"])
        with pytest.raises(ValueError, match="both the second and third axes"):
            plot_series(
                {"a": [1, 2], "b": [2, 3]},
                facet=False,
                second_axis_vars=["b"],
                third_axis_vars=["b"],
            )

    def test_figsize_applies_to_single_axes(self):
        fig, _ax = plot_series(
            {"a": [1, 2, 3]}, facet=False, figsize=(8.0, 4.0)
        )
        assert fig.get_size_inches()[0] == 8.0
        assert fig.get_size_inches()[1] == 4.0
        plt.close(fig)

    def test_facet_rows_and_cols_layout(self):
        data = {"a": [1, 2], "b": [2, 3], "c": [3, 4], "d": [4, 5]}

        fig, axes = plot_series(data, facet=True, facet_rows=2, facet_cols=2)
        assert len(axes) == 4
        assert fig.axes.count is not None
        assert [ax.get_visible() for ax in fig.axes][:4] == [True] * 4
        plt.close(fig)

        fig, axes = plot_series(
            data, facet=True, facet_cols=3, figsize=(12.0, 8.0)
        )
        assert len(axes) == 4
        assert fig.get_size_inches()[0] == 12.0
        assert fig.get_size_inches()[1] == 8.0
        plt.close(fig)

    def test_rejects_facet_grid_too_small(self):
        with pytest.raises(ValueError, match="cannot fit"):
            plot_series(
                {"a": [1], "b": [2], "c": [3]},
                facet=True,
                facet_rows=1,
                facet_cols=2,
            )
        with pytest.raises(ValueError, match="facet_rows must be"):
            plot_series({"a": [1, 2]}, facet=True, facet_rows=0)

    def test_legend_title_and_cols(self):
        data = {"a": [1, 2, 3], "b": [2, 3, 4], "c": [3, 4, 5], "d": [4, 5, 6]}

        fig, ax = plot_series(
            data,
            facet=False,
            legend_title="变量",
            legend_cols=2,
        )
        legend = ax.get_legend()
        assert legend is not None
        assert legend.get_title().get_text() == "变量"
        assert legend._ncols == 2
        plt.close(fig)

        with pytest.raises(ValueError, match="legend_cols must be"):
            plot_series({"a": [1, 2]}, facet=False, legend_cols=0)

    def _legend_window(self, fig, legend):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        x0, y0 = inv.transform(legend.get_window_extent(renderer).p0)
        x1, y1 = inv.transform(legend.get_window_extent(renderer).p1)
        return x0, y0, x1, y1

    def test_legend_default_below_time_axis(self):
        fig, ax = plot_series(
            {"a": [1, 2, 3], "b": [2, 2.5, 3]},
            facet=False,
        )
        legend = ax.get_legend()
        assert legend is not None
        # 默认图例在时间轴（x 轴）下方、绘图区外：图例顶部低于轴底线。
        _, _, _, legend_top = self._legend_window(fig, legend)
        assert legend_top < ax.get_position().y0
        # 底部排布自动把 2 条记录排成一行（2 列），保持高度紧凑。
        assert legend._ncols == 2
        plt.close(fig)

        # 显式传入位置时回到绘图区内（旧默认行为）。
        fig2, ax2 = plot_series(
            {"a": [1, 2, 3], "b": [2, 2.5, 3]},
            facet=False,
            legend_loc="upper left",
        )
        legend2 = ax2.get_legend()
        _, legend2_bottom, _, _ = self._legend_window(fig2, legend2)
        assert legend2_bottom > ax2.get_position().y0
        plt.close(fig2)

    def test_legend_below_year_ruler(self):
        idx = pd.date_range("2020-01", periods=24, freq="MS")
        rng = np.random.default_rng(1)
        fig, ax = plot_series(
            pd.DataFrame(
                {
                    "a": rng.normal(size=24),
                    "b": rng.normal(size=24),
                },
                index=idx,
            ),
            facet=False,
            year_ruler=True,
        )
        legend = ax.get_legend()
        assert legend is not None
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv_ax = ax.transAxes.inverted()
        # year_ruler：图例顶部必须低于年份标尺标签（轴下方最靠下的元素）。
        legend_top_axes = inv_ax.transform(legend.get_window_extent(renderer).p1)[1]
        year_label_bottom = min(
            inv_ax.transform(text.get_window_extent(renderer).p0)[1]
            for text in ax.texts
        )
        assert legend_top_axes < year_label_bottom
        plt.close(fig)

    def test_facet_shares_single_bottom_legend(self):
        fig, axes = plot_series(
            {"a": [1, 2, 3], "b": [2, 3, 4]}, facet=True
        )
        bottom_panel = min(axes, key=lambda a: a.get_position().y0)
        top_panel = max(axes, key=lambda a: a.get_position().y0)
        legend = bottom_panel.get_legend()
        assert legend is not None
        # 分面时整图共享一个底部图例，面板内不再有图例。
        assert top_panel.get_legend() is None
        _, _, _, legend_top = self._legend_window(fig, legend)
        assert legend_top < bottom_panel.get_position().y0
        plt.close(fig)

    def test_bottom_legend_auto_balances_grid(self):
        # 4 条记录自动排成 2×2（近方形网格），避免长文字挤成过宽单行。
        data = {"一": [1, 2, 3], "二": [2, 3, 4], "三": [3, 4, 5], "四": [4, 5, 6]}
        fig, ax = plot_series(data, facet=False)
        legend = ax.get_legend()
        assert legend is not None
        assert legend._ncols == 2
        # 4 个条目配 2 列 ⇒ 2 行，即 2×2 网格。
        assert len(legend.get_texts()) == 4
        plt.close(fig)

        # 显式 legend_cols 优先于自动排布。
        fig2, ax2 = plot_series(data, facet=False, legend_cols=4)
        assert ax2.get_legend()._ncols == 4
        plt.close(fig2)

    def test_yaxis_title_unit_only_by_default(self):
        # 轴标题默认只显示单位，不再显示默认 "Value"。
        fig, ax = plot_series({"a": [1, 2, 3]}, facet=False, unit="亿元")
        assert ax.get_ylabel() == "（单位：亿元）"
        plt.close(fig)

        # 双轴：右轴标题同样只显示单位，不携带变量名。
        fig2, ax2 = plot_series(
            {"a": [1, 2, 3], "b": [100, 200, 300]},
            facet=False,
            unit="亿元",
        )
        assert ax2.get_ylabel() == "（单位：亿元）"
        assert ax2.right_ax.get_ylabel() == "（单位：亿元）"
        plt.close(fig2)

        # 无 unit 且不传 ytitle：不显示 y 轴标题。
        fig3, ax3 = plot_series({"a": [1, 2, 3]}, facet=False)
        assert ax3.get_ylabel() == ""
        plt.close(fig3)

        # 显式 ytitle：变量名/自定义标题生效，单位追加在末尾。
        fig4, ax4 = plot_series(
            {"a": [1, 2, 3]}, facet=False, ytitle="GDP", unit="亿元"
        )
        assert ax4.get_ylabel() == "GDP（单位：亿元）"
        plt.close(fig4)

        # 显式第二轴标题：保留并追加单位。
        fig5, ax5 = plot_series(
            {"a": [1, 2, 3], "b": [100, 200, 300]},
            facet=False,
            second_axis_vars=["b"],
            unit="亿元",
            second_axis_title="右轴标题",
        )
        assert ax5.right_ax.get_ylabel() == "右轴标题（单位：亿元）"
        plt.close(fig5)

    def test_facet_yaxis_unit_only_by_default(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
        fig, axes = plot_series(df, unit="%")
        for panel in axes:
            assert panel.get_ylabel() == "（单位：%）"
        plt.close(fig)

    def test_per_series_units_drive_overlay_axis_titles(self):
        fig, ax = plot_series(
            {"price": [10, 11, 12], "revenue": [100, 110, 120]},
            facet=False,
            axis_groups={"price": "left", "revenue": "right"},
            units={"price": "美元/桶", "revenue": "亿美元"},
        )

        assert ax.get_ylabel() == "美元/桶"
        assert ax.right_ax.get_ylabel() == "亿美元"
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "price（左轴）",
            "revenue（右轴）",
        ]
        plt.close(fig)

    def test_per_series_units_reject_mixed_units_on_one_axis(self):
        with pytest.raises(ValueError, match="same unit"):
            plot_series(
                {"a": [1, 2], "b": [2, 3]},
                facet=False,
                auto_dual_y=False,
                units={"a": "人", "b": "家"},
            )

    def test_per_series_units_work_for_facets(self):
        fig, axes = plot_series(
            {"a": [1, 2], "b": [2, 3]},
            units={"a": "人", "b": "家"},
        )
        assert [axis.get_ylabel() for axis in axes] == ["人", "家"]
        plt.close(fig)

    def test_format_compact_y_axis_applies_descending_domain_rules(self):
        fig, ax = plot_series(
            {"amount": [100_000, 200_000]},
            facet=False,
            units={"amount": "百万迪拉姆"},
        )
        ax.set_ylim(0, 1_000_000)
        format_compact_y_axis(
            ax,
            unit="百万迪拉姆",
            rules={
                "百万迪拉姆": (
                    (100_000, 1_000_000, "万亿迪拉姆"),
                    (0, 100, "亿迪拉姆"),
                )
            },
        )
        assert ax.get_ylabel() == "万亿迪拉姆"
        assert ax.yaxis.get_major_formatter()(1_000_000, 0) == "1"
        plt.close(fig)

    def test_dual_axis_legend_role_suffix(self):
        # 双轴图：图例文字统一为「变量名（左轴/右轴）」。
        fig, ax = plot_series(
            {"a": [1, 2, 3], "b": [100, 200, 300]},
            facet=False,
            axis_groups={"a": "left", "b": "right"},
        )
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            "a（左轴）",
            "b（右轴）",
        ]
        plt.close(fig)

        # 单轴叠加：不带括号后缀。
        fig2, ax2 = plot_series(
            {"a": [1, 2, 3], "b": [2, 3, 4]}, facet=False
        )
        assert [t.get_text() for t in ax2.get_legend().get_texts()] == ["a", "b"]
        plt.close(fig2)

        # 显式 legend_labels：按原文使用，不加后缀。
        fig3, ax3 = plot_series(
            {"a": [1, 2, 3], "b": [100, 200, 300]},
            facet=False,
            axis_groups={"a": "left", "b": "right"},
            legend_labels=["甲", "乙"],
        )
        assert [t.get_text() for t in ax3.get_legend().get_texts()] == ["甲", "乙"]
        plt.close(fig3)

        # 显式 legend_loc（图例回绘图区内）同样应用（左轴/右轴）后缀。
        fig4, ax4 = plot_series(
            {"a": [1, 2, 3], "b": [100, 200, 300]},
            facet=False,
            axis_groups={"a": "left", "b": "right"},
            legend_loc="upper left",
        )
        assert [t.get_text() for t in ax4.get_legend().get_texts()] == [
            "a（左轴）",
            "b（右轴）",
        ]
        plt.close(fig4)

    def test_single_series_has_no_legend_by_default(self):
        """模板契约：单序列图默认不显示图例（一条线/柱无需图例）。"""
        # 单序列默认无图例。
        fig, ax = plot_series({"a": [1, 2, 3]}, facet=False)
        assert ax.get_legend() is None
        plt.close(fig)

        # 单序列 + 显式 legend_labels：仍显示（允许用户自定义单序列图例文字）。
        fig2, ax2 = plot_series(
            {"a": [1, 2, 3]}, facet=False, legend_labels=["自定义甲"]
        )
        legend2 = ax2.get_legend()
        assert legend2 is not None
        assert [t.get_text() for t in legend2.get_texts()] == ["自定义甲"]
        plt.close(fig2)

        # 多序列不受影响，仍显示图例。
        fig3, ax3 = plot_series({"a": [1, 2], "b": [2, 3]}, facet=False)
        assert ax3.get_legend() is not None
        plt.close(fig3)

    def test_multiple_bar_series_are_side_by_side(self):
        """模板契约：同一时间点存在多根柱（多个 bar_series）时并排错开。"""
        # 双 bar 系列分居左右轴：同一时间点上两根柱 x 起点不同（并排而非重叠），
        # 且显式 bar_width 被对半分给两根柱（2 根并排、组宽保持不变）。
        width = 10.0
        fig, ax = plot_series(
            {"a": [1, 2, 3], "b": [100, 200, 300]},
            facet=False,
            axis_groups={"a": "left", "b": "right"},
            bar_series=["a", "b"],
            bar_width=width,
        )
        left_x = [p.get_x() for p in ax.patches]
        right_x = [p.get_x() for p in ax.right_ax.patches]
        assert left_x and right_x
        assert all(l != r for l, r in zip(left_x, right_x))
        assert ax.patches[0].get_width() == pytest.approx(width / 2)
        assert ax.right_ax.patches[0].get_width() == pytest.approx(width / 2)
        plt.close(fig)

        # 单 bar 系列：柱宽不变、不偏移（柱仍居中于数据点）。
        fig2, ax2 = plot_series(
            {"a": [1, 2, 3]},
            facet=False,
            bar_series=["a"],
            bar_width=width,
        )
        assert ax2.patches[0].get_width() == pytest.approx(width)
        # 数据点 2 → 柱中心 2（x 起点 = 2 - width/2）。
        assert ax2.patches[2].get_x() + ax2.patches[2].get_width() / 2 == 2
        plt.close(fig2)

    def test_note_below_legend_in_bottom_margin(self):
        fig, ax = plot_series(
            {"a": [1, 2, 3], "b": [2, 3, 4]},
            facet=False,
            note="数据来源：原始数据",
        )
        legend = ax.get_legend()
        assert legend is not None
        _, legend_bottom, _, _ = self._legend_window(fig, legend)
        texts = [t for t in fig.texts if t.get_text() == "数据来源：原始数据"]
        assert texts
        y = texts[0].get_position()[1]
        # 图注紧跟图例下方，且仍在图形内（图例→图注 自下而上堆叠）。
        assert y < legend_bottom
        assert y > 0
        assert texts[0].get_va() == "top"
        plt.close(fig)

    def test_note_loc_and_prefix(self):
        fig, _ax = plot_series(
            {"a": [1, 2, 3]},
            facet=False,
            note="原始数据",
            note_loc="center",
            note_prefix="数据来源：",
        )
        texts = [
            t for t in fig.texts if t.get_text() == "数据来源：原始数据"
        ]
        assert texts
        assert texts[0].get_horizontalalignment() == "center"
        assert texts[0].get_position()[0] == 0.5
        plt.close(fig)

        with pytest.raises(ValueError, match="note_loc"):
            plot_series({"a": [1, 2]}, facet=False, note="x", note_loc="top")

    def test_note_prefix_facet(self):
        fig, _axes = plot_series(
            {"a": [1, 2], "b": [2, 3]},
            facet=True,
            note="原始数据",
            note_prefix="数据来源：",
        )
        assert any(
            t.get_text() == "数据来源：原始数据" for t in fig.texts
        )
        plt.close(fig)

    @pytest.mark.parametrize(
        ("axis_groups", "max_y_axes", "message"),
        [
            (["a", "b"], 3, "axis_groups must be a mapping"),
            ({"a": 0}, 3, "axis_groups labels must exactly match"),
            ({"a": 0, "b": 0, "extra": 1}, 3, "axis_groups labels must exactly match"),
            ({"a": [], "b": []}, 3, "axis_groups values must be hashable"),
            ({"a": 0, "b": 1}, 1, "axis_groups defines 2 groups"),
        ],
    )
    def test_rejects_invalid_manual_axis_groups(
        self,
        axis_groups,
        max_y_axes,
        message,
    ):
        with pytest.raises((TypeError, ValueError), match=message):
            plot_series(
                {"a": [1, 2], "b": [2, 3]},
                facet=False,
                axis_groups=axis_groups,
                max_y_axes=max_y_axes,
            )

    def test_manual_axis_groups_require_overlay_mode(self):
        with pytest.raises(ValueError, match="axis_groups requires facet=False"):
            plot_series(
                {"a": [1, 2], "b": [2, 3]},
                axis_groups={"a": 0, "b": 1},
            )

    def test_multiple_series_facet_rejects_existing_axes(self):
        fig, ax = plt.subplots()

        with pytest.raises(ValueError, match="facet=False"):
            plot_series({"a": [1, 2], "b": [2, 3]}, ax=ax)

        plt.close(fig)

    @pytest.mark.parametrize(
        ("keyword", "value", "message"),
        [
            ("facet", 1, "facet must be a boolean"),
            ("sharex", "yes", "sharex must be a boolean"),
            ("sharey", None, "sharey must be a boolean"),
            ("auto_dual_y", 0, "auto_dual_y must be a boolean"),
            (
                "scale_ratio_threshold",
                0,
                "scale_ratio_threshold must be positive",
            ),
            ("max_y_axes", 0, "max_y_axes must be at least 1"),
            ("max_y_axes", 1.5, "max_y_axes must be an integer"),
        ],
    )
    def test_rejects_invalid_layout_options(self, keyword, value, message):
        with pytest.raises((TypeError, ValueError), match=message):
            plot_series({"a": [1, 2], "b": [2, 3]}, **{keyword: value})

    def test_title_and_labels(self):
        data = np.random.randn(20)
        fig, ax = plot_series(data, title="Test Title", xtitle="X", ytitle="Y")
        assert ax.get_title() == "Test Title"
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        plt.close(fig)

    def test_grid(self):
        data = np.random.randn(20)
        fig, _ax = plot_series(data, grid=True)
        # grid should be visible
        plt.close(fig)

    def test_grid_axis_controls_direction(self):
        data = np.random.randn(20)
        fig, ax = plot_series(data, grid=True, grid_axis="x")
        assert any(line.get_visible() for line in ax.get_xgridlines())
        assert not any(line.get_visible() for line in ax.get_ygridlines())
        plt.close(fig)

        fig, ax = plot_series(data, grid=True, grid_axis="y")
        assert any(line.get_visible() for line in ax.get_ygridlines())
        assert not any(line.get_visible() for line in ax.get_xgridlines())
        plt.close(fig)

    def test_grid_width_and_linestyle(self):
        data = np.random.randn(20)
        fig, ax = plot_series(
            data, grid=True, grid_linewidth=1.5, grid_linestyle=":"
        )
        fig.canvas.draw()
        line = next(l for l in ax.get_ygridlines() if l.get_visible())
        assert line.get_linewidth() == 1.5
        assert line.get_linestyle() == ":"
        plt.close(fig)

    def test_rejects_invalid_grid_axis(self):
        with pytest.raises(ValueError, match="grid_axis"):
            plot_series([1, 2, 3], grid=True, grid_axis="z")

    def test_vlines_and_shade(self):
        data = np.random.randn(30)
        fig, _ax = plot_series(data, vlines=15, shade=(10, 20))
        plt.close(fig)

    def test_rejects_color_count_mismatch(self):
        data = {"a": [1, 2], "b": [2, 3]}
        with pytest.raises(ValueError, match="colors has 1 entries"):
            plot_series(data, colors=["red"])

    def test_rejects_y_for_non_dataframe_input(self):
        with pytest.raises(TypeError, match="y requires a DataFrame"):
            plot_series([1, 2, 3], y="series")

    def test_bar_series_draws_rectangles_without_lines(self):
        data = {"volume": [1, 2, 3], "price": [10, 20, 30]}

        fig, ax = plot_series(
            data,
            facet=False,
            auto_dual_y=False,
            bar_series=["volume"],
            colors=["#aaaaaa", "#1f4e79"],
        )

        assert len(ax.patches) == 3
        assert [patch.get_height() for patch in ax.patches] == [1, 2, 3]
        assert [line.get_label() for line in ax.lines] == ["price"]
        assert ax.get_ylim()[0] == 0
        plt.close(fig)

    def test_bar_series_on_secondary_axis_starts_at_zero(self):
        data = {
            "volume": [1, 2, 3],
            "price": [1000, 2000, 3000],
        }

        fig, ax = plot_series(
            data,
            facet=False,
            axis_groups={"volume": "left", "price": "right"},
            bar_series=["volume"],
        )

        assert len(ax.patches) == 3
        assert ax.right_ax is not None
        assert [line.get_label() for line in ax.right_ax.lines] == ["price"]
        assert ax.get_ylim()[0] == 0
        plt.close(fig)

    def test_bar_series_in_facet_panels(self):
        data = {"volume": [1, 2, 3], "price": [10, 20, 30]}

        fig, axes = plot_series(data, bar_series=["volume"])

        assert len(axes[0].patches) == 3
        assert len(axes[1].lines) == 1
        assert axes[0].get_ylim()[0] == 0
        plt.close(fig)

    def test_bar_series_auto_width_uses_data_spacing(self):
        dates = pd.date_range("2025-01-01", periods=3, freq="30D")

        fig, ax = plot_series(
            pd.Series([1, 2, 3], index=dates, name="volume"),
            bar_series=["volume"],
            facet=False,
        )

        assert ax.patches[0].get_width() == pytest.approx(0.6 * 30)
        plt.close(fig)

    def test_bar_face_color_overrides_bars_but_keeps_lines(self):
        data = {"volume": [1, 2, 3], "price": [10, 20, 30]}

        fig, ax = plot_series(
            data,
            facet=False,
            auto_dual_y=False,
            bar_series=["volume"],
            bar_face_color="#888888",
        )

        assert len(ax.patches) == 3
        assert all(
            tuple(patch.get_facecolor())[:3]
            == pytest.approx((136 / 255, 136 / 255, 136 / 255))
            for patch in ax.patches
        )
        # 线保持自己的调色板色，不被柱色覆盖。
        line_color = ax.lines[0].get_color()
        assert line_color != "#888888"
        plt.close(fig)

    def test_bar_face_color_applies_in_facet_panels(self):
        data = {"volume": [1, 2, 3], "price": [10, 20, 30]}

        fig, axes = plot_series(
            data,
            bar_series=["volume"],
            bar_face_color="#888888",
        )

        assert len(axes[0].patches) == 3
        assert all(
            tuple(patch.get_facecolor())[:3]
            == pytest.approx((136 / 255, 136 / 255, 136 / 255))
            for patch in axes[0].patches
        )
        assert len(axes[0].lines) == 0
        assert len(axes[1].lines) == 1
        plt.close(fig)

    def test_bar_series_rejects_unknown_labels(self):
        with pytest.raises(ValueError, match="bar_series labels must be plotted"):
            plot_series(
                {"a": [1, 2]},
                bar_series=["missing"],
            )

    def test_year_ruler_places_month_ticks_and_year_labels(self):
        dates = pd.date_range("2025-01-31", periods=24, freq="ME")

        fig, ax = plot_series(
            pd.Series(range(24), index=dates, name="series"),
            year_ruler=True,
        )

        labels = [label.get_text() for label in ax.get_xticklabels()]
        assert "3月" in labels and "12月" in labels
        assert all("2025" not in label for label in labels)
        assert " 2025年 " in {
            text.get_text() for text in ax.texts
        }
        assert " 2026年 " in {text.get_text() for text in ax.texts}
        assert all(label.get_rotation() == 0 for label in ax.get_xticklabels())
        plt.close(fig)

    def test_vlines_outside_data_range_are_skipped(self):
        fig, ax = plot_series(
            pd.Series(range(10), index=range(10), name="series"),
            vlines=50,
        )

        assert len(ax.get_lines()) == 1
        plt.close(fig)

    def test_vlines_inside_data_range_are_drawn(self):
        fig, ax = plot_series(
            pd.Series(range(10), index=range(10), name="series"),
            vlines=5,
        )

        assert len(ax.get_lines()) == 2
        plt.close(fig)

    def test_vlines_with_dates_outside_range_are_skipped(self):
        dates = pd.date_range("2025-01-31", periods=6, freq="ME")

        fig, ax = plot_series(
            pd.Series([1, 2, 3, 4, 5, 6], index=dates, name="series"),
            vlines=pd.Timestamp("2026-03-01"),
        )

        assert len(ax.get_lines()) == 1
        plt.close(fig)


class TestPlotScatter:
    def test_returns_fig_ax(self):
        fig, ax = plot_scatter(x=np.random.randn(30), y=np.random.randn(30))
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_with_fit_line(self):
        fig, _ax = plot_scatter(
            x=np.random.randn(30), y=np.random.randn(30), fit_line=True
        )
        plt.close(fig)

    def test_legend_default_below_axis(self):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6],
                "y": [1, 1, 2, 2, 3, 3],
                "g": ["a"] * 3 + ["b"] * 3,
            }
        )
        fig, ax = plot_scatter(df, x="x", y="y", group="g")
        legend = ax.get_legend()
        assert legend is not None
        # 默认图例在 x 轴下方、绘图区外，与 plot_series 一致。
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        legend_top = inv.transform(legend.get_window_extent(renderer).p1)[1]
        assert legend_top < ax.get_position().y0
        plt.close(fig)

    def test_title(self):
        fig, ax = plot_scatter(
            x=np.random.randn(30), y=np.random.randn(30), title="Test Title"
        )
        assert ax.get_title() == "Test Title"
        plt.close(fig)

    def test_dataframe_with_group(self):
        df = pd.DataFrame(
            {
                "x": np.random.randn(60),
                "y": np.random.randn(60),
                "g": ["A"] * 30 + ["B"] * 30,
            }
        )
        fig, _ax = plot_scatter(df, x="x", y="y", group="g")
        plt.close(fig)

    def test_rejects_nonpositive_tick_steps(self):
        with pytest.raises(ValueError, match="xtick_step must be positive"):
            plot_scatter(x=[1, 2], y=[2, 3], xtick_step=0)
        with pytest.raises(ValueError, match="ytick_step must be positive"):
            plot_scatter(x=[1, 2], y=[2, 3], ytick_step=0)

    def test_rejects_color_count_mismatch(self):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": [1, 2, 3, 4],
                "g": ["a", "a", "b", "b"],
            }
        )
        with pytest.raises(ValueError, match="colors has 1 entries"):
            plot_scatter(df, x="x", y="y", group="g", colors=["red"])


class TestPlotACF:
    def test_default_nlags_adapts_to_sample_size(self):
        data = np.random.randn(60)

        fig, ax = plot_acf(data)

        assert len(ax.patches) == 18  # lag 0 through floor(10 * log10(60))
        plt.close(fig)

    def test_returns_fig_ax(self):
        data = np.random.randn(100)
        fig, _ax = plot_acf(data, nlags=10)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_existing_axes_receive_resolved_fonts(self):
        fig, ax = plt.subplots()
        ax.xaxis.label.set_fontfamily(["DejaVu Sans"])
        ax.yaxis.label.set_fontfamily(["DejaVu Sans"])

        plot_acf(np.random.randn(100), nlags=10, ax=ax, title="Residual ACF")

        assert ax.xaxis.label.get_fontfamily() == _body_font_family()
        assert ax.yaxis.label.get_fontfamily() == _body_font_family()
        assert ax.title.get_fontfamily() == _title_font_family()
        plt.close(fig)

    def test_default_missing_drop_uses_cleaned_sample_for_adaptive_nlags(self):
        data = np.random.default_rng(42).normal(size=63)
        data[[0, 17, 62]] = np.nan

        fig, ax = plot_acf(data)

        assert len(ax.patches) == 18
        plt.close(fig)


class TestPlotPACF:
    def test_default_nlags_adapts_to_sample_size(self):
        data = np.random.randn(60)

        fig, ax = plot_pacf(data)

        assert len(ax.patches) == 17  # lag 0 is computed but not displayed
        plt.close(fig)

    def test_returns_fig_ax(self):
        data = np.random.randn(100)
        fig, _ax = plot_pacf(data, nlags=10)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_explicit_nlags_preserves_statsmodels_validation(self):
        with pytest.raises(ValueError, match="requested nlags 40"):
            plot_pacf(np.random.randn(60), nlags=40)

    def test_default_missing_drop_uses_cleaned_sample_for_adaptive_nlags(self):
        data = np.random.default_rng(42).normal(size=63)
        data[[0, 17, 62]] = np.nan

        fig, ax = plot_pacf(data)

        assert len(ax.patches) == 17
        plt.close(fig)


@pytest.mark.parametrize("plotter", [plot_acf, plot_pacf])
class TestCorrelogramMissingPolicy:
    def test_default_drop_matches_explicitly_cleaned_series(self, plotter):
        data = pd.Series(
            [1.0, -0.5, np.nan, 0.25, np.inf, 1.5, -np.inf, -1.0, 0.75, 2.0]
        )
        original = data.copy()

        fig, ax = plotter(data, nlags=2)
        expected_fig, expected_ax = plotter(data[np.isfinite(data)], nlags=2)

        np.testing.assert_allclose(
            [patch.get_height() for patch in ax.patches],
            [patch.get_height() for patch in expected_ax.patches],
        )
        pd.testing.assert_series_equal(data, original)
        plt.close(fig)
        plt.close(expected_fig)

    def test_raise_reports_original_non_finite_positions(self, plotter):
        data = np.array([1.0, np.nan, 2.0, np.inf, 3.0])

        with pytest.raises(ValueError, match="row positions: 1, 3"):
            plotter(data, missing="raise")

    def test_unknown_policy_is_rejected(self, plotter):
        with pytest.raises(ValueError, match="missing must be 'raise' or 'drop'"):
            plotter(np.arange(20.0), missing="omit")

    def test_all_non_finite_data_is_rejected(self, plotter):
        with pytest.raises(ValueError, match="no finite observations"):
            plotter(np.array([np.nan, np.inf, -np.inf]))


class TestStyleConstants:
    def test_palette_has_eight_colors(self):
        assert len(DEFAULT_PALETTE) == 8

    def test_palette_cycle_led_by_requested_main_colors(self):
        # 调色模板主色：黑 / 深蓝 / 灰 / 深红，必须引导默认 8 色循环。
        assert DEFAULT_PALETTE[:4] == [BLACK, DARK_BLUE, GRAY, DARK_RED]

    def test_linestyles_has_eight(self):
        assert len(DEFAULT_LINESTYLES) == 8

    def test_markers_has_eight(self):
        assert len(DEFAULT_MARKERS) == 8

    def test_markers_are_distinct(self):
        assert len(set(DEFAULT_MARKERS)) == len(DEFAULT_MARKERS)


class TestPlotCorrelogram:
    def test_precomputed_series_uses_supplied_values_and_band(self):
        values = pd.Series([0.1, -0.3, 0.5], index=[0, 1, 2], name="price")

        fig, ax = plot_correlogram(values, confidence_band=0.2)

        np.testing.assert_allclose(
            [patch.get_height() for patch in ax.patches], values.to_numpy()
        )
        assert ax.get_title() == "price"
        assert len(ax.collections) == 1
        plt.close(fig)

    def test_precomputed_correlogram_accepts_external_axis_and_varying_band(self):
        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot_correlogram(
            [0.1, 0.2, -0.1],
            confidence_band=[0.3, 0.25, 0.2],
            ax=ax,
            title="Residual CCF",
        )

        assert returned_fig is fig
        assert returned_ax is ax
        assert ax.get_title() == "Residual CCF"
        plt.close(fig)

    def test_precomputed_multicolumn_correlogram_creates_facets(self):
        values = pd.DataFrame(
            {"price": [0.2, 0.1, -0.1], "income": [-0.2, 0.3, 0.1]},
            index=pd.RangeIndex(3, name="lag"),
        )
        bands = pd.DataFrame(0.18, index=values.index, columns=values.columns)

        fig, axes = plot_correlogram(
            values,
            confidence_band=bands,
            title="Residual cross-correlations",
        )

        assert isinstance(axes, np.ndarray)
        assert axes.shape == (2,)
        assert [axis.get_title() for axis in axes] == ["price", "income"]
        assert fig._suptitle.get_text() == "Residual cross-correlations"
        plt.close(fig)

    @pytest.mark.parametrize(
        ("values", "band", "match"),
        [
            ([0.1, 0.2], [-0.1, 0.2], "non-negative"),
            ([0.1, 0.2], [0.1], "match"),
            (pd.Series([0.1, 0.2], index=[1, 0]), 0.2, "increasing"),
        ],
    )
    def test_precomputed_correlogram_validates_lags_and_bands(
        self, values, band, match
    ):
        with pytest.raises(ValueError, match=match):
            plot_correlogram(values, confidence_band=band)


class TestPlotCorrelationMatrix:
    def test_dataframe_uses_labels_annotations_and_fixed_scale(self):
        matrix = pd.DataFrame(
            [[1.0, -0.75], [-0.75, 1.0]],
            index=["alpha[1]", "beta[1]"],
            columns=["alpha[1]", "beta[1]"],
        )

        fig, ax = plot_correlation_matrix(matrix)

        assert fig is ax.figure
        assert [tick.get_text() for tick in ax.get_xticklabels()] == list(matrix)
        assert len(ax.texts) == 4
        assert ax.images[0].get_clim() == (-1.0, 1.0)
        plt.close(fig)

    def test_array_accepts_labels_and_existing_axes(self):
        fig, ax = plt.subplots()

        returned_fig, returned_ax = plot_correlation_matrix(
            [[1.0, 0.25], [0.25, 1.0]],
            labels=["ar.L1", "ma.L1"],
            annotate=False,
            ax=ax,
        )

        assert returned_fig is fig
        assert returned_ax is ax
        assert not ax.texts
        plt.close(fig)

    @pytest.mark.parametrize(
        ("matrix", "kwargs", "message"),
        [
            ([[1.0, 0.2]], {}, "square"),
            ([[1.0, 1.2], [1.2, 1.0]], {}, "between -1 and 1"),
            ([[1.0, 0.2], [0.3, 1.0]], {}, "symmetric"),
            ([[0.9, 0.2], [0.2, 1.0]], {}, "diagonal"),
            ([[1.0, 0.2], [0.2, 1.0]], {"labels": ["one"]}, "length"),
        ],
    )
    def test_validates_matrix_contract(self, matrix, kwargs, message):
        with pytest.raises(ValueError, match=message):
            plot_correlation_matrix(matrix, **kwargs)
