import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from Ts.TsPlots import plot_bar, plot_series
from Ts.TsPlots.style import (
    BAR_EDGE_COLOR,
    _body_font_family,
    _title_font_family,
)
from matplotlib.colors import to_rgb

BAR_EDGE_RGB = tuple(to_rgb(BAR_EDGE_COLOR))


class TestPlotBar:
    def test_returns_fig_ax(self):
        fig, ax = plot_bar(pd.Series([1, 2, 3], index=["a", "b", "c"]))
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_single_series_bars_start_at_zero(self):
        series = pd.Series([1, 2, 3], index=["a", "b", "c"], name="x")

        fig, ax = plot_bar(series)

        assert len(ax.patches) == 3
        assert [patch.get_height() for patch in ax.patches] == [1, 2, 3]
        assert [patch.get_width() for patch in ax.patches] == [0.6] * 3
        assert ax.get_ylim()[0] == 0
        plt.close(fig)

    def test_unnamed_series_uses_default_label(self):
        fig, ax = plot_bar(pd.Series([1, 2]))
        legend = ax.get_legend()
        assert legend is not None
        assert legend.get_texts()[0].get_text() == "Series 1"
        plt.close(fig)

    def test_dataframe_columns_become_grouped_series(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [10, 20, 30]})

        fig, ax = plot_bar(df)

        assert len(ax.patches) == 6
        # ax.bar batches per series: series A occupies the left offset (-0.15),
        # series B the right (+0.15); each bar is 0.6/2 = 0.3 wide.
        assert [patch.get_x() for patch in ax.patches[:3]] == pytest.approx(
            [-0.30, 0.70, 1.70]
        )
        assert [patch.get_x() for patch in ax.patches[3:]] == pytest.approx(
            [0.00, 1.00, 2.00]
        )
        assert all(patch.get_width() == pytest.approx(0.3) for patch in ax.patches)
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "A",
            "B",
        ]
        plt.close(fig)

    def test_dataframe_x_column_provides_categories(self):
        df = pd.DataFrame(
            {"cat": ["北", "南"], "产量": [120, 90], "销量": [60, 40]}
        )

        fig, ax = plot_bar(df, x="cat")

        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["北", "南"]
        assert len(ax.patches) == 4  # cat column excluded from series
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "产量",
            "销量",
        ]
        plt.close(fig)

    def test_dataframe_y_restricts_series_columns(self):
        df = pd.DataFrame(
            {"a": [1, 2], "b": [3, 4], "c": [5, 6]}, index=["x", "y"]
        )

        fig, ax = plot_bar(df, y=["b", "c"])

        assert len(ax.patches) == 4
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "b",
            "c",
        ]
        plt.close(fig)

    def test_long_form_group_column(self):
        long = pd.DataFrame(
            {
                "地区": ["东部", "东部", "中部", "中部"],
                "年份": ["2024", "2023", "2024", "2023"],
                "产量": [135, 120, 95, 90],
            }
        )

        fig, ax = plot_bar(long, x="地区", y="产量", group="年份")

        # First-appearance order: 东部, 中部; group first-appearance: 2024, 2023.
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["东部", "中部"]
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "2024",
            "2023",
        ]
        # Series 2024: 东部=135, 中部=95 (batched per series).
        assert [patch.get_height() for patch in ax.patches[:2]] == [135, 95]
        assert [patch.get_height() for patch in ax.patches[2:]] == [120, 90]
        plt.close(fig)

    def test_dict_and_2d_array_inputs(self):
        fig, ax = plot_bar({"s1": [1, 2], "s2": [3, 4]})
        assert len(ax.patches) == 4
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "s1",
            "s2",
        ]
        plt.close(fig)

        fig, ax = plot_bar(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert len(ax.patches) == 4
        assert [patch.get_height() for patch in ax.patches] == [1, 3, 2, 4]
        plt.close(fig)

    def test_horizontal_bars(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=["x", "y"])

        fig, ax = plot_bar(df, horizontal=True)

        assert len(ax.patches) == 4
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["x", "y"]
        assert ax.patches[0].get_width() == pytest.approx(1)
        assert ax.patches[0].get_height() == pytest.approx(0.3)
        plt.close(fig)

    def test_stacked_bars_accumulate_bottoms(self):
        df = pd.DataFrame({"a": [1, 2], "b": [10, 20]})

        fig, ax = plot_bar(df, stacked=True)

        assert len(ax.patches) == 4
        # Second series stacks on top of the first: bottom = first heights.
        assert ax.patches[2].get_y() == pytest.approx(1)
        assert ax.patches[2].get_height() == pytest.approx(10)
        assert ax.patches[3].get_y() == pytest.approx(2)
        assert ax.patches[3].get_height() == pytest.approx(20)
        plt.close(fig)

    def test_stacked_horizontal_bars(self):
        df = pd.DataFrame({"a": [1, 2], "b": [10, 20]})

        fig, ax = plot_bar(df, stacked=True, horizontal=True)

        assert ax.patches[2].get_x() == pytest.approx(1)
        assert ax.patches[2].get_width() == pytest.approx(10)
        plt.close(fig)

    def test_legend_default_below_axis(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        fig, ax = plot_bar(df)
        legend = ax.get_legend()
        assert legend is not None
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        legend_top = inv.transform(legend.get_window_extent(renderer).p1)[1]
        assert legend_top < ax.get_position().y0
        plt.close(fig)

    def test_legend_loc_moves_legend_inside_axes(self):
        fig, ax = plot_bar(
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}), legend_loc="upper left"
        )
        legend = ax.get_legend()
        assert legend is not None
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        legend_bottom = inv.transform(legend.get_window_extent(renderer).p0)[1]
        assert legend_bottom > ax.get_position().y0
        plt.close(fig)

    def test_show_values_annotates_every_bar(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        fig, ax = plot_bar(df, show_values=True)

        assert len(ax.texts) == 4
        texts = sorted(text.get_text() for text in ax.texts)
        assert texts == ["1.0", "2.0", "3.0", "4.0"]
        plt.close(fig)

    def test_value_decimals_controls_annotation_format(self):
        fig, ax = plot_bar(
            pd.Series([1.25, 2.5], index=["a", "b"]),
            show_values=True,
            value_decimals=2,
        )
        assert sorted(text.get_text() for text in ax.texts) == [
            "1.25",
            "2.50",
        ]
        plt.close(fig)

    def test_tick_thinning_for_many_categories(self):
        series = pd.Series(np.arange(30.0), index=[f"c{i}" for i in range(30)])

        fig, ax = plot_bar(series, max_ticks=10)

        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        non_empty = [label for label in labels if label != ""]
        assert len(non_empty) <= 10
        assert non_empty[0] == "c0"
        assert len(ax.patches) == 30  # every bar still drawn
        plt.close(fig)

    def test_negative_values_draw_below_zero(self):
        fig, ax = plot_bar(pd.Series([2, -1, 3], index=["a", "b", "c"]))

        patch = ax.patches[1]
        assert patch.get_height() == pytest.approx(-1)
        assert ax.get_ylim()[0] < 0
        plt.close(fig)

    def test_existing_axes_receive_resolved_fonts(self):
        fig, ax = plt.subplots()
        ax.xaxis.label.set_fontfamily(["DejaVu Sans"])
        ax.yaxis.label.set_fontfamily(["DejaVu Sans"])

        returned_fig, returned_ax = plot_bar(
            pd.DataFrame({"a": [1, 2]}), ax=ax, title="产量"
        )

        assert returned_fig is fig
        assert returned_ax is ax
        assert ax.xaxis.label.get_fontfamily() == _body_font_family()
        assert ax.yaxis.label.get_fontfamily() == _body_font_family()
        assert ax.title.get_fontfamily() == _title_font_family()
        plt.close(fig)

    def test_grid_defaults_to_horizontal_lines(self):
        fig, ax = plot_bar(pd.Series([1, 2, 3], index=["a", "b", "c"]), grid=True)
        assert any(line.get_visible() for line in ax.get_ygridlines())
        assert not any(line.get_visible() for line in ax.get_xgridlines())
        plt.close(fig)

    def test_rejects_mismatched_series_lengths(self):
        with pytest.raises(ValueError, match="must have the same length"):
            plot_bar({"a": [1, 2], "b": [1, 2, 3]})

    def test_rejects_series_x_length_mismatch(self):
        with pytest.raises(ValueError, match="values but"):
            plot_bar(pd.Series([1, 2], index=["a", "b"]), x=["a", "b", "c"])

    @pytest.mark.parametrize(
        ("keyword", "value", "message"),
        [
            ("horizontal", 1, "horizontal must be a boolean"),
            ("stacked", "yes", "stacked must be a boolean"),
            ("show_legend", 0, "show_legend must be a boolean"),
            ("show_values", None, "show_values must be a boolean"),
        ],
    )
    def test_rejects_invalid_booleans(self, keyword, value, message):
        with pytest.raises(TypeError, match=message):
            plot_bar(pd.Series([1, 2]), **{keyword: value})

    def test_rejects_color_count_mismatch(self):
        with pytest.raises(ValueError, match="colors has 1 entries"):
            plot_bar(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), colors=["red"])

    def test_rejects_nonpositive_bar_width(self):
        with pytest.raises(ValueError, match="bar_width must be positive"):
            plot_bar(pd.Series([1, 2]), bar_width=0)
        with pytest.raises(TypeError, match="bar_width must be a positive"):
            plot_bar(pd.Series([1, 2]), bar_width="wide")

    def test_rejects_y_for_non_dataframe(self):
        with pytest.raises(TypeError, match="y requires a DataFrame"):
            plot_bar([1, 2, 3], y="series")

    def test_rejects_group_without_dataframe(self):
        with pytest.raises(ValueError, match="group requires a DataFrame"):
            plot_bar([1, 2], group="g")

    def test_rejects_unknown_columns(self):
        with pytest.raises(KeyError, match="not found"):
            plot_bar(pd.DataFrame({"a": [1]}), y="nope")
        with pytest.raises(KeyError, match="not found"):
            plot_bar(pd.DataFrame({"a": [1]}), group="g", x="a", y="a")

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="No categories"):
            plot_bar(pd.Series([], dtype=float))
        with pytest.raises(ValueError, match="no series"):
            plot_bar({})

    def test_duplicate_category_index_is_rejected(self):
        long = pd.DataFrame(
            {
                "cat": ["a", "a"],
                "g": ["x", "x"],
                "v": [1, 2],
            }
        )
        with pytest.raises(ValueError, match="duplicate entries"):
            plot_bar(long, x="cat", y="v", group="g")

    def test_bar_alpha_and_edge_style_apply(self):
        fig, ax = plot_bar(
            pd.Series([1, 2], index=["a", "b"]),
            bar_alpha=0.5,
            bar_edge_color="#888888",
            bar_edge_linewidth=1.5,
        )
        patch = ax.patches[0]
        assert patch.get_alpha() == pytest.approx(0.5)
        rgba = patch.get_edgecolor()
        assert tuple(rgba)[:3] == pytest.approx((136 / 255, 136 / 255, 136 / 255))
        assert patch.get_linewidth() == pytest.approx(1.5)
        plt.close(fig)

    def test_series_bar_delegates_to_shared_bar_drawing(self):
        """plot_series(bar_series=...) reuses the plot_bar bar implementation:

        same face colour for the first series and the same default edge colour
        (light-gray ``BAR_EDGE_COLOR``) on both functions.
        """
        data = {"volume": [1, 2, 3], "price": [10, 20, 30]}

        fig, ax = plot_series(
            data,
            facet=False,
            auto_dual_y=False,
            bar_series=["volume"],
        )

        patch = ax.patches[0]
        face = tuple(patch.get_facecolor())[:3]
        assert tuple(patch.get_edgecolor())[:3] == pytest.approx(BAR_EDGE_RGB)
        plt.close(fig)

        # Same face and edge colours as plot_bar's first series.
        fig2, ax2 = plot_bar(pd.Series([1, 2, 3], index=["a", "b", "c"]))
        assert tuple(ax2.patches[0].get_facecolor())[:3] == pytest.approx(face)
        assert tuple(ax2.patches[0].get_edgecolor())[:3] == pytest.approx(
            BAR_EDGE_RGB
        )
        plt.close(fig2)

        # Explicit None keeps the face-coloured edge override (兼容).
        fig3, ax3 = plot_series(
            data,
            facet=False,
            auto_dual_y=False,
            bar_series=["volume"],
            bar_edge_color=None,
        )
        patch3 = ax3.patches[0]
        assert tuple(patch3.get_edgecolor())[:3] == pytest.approx(
            tuple(patch3.get_facecolor())[:3]
        )
        plt.close(fig3)

    def test_unit_labels_appended(self):
        fig, ax = plot_bar(
            pd.Series([1, 2], index=["a", "b"], name="产量"),
            y_unit="万吨",
        )
        assert ax.get_ylabel() == "Value（单位：万吨）"
        plt.close(fig)