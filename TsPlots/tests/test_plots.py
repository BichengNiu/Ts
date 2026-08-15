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
    DEFAULT_LINESTYLES,
    DEFAULT_MARKERS,
    DEFAULT_PALETTE,
    _body_font_family,
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
        assert [axis.get_title(loc="left") for axis in axes] == ["small", "large"]
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

    def test_vlines_and_shade(self):
        data = np.random.randn(30)
        fig, _ax = plot_series(data, vlines=15, shade=(10, 20))
        plt.close(fig)

    def test_rejects_nonpositive_tick_step(self):
        with pytest.raises(ValueError, match="xtick_step must be positive"):
            plot_series([1, 2, 3], xtick_step=0)

    def test_rejects_color_count_mismatch(self):
        data = {"a": [1, 2], "b": [2, 3]}
        with pytest.raises(ValueError, match="colors has 1 entries"):
            plot_series(data, colors=["red"])

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

    def test_linestyles_has_eight(self):
        assert len(DEFAULT_LINESTYLES) == 8

    def test_markers_has_eight(self):
        assert len(DEFAULT_MARKERS) == 8


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
