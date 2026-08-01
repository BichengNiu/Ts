import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from Ts.TsPlots import plot_series, plot_scatter, plot_acf, plot_pacf
from Ts.TsPlots.style import DEFAULT_PALETTE, DEFAULT_LINESTYLES, DEFAULT_MARKERS


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

    def test_three_series_split_at_largest_scale_gap(self):
        data = {
            "middle": [100, 200, 300],
            "small": [1, 2, 3],
            "large": [1_000_000, 2_000_000, 3_000_000],
        }

        fig, ax = plot_series(data, facet=False)

        assert [line.get_label() for line in ax.lines] == ["middle", "small"]
        assert [line.get_label() for line in ax.right_ax.lines] == ["large"]
        plt.close(fig)

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
    def test_returns_fig_ax(self):
        data = np.random.randn(100)
        fig, _ax = plot_acf(data, nlags=10)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotPACF:
    def test_returns_fig_ax(self):
        data = np.random.randn(100)
        fig, _ax = plot_pacf(data, nlags=10)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestStyleConstants:
    def test_palette_has_eight_colors(self):
        assert len(DEFAULT_PALETTE) == 8

    def test_linestyles_has_eight(self):
        assert len(DEFAULT_LINESTYLES) == 8

    def test_markers_has_eight(self):
        assert len(DEFAULT_MARKERS) == 8
