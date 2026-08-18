"""TsPlots — Shared plotting toolkit.

This package unifies time-series plotting (:mod:`TsPlots.ts_plot`), scatter
plotting (:mod:`TsPlots.sc_plot`), and autocorrelation plotting
(:mod:`TsPlots.acf_plot`) under a common interface, sharing fonts, the default
colour template, and axis styling via :mod:`TsPlots.style`.

Main interfaces
---------------
plot_series
    Time-series line chart; accepts DataFrame / Series / dict / array-like.
plot_scatter
    Scatter chart; accepts DataFrame / dict / arrays; supports grouping and
    trend lines.
plot_acf
    Sample autocorrelation function bar chart with confidence band.
plot_pacf
    Sample partial autocorrelation function bar chart with confidence band.
plot_correlogram
    Precomputed lag correlations with supplied null confidence bands.
plot_lag_response
    Lag-indexed impulse-response bar chart with optional facets.
plot_correlation_matrix
    Labelled heatmap for a precomputed correlation matrix.

Quick start
-----------
>>> from Ts.TsPlots import plot_series, plot_scatter, plot_acf, plot_pacf
>>> fig, axes = plot_series(df, title="GDP Growth Rate")  # multi-column DataFrame
>>> fig, ax = plot_scatter(df, x="Income", y="Consumption", fit_line=True)

Advanced usage — style constants and helpers
--------------------------------------------
The default colour template leads with 黑 / 深蓝 / 灰 / 深红 (``BLACK``,
``DARK_BLUE``, ``GRAY``, ``DARK_RED``) and every other colour — text, reference
lines, shading, grid, annotation boxes — is a named role defined in
:mod:`TsPlots.style`:

>>> from Ts.TsPlots.style import (
...     DEFAULT_PALETTE, BLACK, DARK_BLUE, GRAY, DARK_RED,
...     style_axes, apply_fonts,
... )
"""

from .ts_plot import plot_series
from .sc_plot import plot_scatter
from .acf_plot import plot_acf, plot_correlogram, plot_pacf
from .lag_plot import plot_lag_response
from .matrix_plot import plot_correlation_matrix

__all__ = [
    "plot_acf",
    "plot_correlation_matrix",
    "plot_correlogram",
    "plot_lag_response",
    "plot_pacf",
    "plot_scatter",
    "plot_series",
]
