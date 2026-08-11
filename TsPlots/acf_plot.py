"""Autocorrelation and partial-autocorrelation plots for the plots package.

The two main entry points are :func:`plot_acf` and :func:`plot_pacf`. Both
follow the same style conventions as :mod:`TsPlots.ts_plot` and
:mod:`TsPlots.sc_plot`: colorblind-friendly palette, Times New Roman / FangSong
typography, hidden top/right spines, and optional dashed grid.

Accepted input
--------------
- 1-D numpy array
- pandas Series
- single-column pandas DataFrame

Confidence bands
----------------
The shaded region represents a symmetric confidence band at the requested
level (default 95 %). For ACF, Bartlett's formula is used by default so the
band width varies by lag. For PACF the band is uniform (±z / √n). Pass
``alpha=0.01`` for a 99 % band, ``alpha=0.10`` for a 90 % band, etc.

Examples
--------
>>> from Ts.TsPlots import plot_acf, plot_pacf
>>> fig, ax = plot_acf(residuals, nlags=20)
>>> fig, ax = plot_pacf(residuals, nlags=20, alpha=0.01)
>>> # embed in an existing subplot grid
>>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
>>> plot_acf(series, ax=ax1)
>>> plot_pacf(series, ax=ax2)
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tsa.stattools import pacf as sm_pacf

from .style import (
    AXIS_LABEL_FONTSIZE,
    DEFAULT_PALETTE,
    FIGSIZE,
    TITLE_FONTSIZE,
    _ensure_fonts,
    draw_note_and_bottom_title,
    style_axes,
)
from .lag_plot import _normalise_lag_response


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


def _to_1d(data) -> np.ndarray:
    """Convert supported input types to a 1-D numpy array.

    Parameters
    ----------
    data : array-like, pandas Series, or single-column pandas DataFrame
        Input data.

    Returns
    -------
    numpy.ndarray
        1-D float array.

    Raises
    ------
    ValueError
        If a DataFrame with more than one column is passed, or if the array
        is not 1-D.
    """
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError(
                f"DataFrame must have exactly one column; got {data.shape[1]}."
            )
        return data.iloc[:, 0].to_numpy(dtype=float)
    if isinstance(data, pd.Series):
        return data.to_numpy(dtype=float)
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Data must be 1-D; got shape {arr.shape}.")
    return arr


def _resolve_missing(data: np.ndarray, missing: str) -> np.ndarray:
    """Apply the correlogram missing-value policy to a 1-D array."""
    if missing not in {"raise", "drop"}:
        raise ValueError("missing must be 'raise' or 'drop'")

    finite = np.isfinite(data)
    positions = np.flatnonzero(~finite)
    if positions.size and missing == "raise":
        joined = ", ".join(str(int(position)) for position in positions)
        raise ValueError(f"data contains non-finite values at row positions: {joined}")

    cleaned = data[finite]
    if cleaned.size == 0:
        raise ValueError("data contains no finite observations")
    return cleaned


# ---------------------------------------------------------------------------
# Shared rendering core
# ---------------------------------------------------------------------------


def _draw_correlogram(
    ax,
    lags: np.ndarray,
    values: np.ndarray,
    conf_band: np.ndarray,
    *,
    bar_color: str,
    band_color: str,
    band_alpha: float,
    xtitle: str,
    ytitle: str,
    title: str | None,
    title_position: str,
    max_ticks: int,
    grid: bool,
    note: str | None,
    fig,
):
    """Render bars, zero line, and confidence band onto *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    lags : 1-D array of int
        Lag indices (x positions).
    values : 1-D array of float
        Correlation coefficients (y values).
    conf_band : 1-D array of float
        Half-width of the confidence band at each lag. Always positive. May
        be uniform (all equal) or lag-varying (Bartlett).
    bar_color : str
        Fill colour for the bars.
    band_color : str
        Fill colour of the confidence region.
    band_alpha : float
        Opacity of the confidence region (0–1).
    xtitle : str
        X-axis label.
    ytitle : str
        Y-axis label.
    title : str or None
        Optional top title.
    title_position : str
        ``"top"`` or ``"bottom"``.
    max_ticks : int
        Upper bound on the number of x-axis tick marks. When the number of
        lags exceeds this value, ``MaxNLocator`` is used so tick labels
        remain readable.
    grid : bool
        Whether to draw a dashed grid.
    note : str or None
        Figure-level note placed at the lower left.
    fig : matplotlib.figure.Figure
        Parent figure (needed for tight_layout and note/bottom-title
        placement).
    """
    # --- Confidence band ---------------------------------------------------
    # Extend half a unit beyond the first and last lag so the band covers the
    # full width of the outermost bars (bar width = 0.3, axis margin ≈ 0.5).
    x_fill = np.concatenate([[lags[0] - 0.5], lags, [lags[-1] + 0.5]])
    band_fill = np.concatenate([[conf_band[0]], conf_band, [conf_band[-1]]])
    ax.fill_between(
        x_fill,
        -band_fill,
        +band_fill,
        color=band_color,
        alpha=band_alpha,
        linewidth=0,
        zorder=0,
    )

    # --- Zero reference line -----------------------------------------------
    ax.axhline(0, color="black", linewidth=0.8, zorder=1)

    # --- Correlation bars --------------------------------------------------
    ax.bar(
        lags,
        values,
        width=0.3,
        color=bar_color,
        zorder=2,
    )

    # --- Axis labels -------------------------------------------------------
    ax.set_xlabel(xtitle, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ytitle, fontsize=AXIS_LABEL_FONTSIZE)

    # --- X-axis ticks ------------------------------------------------------
    # Show every lag when there are few enough; otherwise let MaxNLocator
    # choose an integer-valued subset (mirrors ts_plot's max_ticks logic).
    if len(lags) <= max_ticks:
        ax.set_xticks(lags)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks, integer=True))

    # --- Optional title ----------------------------------------------------
    if title is not None and title_position == "top":
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=12)

    style_axes(ax, grid=grid)

    fig.tight_layout()
    draw_note_and_bottom_title(
        fig,
        note=note,
        title=title,
        title_position=title_position,
    )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def _normalise_correlogram_band(confidence_band, frame):
    if isinstance(confidence_band, pd.DataFrame):
        if not confidence_band.index.equals(frame.index):
            raise ValueError("confidence-band index must match correlation lags")
        if tuple(confidence_band.columns) != tuple(frame.columns):
            raise ValueError("confidence-band columns must match correlation names")
        values = confidence_band.to_numpy(dtype=float)
    elif np.isscalar(confidence_band):
        values = np.full(frame.shape, float(confidence_band))
    else:
        values = np.asarray(confidence_band, dtype=float)
        if values.ndim == 1 and len(values) == len(frame):
            values = np.repeat(values[:, None], frame.shape[1], axis=1)
        elif values.shape != frame.shape:
            raise ValueError("confidence_band shape must match the correlation data")
    if values.shape != frame.shape:
        raise ValueError("confidence_band shape must match the correlation data")
    if not np.all(np.isfinite(values)):
        raise ValueError("confidence_band must contain only finite values")
    if np.any(values < 0):
        raise ValueError("confidence_band must be non-negative")
    return values


def plot_correlogram(
    data,
    confidence_band,
    *,
    ax=None,
    title=None,
    xtitle="Lag",
    ytitle="Correlation",
    bar_color=None,
    band_color="#d0d0d0",
    band_alpha=0.4,
    max_ticks=12,
    grid=False,
    note=None,
    title_position="top",
    figsize=None,
):
    """Plot precomputed lag correlations with null confidence bands.

    Series and one-dimensional inputs produce one axis. DataFrames and
    two-dimensional arrays produce one facet per column. This renderer does
    not calculate correlations; statistical packages supply both ``data`` and
    ``confidence_band``.

    Parameters
    ----------
    data : Series, DataFrame, or array-like
        Precomputed correlation values indexed by increasing non-negative lag.
    confidence_band : float, array-like, or DataFrame
        Non-negative confidence-band half-widths. A scalar is shared by every
        lag and response; a one-dimensional array is shared across responses.
    ax : matplotlib.axes.Axes, optional
        Existing axis for a single response. Multi-response data create facets.
    title : str, optional
        Axis title for one response or figure title for multiple responses.
    xtitle, ytitle : str
        Axis labels.
    bar_color : color or sequence of colors, optional
        One shared bar color or one color per response.
    band_color : color, default "#d0d0d0"
        Confidence-region color.
    band_alpha : float, default 0.4
        Confidence-region opacity.
    max_ticks : int, default 12
        Maximum labeled lag ticks before integer thinning.
    grid : bool, default False
        Whether to draw the shared dashed grid.
    note : str, optional
        Figure-level note below the plot.
    title_position : {"top", "bottom"}, default "top"
        Figure-title position.
    figsize : tuple, optional
        Figure size; facet height expands with the row count by default.

    Returns
    -------
    tuple
        ``(fig, ax)`` for one response or ``(fig, axes)`` for facets.

    Examples
    --------
    >>> import pandas as pd
    >>> from Ts.TsPlots import plot_correlogram
    >>> values = pd.Series([0.1, -0.2, 0.05], name="input")
    >>> fig, ax = plot_correlogram(values, confidence_band=0.15)
    >>> len(ax.patches)
    3
    """
    _ensure_fonts()
    if (
        isinstance(max_ticks, (bool, np.bool_))
        or not isinstance(max_ticks, (int, np.integer))
        or int(max_ticks) < 1
    ):
        raise ValueError("max_ticks must be a positive integer")
    frame = _normalise_lag_response(data)
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        frame.columns = [
            "correlation" if frame.shape[1] == 1 else f"correlation_{index + 1}"
            for index in range(frame.shape[1])
        ]
    bands = _normalise_correlogram_band(confidence_band, frame)
    count = frame.shape[1]
    if count > 1 and ax is not None:
        raise ValueError("ax cannot be supplied for multiple correlation sequences")

    if bar_color is None or is_color_like(bar_color):
        colors = [
            bar_color or DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)]
            for index in range(count)
        ]
    else:
        colors = list(bar_color)
        if len(colors) != count:
            raise ValueError("bar_color must contain one value per response")

    if count == 1:
        if ax is None:
            fig, axis = plt.subplots(figsize=figsize or FIGSIZE)
        else:
            axis = ax
            fig = ax.figure
        panel_title = title
        if panel_title is None and frame.columns[0] != "correlation":
            panel_title = str(frame.columns[0])
        _draw_correlogram(
            axis,
            frame.index.to_numpy(dtype=int),
            frame.iloc[:, 0].to_numpy(),
            bands[:, 0],
            bar_color=colors[0],
            band_color=band_color,
            band_alpha=band_alpha,
            xtitle=xtitle,
            ytitle=ytitle,
            title=panel_title,
            title_position=title_position,
            max_ticks=int(max_ticks),
            grid=grid,
            note=note,
            fig=fig,
        )
        return fig, axis

    ncols = min(2, count)
    nrows = math.ceil(count / ncols)
    if figsize is None:
        figsize = (FIGSIZE[0], FIGSIZE[1] * nrows)
    fig, grid_axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flattened = list(grid_axes.ravel())
    axes = flattened[:count]
    for unused in flattened[count:]:
        unused.set_visible(False)
    lags = frame.index.to_numpy(dtype=int)
    for position, (name, axis) in enumerate(zip(frame.columns, axes, strict=True)):
        _draw_correlogram(
            axis,
            lags,
            frame[name].to_numpy(),
            bands[:, position],
            bar_color=colors[position],
            band_color=band_color,
            band_alpha=band_alpha,
            xtitle=xtitle,
            ytitle=ytitle,
            title=str(name),
            title_position="top",
            max_ticks=int(max_ticks),
            grid=grid,
            note=None,
            fig=fig,
        )
    if title is not None and title_position == "top":
        fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96), pad=1.5)
    else:
        fig.tight_layout(pad=1.5)
    draw_note_and_bottom_title(
        fig,
        note=note,
        title=title,
        title_position=title_position,
    )
    return fig, np.asarray(axes, dtype=object)


def plot_acf(
    data,
    nlags: int | None = None,
    *,
    alpha: float = 0.05,
    missing: str = "drop",
    bartlett_confint: bool = True,
    zero_lag: bool = True,
    title: str | None = None,
    xtitle: str = "滞后期数",
    ytitle: str = "ACF值",
    bar_color: str | None = None,
    band_color: str = "#d0d0d0",
    band_alpha: float = 0.4,
    max_ticks: int = 12,
    grid: bool = False,
    note: str | None = None,
    title_position: str = "top",
    ax=None,
):
    """Plot the sample autocorrelation function with a confidence band.

    Parameters
    ----------
    data : array-like or pandas Series or single-column DataFrame
        The time series whose ACF is plotted.
    nlags : int, optional
        Number of lags to compute and display. If None (default), statsmodels
        selects ``min(int(10 * log10(n)), n - 1)`` from the sample size
        ``n``. Pass an integer to set the lag count explicitly.
    alpha : float
        Significance level for the confidence band. ``0.05`` gives a 95 % band
        (default), ``0.01`` gives 99 %, ``0.10`` gives 90 %, etc.
    missing : {"drop", "raise"}
        Non-finite-value policy. ``"drop"`` (default) removes ``NaN`` and
        positive or negative infinity before computing the ACF. ``"raise"``
        reports their original row positions. Dropping an interior gap
        compresses time; use ``"raise"`` when that would be misleading. The
        effective sample after removal controls the adaptive ``nlags`` default.
    bartlett_confint : bool
        Use Bartlett's lag-varying formula for the confidence interval width
        (``True``, default) rather than the uniform ±z / √n formula
        (``False``).
    zero_lag : bool
        Include lag 0 (ACF = 1 by definition) in the plot. Defaults to
        ``True``.
    title : str, optional
        Plot title. Defaults to None (no title).
    xtitle : str
        X-axis label. Defaults to ``"滞后期数"`` (Lag Number).
    ytitle : str
        Y-axis label. Defaults to ``"ACF值"`` (ACF Value).
    bar_color : str, optional
        Bar colour. Defaults to ``DEFAULT_PALETTE[0]`` (deep blue).
    band_color : str
        Confidence-band fill colour. Defaults to ``"#d0d0d0"`` (light gray).
    band_alpha : float
        Confidence-band opacity (0–1). Defaults to 0.4.
    max_ticks : int
        Upper bound on the number of x-axis tick marks. When the number of
        displayed lags exceeds this value, ``MaxNLocator`` selects an integer
        subset. Defaults to 12.
    grid : bool
        Whether to draw a dashed grid. Defaults to False.
    note : str, optional
        Free-text note placed at the lower left of the figure.
    title_position : str
        ``"top"`` (default) or ``"bottom"``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, a new figure and axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If ``missing`` is unknown, ``missing="raise"`` encounters a non-finite
        value, or no finite observations remain.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsPlots import plot_acf
    >>> rng = np.random.default_rng(42)
    >>> fig, ax = plot_acf(rng.normal(size=100), nlags=12, zero_lag=False)
    >>> ax.get_xlabel()
    '滞后期数'
    >>> fig, ax = plot_acf(np.r_[np.nan, rng.normal(size=100)], nlags=12)
    >>> len(ax.patches)
    13
    """
    _ensure_fonts()
    x = _resolve_missing(_to_1d(data), missing)
    color = bar_color if bar_color is not None else DEFAULT_PALETTE[0]

    acf_vals, confint = sm_acf(
        x,
        nlags=nlags,
        alpha=alpha,
        fft=True,
        bartlett_confint=bartlett_confint,
    )

    # confint shape: (nlags+1, 2) — absolute lower/upper bounds.
    # Half-width = upper_bound − centre (symmetric, always positive).
    conf_band = confint[:, 1] - acf_vals

    lags = np.arange(len(acf_vals))  # 0, 1, …, nlags

    if not zero_lag:
        lags = lags[1:]
        acf_vals = acf_vals[1:]
        conf_band = conf_band[1:]

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    else:
        fig = ax.figure

    _draw_correlogram(
        ax,
        lags,
        acf_vals,
        conf_band,
        bar_color=color,
        band_color=band_color,
        band_alpha=band_alpha,
        xtitle=xtitle,
        ytitle=ytitle,
        title=title,
        title_position=title_position,
        max_ticks=max_ticks,
        grid=grid,
        note=note,
        fig=fig,
    )

    return fig, ax


def plot_pacf(
    data,
    nlags: int | None = None,
    *,
    alpha: float = 0.05,
    missing: str = "drop",
    method: str = "ywm",
    title: str | None = None,
    xtitle: str = "滞后期数",
    ytitle: str = "PACF值",
    bar_color: str | None = None,
    band_color: str = "#d0d0d0",
    band_alpha: float = 0.4,
    max_ticks: int = 12,
    grid: bool = False,
    note: str | None = None,
    title_position: str = "top",
    ax=None,
):
    """Plot the sample partial autocorrelation function with a confidence band.

    Parameters
    ----------
    data : array-like or pandas Series or single-column DataFrame
        The time series whose PACF is plotted.
    nlags : int, optional
        Number of lags to compute and display. If None (default), statsmodels
        selects ``min(int(10 * log10(n)), n // 2 - 1)`` from the sample size
        ``n``. Pass an integer to set the lag count explicitly; explicit
        values must satisfy statsmodels' PACF sample-size limit.
    alpha : float
        Significance level for the confidence band. ``0.05`` gives a 95 % band
        (default), ``0.01`` gives 99 %, ``0.10`` gives 90 %, etc.
    missing : {"drop", "raise"}
        Non-finite-value policy. ``"drop"`` (default) removes ``NaN`` and
        positive or negative infinity before computing the PACF. ``"raise"``
        reports their original row positions. Dropping an interior gap
        compresses time; use ``"raise"`` when that would be misleading. The
        effective sample after removal controls the adaptive ``nlags`` default.
    method : str
        PACF estimation method passed to ``statsmodels``. Common choices:
        ``"ywm"`` (Yule-Walker with bias correction, default), ``"ols"``,
        ``"ld"``.
    title : str, optional
        Plot title. Defaults to None (no title).
    xtitle : str
        X-axis label. Defaults to ``"滞后期数"`` (Lag Number).
    ytitle : str
        Y-axis label. Defaults to ``"PACF值"`` (PACF Value).
    bar_color : str, optional
        Bar colour. Defaults to ``DEFAULT_PALETTE[0]`` (deep blue).
    band_color : str
        Confidence-band fill colour. Defaults to ``"#d0d0d0"`` (light gray).
    band_alpha : float
        Confidence-band opacity (0–1). Defaults to 0.4.
    max_ticks : int
        Upper bound on the number of x-axis tick marks. When the number of
        displayed lags exceeds this value, ``MaxNLocator`` selects an integer
        subset. Defaults to 12.
    grid : bool
        Whether to draw a dashed grid. Defaults to False.
    note : str, optional
        Free-text note placed at the lower left of the figure.
    title_position : str
        ``"top"`` (default) or ``"bottom"``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, a new figure and axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If ``missing`` is unknown, ``missing="raise"`` encounters a non-finite
        value, or no finite observations remain.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsPlots import plot_pacf
    >>> rng = np.random.default_rng(42)
    >>> fig, ax = plot_pacf(rng.normal(size=100), nlags=12, method="ywm")
    >>> ax.get_xlabel()
    '滞后期数'
    >>> fig, ax = plot_pacf(np.r_[np.nan, rng.normal(size=100)], nlags=12)
    >>> len(ax.patches)
    12
    """
    _ensure_fonts()
    x = _resolve_missing(_to_1d(data), missing)
    color = bar_color if bar_color is not None else DEFAULT_PALETTE[0]

    pacf_vals, confint = sm_pacf(
        x,
        nlags=nlags,
        alpha=alpha,
        method=method,
    )

    # confint shape: (nlags+1, 2) — absolute lower/upper bounds.
    # Half-width = upper_bound − centre (symmetric, always positive).
    conf_band = confint[:, 1] - pacf_vals

    # PACF at lag 0 is always 1 by construction; skip it for a cleaner plot.
    lags = np.arange(1, len(pacf_vals))
    pacf_vals = pacf_vals[1:]
    conf_band = conf_band[1:]

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    else:
        fig = ax.figure

    _draw_correlogram(
        ax,
        lags,
        pacf_vals,
        conf_band,
        bar_color=color,
        band_color=band_color,
        band_alpha=band_alpha,
        xtitle=xtitle,
        ytitle=ytitle,
        title=title,
        title_position=title_position,
        max_ticks=max_ticks,
        grid=grid,
        note=note,
        fig=fig,
    )

    return fig, ax
