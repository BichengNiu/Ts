"""Lag-indexed plots for impulse responses and dynamic multipliers."""

from __future__ import annotations

from matplotlib.colors import is_color_like
import numpy as np
import pandas as pd

from .style import (
    AXIS_LABEL_FONTSIZE,
    TITLE_FONTSIZE,
    TITLE_PAD,
    TIGHT_PAD,
    INK,
    ZERO_LINE_COLOR,
    _ensure_fonts,
    _facet_grid,
    _fig_axes,
    _finalize_facet_figure,
    _resolve_bar_colors,
    _set_lag_ticks,
    _validate_max_ticks,
    draw_note_and_bottom_title,
    draw_legend,
    style_axes,
)


def _normalise_lag_response(data):
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, pd.Series):
        name = data.name if data.name is not None else "response"
        frame = data.rename(name).to_frame()
    else:
        values = np.asarray(data, dtype=float)
        if values.ndim == 1:
            frame = pd.DataFrame({"response": values})
        elif values.ndim == 2:
            frame = pd.DataFrame(
                values,
                columns=[f"response_{index + 1}" for index in range(values.shape[1])],
            )
        else:
            raise ValueError("data must be one- or two-dimensional")

    if frame.empty or frame.shape[1] == 0:
        raise ValueError("data must contain at least one non-empty response")
    if not frame.columns.is_unique:
        raise ValueError("response names must be unique")
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("lag responses must be numeric") from error
    if not np.all(np.isfinite(frame.to_numpy())):
        raise ValueError("lag responses must contain only finite values")
    if not frame.index.is_unique:
        raise ValueError("time lags must be unique")

    lag_values = np.asarray(frame.index)
    try:
        numeric_lags = lag_values.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("time lags must be non-negative integers") from error
    if not np.all(np.isfinite(numeric_lags)):
        raise ValueError("time lags must be finite integers")
    if np.any(numeric_lags < 0):
        raise ValueError("time lags must be non-negative")
    if not np.all(numeric_lags == np.floor(numeric_lags)):
        raise ValueError("time lags must be integers")
    integer_lags = numeric_lags.astype(int)
    if len(integer_lags) > 1 and np.any(np.diff(integer_lags) <= 0):
        raise ValueError("time lags must be strictly increasing")
    frame.index = pd.Index(integer_lags, name=frame.index.name or "lag")
    return frame


def plot_lag_response(
    data,
    *,
    line_data=None,
    ax=None,
    title=None,
    xtitle="Time lag",
    ytitle="Impulse response",
    color=None,
    line_color=INK,
    zero_line=True,
    grid=True,
    max_ticks=15,
    note=None,
    figsize=None,
):
    """Plot lag-indexed response weights as bars with an optional line.

    A Series or one-dimensional input produces one axis. A multi-column
    DataFrame or two-dimensional array produces one facet per response in
    column order. The lag index must contain unique, increasing,
    non-negative integers.

    Parameters
    ----------
    data : Series, DataFrame, or array-like
        Response weights indexed by time lag.
    line_data : Series, DataFrame, or array-like, optional
        Response weights to overlay as solid lines. The lag index and response
        names must exactly match ``data``.
    ax : matplotlib.axes.Axes, optional
        Existing axis for a single response. Multi-response inputs create
        their own facets.
    title : str, optional
        Axis title for one response or figure title for multiple responses.
    xtitle, ytitle : str
        Axis labels.
    color : color or sequence of colors, optional
        Bar color shared by all responses or one color per response.
    line_color : color, default ``INK``
        Color of the optional response line.
    zero_line : bool, default True
        Whether to draw the zero-response reference line.
    grid : bool, default True
        Whether to draw the shared dashed grid.
    max_ticks : int, default 15
        Maximum number of labeled lag ticks before integer tick thinning.
    note : str, optional
        Figure-level note below the plot.
    figsize : tuple, optional
        Figure size. Multi-response height expands with facet rows.

    Returns
    -------
    tuple
        ``(fig, ax)`` for one response or ``(fig, axes)`` for multiple
        responses, where ``axes`` is a one-dimensional ndarray.

    Examples
    --------
    >>> import pandas as pd
    >>> from Ts.TsPlots import plot_lag_response
    >>> weights = pd.Series([1.0, 0.5, 0.25], name="price")
    >>> fig, ax = plot_lag_response(weights)
    >>> len(ax.patches)
    3
    """
    _ensure_fonts()
    max_ticks = _validate_max_ticks(max_ticks)
    frame = _normalise_lag_response(data)
    line_frame = None
    if line_data is not None:
        line_frame = _normalise_lag_response(line_data)
        if not frame.index.equals(line_frame.index):
            raise ValueError("line_data time lags must exactly match data")
        if not frame.columns.equals(line_frame.columns):
            raise ValueError("line_data response names must exactly match data")
        if not is_color_like(line_color):
            raise ValueError("line_color must be a valid matplotlib color")
    count = frame.shape[1]
    colors = _resolve_bar_colors(color, count)

    if count > 1 and ax is not None:
        raise ValueError("ax cannot be supplied for multiple lag responses")

    if count == 1:
        fig, axis = _fig_axes(ax, figsize)
        axes = [axis]
    else:
        fig, axes = _facet_grid(count, figsize)

    lags = frame.index.to_numpy(dtype=int)
    for position, (name, axis) in enumerate(zip(frame.columns, axes, strict=True)):
        if zero_line:
            axis.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=0.8, zorder=1)
        bars = axis.bar(
            lags,
            frame[name].to_numpy(),
            width=0.65,
            color=colors[position],
            label=("Sample impulse response" if line_frame is not None else None),
            zorder=2,
        )
        if line_frame is not None:
            transfer_line = axis.plot(
                lags,
                line_frame[name].to_numpy(),
                color=line_color,
                linewidth=2.0,
                label="Transfer-function weights",
                zorder=3,
            )[0]
            draw_legend(axis, handles=[bars, transfer_line])
        axis.set_xlabel(xtitle, fontsize=AXIS_LABEL_FONTSIZE)
        axis.set_ylabel(ytitle, fontsize=AXIS_LABEL_FONTSIZE)
        _set_lag_ticks(axis, lags, max_ticks)
        panel_title = title if count == 1 and title is not None else str(name)
        if panel_title != "response" or title is not None:
            axis.set_title(
                panel_title,
                fontsize=TITLE_FONTSIZE,
                fontweight="bold",
                pad=TITLE_PAD,
            )
        style_axes(axis, grid=grid)

    if count > 1:
        _finalize_facet_figure(fig, title=title, note=note)
    else:
        fig.tight_layout(pad=TIGHT_PAD)
        if note is not None:
            draw_note_and_bottom_title(fig, note=note)

    if count == 1:
        return fig, axes[0]
    return fig, np.asarray(axes, dtype=object)
