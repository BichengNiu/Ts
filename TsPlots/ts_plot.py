"""Reusable plotting utilities for time series visualization.

The main entry point is :func:`plot_series`, which draws an arbitrary number
of series. Styling cycles per series: even-indexed series are drawn as solid
lines with filled circles, odd-indexed series as dashed lines with hollow
circles, with colors from a colorblind-friendly palette.

Accepted inputs
---------------
- pandas DataFrame : each column becomes a series. The time axis is the index
  by default, or a column named via ``x``.
- pandas Series    : a single series indexed by its index.
- dict             : ``{label: values}`` for multiple named series.
- 2D array         : each column becomes a series.
- 1D array         : a single series.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from Ts.TsPlots import plot_series
>>> # DataFrame, index as time
>>> df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}, index=[2000, 2001, 2002])
>>> fig, axes = plot_series(df)
>>> # DataFrame, a column as the time variable
>>> df2 = pd.DataFrame({"year": [2000, 2001, 2002], "a": [1, 2, 3], "b": [3, 2, 1]})
>>> fig, axes = plot_series(df2, x="year")
>>> # Overlay named series; large scale gaps automatically create more y-axes
>>> fig, ax = plot_series(
...     {
...         "rate": [1, 2, 3],
...         "level": [1000, 2000, 3000],
...         "population": [1_000_000, 2_000_000, 3_000_000],
...     },
...     facet=False,
... )
>>> right_ax = ax.right_ax
>>> all_right_axes = ax.extra_y_axes
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import pandas as pd

from .style import (
    _ensure_fonts,
    _FigureContext,
    _resolve_colors,
    _validate_positive_step,
    DEFAULT_PALETTE,
    DEFAULT_LINESTYLES,
    DEFAULT_MARKERS,
    FIGSIZE,
    TITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    style_axes,
    draw_unit_label,
    draw_note_and_bottom_title,
    draw_shade,
    draw_vlines,
)


_VALID_FREQS = {"day", "week", "month", "quarter", "year"}


def _make_chinese_formatter(freq: str) -> FuncFormatter:
    """Return a FuncFormatter that renders datetime ticks in Chinese.

    Formats
    -------
    day     : XXXX年X月X日      e.g. 2026年6月27日
    week    : XXXX年X月第X周    e.g. 2026年6月第4周
    month   : XXXX年X月         e.g. 2026年6月
    quarter : XXXX年第X季度     e.g. 2026年第2季度
    year    : XXXX年             e.g. 2026年
    """
    if freq == "day":

        def _fmt(x, pos):
            dt = mdates.num2date(x)
            return f"{dt.year}年{dt.month}月{dt.day}日"

    elif freq == "week":

        def _fmt(x, pos):
            dt = mdates.num2date(x)
            week_of_month = (dt.day - 1) // 7 + 1
            return f"{dt.year}年{dt.month}月第{week_of_month}周"

    elif freq == "month":

        def _fmt(x, pos):
            dt = mdates.num2date(x)
            return f"{dt.year}年{dt.month}月"

    elif freq == "quarter":

        def _fmt(x, pos):
            dt = mdates.num2date(x)
            quarter = (dt.month - 1) // 3 + 1
            return f"{dt.year}年第{quarter}季度"

    elif freq == "year":

        def _fmt(x, pos):
            dt = mdates.num2date(x)
            return f"{dt.year}年"

    else:
        raise ValueError(
            f"freq={freq!r} is not valid. Choose from: "
            + ", ".join(sorted(_VALID_FREQS))
        )

    return FuncFormatter(_fmt)


def _apply_freq_ticks(ax, x_values, freq: str, max_ticks: int = 12) -> None:
    """Apply a matplotlib.dates locator/formatter for the given frequency.

    The locator interval is calculated from the data's date range and
    ``max_ticks`` so that at most ``max_ticks`` ticks are displayed.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x_values : array-like
        The x data (must be datetime-like for this function to have any effect).
    freq : str
        One of 'day', 'week', 'month', 'quarter', 'year'.
    max_ticks : int
        Upper bound on the number of ticks. Defaults to 12.
    """
    if freq not in _VALID_FREQS:
        raise ValueError(
            f"freq={freq!r} is not valid. Choose from: "
            + ", ".join(sorted(_VALID_FREQS))
        )

    x_dt = pd.DatetimeIndex(pd.to_datetime(x_values))
    d_min, d_max = x_dt.min(), x_dt.max()

    if freq == "day":
        n = max(1, (d_max - d_min).days + 1)
        interval = max(1, int(np.ceil(n / max_ticks)))
        locator = mdates.DayLocator(interval=interval)

    elif freq == "week":
        n = max(1, (d_max - d_min).days // 7 + 1)
        interval = max(1, int(np.ceil(n / max_ticks)))
        locator = mdates.WeekdayLocator(byweekday=mdates.MO, interval=interval)

    elif freq == "month":
        n = (d_max.year - d_min.year) * 12 + (d_max.month - d_min.month) + 1
        interval = max(1, int(np.ceil(n / max_ticks)))
        locator = mdates.MonthLocator(interval=interval)

    elif freq == "quarter":
        n_months = (d_max.year - d_min.year) * 12 + (d_max.month - d_min.month) + 1
        n = max(1, int(np.ceil(n_months / 3)))
        interval = max(1, int(np.ceil(n / max_ticks)))
        locator = mdates.MonthLocator(interval=3 * interval)

    elif freq == "year":
        n = max(1, d_max.year - d_min.year + 1)
        interval = max(1, int(np.ceil(n / max_ticks)))
        locator = mdates.YearLocator(base=interval)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(_make_chinese_formatter(freq))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def _resolve_input(data, x, y) -> tuple[np.ndarray, dict, str]:
    """Normalize supported inputs into (x_values, series_dict, default_xlabel).

    series_dict maps label -> values, preserving order.
    """
    if isinstance(data, pd.DataFrame):
        if x is None:
            x_values = data.index.to_numpy()
            default_xlabel = data.index.name or "Time"
            cols = list(data.columns)
        elif isinstance(x, str):
            if x not in data.columns:
                raise KeyError(f"Column {x!r} not found in DataFrame.")
            x_values = data[x].to_numpy()
            default_xlabel = x
            cols = [c for c in data.columns if c != x]
        else:
            x_values = np.asarray(x)
            default_xlabel = "Time"
            cols = list(data.columns)

        if y is not None:
            cols = [y] if isinstance(y, str) else list(y)

        series = {str(c): data[c].to_numpy() for c in cols}
        return x_values, series, default_xlabel

    if isinstance(data, pd.Series):
        x_values = np.asarray(x) if x is not None else data.index.to_numpy()
        default_xlabel = "Time" if x is not None else (data.index.name or "Time")
        label = data.name if data.name is not None else "Series 1"
        return x_values, {str(label): data.to_numpy()}, default_xlabel

    if isinstance(data, dict):
        series = {str(k): np.asarray(v) for k, v in data.items()}
        first_len = len(next(iter(series.values())))
        x_values = np.asarray(x) if x is not None else np.arange(first_len)
        return x_values, series, "Time"

    arr = np.asarray(data)
    if arr.ndim == 1:
        x_values = np.asarray(x) if x is not None else np.arange(len(arr))
        return x_values, {"Series 1": arr}, "Time"
    if arr.ndim == 2:
        n_rows, n_cols = arr.shape
        x_values = np.asarray(x) if x is not None else np.arange(n_rows)
        series = {f"Series {i + 1}": arr[:, i] for i in range(n_cols)}
        return x_values, series, "Time"

    raise TypeError(f"Unsupported data type: {type(data)!r}")


def _validate_bool(name: str, value) -> bool:
    """Return a real boolean or reject ambiguous truthy/falsy values."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _robust_scale(values) -> float | None:
    """Estimate a series magnitude without being dominated by one outlier."""
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    q05, q95 = np.percentile(finite, [5, 95])
    magnitude = float(np.percentile(np.abs(finite), 95))
    scale = max(magnitude, float(q95 - q05))
    return scale if np.isfinite(scale) and scale > 0 else None


def _validate_max_y_axes(value) -> int:
    """Validate the total number of y-axes allowed in overlay mode."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError("max_y_axes must be an integer")
    if value < 1:
        raise ValueError("max_y_axes must be at least 1")
    return int(value)


def _manual_axis_groups(
    series: dict,
    axis_groups,
    max_y_axes: int,
) -> list[list[str]]:
    """Validate and order an explicit ``series label -> group id`` mapping."""
    if not isinstance(axis_groups, Mapping):
        raise TypeError("axis_groups must be a mapping")

    series_labels = list(series)
    if set(axis_groups) != set(series_labels):
        missing = sorted(set(series_labels) - set(axis_groups))
        unknown = sorted(set(axis_groups) - set(series_labels))
        raise ValueError(
            "axis_groups labels must exactly match the plotted series; "
            f"missing={missing}, unknown={unknown}"
        )

    grouped: dict[object, list[str]] = {}
    for label in series_labels:
        group_id = axis_groups[label]
        try:
            hash(group_id)
        except TypeError as error:
            raise TypeError("axis_groups values must be hashable") from error
        grouped.setdefault(group_id, []).append(label)

    groups = list(grouped.values())
    if len(groups) > max_y_axes:
        raise ValueError(
            f"axis_groups defines {len(groups)} groups but max_y_axes is {max_y_axes}"
        )
    return groups


def _merge_closest_scale_groups(
    groups: list[dict],
    max_y_axes: int,
) -> list[dict]:
    """Merge neighboring scale groups at their smallest remaining gap."""
    while len(groups) > max_y_axes:
        ratios = [
            groups[index + 1]["scales"][0] / groups[index]["scales"][-1]
            for index in range(len(groups) - 1)
        ]
        merge_index = int(np.argmin(ratios))
        groups[merge_index]["labels"].extend(groups[merge_index + 1]["labels"])
        groups[merge_index]["scales"].extend(groups[merge_index + 1]["scales"])
        del groups[merge_index + 1]
    return groups


def _automatic_axis_groups(
    series: dict,
    threshold: float,
    max_y_axes: int,
) -> list[list[str]]:
    """Group adjacent robust scales, preserving the first series on the left."""
    series_labels = list(series)
    scaled = [
        (label, scale)
        for label, values in series.items()
        if (scale := _robust_scale(values)) is not None
    ]
    if len(scaled) < 2 or max_y_axes == 1:
        return [series_labels]

    scaled.sort(key=lambda item: item[1])
    groups = [{"labels": [scaled[0][0]], "scales": [scaled[0][1]]}]
    for label, scale in scaled[1:]:
        previous_scale = groups[-1]["scales"][-1]
        if scale / previous_scale < threshold:
            groups[-1]["labels"].append(label)
            groups[-1]["scales"].append(scale)
        else:
            groups.append({"labels": [label], "scales": [scale]})

    groups = _merge_closest_scale_groups(groups, max_y_axes)

    scaled_labels = {label for label, _scale in scaled}
    unscaled_labels = [label for label in series_labels if label not in scaled_labels]
    if unscaled_labels:
        groups[0]["labels"].extend(unscaled_labels)

    order = {label: index for index, label in enumerate(series_labels)}
    resolved = [sorted(group["labels"], key=order.get) for group in groups]
    primary_index = next(
        index for index, labels in enumerate(resolved) if series_labels[0] in labels
    )
    return [
        resolved[primary_index],
        *resolved[:primary_index],
        *resolved[primary_index + 1 :],
    ]


def _resolve_axis_groups(
    series: dict,
    *,
    axis_groups,
    auto_dual_y: bool,
    threshold: float,
    max_y_axes: int,
) -> list[list[str]]:
    """Resolve manual groups or automatically group series by robust scale."""
    if axis_groups is not None:
        return _manual_axis_groups(series, axis_groups, max_y_axes)
    if len(series) < 2 or not auto_dual_y:
        return [list(series)]
    return _automatic_axis_groups(series, threshold, max_y_axes)


def _series_color(colors, index: int) -> str:
    if colors is not None:
        return colors[index]
    return DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)]


def _plot_one_series(
    ax,
    x_values,
    values,
    label,
    index,
    *,
    colors,
    linewidth,
    markersize,
    marker_edge_width,
    show_values,
    value_decimals,
):
    """Draw one series and its optional point-value annotations."""
    color = _series_color(colors, index)
    linestyle = DEFAULT_LINESTYLES[index % len(DEFAULT_LINESTYLES)]
    marker = DEFAULT_MARKERS[index % len(DEFAULT_MARKERS)]
    is_even = index % 2 == 0
    line = ax.plot(
        x_values,
        values,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markersize=markersize,
        color=color,
        markerfacecolor=color if is_even else "white",
        markeredgecolor=color,
        markeredgewidth=marker_edge_width,
        label=label,
    )[0]

    if not show_values:
        return line

    fmt = f".{value_decimals}f"
    n_points = len(values)
    for point_index, (x_value, y_value) in enumerate(
        zip(x_values, values, strict=False)
    ):
        previous = float(values[point_index - 1]) if point_index > 0 else float(y_value)
        following = (
            float(values[point_index + 1])
            if point_index < n_points - 1
            else float(y_value)
        )
        if float(y_value) >= (previous + following) / 2:
            y_offset, vertical_alignment = -(markersize + 10), "top"
        else:
            y_offset, vertical_alignment = markersize + 4, "bottom"
        ax.annotate(
            f"{y_value:{fmt}}",
            xy=(x_value, y_value),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va=vertical_alignment,
            fontsize=11,
            color=color,
        )
    return line


def _configure_x_axis(ax, x_values, *, xtick_step, max_ticks, freq) -> None:
    """Apply datetime or numeric tick selection to one axes."""
    x_array = np.asarray(x_values)
    is_datetime = np.issubdtype(x_array.dtype, np.datetime64) or (
        pd.api.types.is_datetime64_any_dtype(x_array)
    )
    is_numeric = np.issubdtype(x_array.dtype, np.number)

    if is_datetime and freq is not None:
        _apply_freq_ticks(ax, x_values, freq, max_ticks=max_ticks)
    elif is_numeric and xtick_step is not None:
        x_min, x_max = int(np.min(x_values)), int(np.max(x_values))
        ax.set_xticks(range(x_min, x_max + 1, xtick_step))
    elif is_numeric:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks))


def _validate_legend_labels(legend_labels, series_count: int):
    if legend_labels is None:
        return None
    resolved = list(legend_labels)
    if len(resolved) != series_count:
        raise ValueError(
            f"legend_labels has {len(resolved)} entries but there are "
            f"{series_count} series."
        )
    return resolved


def plot_series(
    data,
    x=None,
    y=None,
    *,
    labels=None,
    colors=None,
    title: str | None = None,
    xtitle: str | None = None,
    ytitle: str = "Value",
    linewidth: float = 3,
    markersize: float = 7,
    marker_edge_width: float = 2.5,
    xtick_step: int | None = None,
    max_ticks: int = 12,
    freq: str | None = None,
    ymin: float | None = None,
    show_legend: bool = True,
    legend_labels=None,
    legend_loc: str = "best",
    legend_bbox=None,
    title_loc: str = "center",
    title_pad: float = 12,
    title_position: str = "top",
    note: str | None = None,
    grid: bool = False,
    show_values: bool = False,
    value_decimals: int = 1,
    vlines=None,
    vline_color: str = "#d9534f",
    vline_linestyle: str = "--",
    vline_linewidth: float = 1.5,
    shade=None,
    shade_color: str = "#d0d0d0",
    shade_alpha: float = 0.3,
    facet: bool = True,
    sharex: bool = True,
    sharey: bool = False,
    auto_dual_y: bool = True,
    scale_ratio_threshold: float = 10.0,
    axis_groups: Mapping[str, object] | None = None,
    max_y_axes: int = 3,
    ax=None,
    unit: str | None = None,
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    """Plot an arbitrary number of time series with cycling styles.

    Styling cycles per series so that each added line differs in color, line
    style, and marker shape, keeping series distinguishable even in grayscale
    or black-and-white print. Even-indexed series use filled markers, odd-
    indexed series use hollow markers.

    Parameters
    ----------
    data : DataFrame, Series, dict, or array-like
        The series to plot. See module docstring for accepted shapes.
    x : str or array-like, optional
        Time variable. For a DataFrame, a string selects a column to use as
        the time axis; otherwise an array-like provides explicit x values.
        If None, the DataFrame/Series index (or a 0-based range) is used.
    y : str or sequence of str, optional
        For a DataFrame, restricts which columns to plot.
    labels : sequence of str, optional
        Override the legend labels (must match the number of series).
    colors : sequence of str, optional
        Override the per-series colors.
    title : str, optional
        Plot title. Defaults to None (no title).
    xtitle : str, optional
        Label for the x-axis. Defaults to None (detected from data).
    ytitle : str, optional
        Label for the y-axis. Defaults to ``"Value"``.
    linewidth : float
        Width of every line. Defaults to 3.
    markersize : float
        Size of the markers. Defaults to 7.
    marker_edge_width : float
        Width of the marker outlines. Defaults to 2.5.
    xtick_step : int, optional
        Explicit spacing between x ticks (numeric x only). If None, ticks
        are chosen automatically (at most ``max_ticks``).
    max_ticks : int
        Upper bound on the number of automatic x ticks. Defaults to 12.
    freq : str, optional
        Tick frequency for datetime x-axes. One of ``'day'``, ``'week'``,
        ``'month'``, ``'quarter'``, or ``'year'``. When set, the appropriate
        ``matplotlib.dates`` locator and formatter are applied and tick labels
        are auto-rotated 45°.
    ymin : float or None
        Lower limit of the y-axis. Defaults to None (automatic); pass a float to set a fixed minimum.
    show_legend : bool
        Whether to display the legend. Defaults to ``True``.
    legend_labels : sequence of str, optional
        Override the text of the legend entries. Must match the number of
        series.
    legend_loc : str
        Legend location (e.g. ``"upper left"``, ``"best"``). Defaults to
        ``"best"``.
    legend_bbox : tuple, optional
        ``bbox_to_anchor`` for the legend, e.g. ``(1.02, 1)``.
    title_loc : str
        Title horizontal alignment: ``"center"``, ``"left"``, or ``"right"``.
    title_pad : float
        Padding between the title and the axes. Defaults to 12.
    title_position : str
        ``"top"`` (default) or ``"bottom"``.
    note : str, optional
        Free-text note placed at the lower-left of the figure.
    grid : bool
        Whether to show a dashed grid on both axes. Defaults to False.
    show_values : bool
        Annotate each data point with its numeric value. Defaults to False.
        Labels are placed **above** local minima and **below** local maxima so
        they do not overlap with adjacent line segments.
    value_decimals : int
        Decimal places for ``show_values`` annotations. Defaults to 1.
    vlines : float or list of float, optional
        X-axis position(s) for vertical reference lines.
    vline_color : str
        Color of vertical lines. Defaults to ``"#d9534f"``.
    vline_linestyle : str
        Line style of vertical lines. Defaults to ``"--"``.
    vline_linewidth : float
        Width of vertical lines. Defaults to 1.5.
    shade : tuple or list of tuple, optional
        ``(xmin, xmax)`` interval(s) to shade, e.g. ``[(2008, 2009)]``.
    shade_color : str
        Fill color for shaded regions. Defaults to ``"#d0d0d0"``.
    shade_alpha : float
        Opacity of shaded regions (0–1). Defaults to 0.3.
    facet : bool
        For two or more series, draw one vertical panel per series. Defaults
        to ``True``. A single series always uses one axes.
    sharex : bool
        Whether faceted panels share the same x-axis scale. Defaults to
        ``True``.
    sharey : bool
        Whether faceted panels share the same y-axis scale. Defaults to
        ``False``.
    auto_dual_y : bool
        When ``facet=False`` and ``axis_groups`` is not supplied,
        automatically group similar robust scales and create additional
        y-axes as needed. The name is retained for backward compatibility.
        Defaults to ``True``.
    scale_ratio_threshold : float
        Positive adjacent scale ratio that starts a new automatic axis group.
        Defaults to 10.
    axis_groups : mapping, optional
        Explicit ``{series_label: group_id}`` assignment used when
        ``facet=False``. Series with the same hashable group identifier share
        one y-axis. The mapping must contain every plotted label exactly once
        and overrides ``auto_dual_y``. Defaults to automatic scale grouping.
    max_y_axes : int
        Maximum total number of y-axes, including the primary left axes.
        Automatic groups beyond this limit are merged by the closest scale
        gap; explicit groups beyond the limit raise an error. Defaults to 3.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. Multi-series faceting requires ``ax=None``;
        pass ``facet=False`` to overlay multiple series on an existing axes.
    unit : str, optional
        Unit label appended to the y-axis label, formatted as
        ``（单位：XX）``. Defaults to None.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes or numpy.ndarray
        A single axes for a single series or overlay. Faceting returns a
        one-dimensional array of axes. In multi-axis overlay mode, the first
        secondary axes is available as ``ax.right_ax``. All
        right-side axes are available through ``ax.extra_y_axes`` and all
        axes through ``fig.axes``.

    Examples
    --------
    Plot two columns in separate panels, using the DataFrame index as time.

    >>> import pandas as pd
    >>> from Ts.TsPlots import plot_series
    >>> frame = pd.DataFrame(
    ...     {"output": [100, 104, 109], "prices": [95, 99, 103]},
    ...     index=pd.date_range("2024-01-01", periods=3, freq="YS"),
    ... )
    >>> fig, axes = plot_series(frame, title="Annual indicators")
    >>> len(axes)
    2

    Overlay differently scaled series and assign axes automatically. Manual
    ``axis_groups`` takes precedence when the grouping is substantively known.

    >>> fig, ax = plot_series(
    ...     {"rate": [1, 2, 3], "level": [1000, 2000, 3000]},
    ...     facet=False,
    ... )
    >>> len(ax.extra_y_axes)
    1
    >>> fig, ax = plot_series(
    ...     frame,
    ...     facet=False,
    ...     axis_groups={"output": "real", "prices": "real"},
    ... )
    >>> len(ax.extra_y_axes)
    0
    """
    x_values, series, default_xlabel = _resolve_input(data, x, y)
    colors = _resolve_colors(colors, len(series))
    xtick_step = _validate_positive_step(
        "xtick_step",
        xtick_step,
        integer=True,
    )
    facet = _validate_bool("facet", facet)
    sharex = _validate_bool("sharex", sharex)
    sharey = _validate_bool("sharey", sharey)
    auto_dual_y = _validate_bool("auto_dual_y", auto_dual_y)
    scale_ratio_threshold = _validate_positive_step(
        "scale_ratio_threshold",
        scale_ratio_threshold,
    )
    max_y_axes = _validate_max_y_axes(max_y_axes)

    if labels is not None:
        labels = list(labels)
        if len(labels) != len(series):
            raise ValueError(
                f"labels has {len(labels)} entries but there are {len(series)} series."
            )
        series = {labels[i]: v for i, v in enumerate(series.values())}

    legend_labels = _validate_legend_labels(legend_labels, len(series))

    if axis_groups is not None and facet and len(series) >= 2:
        raise ValueError("axis_groups requires facet=False for multiple series")

    if facet and len(series) >= 2:
        if ax is not None:
            raise ValueError(
                "Multi-series faceting cannot draw on one existing ax; "
                "pass facet=False to overlay the series on that axes."
            )

        _ensure_fonts()
        figure_height = max(FIGSIZE[1], 3.5 * len(series))
        fig, axes = plt.subplots(
            len(series),
            1,
            figsize=(FIGSIZE[0], figure_height),
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
        )
        axes = axes[:, 0]
        x_label = xtitle if xtitle is not None else default_xlabel
        display_labels = legend_labels or list(series)

        for index, ((label, values), panel_ax) in enumerate(
            zip(series.items(), axes, strict=True)
        ):
            draw_shade(panel_ax, shade, shade_color, shade_alpha)
            draw_vlines(
                panel_ax,
                vlines,
                vline_color,
                vline_linestyle,
                vline_linewidth,
            )
            line = _plot_one_series(
                panel_ax,
                x_values,
                values,
                label,
                index,
                colors=colors,
                linewidth=linewidth,
                markersize=markersize,
                marker_edge_width=marker_edge_width,
                show_values=show_values,
                value_decimals=value_decimals,
            )
            panel_ax.set_title(
                display_labels[index],
                fontsize=AXIS_LABEL_FONTSIZE,
                loc="left",
                pad=6,
            )
            if ytitle is not None:
                panel_ax.set_ylabel(ytitle, fontsize=AXIS_LABEL_FONTSIZE)
            if unit is not None:
                draw_unit_label(panel_ax, unit, axis="y")
            if not sharex or index == len(axes) - 1:
                panel_ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
            _configure_x_axis(
                panel_ax,
                x_values,
                xtick_step=xtick_step,
                max_ticks=max_ticks,
                freq=freq,
            )
            if ymin is not None:
                panel_ax.set_ylim(bottom=ymin)
            style_axes(panel_ax, grid=grid)
            if show_legend:
                panel_ax.legend(
                    [line],
                    [display_labels[index]],
                    frameon=False,
                    fontsize=LEGEND_FONTSIZE,
                    loc=legend_loc,
                    bbox_to_anchor=legend_bbox,
                )

        if title and title_position == "top":
            title_positions = {
                "left": (0.01, "left"),
                "center": (0.5, "center"),
                "right": (0.99, "right"),
            }
            if title_loc not in title_positions:
                raise ValueError("title_loc must be 'left', 'center', or 'right'")
            title_x, title_alignment = title_positions[title_loc]
            fig.suptitle(
                title,
                fontsize=TITLE_FONTSIZE,
                fontweight="bold",
                x=title_x,
                ha=title_alignment,
            )
        tight_rect = (0, 0, 1, 0.97) if title and title_position == "top" else None
        fig.tight_layout(pad=1.5, rect=tight_rect)
        if note or (title and title_position == "bottom"):
            draw_note_and_bottom_title(
                fig,
                note=note,
                title=title,
                title_position=title_position,
            )
        return fig, axes

    _ensure_fonts()
    ctx = _FigureContext(ax=ax)
    fig, ax = ctx.fig, ctx.ax

    draw_shade(ax, shade, shade_color, shade_alpha)
    draw_vlines(ax, vlines, vline_color, vline_linestyle, vline_linewidth)

    resolved_groups = _resolve_axis_groups(
        series,
        axis_groups=axis_groups,
        auto_dual_y=auto_dual_y,
        threshold=scale_ratio_threshold,
        max_y_axes=max_y_axes,
    )
    right_axes = []
    for group_index in range(1, len(resolved_groups)):
        right_axis = ax.twinx()
        right_axis.spines["right"].set_position(
            ("axes", 1.0 + 0.12 * (group_index - 1))
        )
        right_axes.append(right_axis)

    ax.extra_y_axes = right_axes
    if right_axes:
        ax.right_ax = right_axes[0]

    axis_by_label = {
        label: ([ax, *right_axes][group_index])
        for group_index, group_labels in enumerate(resolved_groups)
        for label in group_labels
    }

    lines = []
    for index, (label, values) in enumerate(series.items()):
        target_ax = axis_by_label[label]
        lines.append(
            _plot_one_series(
                target_ax,
                x_values,
                values,
                label,
                index,
                colors=colors,
                linewidth=linewidth,
                markersize=markersize,
                marker_edge_width=marker_edge_width,
                show_values=show_values,
                value_decimals=value_decimals,
            )
        )

    ax.set_xlabel(
        xtitle if xtitle is not None else default_xlabel,
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    if ytitle is not None:
        ax.set_ylabel(ytitle, fontsize=AXIS_LABEL_FONTSIZE)

    _configure_x_axis(
        ax,
        x_values,
        xtick_step=xtick_step,
        max_ticks=max_ticks,
        freq=freq,
    )

    if ymin is not None:
        for y_axis in [ax, *right_axes]:
            y_axis.set_ylim(bottom=ymin)

    display_labels = legend_labels or list(series)
    display_name_by_label = dict(zip(series, display_labels, strict=True))
    for group_labels, right_axis in zip(
        resolved_groups[1:],
        right_axes,
        strict=True,
    ):
        right_axis.set_ylabel(
            " / ".join(display_name_by_label[label] for label in group_labels),
            fontsize=AXIS_LABEL_FONTSIZE,
        )
        if unit is not None:
            draw_unit_label(right_axis, unit, axis="y")
        style_axes(right_axis, grid=False)
        right_axis.spines["right"].set_visible(True)
        right_axis.yaxis.set_label_position("right")
        right_axis.yaxis.tick_right()

    if show_legend and lines:
        ax.legend(
            lines,
            display_labels,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
        )

    ctx.finalize(
        title=title,
        title_position=title_position,
        title_loc=title_loc,
        title_pad=title_pad,
        note=note,
        grid=grid,
        show_legend=False,
        labels=list(series),
        unit=unit,
    )

    if len(right_axes) >= 2:
        right_margin = max(0.5, 0.88 - 0.10 * (len(right_axes) - 1))
        fig.subplots_adjust(right=right_margin)

    return fig, ax
