"""Correlation-matrix heatmaps using the shared TsPlots style contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .style import (
    NOTE_FONTSIZE,
    TICK_LABELSIZE,
    TITLE_FONTSIZE,
    _body_font_family,
    _ensure_fonts,
    _fig_axes,
    style_axes,
)


def _resolve_correlation_input(matrix, labels):
    """Return a validated correlation array and display labels."""
    if isinstance(matrix, pd.DataFrame):
        if labels is not None:
            raise ValueError("labels cannot be supplied with a DataFrame")
        if not matrix.index.equals(matrix.columns):
            raise ValueError("DataFrame index and columns must match")
        resolved_labels = [str(label) for label in matrix.index]
        values = matrix.to_numpy(dtype=float, copy=True)
    else:
        values = np.array(matrix, dtype=float, copy=True)
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError("matrix must be a square two-dimensional array")
        if labels is None:
            resolved_labels = [str(index) for index in range(values.shape[0])]
        else:
            if isinstance(labels, str):
                raise TypeError("labels must be a sequence of strings")
            resolved_labels = list(labels)

    if values.shape[0] == 0:
        raise ValueError("matrix must not be empty")
    if len(resolved_labels) != values.shape[0]:
        raise ValueError("labels length must match the matrix size")
    if any(not isinstance(label, str) for label in resolved_labels):
        raise TypeError("every label must be a string")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must contain only finite values")
    tolerance = np.finfo(float).eps * 100.0
    if np.any(values < -1.0 - tolerance) or np.any(values > 1.0 + tolerance):
        raise ValueError("correlations must be between -1 and 1")
    if not np.allclose(values, values.T, rtol=1e-7, atol=tolerance):
        raise ValueError("correlation matrix must be symmetric")
    if not np.allclose(np.diag(values), 1.0, rtol=1e-7, atol=tolerance):
        raise ValueError("correlation matrix diagonal must equal 1")
    return np.clip((values + values.T) / 2.0, -1.0, 1.0), resolved_labels


def plot_correlation_matrix(
    matrix,
    *,
    labels=None,
    annotate=True,
    decimals=2,
    title="Correlation Matrix",
    ax=None,
):
    """Plot a labelled correlation matrix on a fixed ``[-1, 1]`` colour scale.

    Parameters
    ----------
    matrix : array-like or pandas.DataFrame
        Square, finite, symmetric correlation matrix with a unit diagonal.
        A DataFrame supplies its own labels and requires identical index and
        columns.
    labels : sequence of str, optional
        Axis labels for array input. Omit for integer labels. Must not be
        supplied when *matrix* is a DataFrame.
    annotate : bool, default True
        Whether to write the numeric value inside each cell.
    decimals : int, default 2
        Number of decimal places used for annotations.
    title : str, default "Correlation Matrix"
        Chart title.
    ax : matplotlib.axes.Axes, optional
        Existing axes. A new figure and axes are created when omitted.

    Returns
    -------
    tuple
        ``(figure, axes)`` containing the heatmap. The figure also contains a
        colour-bar axes.

    Examples
    --------
    >>> from Ts.TsPlots import plot_correlation_matrix
    >>> matrix = [[1.0, -0.4], [-0.4, 1.0]]
    >>> fig, ax = plot_correlation_matrix(matrix, labels=["ar.L1", "ma.L1"])
    """
    _ensure_fonts()
    values, resolved_labels = _resolve_correlation_input(matrix, labels)
    if not isinstance(annotate, (bool, np.bool_)):
        raise TypeError("annotate must be a boolean")
    if isinstance(decimals, (bool, np.bool_)) or not isinstance(
        decimals, (int, np.integer)
    ):
        raise TypeError("decimals must be a non-negative integer")
    if decimals < 0:
        raise ValueError("decimals must be non-negative")
    if title is not None and not isinstance(title, str):
        raise TypeError("title must be a string or None")

    size = min(14.0, max(5.5, 0.72 * len(resolved_labels) + 2.5))
    fig, ax = _fig_axes(ax, (size, size))

    image = ax.imshow(
        values,
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=-1.0,
        vmax=1.0,
        aspect="equal",
    )
    positions = np.arange(len(resolved_labels))
    ax.set_xticks(positions, labels=resolved_labels)
    ax.set_yticks(positions, labels=resolved_labels)
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    if annotate:
        body_family = _body_font_family()
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                color = "white" if abs(value) >= 0.55 else "black"
                ax.text(
                    column,
                    row,
                    f"{value:.{int(decimals)}f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=NOTE_FONTSIZE,
                    fontfamily=body_family,
                )

    style_axes(ax, tick_labelsize=TICK_LABELSIZE)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Correlation", fontfamily=_body_font_family())
    colorbar.ax.tick_params(labelsize=TICK_LABELSIZE)
    for label in colorbar.ax.get_yticklabels():
        label.set_fontfamily(_body_font_family())
    fig.tight_layout(pad=1.5)
    return fig, ax
