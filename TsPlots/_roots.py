"""Shared inverse-root (unit-circle) plotting for model diagnostics."""

from __future__ import annotations

import numpy as np


def _plot_inverse_roots(
    groups: dict[str, np.ndarray],
    *,
    ax=None,
    title: str | None = None,
    margin: float | None = None,
):
    """Plot inverse characteristic roots on the complex unit circle.

    Each entry of *groups* maps a legend label to an array of complex
    inverse roots.  The first group uses the primary palette marker, the
    second (if present) the secondary marker.  A legend is drawn only when
    more than one group is supplied.

    Parameters
    ----------
    groups : dict of str to numpy.ndarray
        Legend label to complex inverse-root array.
    ax : matplotlib.axes.Axes, optional
        Target axes; a new 6-by-6 figure is created when omitted.
    title : str, optional
        Chart title.
    margin : float, optional
        Absolute axis limit.  When ``None``, the limit is sized from the
        largest root modulus with a ``1.5`` floor.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    from Ts.TsPlots.style import (
        AXIS_LABEL_FONTSIZE,
        DEFAULT_PALETTE,
        LEGEND_FONTSIZE,
        TICK_LABELSIZE,
        TITLE_FONTSIZE,
        _ensure_fonts,
        style_axes,
    )

    _ensure_fonts()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(
        np.cos(theta),
        np.sin(theta),
        color=DEFAULT_PALETTE[1],
        linewidth=1.0,
        linestyle="--",
    )
    ax.axhline(0, color=DEFAULT_PALETTE[1], linewidth=0.5, alpha=0.5)
    ax.axvline(0, color=DEFAULT_PALETTE[1], linewidth=0.5, alpha=0.5)

    for index, (label, roots) in enumerate(groups.items()):
        ax.scatter(
            roots.real,
            roots.imag,
            color=DEFAULT_PALETTE[0] if index == 0 else DEFAULT_PALETTE[4],
            marker="o" if index == 0 else "^",
            s=50,
            edgecolors=DEFAULT_PALETTE[7],
            linewidth=0.5,
            zorder=5,
            label=label,
        )

    ax.set_aspect("equal")
    style_axes(ax)

    if margin is None:
        modulus = [abs(root) for roots in groups.values() for root in roots]
        margin = max(1.5, (max(modulus) if modulus else 0.0) * 1.15)
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)

    ax.set_xlabel("Real", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Imaginary", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABELSIZE)

    if len(groups) > 1:
        ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)

    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")

    fig.tight_layout(pad=1.5)
    return fig, ax