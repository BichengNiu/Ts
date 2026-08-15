"""Backward-compatible shim for unit-root diagnostic plots.

The rendering helpers now live in :mod:`Ts.TsPlots.unitroot_plot` so that all
plotting code stays inside the TsPlots package (AGENTS.md boundary rule).
This module re-exports them so existing internal imports keep working.
"""

from __future__ import annotations

from Ts.TsPlots.unitroot_plot import (
    _render_critical_value_plot,
    _render_ic_plot,
    _render_tstat_plot,
)

__all__ = [
    "_render_critical_value_plot",
    "_render_ic_plot",
    "_render_tstat_plot",
]
