"""Seasonal-Trend decomposition using LOESS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.seasonal import STL as _StatsmodelsSTL

from ._validation import _resolve_missing_rows


@dataclass
class STLResult:
    """Result container for STL decomposition components."""

    observed: np.ndarray
    trend: np.ndarray
    seasonal: np.ndarray
    residuals: np.ndarray
    weights: np.ndarray
    period: int
    config: dict

    @property
    def nobs(self):
        """Number of observations in the decomposition."""
        return self.observed.size

    @property
    def fitted_values(self):
        """Trend plus seasonal component."""
        return self.trend + self.seasonal

    def summary(self):
        """Return a formatted decomposition summary."""
        robust = self.config["robust"]
        seasonal = self.config["seasonal"]
        trend = self.config["trend"]
        low_pass = self.config["low_pass"]
        lines = [
            "STL Decomposition Result",
            "=" * 50,
            f"Observations       : {self.nobs}",
            f"Period             : {self.period}",
            f"Robust             : {robust}",
            f"Seasonal smoother  : {seasonal}",
            f"Trend smoother     : {trend}",
            f"Low-pass smoother  : {low_pass}",
            f"Residual std. dev. : {np.std(self.residuals):.6f}",
        ]
        return "\n".join(lines)

    def plot(self, title=None):
        """Plot observed, trend, seasonal, and residual components."""
        import matplotlib.pyplot as plt

        from Ts.TsPlots import plot_series
        from Ts.TsPlots.style import AXIS_LABEL_FONTSIZE, TITLE_FONTSIZE

        fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        panels = [
            (self.observed, "Observed", "Value"),
            (self.trend, "Trend", "Trend"),
            (self.seasonal, "Seasonal", "Seasonal"),
            (self.residuals, "Residual", "Residual"),
        ]
        for axis, (values, panel_title, ytitle) in zip(axes, panels):
            plot_series(
                values,
                ax=axis,
                title=panel_title,
                ytitle=ytitle,
                linewidth=1.5,
                markersize=0,
                show_legend=False,
            )
        axes[-1].set_xlabel("Time", fontsize=AXIS_LABEL_FONTSIZE)
        fig.suptitle(
            title or "STL Decomposition",
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return fig, axes


class STL:
    """Seasonal-Trend decomposition using LOESS.

    ``missing="raise"`` rejects non-finite input. Explicit ``"drop"`` removes
    affected observations and records their original zero-based positions in
    :attr:`dropped_positions`.
    """

    def __init__(
        self,
        data,
        period,
        seasonal=7,
        trend=None,
        low_pass=None,
        seasonal_deg=1,
        trend_deg=1,
        low_pass_deg=1,
        robust=False,
        seasonal_jump=1,
        trend_jump=1,
        low_pass_jump=1,
        missing="raise",
    ):
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim != 1:
            raise ValueError("data must be one-dimensional")
        finite_rows = np.isfinite(self.data)
        dropped_positions = _resolve_missing_rows(finite_rows, missing)
        if missing == "drop":
            self.data = self.data[finite_rows]
        else:
            self.data = self.data.copy()
        self.missing = missing
        self.dropped_positions = dropped_positions
        if (
            isinstance(period, (bool, np.bool_))
            or not isinstance(period, (int, np.integer))
            or period < 2
        ):
            raise ValueError("period must be a non-boolean integer >= 2")
        if self.data.size < 2 * period:
            raise ValueError(
                "data must contain at least two complete cycles: "
                f"need {2 * period} observations, got {self.data.size}"
            )
        self._model = _StatsmodelsSTL(
            self.data,
            period=int(period),
            seasonal=seasonal,
            trend=trend,
            low_pass=low_pass,
            seasonal_deg=seasonal_deg,
            trend_deg=trend_deg,
            low_pass_deg=low_pass_deg,
            robust=robust,
            seasonal_jump=seasonal_jump,
            trend_jump=trend_jump,
            low_pass_jump=low_pass_jump,
        )
        self.config = dict(self._model.config)
        self.period = self.config["period"]
        self.result_ = None

    def fit(self, inner_iter=None, outer_iter=None):
        """Estimate trend, seasonal, and residual components."""
        decomposed = self._model.fit(
            inner_iter=inner_iter,
            outer_iter=outer_iter,
        )
        result = STLResult(
            observed=np.asarray(decomposed.observed, dtype=float),
            trend=np.asarray(decomposed.trend, dtype=float),
            seasonal=np.asarray(decomposed.seasonal, dtype=float),
            residuals=np.asarray(decomposed.resid, dtype=float),
            weights=np.asarray(decomposed.weights, dtype=float),
            period=self.period,
            config=dict(self.config),
        )
        self.result_ = result
        return result

    def summary(self):
        """Return the fitted decomposition summary."""
        if self.result_ is None:
            self.fit()
        return self.result_.summary()
