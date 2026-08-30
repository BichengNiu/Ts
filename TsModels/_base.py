"""Base classes for all time series models in TsModel.

Provides:

- :class:`BaseModelResult` — common result container (dataclass).
- :class:`BaseModel` — abstract base class enforcing the ``fit()`` / ``summary()``
  / ``result_`` contract.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from Ts.TsUtils._calendar import validate_time_index


# Shared constants used by _garch.py and _auto.py
_VOL_TYPES = frozenset({"GARCH", "EGARCH"})
_GARCH_M_FORMS = frozenset({"vol", "var", "log"})


def _normalise_model_dates(data, dates, expected_length):
    """Return a strict copied DatetimeIndex for a public model input."""
    values = dates
    if values is None:
        index = getattr(data, "index", None)
        if isinstance(index, pd.DatetimeIndex):
            values = index
    if values is None:
        return None
    try:
        index = pd.DatetimeIndex(values)
    except (TypeError, ValueError) as error:
        raise TypeError("dates must be datetime-like") from error
    if len(index) != expected_length:
        raise ValueError(
            f"dates must contain {expected_length} entries, got {len(index)}"
        )
    validate_time_index(index, name="dates")
    return index.copy()


@dataclass(frozen=True)
class _PredictionWindow:
    """Resolved positions for a unified prediction request."""

    start: int
    end: int
    size: int
    in_sample_size: int
    forecast_steps: int
    forecast_skip: int

    @property
    def has_forecast(self):
        """Return whether the request reaches beyond the fitted sample."""
        return self.forecast_steps > 0


def _resolve_prediction_window(nobs, start, end):
    """Validate a prediction range and resolve sample/forecast slices."""
    for name, value in (("start", start), ("end", end)):
        if value is None and name == "end":
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise TypeError(f"{name} must be a non-negative integer")

    start = int(start)
    end = nobs - 1 if end is None else int(end)
    if start < 0 or end < 0:
        raise ValueError("start and end must be non-negative")
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    in_sample_end = min(end, nobs - 1)
    in_sample_size = max(0, in_sample_end - start + 1)
    return _PredictionWindow(
        start=start,
        end=end,
        size=end - start + 1,
        in_sample_size=in_sample_size,
        forecast_steps=max(0, end - nobs + 1),
        forecast_skip=max(0, start - nobs),
    )


def _autoscale_y_to_visible_data(axis) -> None:
    """Fit the y-axis limits to data visible within the current x-limits."""
    x_min, x_max = axis.get_xlim()
    x_min, x_max = sorted((float(x_min), float(x_max)))
    visible_values = []

    for line in axis.lines:
        if line.get_transform() != axis.transData:
            continue
        try:
            x_values = np.ma.asarray(line.get_xdata(orig=False), dtype=float)
            y_values = np.ma.asarray(line.get_ydata(orig=False), dtype=float)
            if x_values.shape != y_values.shape:
                continue
            mask = (
                np.isfinite(x_values)
                & np.isfinite(y_values)
                & (x_values >= x_min)
                & (x_values <= x_max)
            )
            visible_values.extend(np.asarray(y_values[mask], dtype=float))
        except (TypeError, ValueError):
            continue

    for collection in axis.collections:
        if collection.get_transform() != axis.transData:
            continue
        for path in collection.get_paths():
            vertices = np.asarray(path.vertices, dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] < 2:
                continue
            mask = (
                np.isfinite(vertices[:, 0])
                & np.isfinite(vertices[:, 1])
                & (vertices[:, 0] >= x_min)
                & (vertices[:, 0] <= x_max)
            )
            visible_values.extend(vertices[mask, 1])

    if not visible_values:
        return
    values = np.asarray(visible_values, dtype=float)
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    padding = span * 0.05
    if padding == 0.0:
        padding = abs(lower) * 0.05
    if padding == 0.0:
        padding = 1.0
    axis.set_ylim(lower - padding, upper + padding)


@dataclass
class PredictResult:
    """Container for unified prediction output.

    Returned by :meth:`BaseModelResult.predict` across all model types.

    Parameters
    ----------
    mean : np.ndarray
        Predicted values, shape ``(n_periods,)`` or ``(n_periods, k)`` for VAR.
    lower : np.ndarray or None
        Lower bound of confidence interval, same shape as *mean*.
    upper : np.ndarray or None
        Upper bound of confidence interval, same shape as *mean*.
    is_oos : np.ndarray
        Boolean mask marking periods beyond the fitted sample.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsModels import PredictResult
    >>> result = PredictResult(
    ...     mean=np.array([1.0, 2.0]),
    ...     lower=np.array([0.5, 1.5]),
    ...     upper=np.array([1.5, 2.5]),
    ...     is_oos=np.array([False, True]),
    ... )
    >>> result.is_oos.tolist()
    [False, True]
    """

    mean: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    is_oos: np.ndarray
    _full_data: np.ndarray | None = None
    _full_fitted: np.ndarray | None = None
    _full_lower: np.ndarray | None = None
    _full_upper: np.ndarray | None = None
    _start: int | None = None

    def plot(self, ci=False, title=None, xlim=None):
        """Plot prediction results: actual, fitted, forecast, and CI.

        Parameters
        ----------
        ci : bool
            Whether to draw confidence interval bands (default ``False``).
        title : str, optional
            Chart title. Auto-generated when ``None``.
        xlim : tuple of (float, float), optional
            Set x-axis limits, e.g. ``(9800, 10020)`` to zoom in on a
            specific time window. The y-axis is automatically fitted to the
            data visible in that window.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes

        Examples
        --------
        >>> from Ts.TsModels import SARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(n=60, seed=42).data
        >>> result = SARIMAX(data).fit()
        >>> prediction = result.predict(start=60, end=62)
        >>> fig, ax = prediction.plot(ci=True)
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        from Ts.TsPlots.style import (
            AXIS_LABEL_FONTSIZE,
            BLACK,
            DARK_RED,
            GRAY,
            GRID_GRAY,
            FIGSIZE,
            LEGEND_FONTSIZE,
            TITLE_FONTSIZE,
            TIGHT_PAD,
            _ensure_fonts,
            style_axes,
        )

        _ensure_fonts()
        fig, ax = plt.subplots(figsize=FIGSIZE)

        has_any = False

        # Layer 1: full historical data
        if self._full_data is not None:
            ax.plot(
                np.arange(len(self._full_data)),
                self._full_data,
                color=BLACK,
                linestyle="-",
                linewidth=1.5,
                label="Actual",
            )
            has_any = True

        # Layer 2: full in-sample fitted values
        if self._full_fitted is not None:
            fitted_x = np.arange(len(self._full_fitted))
            ax.plot(
                fitted_x,
                self._full_fitted,
                color=DARK_RED,
                linestyle="-",
                linewidth=1.5,
                label="Fitted",
            )
            has_any = True

        # Layer 3: forecast (OOS portion only)
        oos_mask = np.asarray(self.is_oos, dtype=bool)
        if np.any(oos_mask) and self._start is not None:
            oos_indices = np.where(oos_mask)[0]
            first_oos = oos_indices[0]
            mean_arr = np.asarray(self.mean)

            fc_x = np.arange(self._start, self._start + len(mean_arr))[oos_mask]
            fc_mean = mean_arr[oos_mask]
            forecast_start = self._start + first_oos
            anchor_x = forecast_start - 1

            # Bridge: connect the period immediately before the forecast to
            # its first point.  This makes the forecast line continuous even
            # when prediction begins in the middle of the full sample.
            if first_oos > 0:
                bridge_x = np.concatenate([[anchor_x], fc_x])
                bridge_y = np.concatenate(
                    [
                        [mean_arr[first_oos - 1]],
                        fc_mean,
                    ]
                )
            elif self._full_data is not None and 0 <= anchor_x < len(self._full_data):
                bridge_x = np.concatenate([[anchor_x], fc_x])
                bridge_y = np.concatenate([[self._full_data[anchor_x]], fc_mean])
            else:
                bridge_x, bridge_y = fc_x, fc_mean

            ax.plot(
                bridge_x,
                bridge_y,
                color=DARK_RED,
                linestyle="--",
                linewidth=1.5,
                label="Forecast",
            )
            has_any = True

            # Layer 4: forecast confidence interval
            if ci and self.lower is not None and self.upper is not None:
                fc_lower = np.asarray(self.lower)[oos_mask]
                fc_upper = np.asarray(self.upper)[oos_mask]
                ci_x = fc_x
                ci_lower = fc_lower
                ci_upper = fc_upper

                ax.fill_between(
                    ci_x,
                    ci_lower,
                    ci_upper,
                    color=GRAY,
                    alpha=0.15,
                    linewidth=0,
                    label="Forecast 95% CI",
                )

            # Layer 5: forecast start divider
            ax.axvline(
                x=forecast_start,
                color=GRID_GRAY,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
            )
        elif not has_any:
            ax.plot(
                np.arange(len(self.mean)),
                np.asarray(self.mean),
                color=BLACK,
                linestyle="-",
                linewidth=1.5,
                label="Predicted",
            )
            has_any = True

        style_axes(ax)
        ax.set_xlabel("Time", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Value", fontsize=AXIS_LABEL_FONTSIZE)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if title is None:
            title = "Prediction Results"
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="normal")

        if xlim is not None:
            ax.set_xlim(xlim)
            _autoscale_y_to_visible_data(ax)

        if has_any:
            ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)

        fig.tight_layout(pad=TIGHT_PAD)
        return fig, ax


@dataclass
class BaseModelResult:
    """Common result container for all TsModel model estimation results.

    Subclasses add model-specific fields and methods (predict, forecast, etc.).

    Parameters
    ----------
    model_type : str
        Model identifier (e.g. ``"SARIMAX"``, ``"ARCH"``, ``"GARCH"``).
    params : dict
        Estimated parameter values keyed by name.
    std_errors : dict
        Standard errors keyed by parameter name.
    p_values : dict
        p-values keyed by parameter name.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    log_likelihood : float
        Maximised log-likelihood.
    residuals : np.ndarray
        Statistically valid residual series. Its length may be less than
        *nobs* when a model discards state-initialization periods. Use
        :attr:`standardized_residuals` for residuals divided by their
        population standard deviation.
    fitted_values : np.ndarray or None
        Fitted values (length *nobs*). May be ``None`` for pure volatility
        models.
    nobs : int
        Effective number of observations used in estimation.
    data : np.ndarray
        Original input data series.

    Examples
    --------
    >>> from Ts.TsModels import BaseModelResult, SARIMAX
    >>> from Ts.TsSims import simulate_sarima
    >>> data = simulate_sarima(n=50, order=(1, 0, 0), seed=42).data
    >>> result = SARIMAX(data, order=(1, 0, 0)).fit()
    >>> isinstance(result, BaseModelResult)
    True
    """

    model_type: str
    params: dict
    std_errors: dict
    p_values: dict
    aic: float
    bic: float
    log_likelihood: float
    residuals: np.ndarray
    fitted_values: np.ndarray | None
    nobs: int
    data: np.ndarray
    _parameter_covariance: np.ndarray | None = None
    _parameter_names: tuple[str, ...] = ()

    def __post_init__(self):
        """Copy and validate optional parameter-inference metadata."""
        self._parameter_names = tuple(self._parameter_names)
        if self._parameter_covariance is None:
            if self._parameter_names:
                raise ValueError(
                    "parameter names require a parameter covariance matrix"
                )
            return

        covariance = np.array(self._parameter_covariance, dtype=float, copy=True)
        expected = len(self._parameter_names)
        if covariance.shape != (expected, expected):
            raise ValueError(
                "parameter covariance shape must match the parameter names"
            )
        if len(set(self._parameter_names)) != expected:
            raise ValueError("parameter names must be unique")
        self._parameter_covariance = covariance

    def _parameter_positions(self, parameters):
        """Resolve an optional public parameter subset to covariance positions."""
        if self._parameter_covariance is None:
            raise NotImplementedError(
                f"parameter correlation is not available for {self.model_type}"
            )
        if parameters is None:
            return list(range(len(self._parameter_names)))
        if isinstance(parameters, str):
            raise TypeError("parameters must be a sequence of parameter names")
        try:
            selected = tuple(parameters)
        except TypeError as error:
            raise TypeError(
                "parameters must be a sequence of parameter names"
            ) from error
        if not selected:
            raise ValueError("parameters must not be empty")
        if any(not isinstance(name, str) for name in selected):
            raise TypeError("every parameter name must be a string")
        if len(set(selected)) != len(selected):
            raise ValueError("parameters must not contain duplicates")

        positions = {name: index for index, name in enumerate(self._parameter_names)}
        unknown = [name for name in selected if name not in positions]
        if unknown:
            raise ValueError("unknown parameter names: " + ", ".join(unknown))
        return [positions[name] for name in selected]

    def parameter_correlation(self, parameters=None) -> pd.DataFrame:
        """Return the estimated-parameter correlation matrix.

        The matrix describes joint estimation uncertainty. Large absolute
        correlations can indicate weak parameter identification or redundant
        dynamics, but they do not replace stationarity roots, invertibility
        roots, or volatility-persistence diagnostics.

        Parameters
        ----------
        parameters : sequence of str, optional
            Ordered subset of estimated parameter names. By default all
            estimated parameters are returned.

        Returns
        -------
        pandas.DataFrame
            Symmetric correlation matrix with parameter names on both axes.

        Raises
        ------
        NotImplementedError
            If the fitted model does not expose parameter covariance.
        ValueError
            If covariance is non-finite, non-symmetric, has non-positive
            marginal variances, or a selected parameter name is unknown.

        Examples
        --------
        >>> from Ts.TsModels import SARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(n=80, order=(1, 0, 0), seed=42).data
        >>> result = SARIMAX(data, order=(1, 0, 0)).fit()
        >>> result.parameter_correlation().shape[0] == len(result.params)
        True
        """
        selected = self._parameter_positions(parameters)
        covariance = self._parameter_covariance[np.ix_(selected, selected)].copy()
        names = [self._parameter_names[index] for index in selected]

        if not np.all(np.isfinite(covariance)):
            raise ValueError("parameter covariance must contain only finite values")
        scale = max(1.0, float(np.max(np.abs(covariance))))
        tolerance = np.finfo(float).eps * scale * max(100.0, len(names) * 10.0)
        if not np.allclose(covariance, covariance.T, rtol=1e-7, atol=tolerance):
            raise ValueError("parameter covariance must be symmetric")
        covariance = (covariance + covariance.T) / 2.0

        variances = np.diag(covariance)
        if np.any(variances <= 0.0):
            raise ValueError("parameter variances must be positive")
        standard_errors = np.sqrt(variances)
        correlation = covariance / np.outer(standard_errors, standard_errors)
        correlation = np.clip(correlation, -1.0, 1.0)
        np.fill_diagonal(correlation, 1.0)
        return pd.DataFrame(correlation, index=names, columns=names)

    def plot_parameter_correlation(
        self,
        parameters=None,
        *,
        annotate=True,
        decimals=2,
        title=None,
        ax=None,
    ):
        """Plot the estimated-parameter correlation matrix.

        Parameters
        ----------
        parameters : sequence of str, optional
            Ordered subset of estimated parameter names. By default all
            estimated parameters are plotted.
        annotate : bool, default True
            Whether to write each correlation value inside its cell.
        decimals : int, default 2
            Number of decimal places used for cell annotations.
        title : str, optional
            Chart title. Defaults to ``"<model> Parameter Correlation"``.
        ax : matplotlib.axes.Axes, optional
            Existing axes. A new figure and axes are created when omitted.

        Returns
        -------
        tuple
            ``(figure, axes)`` containing the correlation heatmap.

        Examples
        --------
        >>> from Ts.TsModels import GARCH
        >>> from Ts.TsSims import simulate_garch
        >>> data = simulate_garch(n=150, seed=42).data
        >>> result = GARCH(data).fit()
        >>> fig, ax = result.plot_parameter_correlation(annotate=False)
        """
        from Ts.TsPlots import plot_correlation_matrix

        matrix = self.parameter_correlation(parameters=parameters)
        resolved_title = title or f"{self.model_type} Parameter Correlation"
        return plot_correlation_matrix(
            matrix,
            annotate=annotate,
            decimals=decimals,
            title=resolved_title,
            ax=ax,
        )

    @property
    def standardized_residuals(self) -> np.ndarray:
        """Return residuals divided by their population standard deviation.

        The residual mean is not subtracted. One-dimensional residuals use one
        scale for the full statistically valid sample. Two-dimensional
        residuals are scaled separately by equation (column). The original
        :attr:`residuals` array is not modified.

        Returns
        -------
        numpy.ndarray
            Floating-point residuals divided by ``std(ddof=0)``.

        Raises
        ------
        ValueError
            If a required residual standard deviation is zero or non-finite.
        """
        residuals = np.asarray(self.residuals, dtype=float)
        axis = 0 if residuals.ndim == 2 else None
        scale = np.std(residuals, axis=axis, ddof=0)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError(
                "residual standard deviation must be positive and finite"
            )
        return residuals / scale

    def summary(self) -> str:
        """Return a formatted parameter summary string.

        Returns
        -------
        str

        Examples
        --------
        >>> from Ts.TsModels import SARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> result = SARIMAX(simulate_sarima(n=40, seed=42).data).fit()
        >>> "SARIMAX" in result.summary()
        True
        """
        lines = [
            f"{self.model_type} Model Estimation Result",
            "=" * 50,
            f"Observations       : {self.nobs}",
            f"Log-Likelihood     : {self.log_likelihood:.4f}",
            f"AIC                : {self.aic:.4f}",
            f"BIC                : {self.bic:.4f}",
            "",
            "Parameter Estimates:",
            "-" * 50,
        ]
        for name in self.params:
            val = self.params[name]
            se = self.std_errors.get(name)
            pv = self.p_values.get(name)
            se_str = f"{se:.4f}" if se is not None else "N/A"
            pv_str = f"{pv:.4f}" if pv is not None else "N/A"
            lines.append(f"  {name:<20s} {val:>10.4f}  ({se_str})  p={pv_str}")
        return "\n".join(lines)

    def _fitted_values_for_plot(self):
        """Return fitted values with any model-specific display masking."""
        return self.fitted_values

    def _diagnostic_label(self, name):
        """Return the display name of one equation in 2D diagnostics panels."""
        return name

    def _variable_names(self):
        """Return display names for the columns of a 2D model result."""
        k = self.data.shape[1]
        names = getattr(self, "_data_names", None)
        if names is None:
            names = [f"Variable {i + 1}" for i in range(k)]
        return [str(name) for name in names]

    def plot_fit(self, title=None):
        """Plot actual vs fitted values.

        One-dimensional series are drawn in a single panel; multi-variable
        models (``data`` with shape ``(n, k)``) produce one subplot per
        variable, aligned to the fitted sample.

        Uses :func:`TsPlots.plot_series` for unified styling.

        Parameters
        ----------
        title : str, optional
            Chart title.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes or numpy.ndarray of matplotlib.axes.Axes

        Examples
        --------
        >>> from Ts.TsModels import SARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> result = SARIMAX(simulate_sarima(n=40, seed=42).data).fit()
        >>> fig, ax = result.plot_fit(title="Observed and fitted")
        >>> ax.get_title()
        'Observed and fitted'
        """
        import matplotlib.pyplot as plt

        from Ts.TsPlots import plot_series
        from Ts.TsPlots.style import TITLE_FONTSIZE, FIGSIZE, draw_suptitle

        if title is None:
            title = f"{self.model_type}: Actual vs Fitted"

        if np.ndim(self.data) == 1:
            plot_data = {"Actual": self.data}
            if self.fitted_values is not None:
                plot_data["Fitted"] = self._fitted_values_for_plot()
            fig, ax = plot_series(
                plot_data,
                facet=False,
                title=title,
                ytitle="Value",
                xtitle="Time",
            )
            return fig, ax

        names = self._variable_names()
        k = len(names)
        fig, axes = plt.subplots(k, 1, figsize=(FIGSIZE[0], 3 * k), squeeze=False)
        axes = axes[:, 0]

        fitted = self._fitted_values_for_plot()
        data_aligned = (
            self.data[-fitted.shape[0] :] if fitted is not None else self.data
        )

        for i, name in enumerate(names):
            plot_data = {f"{name} (actual)": data_aligned[:, i]}
            if fitted is not None:
                plot_data[f"{name} (fitted)"] = fitted[:, i]
            plot_series(
                plot_data,
                ax=axes[i],
                facet=False,
                title=f"Variable: {name}",
                ytitle="Value",
            )

        draw_suptitle(fig, title)
        fig.tight_layout()
        return fig, axes

    def plot_diagnostics(self, title=None):
        """Plot standardized-residual diagnostics.

        One-dimensional residuals produce a 2-by-2 figure: standardized
        residuals over time, their histogram, ACF, and PACF.  The histogram
        and correlogram panels contain only their respective plots.
        Multi-variable residuals (shape ``(n, k)``)
        produce a ``k``-by-3 grid with standardized residuals, ACF, and PACF
        for each variable.  Diagnostic panels omit y-axis titles and legends
        by default.

        Parameters
        ----------
        title : str, optional
            Suptitle for the figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : tuple or numpy.ndarray of matplotlib.axes.Axes
            Flat row-major tuple (1D) or ``k``-by-3 array (2D) of axes.

        Examples
        --------
        >>> from Ts.TsModels import SARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> result = SARIMAX(simulate_sarima(n=120, seed=42).data).fit()
        >>> fig, axes = result.plot_diagnostics()
        >>> len(axes)
        4
        """
        import matplotlib.pyplot as plt

        from Ts.TsPlots import plot_acf, plot_pacf, plot_series
        from Ts.TsPlots.style import (
            AXIS_LABEL_FONTSIZE,
            BLACK,
            WHITE,
            TITLE_PAD,
            TITLE_FONTSIZE,
            _ensure_fonts,
            draw_suptitle,
            style_axes,
        )

        if title is None:
            title = f"{self.model_type}: Diagnostic Plots"

        diagnostic_residuals = self.standardized_residuals

        def suppress_axis_decorations(figure):
            """Hide diagnostic y-axis titles and legend handles."""
            for axis in figure.axes:
                axis.set_ylabel("")
                for line in axis.lines:
                    line.set_label("_nolegend_")
                if axis.get_legend() is not None:
                    axis.get_legend().remove()
            for legend in list(figure.legends):
                legend.remove()

        if np.ndim(self.residuals) > 1:
            k = self.residuals.shape[1]
            names = self._variable_names()
            fig, axes = plt.subplots(k, 3, figsize=(14, 3 * k), squeeze=False)

            for i, name in enumerate(names):
                label = self._diagnostic_label(name)
                plot_series(
                    diagnostic_residuals[:, i],
                    ax=axes[i, 0],
                    title=f"{label} Standardized Residuals",
                    xtitle="",
                    ytitle="Standardized Residual",
                    show_legend=False,
                )
                plot_acf(
                    diagnostic_residuals[:, i],
                    ax=axes[i, 1],
                    title=f"{label} Standardized Residual ACF",
                )
                plot_pacf(
                    diagnostic_residuals[:, i],
                    ax=axes[i, 2],
                    title=f"{label} Standardized Residual PACF",
                )

            draw_suptitle(fig, title)
            suppress_axis_decorations(fig)
            fig.tight_layout()
            return fig, axes

        _ensure_fonts()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax_residuals, ax_histogram, ax_acf, ax_pacf = axes.flat

        plot_series(
            diagnostic_residuals,
            ax=ax_residuals,
            title="Standardized Residuals",
            xtitle="",
            ytitle="Standardized Residual",
            show_legend=False,
        )

        ax_histogram.hist(
            diagnostic_residuals,
            bins="auto",
            color=BLACK,
            edgecolor=WHITE,
            alpha=0.85,
        )
        ax_histogram.set_title(
            "Standardized Residual Histogram",
            fontsize=TITLE_FONTSIZE,
            fontweight="normal",
            pad=TITLE_PAD,
        )
        ax_histogram.set_xlabel(
            "Standardized Residual", fontsize=AXIS_LABEL_FONTSIZE
        )
        ax_histogram.set_ylabel("Frequency", fontsize=AXIS_LABEL_FONTSIZE)
        style_axes(ax_histogram)

        plot_acf(
            diagnostic_residuals,
            ax=ax_acf,
            title="Standardized Residual ACF",
            zero_lag=False,
        )

        plot_pacf(
            diagnostic_residuals,
            ax=ax_pacf,
            title="Standardized Residual PACF",
        )

        draw_suptitle(fig, title)
        suppress_axis_decorations(fig)
        fig.tight_layout()
        return fig, (ax_residuals, ax_histogram, ax_acf, ax_pacf)

    def _white_noise_and_normality(self, residuals, lags):
        """Run the shared Ljung-Box white-noise and Jarque-Bera tests."""
        from Ts.TsTests import LjungBoxTest, NormalityTest

        wn = LjungBoxTest(residuals, lags=lags, apply_squared=False)
        norm = NormalityTest(residuals)
        return wn.fit(), norm.fit()

    def test_residuals(self, lags=10):
        """Run residual diagnostic tests.

        For one-dimensional residuals this executes four tests:
        - White noise: Ljung-Box on raw residuals (H0: no autocorrelation)
        - Normality: Jarque-Bera (H0: normally distributed)
        - ARCH effects: Ljung-Box on squared residuals (H0: no ARCH)
        - ARCH effects: Engle LM (H0: no ARCH)

        For multi-variable residuals (shape ``(n, k)``) the same four tests
        run for each equation and are returned keyed by variable name.

        Parameters
        ----------
        lags : int
            Number of lags for autocorrelation-based tests.

        Returns
        -------
        ResidualTestResults or dict
            Container with ``.white_noise``, ``.normality``, ``.ljung_box``,
            ``.engle_lm`` attributes for 1D residuals, or a mapping from
            variable name to :class:`ResidualTestResults` for 2D residuals.

        Examples
        --------
        >>> from Ts.TsModels import SARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> result = SARIMAX(simulate_sarima(n=60, seed=42).data).fit()
        >>> tests = result.test_residuals(lags=5)
        >>> tests.white_noise.lags
        5
        """
        from Ts.TsTests import EngleLMTest, LjungBoxTest

        diagnostic_residuals = self.residuals

        if np.ndim(diagnostic_residuals) > 1:
            results = {}
            for i, name in enumerate(self._variable_names()):
                resid_i = diagnostic_residuals[:, i]
                wn_result, norm_result = self._white_noise_and_normality(
                    resid_i, lags
                )
                lb_result = LjungBoxTest(resid_i, lags=lags).fit()
                lm_result = EngleLMTest(resid_i, lags=lags).fit()
                results[name] = ResidualTestResults(
                    white_noise=wn_result,
                    normality=norm_result,
                    ljung_box=lb_result,
                    engle_lm=lm_result,
                )
            return results

        wn_result, norm_result = self._white_noise_and_normality(
            diagnostic_residuals, lags
        )
        lb_result = LjungBoxTest(diagnostic_residuals, lags=lags).fit()
        lm_result = EngleLMTest(diagnostic_residuals, lags=lags).fit()

        return ResidualTestResults(
            white_noise=wn_result,
            normality=norm_result,
            ljung_box=lb_result,
            engle_lm=lm_result,
        )

    def long_run_equilibrium(self):
        """Return the long-run equilibrium of the estimated model.

        Subclasses override this to return the appropriate long-run concept:

        - **SARIMAX** (stationary, no differencing): unconditional mean
          :math:`E[y_t]` (``float``).
        - **VAR** (stable, no time trend): unconditional mean vector
          :math:`E[\\mathbf{y}_t]` (``np.ndarray`` of shape ``(k,)``).
        - **GARCH** / **GJR-GARCH** (covariance stationary): unconditional
          variance :math:`\\operatorname{Var}[\\varepsilon_t]` (``float``).

        Returns ``None`` when the concept is not applicable (non-stationary,
        deterministic trend, differenced series, EGARCH, IGARCH, etc.).

        Returns
        -------
        float or np.ndarray or None
            The long-run equilibrium value, or ``None`` if not applicable.

        Examples
        --------
        >>> from Ts.TsModels import SARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(n=100, order=(1, 0, 0), ar=[.5], seed=42).data
        >>> result = SARIMAX(data, order=(1, 0, 0), trend="c").fit()
        >>> result.long_run_equilibrium() is None
        False
        """
        return


@dataclass
class ResidualTestResults:
    """Container for residual diagnostic test results.

    Wraps white noise, normality, Ljung-Box, and Engle LM test results.

    Parameters
    ----------
    white_noise : LjungBoxTestResult
        Ljung-Box Q-test on raw residuals (H0: no autocorrelation).
    normality : NormalityTestResult
        Jarque-Bera normality test.
    ljung_box : LjungBoxTestResult
        Ljung-Box Q-test on squared residuals (H0: no ARCH effects).
    engle_lm : EngleLMTestResult
        Engle LM test for ARCH effects.

    Examples
    --------
    >>> from Ts.TsModels import ResidualTestResults, SARIMAX
    >>> from Ts.TsSims import simulate_sarima
    >>> fitted = SARIMAX(simulate_sarima(n=60, seed=42).data).fit()
    >>> diagnostics = fitted.test_residuals(lags=5)
    >>> isinstance(diagnostics, ResidualTestResults)
    True
    """

    white_noise: object
    normality: object
    ljung_box: object
    engle_lm: object

    def __str__(self) -> str:
        wn = self.white_noise
        nm = self.normality
        lb = self.ljung_box
        lm = self.engle_lm

        wn_conc = "Autocorrelation" if wn.pvalue < 0.05 else "White noise"
        nm_conc = "Non-normal" if nm.pvalue < 0.05 else "Normal"
        lb_conc = "ARCH effects" if lb.pvalue < 0.05 else "No ARCH effects"
        lm_conc = "ARCH effects" if lm.pvalue < 0.05 else "No ARCH effects"

        lines = [
            "Residual Diagnostic Tests",
            "=" * 75,
            f"{'Test':<28s} {'Statistic':>10s} {'p-value':>10s}  Conclusion",
            "-" * 75,
            f"{'White Noise (Ljung-Box)':<28s} {wn.statistic:>10.3f} {wn.pvalue:>10.4f}  {wn_conc}",
            f"{'Normality (Jarque-Bera)':<28s} {nm.statistic:>10.3f} {nm.pvalue:>10.4f}  {nm_conc}",
            f"{'ARCH Effect (Ljung-Box)':<28s} {lb.statistic:>10.3f} {lb.pvalue:>10.4f}  {lb_conc}",
            f"{'ARCH Effect (Engle LM)':<28s} {lm.statistic:>10.3f} {lm.pvalue:>10.4f}  {lm_conc}",
            "",
            "-" * 75,
            "",
            str(wn),
            "",
            str(nm),
            "",
            str(lb),
            "",
            str(lm),
        ]
        return "\n".join(lines)

    def summary(self) -> str:
        """Return formatted summary string (same as ``__str__``).

        Returns
        -------
        str

        Examples
        --------
        >>> from Ts.TsModels import SARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> result = SARIMAX(simulate_sarima(n=80, seed=42).data).fit()
        >>> tests = result.test_residuals(lags=5)
        >>> "Residual Diagnostic Tests" in tests.summary()
        True
        """
        return str(self)


class BaseModel(ABC):
    """Abstract base class for all time series models in TsModel.

    Every model must implement :meth:`fit` and expose a :attr:`result_`
    attribute. The :meth:`summary` method returns a formatted string.

    Attributes
    ----------
    result_ : BaseModelResult or None
        Fitted result stored by concrete subclasses.

    Examples
    --------
    Use a concrete estimator rather than instantiating this abstract class.

    >>> from Ts.TsModels import BaseModel, SARIMAX
    >>> model = SARIMAX([1.0] * 20)
    >>> isinstance(model, BaseModel)
    True
    """

    result_: BaseModelResult | None = None

    @abstractmethod
    def fit(self) -> BaseModelResult:
        """Estimate the model and return a result object.

        The result is also stored in :attr:`result_`.

        Returns
        -------
        BaseModelResult

        Examples
        --------
        Use a concrete implementation such as :class:`SARIMAX`:

        >>> from Ts.TsModels import SARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> model = SARIMAX(simulate_sarima(n=60, seed=42).data)
        >>> result = model.fit()
        >>> model.result_ is result
        True
        """
        ...

    _evaluation_target_name = "observed"
    _backcast_target_name = "observed"

    def _clone_for_evaluation(self, data, exog=None, *, dates=None):
        """Clone this configuration with an isolated evaluation data window."""
        cloned = copy.copy(self)
        cloned.data = np.array(data, dtype=float, copy=True)
        cloned.result_ = None
        if hasattr(cloned, "dropped_positions"):
            cloned.dropped_positions = ()
        if hasattr(cloned, "exog"):
            cloned.exog = (
                None if exog is None else np.array(exog, dtype=float, copy=True)
            )
        if hasattr(cloned, "dates"):
            cloned.dates = None if dates is None else dates.copy()
        return cloned

    def _evaluation_predict_kwargs(self, start, stop):
        """Return model-specific context for an evaluation forecast window."""
        del start, stop
        return {}

    def _evaluation_actual(self, observed, train_data):
        """Return the observable target used to score forecasts."""
        del train_data
        return np.array(observed, dtype=float, copy=True)

    def _validate_evaluation(self, context):
        """Validate model-specific requirements for an evaluation method."""
        del context

    def backcast(self, steps, alpha=0.05):
        """Estimate pre-sample values by reverse-time refitting.

        The observed series is reversed, the same configured model is fitted
        to that reversed series, and its forecasts are reversed back into
        chronological pre-sample order. Deterministic trends are therefore
        re-estimated in reverse time. This is a statistical reconstruction,
        not a causal claim about unobserved history.

        Parameters
        ----------
        steps : int
            Number of pre-sample periods to estimate.
        alpha : float
            Significance level used for prediction intervals.

        Returns
        -------
        BackcastResult

        Examples
        --------
        >>> from Ts.TsModels import SARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> model = SARIMAX(simulate_sarima(n=40, seed=42).data)
        >>> result = model.backcast(steps=3)
        >>> result.mean.shape
        (3,)
        """
        from Ts.TsModels._backcast import backcast_model

        return backcast_model(self, steps=steps, alpha=alpha)

    def summary(self) -> str:
        """Return a formatted summary string.

        Automatically calls :meth:`fit` if :attr:`result_` is ``None``.

        Returns
        -------
        str
            Formatted summary from the concrete fitted result.

        Examples
        --------
        >>> from Ts.TsModels import SARIMAX
        >>> model = SARIMAX([1.0] * 20)
        >>> "SARIMAX" in model.summary()
        True
        """
        if self.result_ is None:
            self.fit()
        return self.result_.summary()
