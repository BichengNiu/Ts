"""SARIMA model estimation via statsmodels SARIMAX.

Provides :class:`SARIMA` and :class:`SARIMAResult`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from Ts.TsModels._base import (
    BaseModel,
    BaseModelResult,
    PredictResult,
    _resolve_prediction_window,
    _validate_prediction_alpha,
)
from Ts.TsModels._intervention import _validate_datetime_index


@dataclass(frozen=True)
class _SARIMAInputs:
    endog: np.ndarray
    dates: pd.DatetimeIndex | None
    exog: np.ndarray | None
    exog_names: tuple[str, ...]
    future_exog: pd.DataFrame | None


def _numeric_array(values, name):
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error


def _normalise_exog_names(names, width):
    if names is None:
        raise ValueError("exog_names is required for array exog")
    names = tuple(names)
    if len(names) != width:
        raise ValueError(
            f"exog_names must contain one name per exog column ({width})"
        )
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("exog_names must contain non-empty strings")
    names = tuple(name.strip() for name in names)
    if len(set(names)) != len(names):
        raise ValueError("exog_names must be unique")
    return names


def _normalise_data_and_dates(data, dates):
    if isinstance(data, pd.Series):
        if dates is not None:
            raise ValueError(
                "dates must not be provided when data is a pandas Series"
            )
        values = _numeric_array(data.to_numpy(), "data")
        data_dates = (
            _validate_datetime_index(data.index, "data dates")
            if isinstance(data.index, pd.DatetimeIndex)
            else None
        )
    else:
        values = _numeric_array(data, "data")
        data_dates = (
            None
            if dates is None
            else _validate_datetime_index(dates, "dates")
        )
    if values.ndim != 1:
        raise ValueError(
            f"data must be one-dimensional, got shape {values.shape}"
        )
    if data_dates is not None and len(data_dates) != len(values):
        raise ValueError(
            "dates must contain exactly one value per data observation"
        )
    return values.copy(), data_dates


def _normalise_dataframe_exog(exog, dates, exog_names):
    if exog_names is not None:
        raise ValueError(
            "exog_names must not be provided when exog is a DataFrame"
        )
    if dates is None:
        raise ValueError(
            "DataFrame exog requires dated data or an explicit dates argument"
        )
    index = _validate_datetime_index(exog.index, "exog dates")
    if str(index.tz) != str(dates.tz):
        raise ValueError("exog dates timezone must match data dates timezone")
    names = _normalise_exog_names(tuple(exog.columns), exog.shape[1])

    missing_dates = dates.difference(index)
    if len(missing_dates):
        raise ValueError(
            "exog is missing historical date "
            f"{missing_dates[0].isoformat()}"
        )
    historical_mask = index <= dates[-1]
    extra_historical = index[historical_mask & ~index.isin(dates)]
    if len(extra_historical):
        raise ValueError(
            "exog contains extra historical date "
            f"{extra_historical[0].isoformat()}"
        )

    frame = exog.copy()
    frame.columns = names
    historical = _numeric_array(frame.loc[dates, list(names)], "exog")
    future_frame = frame.loc[index > dates[-1], list(names)].copy()
    if future_frame.empty:
        future_frame = None
    else:
        future_values = _numeric_array(future_frame, "future exog")
        if not np.all(np.isfinite(future_values)):
            raise ValueError("future exog contains non-finite values")
        future_frame = pd.DataFrame(
            future_values,
            index=future_frame.index.copy(),
            columns=names,
        )
    return historical, names, future_frame


def _normalise_array_exog(exog, endog_length, exog_names):
    values = _numeric_array(exog, "exog")
    if values.ndim != 2:
        raise ValueError(
            f"exog must be two-dimensional, got shape {values.shape}"
        )
    if len(values) != endog_length:
        raise ValueError(
            f"exog has {len(values)} observations; expected "
            f"{endog_length} observations"
        )
    names = _normalise_exog_names(exog_names, values.shape[1])
    return values.copy(), names


def _normalise_sarima_inputs(
    data,
    *,
    dates=None,
    exog=None,
    exog_names=None,
    missing="raise",
):
    if missing not in {"raise", "drop"}:
        raise ValueError("missing must be 'raise' or 'drop'")
    endog, data_dates = _normalise_data_and_dates(data, dates)

    future_exog = None
    if exog is None:
        if exog_names is not None:
            raise ValueError("exog_names requires exog")
        historical_exog = None
        names = ()
    elif isinstance(exog, pd.DataFrame):
        historical_exog, names, future_exog = _normalise_dataframe_exog(
            exog,
            data_dates,
            exog_names,
        )
    else:
        historical_exog, names = _normalise_array_exog(
            exog,
            len(endog),
            exog_names,
        )

    finite = np.isfinite(endog)
    if historical_exog is not None:
        finite &= np.all(np.isfinite(historical_exog), axis=1)
    if missing == "raise" and not np.all(finite):
        raise ValueError("data or historical exog contains non-finite values")
    if missing == "drop":
        endog = endog[finite]
        if data_dates is not None:
            data_dates = data_dates[finite].copy()
        if historical_exog is not None:
            historical_exog = historical_exog[finite]

    return _SARIMAInputs(
        endog=endog.copy(),
        dates=data_dates,
        exog=(
            None
            if historical_exog is None
            else np.array(historical_exog, dtype=float, copy=True)
        ),
        exog_names=names,
        future_exog=future_exog,
    )


@dataclass
class SARIMAResult(BaseModelResult):
    """Result container for SARIMA model estimation.

    Inherits all fields from :class:`BaseModelResult` and adds SARIMA-specific
    prediction and forecasting methods.

    Parameters
    ----------
    _order : tuple or None
        SARIMA order (p, d, q).
    _seasonal_order : tuple or None
        Seasonal order (P, D, Q, s).
    _statsmodels_result : object
        Raw statsmodels SARIMAXResultsWrapper, stored for internal
        predict / forecast delegation.
    """

    _order: tuple | None = None
    _seasonal_order: tuple | None = None
    _statsmodels_result: object = None
    _trend: str = "c"

    def summary(self) -> str:
        """Return a formatted parameter summary string.

        Overrides BaseModelResult to add SARIMA-specific details (order,
        seasonal_order).
        """
        base = super().summary()
        lines = base.split("\n")
        header_lines = [lines[0]]
        if self._order:
            header_lines.append(f"Order: SARIMA{self._order}")
        if self._seasonal_order and self._seasonal_order != (0, 0, 0, 0):
            header_lines.append(f"Seasonal Order: {self._seasonal_order}")
        return "\n".join(header_lines + lines[1:])

    def predict(self, start=0, end=None, dynamic=False, alpha=0.05):
        """Return in-sample predictions and forecasts beyond the sample.

        Parameters
        ----------
        start : int
            Start index for prediction (0-based).
        end : int, optional
            End index. If > nobs-1, performs out-of-sample forecast.
            Default: nobs-1.
        dynamic : bool
            If True, use dynamic (multi-step) predictions throughout
            the entire prediction range.
        alpha : float
            Significance level for confidence intervals (default 0.05).

        Returns
        -------
        PredictResult
        """
        if self._statsmodels_result is None:
            raise RuntimeError("No fitted statsmodels result available")

        nobs = self.nobs
        window = _resolve_prediction_window(nobs, start, end)
        alpha = _validate_prediction_alpha(alpha)
        start, end = window.start, window.end
        mean = np.full(window.size, np.nan)
        lower = np.full(window.size, np.nan)
        upper = np.full(window.size, np.nan)
        is_oos = np.zeros(window.size, dtype=bool)
        has_ci = False

        if window.has_forecast:
            n_in = window.in_sample_size
            if n_in > 0:
                pred_in = self._statsmodels_result.get_prediction(
                    start=start, end=nobs - 1, dynamic=dynamic
                )
                summary_in = pred_in.summary_frame(alpha=alpha)
                mean[:n_in] = np.asarray(summary_in["mean"])
                lower[:n_in] = np.asarray(summary_in["mean_ci_lower"])
                upper[:n_in] = np.asarray(summary_in["mean_ci_upper"])
                has_ci = True

            fc = self._statsmodels_result.get_forecast(
                steps=window.forecast_steps
            )
            fc_frame = fc.summary_frame(alpha=alpha)
            forecast_slice = slice(window.forecast_skip, None)
            mean[n_in:] = np.asarray(fc_frame["mean"])[forecast_slice]
            lower[n_in:] = np.asarray(
                fc_frame["mean_ci_lower"]
            )[forecast_slice]
            upper[n_in:] = np.asarray(
                fc_frame["mean_ci_upper"]
            )[forecast_slice]
            has_ci = True
            is_oos[n_in:] = True

        else:
            # Pure in-sample prediction
            pred = self._statsmodels_result.get_prediction(
                start=start, end=end, dynamic=dynamic
            )
            summary = pred.summary_frame(alpha=alpha)
            mean = np.asarray(summary["mean"])
            lower = np.asarray(summary["mean_ci_lower"])
            upper = np.asarray(summary["mean_ci_upper"])
            has_ci = True

        if not has_ci:
            lower = None
            upper = None

        # Compute in-sample fitted CI for plot
        _full_lower = None
        _full_upper = None
        if self._statsmodels_result is not None and self.fitted_values is not None:
            full_pred = self._statsmodels_result.get_prediction(
                start=0, end=self.nobs - 1
            )
            full_frame = full_pred.summary_frame(alpha=alpha)
            _full_lower = np.asarray(full_frame["mean_ci_lower"])
            _full_upper = np.asarray(full_frame["mean_ci_upper"])

        return PredictResult(
            mean=mean,
            lower=lower,
            upper=upper,
            is_oos=is_oos,
            _full_data=self.data,
            _full_fitted=self.fitted_values,
            _full_lower=_full_lower,
            _full_upper=_full_upper,
            _start=start,
        )

    @property
    def arroots(self):
        """Autoregressive (AR) polynomial roots.

        These are the roots of the AR lag polynomial. For stationarity,
        the inverse roots (1 / arroots) must lie inside the unit circle.

        Returns
        -------
        np.ndarray
            AR polynomial roots. Empty array if the model has no AR terms.
        """
        if self._statsmodels_result is None:
            raise RuntimeError("No fitted statsmodels result available")
        return np.asarray(self._statsmodels_result.arroots)

    @property
    def maroots(self):
        """Moving-average (MA) polynomial roots.

        These are the roots of the MA lag polynomial. For invertibility,
        the inverse roots (1 / maroots) must lie inside the unit circle.

        Returns
        -------
        np.ndarray
            MA polynomial roots. Empty array if the model has no MA terms.
        """
        if self._statsmodels_result is None:
            raise RuntimeError("No fitted statsmodels result available")
        return np.asarray(self._statsmodels_result.maroots)

    def plot_roots(self, title=None):
        """Plot inverse AR and MA roots on the complex unit circle.

        Inverse AR roots are shown as blue circles (``"o"``); inverse MA
        roots as orange triangles (``"^"``). Stationarity requires all
        inverse AR roots to lie inside the unit circle. Invertibility
        requires all inverse MA roots to lie inside the unit circle.

        Uses TsPlots global style settings.

        Parameters
        ----------
        title : str, optional
            Chart title. If None, a default title is generated from the
            model order.

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

        if self._statsmodels_result is None:
            raise RuntimeError("No fitted statsmodels result available")

        ar_roots = np.asarray(self._statsmodels_result.arroots)
        ma_roots = np.asarray(self._statsmodels_result.maroots)

        fig, ax = plt.subplots(figsize=(6, 6))

        theta = np.linspace(0, 2 * np.pi, 400)
        ax.plot(
            np.cos(theta), np.sin(theta),
            color=DEFAULT_PALETTE[1], linewidth=1.0, linestyle="--",
        )

        ax.axhline(0, color=DEFAULT_PALETTE[1], linewidth=0.5, alpha=0.5)
        ax.axvline(0, color=DEFAULT_PALETTE[1], linewidth=0.5, alpha=0.5)

        if len(ar_roots) > 0:
            inv_ar = 1.0 / ar_roots
            ax.scatter(
                inv_ar.real, inv_ar.imag,
                color=DEFAULT_PALETTE[0], marker="o", s=50,
                edgecolors=DEFAULT_PALETTE[7], linewidth=0.5, zorder=5,
                label="AR roots",
            )

        if len(ma_roots) > 0:
            inv_ma = 1.0 / ma_roots
            ax.scatter(
                inv_ma.real, inv_ma.imag,
                color=DEFAULT_PALETTE[4], marker="^", s=50,
                edgecolors=DEFAULT_PALETTE[7], linewidth=0.5, zorder=5,
                label="MA roots",
            )

        ax.set_aspect("equal")
        style_axes(ax)

        # Auto-scale limits to keep unit circle and all points visible
        all_re = []
        all_im = []
        for r in list(ar_roots) + list(ma_roots):
            inv = 1.0 / r
            all_re.append(abs(inv.real))
            all_im.append(abs(inv.imag))
        margin = max(1.5, max(all_re + all_im + [0]) * 1.15)
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)

        ax.set_xlabel("Real", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Imaginary", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(labelsize=TICK_LABELSIZE)

        if len(ar_roots) > 0 or len(ma_roots) > 0:
            ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)

        if title is None:
            order_str = f"SARIMA{self._order}"
            if self._seasonal_order and self._seasonal_order != (0, 0, 0, 0):
                order_str += str(self._seasonal_order)
            title = f"{order_str}: Inverse AR and MA Roots"
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")

        fig.tight_layout(pad=1.5)
        return fig, ax

    def long_run_equilibrium(self):
        """Return the unconditional mean (long-run equilibrium) of the
        estimated SARIMA process.

        For a stationary ARMA(p,q) with constant :math:`c`:

        .. math::

            \\mu = \\frac{c}{1 - \\phi_1 - \\dots - \\phi_p}

        where :math:`c` is the ``intercept`` parameter (the constant term in
        the AR representation) and :math:`\\phi_i` are the AR coefficients
        (including seasonal AR terms).

        Returns ``None`` when the concept is not applicable:

        - ``d > 0`` or ``D > 0`` — differencing removes the mean.
        - ``trend`` includes a time component (``"t"``, ``"ct"``, ``"ctt"``)
          — trend-stationary series have no constant equilibrium.
        - AR polynomial is non-stationary (inverse roots on or outside the
          unit circle) or :math:`1 - \\sum\\phi_i \\approx 0`.

        Returns
        -------
        float or None
        """
        if self._statsmodels_result is None:
            raise RuntimeError("No fitted statsmodels result available")

        p, d, _q = self._order
        P, D, _Q, _s = self._seasonal_order

        # Differencing removes the unconditional mean
        if (d or 0) + (D or 0) > 0:
            return None

        # Trend-stationary — no constant long-run equilibrium
        if self._trend in ("t", "ct", "ctt"):
            return None

        # Check AR stationarity
        ar_roots = np.asarray(self._statsmodels_result.arroots)
        if len(ar_roots) > 0:
            inv_roots = 1.0 / ar_roots
            if np.any(np.abs(inv_roots) >= 1.0 - 1e-10):
                return None

        # trend="n" — zero-mean process
        if self._trend == "n":
            return 0.0

        # Extract AR coefficients (non-seasonal + seasonal)
        ar_sum = 0.0
        for i in range(1, p + 1):
            ar_sum += self.params.get(f"ar.L{i}", 0.0)
        for i in range(1, P + 1):
            ar_sum += self.params.get(f"ar.S.L{i}", 0.0)

        intercept = self.params.get("intercept", 0.0)
        denom = 1.0 - ar_sum
        if abs(denom) < 1e-10:
            return None

        return intercept / denom


class SARIMA(BaseModel):
    """SARIMA model estimation via statsmodels SARIMAX.

    Parameters
    ----------
    data : array-like
        Time series data (1-D).
    order : tuple
        ``(p, d, q)`` non-seasonal order.
    seasonal_order : tuple
        ``(P, D, Q, s)`` seasonal order. Default ``(0, 0, 0, 0)``.
    trend : str
        Trend specification: ``"n"`` (none), ``"c"`` (constant),
        ``"t"`` (linear), ``"ct"`` (both). Default ``"c"``.
    enforce_stationarity : bool
        Whether to enforce stationarity of the AR polynomial. Default ``True``.
    enforce_invertibility : bool
        Whether to enforce invertibility of the MA polynomial. Default ``True``.
    """

    def __init__(
        self,
        data,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
        *,
        dates=None,
        exog=None,
        exog_names=None,
        events=None,
        missing="raise",
    ):
        inputs = _normalise_sarima_inputs(
            data,
            dates=dates,
            exog=exog,
            exog_names=exog_names,
            missing=missing,
        )

        if not isinstance(order, (tuple, list)) or len(order) != 3:
            raise ValueError(
                f"order must be a tuple of (p, d, q), got {order}"
            )
        if not isinstance(seasonal_order, (tuple, list)) or len(seasonal_order) != 4:
            raise ValueError(
                f"seasonal_order must be a tuple of (P, D, Q, s), "
                f"got {seasonal_order}"
            )
        if len(inputs.endog) < 10:
            raise ValueError(
                f"Need at least 10 observations, got {len(inputs.endog)}"
            )

        self.data = inputs.endog
        self.dates = inputs.dates
        self.exog = inputs.exog
        self.exog_names = inputs.exog_names
        self.future_exog = inputs.future_exog
        self.events = tuple(events or ())
        self.missing = missing
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

    def fit(self):
        """Estimate the SARIMA model via maximum likelihood.

        Returns
        -------
        SARIMAResult
        """
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order

        model = SARIMAX(
            self.data,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        fitted = model.fit(disp=False)

        params = {}
        std_errors = {}
        p_values = {}
        for name, param, bse_val, pval in zip(
            fitted.param_names, fitted.params, fitted.bse, fitted.pvalues
        ):
            params[name] = float(param)
            std_errors[name] = float(bse_val)
            p_values[name] = float(pval)

        resid = np.asarray(fitted.resid)
        fitted_vals = np.asarray(fitted.fittedvalues)

        result = SARIMAResult(
            model_type="SARIMA",
            params=params,
            std_errors=std_errors,
            p_values=p_values,
            aic=float(fitted.aic),
            bic=float(fitted.bic),
            log_likelihood=float(fitted.llf),
            residuals=resid,
            fitted_values=fitted_vals,
            nobs=int(fitted.nobs),
            data=self.data,
            _order=self.order,
            _seasonal_order=self.seasonal_order,
            _statsmodels_result=fitted,
            _trend=self.trend,
        )

        self.result_ = result
        return result
