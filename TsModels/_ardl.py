"""Autoregressive distributed-lag estimation and order selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
from statsmodels.tsa.ardl import ARDL as StatsmodelsARDL
from statsmodels.tsa.ardl import ardl_select_order

from Ts.TsModels._base import BaseModel, BaseModelResult, PredictResult
from Ts.TsModels._base import _resolve_prediction_window
from Ts.TsModels._sarimax import _normalise_log, _normalise_sarimax_inputs


_ARDL_CRITERIA = frozenset({"aic", "bic"})
_ARDL_SEARCH_METHODS = frozenset({"hierarchical", "global"})


@dataclass(frozen=True)
class _ARDLInputs:
    endog: np.ndarray
    model_endog: np.ndarray
    dates: pd.DatetimeIndex | None
    exog: pd.DataFrame | None
    future_exog: pd.DataFrame | None
    exog_names: tuple[str, ...]
    dropped_positions: tuple[int, ...]
    log: bool


def _validate_bool(name, value):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _validate_nonnegative_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_consecutive_drop(dropped_positions, retained_count):
    if not dropped_positions:
        return
    total = retained_count + len(dropped_positions)
    dropped = set(dropped_positions)
    retained = [position for position in range(total) if position not in dropped]
    if retained and any(retained[0] < position < retained[-1] for position in dropped):
        raise ValueError(
            "ARDL requires consecutive observations; missing='drop' removed "
            "an internal row"
        )


def _prepare_inputs(
    data,
    *,
    dates=None,
    exog=None,
    exog_names=None,
    missing="drop",
    log=False,
):
    inputs = _normalise_sarimax_inputs(
        data,
        dates=dates,
        exog=exog,
        exog_names=exog_names,
        missing=missing,
    )
    if len(inputs.endog) < 10:
        raise ValueError(f"Need at least 10 observations, got {len(inputs.endog)}")
    _validate_consecutive_drop(inputs.dropped_positions, len(inputs.endog))
    log = _normalise_log(log)
    if log and np.any(inputs.endog <= 0.0):
        raise ValueError("log transformation requires strictly positive data")

    index = inputs.dates if inputs.dates is not None else pd.RangeIndex(len(inputs.endog))
    exog_frame = None
    if inputs.exog is not None:
        exog_frame = pd.DataFrame(
            inputs.exog,
            index=index,
            columns=inputs.exog_names,
        )
    return _ARDLInputs(
        endog=inputs.endog.copy(),
        model_endog=np.log(inputs.endog) if log else inputs.endog.copy(),
        dates=None if inputs.dates is None else inputs.dates.copy(),
        exog=exog_frame,
        future_exog=(
            None if inputs.future_exog is None else inputs.future_exog.copy()
        ),
        exog_names=inputs.exog_names,
        dropped_positions=inputs.dropped_positions,
        log=log,
    )


def _prediction_arrays(prediction, alpha, *, log):
    mean = np.asarray(prediction.predicted_mean, dtype=float)
    interval = np.asarray(prediction.conf_int(alpha=alpha), dtype=float)
    lower = interval[:, 0]
    upper = interval[:, 1]
    if log:
        variance = np.asarray(prediction.var_pred_mean, dtype=float)
        mean = np.exp(mean + 0.5 * variance)
        lower = np.exp(lower)
        upper = np.exp(upper)
    return mean, lower, upper


def _parameter_dict(values):
    if isinstance(values, pd.Series):
        return {str(name): float(value) for name, value in values.items()}
    return {str(index): float(value) for index, value in enumerate(values)}


def _normalize_lag_mapping(mapping):
    return {
        str(name): tuple(int(lag) for lag in lags)
        for name, lags in mapping.items()
    }


@dataclass
class ARDLResult(BaseModelResult):
    """Result container for a fitted autoregressive distributed-lag model.

    Parameters
    ----------
    model_type, params, std_errors, p_values : see BaseModelResult
        Shared parameter estimates and model identifier.
    aic, bic, log_likelihood, residuals, fitted_values, nobs, data : see BaseModelResult
        Shared estimation output.

    Examples
    --------
    >>> import pandas as pd
    >>> from Ts.TsModels import ARDL, ARDLResult
    >>> y = pd.Series([1.0 + 0.1 * i for i in range(30)], name="y")
    >>> x = pd.DataFrame({"x": [0.2 * i for i in range(30)]})
    >>> result = ARDL(y, lags=1, exog=x, order=0, trend="n").fit()
    >>> isinstance(result, ARDLResult)
    True
    """

    _statsmodels_result: object = None
    _ar_lags: tuple[int, ...] = ()
    _distributed_lags: dict[str, tuple[int, ...]] = field(default_factory=dict)
    _hold_back: int = 0
    _dates: pd.DatetimeIndex | None = None
    _exog: pd.DataFrame | None = None
    _exog_names: tuple[str, ...] = ()
    _default_future_exog: pd.DataFrame | None = None
    _log_transform: bool = False

    @property
    def log(self):
        """Whether the response was fitted on the natural-log scale."""
        return bool(self._log_transform)

    @property
    def dates(self):
        """Return a copy of the fitted observation dates."""
        return None if self._dates is None else self._dates.copy()

    @property
    def exog_names(self):
        """Return explanatory-variable names in fitted column order."""
        return tuple(self._exog_names)

    @property
    def ar_lags(self):
        """Return the exact fitted target-variable lags."""
        return tuple(self._ar_lags)

    @property
    def distributed_lags(self):
        """Return exact finite lags for every explanatory input."""
        return dict(self._distributed_lags)

    @property
    def ardl_order(self):
        """Return maximum target and input orders in explanatory-column order."""
        target = max(self.ar_lags, default=0)
        inputs = tuple(
            max(self._distributed_lags.get(name, ()), default=0)
            for name in self.exog_names
        )
        return (target, *inputs)

    @property
    def hold_back(self):
        """Return leading observations reserved for lag initialization."""
        return int(self._hold_back)

    @property
    def effective_nobs(self):
        """Return observations used by conditional maximum likelihood."""
        return self.nobs - self.hold_back

    @property
    def roots(self):
        """Return roots of the fitted target autoregressive polynomial."""
        if self._statsmodels_result is None:
            return np.array([], dtype=float)
        return np.asarray(self._statsmodels_result.roots).copy()

    @property
    def is_stationary(self):
        """Whether every target autoregressive root lies outside the unit circle."""
        return bool(len(self.roots) == 0 or np.all(np.abs(self.roots) > 1.0))

    @property
    def converged(self):
        """Whether conditional maximum-likelihood estimation completed."""
        return self._statsmodels_result is not None

    @property
    def optimizer(self):
        """Return the conditional-MLE estimator label."""
        return "conditional_mle"

    def _normalise_future_exog(self, steps, future_exog):
        values = self._default_future_exog if future_exog is None else future_exog
        if values is None:
            raise ValueError("future exog is required for ARDL forecasting")
        if isinstance(values, pd.Series):
            values = values.to_frame()
        if isinstance(values, pd.DataFrame):
            columns = tuple(str(name) for name in values.columns)
            if columns != self.exog_names:
                raise ValueError(
                    "future exog columns must exactly match " + repr(self.exog_names)
                )
            frame = values.copy()
        else:
            array = np.asarray(values, dtype=float)
            if array.ndim == 1 and len(self.exog_names) == 1:
                array = array.reshape(-1, 1)
            if array.ndim != 2 or array.shape[1] != len(self.exog_names):
                raise ValueError("future exog columns must match fitted inputs")
            frame = pd.DataFrame(array, columns=self.exog_names)
        if len(frame) < steps:
            raise ValueError(f"future exog must contain at least {steps} rows")
        frame = frame.iloc[:steps].copy()
        if not np.all(np.isfinite(frame.to_numpy(dtype=float))):
            raise ValueError("future exog contains non-finite values")
        return frame

    def predict(
        self,
        start=0,
        end=None,
        dynamic=False,
        alpha=0.05,
        future_exog=None,
    ):
        """Return unified in-sample predictions and out-of-sample forecasts.

        Parameters
        ----------
        start : int, default 0
            Zero-based first prediction position.
        end : int, optional
            Inclusive final prediction position. Defaults to the sample end.
        dynamic : bool, default False
            Whether forecasts recursively use earlier predicted responses.
        alpha : float, default 0.05
            Significance level for prediction intervals.
        future_exog : pandas.DataFrame or array-like, optional
            Complete future explanatory path from the first post-sample period
            through the requested forecast end.

        Returns
        -------
        PredictResult
            Prediction means, intervals, and out-of-sample mask.

        Examples
        --------
        >>> import pandas as pd
        >>> from Ts.TsModels import ARDL
        >>> y = pd.Series([1.0 + 0.1 * i for i in range(30)])
        >>> x = pd.DataFrame({"x": [0.2 * i for i in range(30)]})
        >>> fitted = ARDL(y, lags=1, exog=x, order=0, trend="n").fit()
        >>> future = pd.DataFrame({"x": [6.0, 6.2]})
        >>> fitted.predict(start=30, end=31, future_exog=future).mean.shape
        (2,)
        """
        if not isinstance(dynamic, (bool, np.bool_)):
            raise TypeError("dynamic must be a boolean")
        if not isinstance(alpha, (int, float, np.integer, np.floating)):
            raise TypeError("alpha must be numeric")
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        window = _resolve_prediction_window(self.nobs, start, end)
        kwargs = {}
        if window.has_forecast and self.exog_names:
            kwargs["exog_oos"] = self._normalise_future_exog(
                window.forecast_steps,
                future_exog,
            )
        prediction = self._statsmodels_result.get_prediction(
            start=window.start,
            end=window.end,
            dynamic=bool(dynamic),
            **kwargs,
        )
        mean, lower, upper = _prediction_arrays(prediction, alpha, log=self.log)
        is_oos = np.arange(window.start, window.end + 1) >= self.nobs

        full_lower = np.full(self.nobs, np.nan)
        full_upper = np.full(self.nobs, np.nan)
        full = self._statsmodels_result.get_prediction(
            start=self.hold_back,
            end=self.nobs - 1,
        )
        _, valid_lower, valid_upper = _prediction_arrays(full, alpha, log=self.log)
        full_lower[self.hold_back :] = valid_lower
        full_upper[self.hold_back :] = valid_upper
        return PredictResult(
            mean=mean,
            lower=lower,
            upper=upper,
            is_oos=is_oos,
            _full_data=self.data,
            _full_fitted=self.fitted_values,
            _full_lower=full_lower,
            _full_upper=full_upper,
            _start=window.start,
        )

    def summary(self):
        """Return a formatted ARDL estimation summary.

        Returns
        -------
        str
            Model structure, stability, fit statistics, and coefficients.

        Examples
        --------
        >>> import pandas as pd
        >>> from Ts.TsModels import ARDL
        >>> y = pd.Series([1.0 + 0.1 * i for i in range(30)])
        >>> x = pd.DataFrame({"x": [0.2 * i for i in range(30)]})
        >>> "Target Lags" in ARDL(y, 1, x, 0, trend="n").fit().summary()
        True
        """
        lines = super().summary().splitlines()
        header = [
            lines[0],
            "Target Lags       : " + (repr(self.ar_lags) if self.ar_lags else "none"),
            "Input Lags        : " + repr(self.distributed_lags),
            f"Hold Back         : {self.hold_back}",
            f"AR Stability      : {'Passed' if self.is_stationary else 'Failed'}",
        ]
        if self.log:
            header.append("Response Scale     : original (log fit; bias-adjusted mean)")
        return "\n".join(header + lines[1:])


def _make_result(estimator, fitted):
    names = tuple(str(name) for name in fitted.params.index)
    hold_back = int(fitted.model.hold_back)
    fitted_values = np.full(len(estimator.data), np.nan)
    prediction = fitted.get_prediction(start=hold_back, end=len(estimator.data) - 1)
    means, _, _ = _prediction_arrays(prediction, 0.05, log=estimator.log)
    fitted_values[hold_back:] = means
    return ARDLResult(
        model_type="ARDL",
        params=_parameter_dict(fitted.params),
        std_errors=_parameter_dict(fitted.bse),
        p_values=_parameter_dict(fitted.pvalues),
        aic=float(fitted.aic),
        bic=float(fitted.bic),
        log_likelihood=float(fitted.llf),
        residuals=np.asarray(fitted.resid, dtype=float).copy(),
        fitted_values=fitted_values,
        nobs=len(estimator.data),
        data=estimator.data.copy(),
        _parameter_covariance=np.asarray(fitted.cov_params(), dtype=float),
        _parameter_names=names,
        _statsmodels_result=fitted,
        _ar_lags=tuple(fitted.model.ar_lags or ()),
        _distributed_lags=_normalize_lag_mapping(fitted.model.dl_lags),
        _hold_back=hold_back,
        _dates=None if estimator.dates is None else estimator.dates.copy(),
        _exog=None if estimator.exog is None else estimator.exog.copy(),
        _exog_names=estimator.exog_names,
        _default_future_exog=(
            None if estimator.future_exog is None else estimator.future_exog.copy()
        ),
        _log_transform=estimator.log,
    )


class ARDL(BaseModel):
    """Estimate a standard autoregressive distributed-lag model.

    Parameters
    ----------
    data : array-like
        One-dimensional response series.
    lags : int, sequence of int, or None, default 1
        Target-variable autoregressive lags.
    exog : pandas.DataFrame or array-like, optional
        Explanatory inputs.
    order : int, sequence of int, or mapping, default 0
        Finite explanatory-variable lags. A mapping specifies each input.
    trend : {"n", "c", "t", "ct"}, default "c"
        Deterministic trend terms.
    dates : datetime-like sequence, optional
        Strict observation dates; inferred from a Series DatetimeIndex.
    exog_names : sequence of str, optional
        Required names for array inputs.
    causal : bool, default False
        If True, exclude contemporaneous explanatory values.
    seasonal : bool, default False
        Whether to include seasonal deterministic indicators.
    period : int, optional
        Seasonal period when seasonal indicators are enabled.
    hold_back : int, optional
        Leading observations excluded from conditional likelihood.
    missing : {"raise", "drop"}, default "drop"
        Non-finite input policy. Internal gaps are rejected because they break
        lag adjacency; leading or trailing incomplete rows may be dropped.
    log : bool, default False
        Fit the response on the natural-log scale and return bias-adjusted
        prediction means on the original scale.

    Examples
    --------
    >>> import pandas as pd
    >>> from Ts.TsModels import ARDL
    >>> y = pd.Series([1.0 + 0.1 * i for i in range(30)])
    >>> x = pd.DataFrame({"x": [0.2 * i for i in range(30)]})
    >>> model = ARDL(y, lags=1, exog=x, order={"x": [0, 1]}, trend="n")
    >>> model.fit().ardl_order
    (1, 1)
    """

    def __init__(
        self,
        data,
        lags=1,
        exog=None,
        order=0,
        trend="c",
        *,
        dates=None,
        exog_names=None,
        causal=False,
        seasonal=False,
        period=None,
        hold_back=None,
        missing="drop",
        log=False,
    ):
        prepared = _prepare_inputs(
            data,
            dates=dates,
            exog=exog,
            exog_names=exog_names,
            missing=missing,
            log=log,
        )
        self.data = prepared.endog
        self._model_data = prepared.model_endog
        self.dates = prepared.dates
        self.exog = prepared.exog
        self.future_exog = prepared.future_exog
        self.exog_names = prepared.exog_names
        self.dropped_positions = prepared.dropped_positions
        self.log = prepared.log
        self.lags = lags
        self.order = order
        self.trend = trend
        self.causal = _validate_bool("causal", causal)
        self.seasonal = _validate_bool("seasonal", seasonal)
        self.period = period
        self.hold_back = hold_back
        self._model = self._build_model()

    def _build_model(self):
        index = self.dates if self.dates is not None else pd.RangeIndex(len(self.data))
        endog = pd.Series(self._model_data, index=index, name="y")
        return StatsmodelsARDL(
            endog,
            lags=self.lags,
            exog=self.exog,
            order=self.order,
            trend=self.trend,
            causal=self.causal,
            seasonal=self.seasonal,
            period=self.period,
            hold_back=self.hold_back,
            missing="none",
        )

    def fit(self, *, cov_type="nonrobust", cov_kwds=None, use_t=True):
        """Estimate the configured ARDL using conditional maximum likelihood.

        Parameters
        ----------
        cov_type : str, default "nonrobust"
            Statsmodels OLS covariance estimator, including HC0-HC3 and HAC.
        cov_kwds : mapping, optional
            Additional covariance-estimator options.
        use_t : bool, default True
            Whether inference uses the Student t distribution.

        Returns
        -------
        ARDLResult
            Unified fitted result with lag structure and prediction methods.

        Examples
        --------
        >>> import pandas as pd
        >>> from Ts.TsModels import ARDL
        >>> y = pd.Series([1.0 + 0.1 * i for i in range(30)])
        >>> x = pd.DataFrame({"x": [0.2 * i for i in range(30)]})
        >>> ARDL(y, 1, x, 0, trend="n").fit().effective_nobs > 0
        True
        """
        fitted = self._model.fit(
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            use_t=_validate_bool("use_t", use_t),
        )
        result = _make_result(self, fitted)
        self.result_ = result
        return result


@dataclass
class AutoARDLResult(BaseModelResult):
    """Result of automatic ARDL lag selection.

    Parameters
    ----------
    best_result : ARDLResult
        Selected fitted ARDL result.
    selection_criterion : str
        Information criterion used for selection.
    search_method : str
        Hierarchical or global subset search.
    criterion_table : pandas.DataFrame
        Evaluated lag structures and criterion values.
    model_type, params, std_errors, p_values : see BaseModelResult
        Shared parameter estimates and model identifier.
    aic, bic, log_likelihood, residuals, fitted_values, nobs, data : see BaseModelResult
        Shared output copied from the selected model.

    Examples
    --------
    >>> import pandas as pd
    >>> from Ts.TsModels import AutoARDL, AutoARDLResult
    >>> y = pd.Series([1.0 + 0.1 * i for i in range(40)])
    >>> x = pd.DataFrame({"x": [0.2 * i for i in range(40)]})
    >>> result = AutoARDL(y, 1, x, 1, trend="n").fit()
    >>> isinstance(result, AutoARDLResult)
    True
    """

    best_result: ARDLResult | None = field(default=None, repr=False)
    selection_criterion: str = "bic"
    search_method: str = "hierarchical"
    criterion_table: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    @property
    def ar_lags(self):
        """Return selected target-variable lags."""
        return self.best_result.ar_lags

    @property
    def distributed_lags(self):
        """Return selected explanatory-variable lags."""
        return self.best_result.distributed_lags

    @property
    def ardl_order(self):
        """Return selected maximum ARDL orders."""
        return self.best_result.ardl_order

    @property
    def dates(self):
        """Return fitted observation dates."""
        return self.best_result.dates

    @property
    def exog_names(self):
        """Return fitted explanatory-variable names."""
        return self.best_result.exog_names

    @property
    def hold_back(self):
        """Return the selected model hold-back count."""
        return self.best_result.hold_back

    @property
    def effective_nobs(self):
        """Return selected model effective observations."""
        return self.best_result.effective_nobs

    @property
    def is_stationary(self):
        """Whether the selected target autoregression is stable."""
        return self.best_result.is_stationary

    @property
    def converged(self):
        """Whether the selected conditional-MLE fit completed."""
        return self.best_result.converged

    @property
    def optimizer(self):
        """Return the selected estimator label."""
        return self.best_result.optimizer

    @property
    def log(self):
        """Whether the selected response used a log transform."""
        return self.best_result.log

    def predict(self, **kwargs):
        """Delegate prediction to the selected ARDL result.

        Parameters
        ----------
        **kwargs
            Prediction arguments accepted by :meth:`ARDLResult.predict`.

        Returns
        -------
        PredictResult
            Prediction output from the selected model.

        Examples
        --------
        >>> import pandas as pd
        >>> from Ts.TsModels import AutoARDL
        >>> y = pd.Series([1.0 + 0.1 * i for i in range(40)])
        >>> x = pd.DataFrame({"x": [0.2 * i for i in range(40)]})
        >>> result = AutoARDL(y, 1, x, 1, trend="n").fit()
        >>> result.predict(start=10, end=11).mean.shape
        (2,)
        """
        return self.best_result.predict(**kwargs)

    def summary(self):
        """Return selection metadata followed by the selected ARDL summary.

        Returns
        -------
        str
            Stable human-readable selection and estimation report.

        Examples
        --------
        >>> import pandas as pd
        >>> from Ts.TsModels import AutoARDL
        >>> y = pd.Series([1.0 + 0.1 * i for i in range(40)])
        >>> x = pd.DataFrame({"x": [0.2 * i for i in range(40)]})
        >>> "Selection Criterion" in AutoARDL(y, 1, x, 1, trend="n").fit().summary()
        True
        """
        return "\n".join(
            [
                "Auto ARDL Model Selection Result",
                "=" * 50,
                f"Selection Criterion: {self.selection_criterion.upper()}",
                f"Search Method      : {self.search_method}",
                f"Selected AR Lags   : {self.ar_lags}",
                f"Selected Input Lags: {self.distributed_lags}",
                "",
                self.best_result.summary(),
            ]
        )


class AutoARDL(BaseModel):
    """Select target and explanatory lags for a standard ARDL model.

    Parameters
    ----------
    data : array-like
        One-dimensional response series.
    maxlag : int
        Maximum target-variable autoregressive lag.
    exog : pandas.DataFrame or array-like
        Explanatory inputs.
    maxorder : int or mapping
        Maximum finite lag for every input or separately by input name.
    trend : {"n", "c", "t", "ct"}, default "c"
        Deterministic trend terms.
    criterion : {"aic", "bic"}, default "bic"
        Information criterion minimized during selection.
    search_method : {"hierarchical", "global"}, default "hierarchical"
        Hierarchical order search or global subset search.
    dates : datetime-like sequence, optional
        Strict observation dates.
    exog_names : sequence of str, optional
        Required names for array explanatory inputs.
    causal : bool, default False
        Whether explanatory lags exclude the contemporaneous value.
    seasonal : bool, default False
        Whether to include seasonal deterministic indicators.
    period : int, optional
        Seasonal period.
    hold_back : int, optional
        Common leading sample excluded from every candidate.
    missing : {"raise", "drop"}, default "drop"
        Non-finite input policy; internal gaps are rejected.
    log : bool, default False
        Select and fit on the log response, returning original-scale forecasts.

    Examples
    --------
    >>> import pandas as pd
    >>> from Ts.TsModels import AutoARDL
    >>> y = pd.Series([1.0 + 0.1 * i for i in range(40)])
    >>> x = pd.DataFrame({"x": [0.2 * i for i in range(40)]})
    >>> selected = AutoARDL(y, maxlag=1, exog=x, maxorder=1, trend="n").fit()
    >>> len(selected.criterion_table) > 0
    True
    """

    def __init__(
        self,
        data,
        maxlag,
        exog,
        maxorder,
        trend="c",
        criterion="bic",
        search_method="hierarchical",
        *,
        dates=None,
        exog_names=None,
        causal=False,
        seasonal=False,
        period=None,
        hold_back=None,
        missing="drop",
        log=False,
    ):
        prepared = _prepare_inputs(
            data,
            dates=dates,
            exog=exog,
            exog_names=exog_names,
            missing=missing,
            log=log,
        )
        if prepared.exog is None:
            raise ValueError("AutoARDL requires exog input data")
        self.data = prepared.endog
        self._model_data = prepared.model_endog
        self.dates = prepared.dates
        self.exog = prepared.exog
        self.future_exog = prepared.future_exog
        self.exog_names = prepared.exog_names
        self.dropped_positions = prepared.dropped_positions
        self.log = prepared.log
        self.maxlag = _validate_nonnegative_int("maxlag", maxlag)
        self.maxorder = self._validate_maxorder(maxorder)
        self.trend = trend
        if criterion not in _ARDL_CRITERIA:
            raise ValueError("criterion must be 'aic' or 'bic'")
        self.criterion = criterion
        if search_method not in _ARDL_SEARCH_METHODS:
            raise ValueError("search_method must be 'hierarchical' or 'global'")
        self.search_method = search_method
        self.causal = _validate_bool("causal", causal)
        self.seasonal = _validate_bool("seasonal", seasonal)
        self.period = period
        self.hold_back = hold_back

    def _validate_maxorder(self, maxorder):
        if isinstance(maxorder, Mapping):
            unknown = [name for name in maxorder if name not in self.exog_names]
            if unknown:
                raise ValueError(f"maxorder contains unknown input {unknown[0]!r}")
            missing = [name for name in self.exog_names if name not in maxorder]
            if missing:
                raise ValueError(f"maxorder is missing input {missing[0]!r}")
            return {
                name: _validate_nonnegative_int(f"maxorder[{name!r}]", maxorder[name])
                for name in self.exog_names
            }
        return _validate_nonnegative_int("maxorder", maxorder)

    def fit(self, *, cov_type="nonrobust", cov_kwds=None, use_t=True):
        """Run lag selection and fit the selected ARDL model.

        Parameters
        ----------
        cov_type : str, default "nonrobust"
            Covariance estimator used for the selected final model.
        cov_kwds : mapping, optional
            Additional covariance-estimator options.
        use_t : bool, default True
            Whether final-model inference uses the Student t distribution.

        Returns
        -------
        AutoARDLResult
            Selected model, exact lag structure, and criterion table.

        Examples
        --------
        >>> import pandas as pd
        >>> from Ts.TsModels import AutoARDL
        >>> y = pd.Series([1.0 + 0.1 * i for i in range(40)])
        >>> x = pd.DataFrame({"x": [0.2 * i for i in range(40)]})
        >>> AutoARDL(y, 1, x, 1, trend="n").fit().criterion_table.empty
        False
        """
        index = self.dates if self.dates is not None else pd.RangeIndex(len(self.data))
        response = pd.Series(self._model_data, index=index, name="y")
        selected = ardl_select_order(
            response,
            maxlag=self.maxlag,
            exog=self.exog,
            maxorder=self.maxorder,
            trend=self.trend,
            causal=self.causal,
            ic=self.criterion,
            glob=self.search_method == "global",
            seasonal=self.seasonal,
            period=self.period,
            hold_back=self.hold_back,
            missing="none",
        )
        selected_estimator = ARDL(
            self.data,
            lags=selected.model.ar_lags,
            exog=self.exog,
            order=selected.model.dl_lags,
            trend=self.trend,
            dates=self.dates,
            causal=self.causal,
            seasonal=self.seasonal,
            period=self.period,
            hold_back=selected.model.hold_back,
            missing="raise",
            log=self.log,
        )
        best = selected_estimator.fit(
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            use_t=use_t,
        )
        best._default_future_exog = (
            None if self.future_exog is None else self.future_exog.copy()
        )
        scores = getattr(selected, self.criterion)
        rows = []
        for score, specification in scores.items():
            target_order, input_orders = specification
            rows.append(
                {
                    "criterion": float(score),
                    "target_lags": target_order,
                    "input_lags": dict(input_orders),
                }
            )
        table = pd.DataFrame(
            rows,
            columns=["criterion", "target_lags", "input_lags"],
        )
        result = AutoARDLResult(
            model_type=best.model_type,
            params=dict(best.params),
            std_errors=dict(best.std_errors),
            p_values=dict(best.p_values),
            aic=best.aic,
            bic=best.bic,
            log_likelihood=best.log_likelihood,
            residuals=best.residuals.copy(),
            fitted_values=best.fitted_values.copy(),
            nobs=best.nobs,
            data=best.data.copy(),
            _parameter_covariance=(
                None
                if best._parameter_covariance is None
                else best._parameter_covariance.copy()
            ),
            _parameter_names=best._parameter_names,
            best_result=best,
            selection_criterion=self.criterion,
            search_method=self.search_method,
            criterion_table=table,
        )
        self.result_ = result
        return result


__all__ = ["ARDL", "ARDLResult", "AutoARDL", "AutoARDLResult"]
