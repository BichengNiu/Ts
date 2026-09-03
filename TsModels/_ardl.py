"""Autoregressive distributed-lag estimation and order selection."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product
from math import prod
from typing import Mapping

import numpy as np
import pandas as pd
from statsmodels.tsa.ardl import ARDL as StatsmodelsARDL

from Ts.TsModels._base import BaseModel, BaseModelResult, PredictResult
from Ts.TsModels._base import _resolve_prediction_window
from Ts.TsModels._parallel import _map_candidates, _validate_n_jobs
from Ts.TsModels._sarimax import (
    SARIMAX,
    _normalise_cov_type,
    _normalise_fit_method,
    _normalise_log,
    _normalise_maxiter,
    _normalise_order,
    _normalise_require_convergence,
    _normalise_sarimax_inputs,
    _normalise_seasonal_order,
    _validate_datetime_index,
)


_ARDL_CRITERIA = frozenset({"aic", "bic"})
_ARDL_SEARCH_METHODS = frozenset({"hierarchical", "global"})


@dataclass(frozen=True)
class _ARDLSelectionContext:
    """Immutable, serializable design data shared by ARDL workers."""

    y: np.ndarray
    blocks: tuple[np.ndarray, ...]
    all_x: np.ndarray
    always_df: int
    var_names: tuple[str, ...]
    causal: bool


@dataclass(frozen=True)
class _ARDLSelectionTask:
    """One hierarchical count or global column permutation to evaluate."""

    context: _ARDLSelectionContext
    key: tuple
    counts: tuple[int, ...] | None = None
    columns: tuple[int, ...] | None = None


def _compute_ardl_ics(task: _ARDLSelectionTask) -> tuple[tuple, tuple[float, float, float]]:
    """Compute the same AIC/BIC/HQIC triple as statsmodels ARDL selection."""
    from statsmodels.tools import eval_measures

    context = task.context
    if task.columns is not None:
        x = context.all_x[:, task.columns]
    else:
        assert task.counts is not None
        x = np.column_stack(
            [
                block[:, :count]
                for block, count in zip(context.blocks, task.counts, strict=True)
            ]
        )
    if x.shape[1]:
        resid = context.y - x @ np.linalg.lstsq(x, context.y, rcond=None)[0]
    else:
        resid = context.y
    nobs = resid.shape[0]
    sigma2 = 1.0 / nobs * np.sum(resid**2)
    llf = -nobs * (np.log(2 * np.pi * sigma2) + 1) / 2
    df_model = context.always_df + x.shape[1]
    total_params = df_model + 1
    return task.key, (
        float(eval_measures.aic(llf, nobs, total_params)),
        float(eval_measures.bic(llf, nobs, total_params)),
        float(eval_measures.hqic(llf, nobs, total_params)),
    )


def _perm_to_ardl_key(keys, perm):
    """Convert statsmodels-style selected columns into an ARDL lag key."""
    if perm == ():
        d = {key: 0 for key, _ in keys if key is not None}
        return (0, tuple((key, value) for key, value in d.items()))

    d = defaultdict(list)
    y_lags = []
    for index in perm:
        key = keys[index]
        if key[0] is None:
            y_lags.append(key[1])
        else:
            d[key[0]].append(key[1])
    d = dict(d)
    y_lags = 0 if not y_lags or y_lags == [0] else tuple(y_lags)
    for key in keys:
        if key[0] not in d and key[0] is not None:
            d[key[0]] = None
    for key in d:
        if d[key] is not None:
            d[key] = tuple(d[key])
    return (y_lags, tuple(d.items()))


def _build_ardl_selection_context(
    response: pd.Series,
    *,
    maxlag: int,
    exog: pd.DataFrame,
    maxorder: int | Mapping,
    trend: str,
    causal: bool,
    seasonal: bool,
    period: int | None,
    hold_back: int | None,
) -> tuple[_ARDLSelectionContext, tuple[tuple[int, ...], ...], tuple[str, ...]]:
    """Build the design blocks used by statsmodels ARDL order selection."""
    base = StatsmodelsARDL(
        response,
        maxlag,
        exog,
        maxorder,
        trend,
        causal=causal,
        seasonal=seasonal,
        hold_back=hold_back,
        period=period,
        missing="none",
    )
    effective_hold_back = base.hold_back
    blocks = base._blocks
    always = np.column_stack([blocks["deterministic"], blocks["fixed"]])
    always = always[effective_hold_back:]
    select = [blocks["endog"][effective_hold_back:]]
    iter_orders = [tuple(range(blocks["endog"].shape[1] + 1))]
    var_names = []
    for name in blocks["exog"]:
        block = blocks["exog"][name][effective_hold_back:]
        select.append(block)
        iter_orders.append(tuple(range(block.shape[1] + 1)))
        var_names.append(name)
    y = base._y
    if always.shape[1]:
        pinv_always = np.linalg.pinv(always)
        for index, block in enumerate(select):
            select[index] = block - always @ (pinv_always @ block)
        y = y - always @ (pinv_always @ y)

    blocks_tuple = tuple(select)
    all_x = np.column_stack(blocks_tuple)
    context = _ARDLSelectionContext(
        y=y,
        blocks=blocks_tuple,
        all_x=all_x,
        always_df=always.shape[1],
        var_names=tuple(var_names),
        causal=causal,
    )
    return context, tuple(iter_orders), tuple(var_names)


def _iter_ardl_selection_tasks(
    context: _ARDLSelectionContext,
    iter_orders: tuple[tuple[int, ...], ...],
    var_names: tuple[str, ...],
    *,
    search_method: str,
) -> tuple[object, int]:
    """Return a stable task iterator and its exact candidate count."""
    if search_method == "hierarchical":
        def hierarchical_tasks():
            for counts in product(*iter_orders):
                input_key = []
                for index, value in enumerate(counts[1:]):
                    name = var_names[index]
                    if context.causal:
                        input_key.append((name, None if value == 0 else value))
                    else:
                        input_key.append(
                            (name, value - 1 if value - 1 >= 0 else None)
                        )
                key = (
                    counts[0] if counts[0] else None,
                    tuple(input_key),
                )
                yield _ARDLSelectionTask(context, key, tuple(counts))

        return hierarchical_tasks(), prod(len(values) for values in iter_orders)

    keys = [(None, index) for index in range(context.blocks[0].shape[1])]
    for name, block in zip(var_names, context.blocks[1:], strict=True):
        keys.extend((name, index) for index in range(block.shape[1]))
    all_columns = range(context.all_x.shape[1])

    def global_tasks():
        for size in range(context.all_x.shape[1] + 1):
            for permutation in combinations(all_columns, size):
                yield _ARDLSelectionTask(
                    context,
                    _perm_to_ardl_key(keys, permutation),
                    columns=tuple(permutation),
                )

    return global_tasks(), 1 << context.all_x.shape[1]


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


def _normalise_error_orders(error_order, error_seasonal_order):
    """Normalize the optional SARIMA error orders and their dependency."""
    seasonal_order = _normalise_seasonal_order(error_seasonal_order)
    if error_order is None:
        if seasonal_order != (0, 0, 0, 0):
            raise ValueError(
                "error_seasonal_order requires error_order to be specified"
            )
        return None, seasonal_order
    return _normalise_order(error_order), seasonal_order


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
    _model_data: np.ndarray | None = field(default=None, repr=False)
    _ardl_model: object = field(default=None, repr=False)
    _error_result: object = field(default=None, repr=False)
    _error_order: tuple | None = None
    _error_seasonal_order: tuple = (0, 0, 0, 0)

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
        if self._error_result is not None:
            return self._error_result.effective_nobs
        return self.nobs - self.hold_back

    @property
    def likelihood_burn(self):
        """Return the original-sample position of the first valid fit value."""
        if self._error_result is None:
            return self.hold_back
        return self.hold_back + self._error_result.likelihood_burn

    @property
    def error_order(self):
        """Return the manually configured non-seasonal SARIMA error order."""
        return None if self._error_order is None else tuple(self._error_order)

    @property
    def error_seasonal_order(self):
        """Return the manually configured seasonal SARIMA error order."""
        return tuple(self._error_seasonal_order)

    @property
    def error_likelihood_burn(self):
        """Return state-space initialization burn for the SARIMA error."""
        if self._error_result is None:
            return 0
        return int(self._error_result.likelihood_burn)

    @property
    def error_arroots(self):
        """Return roots of the fitted SARIMA error AR polynomial."""
        if self._error_result is None:
            return np.array([], dtype=float)
        return self._error_result.arroots.copy()

    @property
    def error_marroots(self):
        """Return roots of the fitted SARIMA error MA polynomial."""
        if self._error_result is None:
            return np.array([], dtype=float)
        return self._error_result.marroots.copy()

    @property
    def error_is_stationary(self):
        """Whether the SARIMA error AR polynomial is stationary."""
        if self._error_result is None:
            return True
        return self._error_result.is_stationary

    @property
    def error_is_invertible(self):
        """Whether the SARIMA error MA polynomial is invertible."""
        if self._error_result is None:
            return True
        return self._error_result.is_invertible

    @property
    def roots(self):
        """Return roots of the fitted target autoregressive polynomial."""
        if self._statsmodels_result is not None:
            return np.asarray(self._statsmodels_result.roots).copy()
        if not self.ar_lags:
            return np.array([], dtype=float)
        coefficients = [1.0]
        for lag in range(1, max(self.ar_lags) + 1):
            coefficients.append(-self.params.get(f"y.L{lag}", 0.0))
        return np.roots(coefficients)

    @property
    def is_stationary(self):
        """Whether every target autoregressive root lies outside the unit circle."""
        return bool(len(self.roots) == 0 or np.all(np.abs(self.roots) > 1.0))

    @property
    def converged(self):
        """Whether conditional maximum-likelihood estimation completed."""
        if self._error_result is not None:
            return self._error_result.converged
        return self._statsmodels_result is not None

    @property
    def optimizer(self):
        """Return the conditional-MLE estimator label."""
        if self._error_result is not None:
            return self._error_result.optimizer
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

    def _resolve_prediction_bounds(self, start, end, future_dates):
        """Resolve integer or dated bounds against the original ARDL sample."""
        date_bounds = [
            value
            for value in (start, end)
            if value is not None and not isinstance(value, (int, np.integer))
        ]
        if not date_bounds:
            return start, end
        if self._dates is None:
            raise TypeError("date prediction bounds require dated model data")

        parsed_bounds = []
        for value in date_bounds:
            try:
                timestamp = pd.Timestamp(value)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "prediction bounds must be integer positions or dates"
                ) from error
            if str(timestamp.tz) != str(self._dates.tz):
                raise ValueError("prediction date timezone must match model dates")
            parsed_bounds.append(timestamp)

        future_bound = max(
            (timestamp for timestamp in parsed_bounds if timestamp > self._dates[-1]),
            default=None,
        )
        future_index = None
        if future_bound is not None:
            if future_dates is not None:
                future_index = _validate_datetime_index(
                    future_dates,
                    "future_dates",
                )
            else:
                frequency = self._dates.freq or pd.infer_freq(self._dates)
                if frequency is None:
                    raise ValueError(
                        "future_dates is required when date frequency cannot be inferred"
                    )
                future_index = pd.date_range(
                    start=self._dates[-1],
                    end=future_bound,
                    freq=frequency,
                )[1:]
        calendar = (
            self._dates if future_index is None else self._dates.append(future_index)
        )

        def position(value, name):
            if value is None or isinstance(value, (int, np.integer)):
                return value
            timestamp = pd.Timestamp(value)
            location = int(calendar.get_indexer([timestamp])[0])
            if location < 0:
                raise ValueError(
                    f"prediction date {timestamp.isoformat()} for {name} is absent "
                    "from the prediction calendar"
                )
            return location

        return position(start, "start"), position(end, "end")

    def _build_error_future_design(self, steps, future_exog, future_dates):
        """Build future ARDL lag columns for the SARIMA error backend."""
        if self._error_result is None or self._ardl_model is None:
            raise RuntimeError("SARIMA error state is unavailable")
        future_inputs = None
        if self.exog_names:
            future_inputs = self._normalise_future_exog(steps, future_exog)
        elif future_exog is not None:
            raise ValueError(
                "future_exog is invalid because the model has no explanatory inputs"
            )

        index = (
            future_dates.copy()
            if future_dates is not None
            else pd.RangeIndex(steps)
        )
        columns = tuple(self._error_result.exog_names)
        future_design = pd.DataFrame(index=index, columns=columns, dtype=float)

        deterministics = getattr(self._ardl_model, "_deterministics", None)
        if deterministics is not None:
            if future_dates is None:
                deterministic = deterministics.out_of_sample(steps)
            else:
                deterministic = deterministics.out_of_sample(
                    steps,
                    forecast_index=future_dates,
                )
            for name in deterministic.columns:
                if name in future_design.columns:
                    future_design[name] = np.asarray(deterministic[name], dtype=float)

        historical_response = np.asarray(self._model_data, dtype=float)
        response_path = np.full(self.nobs + steps, np.nan, dtype=float)
        response_path[: self.nobs] = historical_response
        exog_path = None
        if self._exog is not None:
            exog_path = np.vstack(
                [
                    self._exog.to_numpy(dtype=float),
                    future_inputs.to_numpy(dtype=float),
                ]
            )

        lag_columns = {
            f"y.L{lag}": ("response", lag)
            for lag in self.ar_lags
        }
        for name, lags in self.distributed_lags.items():
            for lag in lags:
                lag_columns[f"{name}.L{lag}"] = (name, lag)

        for step in range(steps):
            for column, (source, lag) in lag_columns.items():
                if column not in future_design.columns:
                    continue
                source_position = self.nobs + step - lag
                if source == "response":
                    value = response_path[source_position]
                else:
                    if exog_path is None:
                        raise RuntimeError("fitted explanatory input history is missing")
                    value = exog_path[source_position, self.exog_names.index(source)]
                future_design.loc[index[step], column] = value

            partial_design = future_design.iloc[: step + 1].copy()
            future_prediction = self._error_result.predict(
                start=self._error_result.nobs,
                end=self._error_result.nobs + step,
                dynamic=False,
                alpha=0.05,
                future_exog=partial_design,
                future_dates=(
                    None if future_dates is None else future_dates[: step + 1]
                ),
            )
            response_path[self.nobs + step] = float(future_prediction.mean[-1])

        if not np.all(np.isfinite(future_design.to_numpy(dtype=float))):
            raise RuntimeError("future ARDL design contains non-finite values")
        return future_design

    def _predict_with_error(
        self,
        window,
        *,
        dynamic,
        alpha,
        future_exog,
        future_dates,
    ):
        """Predict through the SARIMA error result and remap to ARDL positions."""
        mean = np.full(window.size, np.nan, dtype=float)
        lower = np.full(window.size, np.nan, dtype=float)
        upper = np.full(window.size, np.nan, dtype=float)

        if window.in_sample_size:
            valid_start = max(window.start, self.hold_back)
            valid_end = min(window.end, self.nobs - 1)
            if valid_start <= valid_end:
                prediction = self._error_result.predict(
                    start=valid_start - self.hold_back,
                    end=valid_end - self.hold_back,
                    dynamic=dynamic,
                    alpha=alpha,
                )
                destination = valid_start - window.start
                size = valid_end - valid_start + 1
                mean[destination : destination + size] = prediction.mean[:size]
                lower[destination : destination + size] = prediction.lower[:size]
                upper[destination : destination + size] = prediction.upper[:size]

        if window.has_forecast:
            resolved_dates = self._error_result._resolve_future_dates(
                window.forecast_steps,
                future_dates,
            )
            future_design = self._build_error_future_design(
                window.forecast_steps,
                future_exog,
                resolved_dates,
            )
            prediction = self._error_result.predict(
                start=self._error_result.nobs,
                end=self._error_result.nobs + window.forecast_steps - 1,
                dynamic=dynamic,
                alpha=alpha,
                future_exog=future_design,
                future_dates=resolved_dates,
            )
            destination = window.in_sample_size
            source = window.forecast_skip
            mean[destination:] = prediction.mean[source:]
            lower[destination:] = prediction.lower[source:]
            upper[destination:] = prediction.upper[source:]

        full = self._error_result.predict(
            start=0,
            end=self._error_result.nobs - 1,
            alpha=alpha,
        )
        full_lower = np.full(self.nobs, np.nan, dtype=float)
        full_upper = np.full(self.nobs, np.nan, dtype=float)
        if full._full_lower is not None:
            full_lower[self.hold_back :] = full._full_lower
            full_upper[self.hold_back :] = full._full_upper
        return PredictResult(
            mean=mean,
            lower=lower,
            upper=upper,
            is_oos=np.arange(window.start, window.end + 1) >= self.nobs,
            _full_data=self.data,
            _full_fitted=self.fitted_values,
            _full_lower=full_lower,
            _full_upper=full_upper,
            _start=window.start,
        )

    def predict(
        self,
        start=0,
        end=None,
        dynamic=False,
        alpha=0.05,
        future_exog=None,
        future_dates=None,
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
        future_dates : datetime-like sequence, optional
            Exact forecast dates when a dated model has no inferable frequency.

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
        start, end = self._resolve_prediction_bounds(start, end, future_dates)
        window = _resolve_prediction_window(self.nobs, start, end)
        if self._error_result is not None:
            if not window.has_forecast and (
                future_exog is not None or future_dates is not None
            ):
                raise ValueError(
                    "future_exog and future_dates require an out-of-sample range"
                )
            return self._predict_with_error(
                window,
                dynamic=bool(dynamic),
                alpha=alpha,
                future_exog=future_exog,
                future_dates=future_dates,
            )
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
        if self.error_order is not None:
            header.extend(
                [
                    f"Error SARIMA      : {self.error_order}",
                    f"Error Seasonal    : {self.error_seasonal_order}",
                    f"Error Burn        : {self.error_likelihood_burn}",
                    "Error Stationarity: "
                    + ("Passed" if self.error_is_stationary else "Failed"),
                    "Error Invertibility: "
                    + ("Passed" if self.error_is_invertible else "Failed"),
                    "Optimizer         : " + (self.optimizer or "Unknown"),
                    "Converged         : " + ("Yes" if self.converged else "No"),
                ]
            )
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
        _model_data=estimator._model_data.copy(),
        _ardl_model=estimator._model,
        _error_order=None,
        _error_seasonal_order=(0, 0, 0, 0),
    )


def _make_error_result(estimator, error_result):
    """Build an ARDL result backed by a manually specified SARIMA error."""
    hold_back = int(estimator._model.hold_back)
    fitted_values = np.full(len(estimator.data), np.nan, dtype=float)
    prediction = error_result._statsmodels_result.get_prediction(
        start=0,
        end=error_result.nobs - 1,
    )
    means, _, _ = _prediction_arrays(prediction, 0.05, log=estimator.log)
    burn = error_result.likelihood_burn
    fitted_values[hold_back + burn :] = means[burn:]
    model = estimator._model
    return ARDLResult(
        model_type="ARDL",
        params=dict(error_result.params),
        std_errors=dict(error_result.std_errors),
        p_values=dict(error_result.p_values),
        aic=float(error_result.aic),
        bic=float(error_result.bic),
        log_likelihood=float(error_result.log_likelihood),
        residuals=np.asarray(error_result.residuals, dtype=float).copy(),
        fitted_values=fitted_values,
        nobs=len(estimator.data),
        data=estimator.data.copy(),
        _parameter_covariance=(
            None
            if error_result._parameter_covariance is None
            else error_result._parameter_covariance.copy()
        ),
        _parameter_names=tuple(error_result._parameter_names),
        _ar_lags=tuple(model.ar_lags or ()),
        _distributed_lags=_normalize_lag_mapping(model.dl_lags),
        _hold_back=hold_back,
        _dates=None if estimator.dates is None else estimator.dates.copy(),
        _exog=None if estimator.exog is None else estimator.exog.copy(),
        _exog_names=estimator.exog_names,
        _default_future_exog=(
            None if estimator.future_exog is None else estimator.future_exog.copy()
        ),
        _log_transform=estimator.log,
        _model_data=estimator._model_data.copy(),
        _ardl_model=model,
        _error_result=error_result,
        _error_order=estimator.error_order,
        _error_seasonal_order=estimator.error_seasonal_order,
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
    error_order : tuple or None, default None
        Optional manually specified SARIMA error order ``(p, d, q)``. When
        omitted, the original conditional OLS ARDL path is used.
    error_seasonal_order : tuple, default ``(0, 0, 0, 0)``
        Manually specified seasonal SARIMA error order ``(P, D, Q, s)``.
        It requires ``error_order`` to be specified.
    error_enforce_stationarity : bool, default True
        Whether the SARIMA error AR polynomial is constrained to be stationary.
    error_enforce_invertibility : bool, default True
        Whether the SARIMA error MA polynomial is constrained to be invertible.

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
        error_order=None,
        error_seasonal_order=(0, 0, 0, 0),
        error_enforce_stationarity=True,
        error_enforce_invertibility=True,
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
        (
            self.error_order,
            self.error_seasonal_order,
        ) = _normalise_error_orders(error_order, error_seasonal_order)
        self.error_enforce_stationarity = _validate_bool(
            "error_enforce_stationarity",
            error_enforce_stationarity,
        )
        self.error_enforce_invertibility = _validate_bool(
            "error_enforce_invertibility",
            error_enforce_invertibility,
        )
        self._model = self._build_model()

    def _clone_for_evaluation(self, data, exog=None, *, dates=None):
        """Rebuild the ARDL design for one isolated evaluation window."""
        return type(self)(
            data,
            lags=self.lags,
            exog=exog,
            order=self.order,
            trend=self.trend,
            dates=dates,
            exog_names=self.exog_names if exog is not None else None,
            causal=self.causal,
            seasonal=self.seasonal,
            period=self.period,
            hold_back=self.hold_back,
            missing="raise",
            log=self.log,
            error_order=self.error_order,
            error_seasonal_order=self.error_seasonal_order,
            error_enforce_stationarity=self.error_enforce_stationarity,
            error_enforce_invertibility=self.error_enforce_invertibility,
        )

    def _evaluation_predict_kwargs(self, start, stop):
        """Return the observed explanatory path for one historical window."""
        kwargs = {}
        if self.exog is not None:
            future_exog = self.exog.iloc[start:stop].copy()
            if len(future_exog) != stop - start:
                raise ValueError("future exog does not cover the evaluation window")
            kwargs["future_exog"] = future_exog
        if self.dates is not None:
            future_dates = self.dates[start:stop]
            if len(future_dates) != stop - start:
                raise ValueError("future dates do not cover the evaluation window")
            kwargs["future_dates"] = future_dates.copy()
        return kwargs

    def _build_error_model(self):
        """Build the SARIMAX backend over the effective ARDL design."""
        hold_back = int(self._model.hold_back)
        index = (
            self.dates[hold_back:]
            if self.dates is not None
            else pd.RangeIndex(len(self._model._y))
        )
        response = pd.Series(
            np.asarray(self._model._y, dtype=float),
            index=index,
            name="y",
        )
        design = pd.DataFrame(
            np.asarray(self._model._x, dtype=float),
            index=index,
            columns=self._model.exog_names,
        )
        return SARIMAX(
            response,
            exog=design,
            order=self.error_order,
            seasonal_order=self.error_seasonal_order,
            trend="n",
            enforce_stationarity=self.error_enforce_stationarity,
            enforce_invertibility=self.error_enforce_invertibility,
            missing="raise",
        )

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

    def fit(
        self,
        *,
        cov_type="nonrobust",
        cov_kwds=None,
        use_t=True,
        error_method="bfgs",
        error_maxiter=500,
        error_cov_type="oim",
        error_require_convergence=True,
    ):
        """Estimate the configured ARDL using conditional maximum likelihood.

        Parameters
        ----------
        cov_type : str, default "nonrobust"
            Statsmodels OLS covariance estimator, including HC0-HC3 and HAC.
        cov_kwds : mapping, optional
            Additional covariance-estimator options.
        use_t : bool, default True
            Whether inference uses the Student t distribution.
        error_method : str, default ``"bfgs"``
            Optimizer used for the SARIMA error maximum likelihood fit.
        error_maxiter : int, default 500
            Positive optimizer iteration limit for the SARIMA error fit.
        error_cov_type : str, default ``"oim"``
            Covariance estimator used for the SARIMA error fit.
        error_require_convergence : bool, default True
            Whether a non-converged SARIMA error fit raises ``RuntimeError``.

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
        if self.error_order is None:
            fitted = self._model.fit(
                cov_type=cov_type,
                cov_kwds=cov_kwds,
                use_t=_validate_bool("use_t", use_t),
            )
            result = _make_result(self, fitted)
        else:
            error_method = _normalise_fit_method(error_method)
            error_maxiter = _normalise_maxiter(error_maxiter)
            error_cov_type = _normalise_cov_type(error_cov_type)
            error_require_convergence = _normalise_require_convergence(
                error_require_convergence
            )
            error_result = self._build_error_model().fit(
                method=error_method,
                maxiter=error_maxiter,
                cov_type=error_cov_type,
                require_convergence=error_require_convergence,
            )
            result = _make_error_result(self, error_result)
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
    n_jobs : int, keyword-only
        Candidate worker count. ``-1`` uses at most CPU count minus one,
        ``1`` forces serial execution, and positive values request a bounded
        number of workers. Default ``-1``.

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
        n_jobs=-1,
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
        self.n_jobs = _validate_n_jobs(n_jobs)

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

    def fit(
        self,
        *,
        cov_type="nonrobust",
        cov_kwds=None,
        use_t=True,
        progress_callback=None,
    ):
        """Run lag selection and fit the selected ARDL model.

        Parameters
        ----------
        cov_type : str, default "nonrobust"
            Covariance estimator used for the selected final model.
        cov_kwds : mapping, optional
            Additional covariance-estimator options.
        use_t : bool, default True
            Whether final-model inference uses the Student t distribution.
        progress_callback : callable, optional
            Parent-process callback invoked as ``callback(completed, total)``
            after each candidate finishes. The callback is never sent to a
            worker process.

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
        context, iter_orders, var_names = _build_ardl_selection_context(
            response,
            maxlag=self.maxlag,
            exog=self.exog,
            maxorder=self.maxorder,
            trend=self.trend,
            causal=self.causal,
            seasonal=self.seasonal,
            period=self.period,
            hold_back=self.hold_back,
        )
        tasks, n_tasks = _iter_ardl_selection_tasks(
            context,
            iter_orders,
            var_names,
            search_method=self.search_method,
        )
        evaluated = _map_candidates(
            tasks,
            _compute_ardl_ics,
            n_jobs=self.n_jobs,
            n_tasks=n_tasks,
            progress_callback=progress_callback,
        )
        ics = dict(evaluated)
        criterion_index = {"aic": 0, "bic": 1}[self.criterion]
        selected_key = min(
            ics,
            key=lambda key: ics[key][criterion_index],
        )
        selected_model = StatsmodelsARDL(
            response,
            selected_key[0],
            self.exog,
            dict(selected_key[1]),
            self.trend,
            causal=self.causal,
            seasonal=self.seasonal,
            hold_back=self.hold_back,
            period=self.period,
            missing="none",
        )
        selected_estimator = ARDL(
            self.data,
            lags=selected_model.ar_lags,
            exog=self.exog,
            # statsmodels represents an excluded input as ``None`` in the
            # selection criterion, but omits that key from ``dl_lags``.
            # Preserve the exclusion explicitly: an empty mapping would make
            # ARDL restore its default contemporaneous input lag on refit.
            order={
                name: selected_model.dl_lags.get(name)
                for name in self.exog_names
            },
            trend=self.trend,
            dates=self.dates,
            causal=self.causal,
            seasonal=self.seasonal,
            period=self.period,
            hold_back=selected_model.hold_back,
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
        scores = sorted(
            ics.items(),
            key=lambda item: item[1][criterion_index],
        )
        rows = []
        for key, values in scores:
            score = values[criterion_index]
            specification = key
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
