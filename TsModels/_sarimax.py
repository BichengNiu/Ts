"""SARIMAX model estimation via statsmodels SARIMAX.

Provides :class:`SARIMAX` and :class:`SARIMAXResult`.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX as StatsmodelsSARIMAX

from Ts.TsUtils._validation import (
    _resolve_missing_rows,
    validate_alpha as _validate_prediction_alpha,
)

from Ts.TsModels._base import (
    BaseModel,
    BaseModelResult,
    PredictResult,
    _resolve_prediction_window,
)
from Ts.TsModels._distributed_lag import (
    RationalLagResult,
    RationalLagSpec,
    _make_rational_lag_results,
    _normalise_nonnegative_integer,
    _RationalLagSARIMAX,
)
from Ts.TsModels._intervention import (
    EventColumns,
    EventSpec,
    _validate_datetime_index,
    build_event_matrix,
)


_SARIMAX_OPTIMIZERS = frozenset(
    {
        "basinhopping",
        "bfgs",
        "cg",
        "lbfgs",
        "ncg",
        "newton",
        "nm",
        "powell",
    }
)
_SARIMAX_COVARIANCE_TYPES = frozenset(
    {"approx", "oim", "opg", "robust", "robust_approx"}
)


@dataclass(frozen=True)
class _SARIMAXInputs:
    endog: np.ndarray
    dates: pd.DatetimeIndex | None
    exog: np.ndarray | None
    exog_names: tuple[str, ...]
    future_exog: pd.DataFrame | None
    dropped_positions: tuple[int, ...]


def _numeric_array(values, name):
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error


def _normalise_lag_order(value, name):
    """Return an integer order or immutable tuple of active positive lags."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return _normalise_nonnegative_integer(value, name)
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a non-negative integer or an iterable of lags")
    try:
        lags = tuple(value)
    except TypeError as error:
        raise TypeError(
            f"{name} must be a non-negative integer or an iterable of lags"
        ) from error
    if not lags:
        return 0

    normalised = []
    for lag in lags:
        if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)):
            raise TypeError(f"{name} lags must be positive integers")
        lag = int(lag)
        if lag <= 0:
            raise ValueError(f"{name} lags must be positive integers")
        normalised.append(lag)
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"{name} lags must be unique")
    return tuple(sorted(normalised))


def _normalise_order(order):
    if not isinstance(order, (tuple, list)) or len(order) != 3:
        raise ValueError(f"order must be a tuple of (p, d, q), got {order}")
    p, d, q = order
    return (
        _normalise_lag_order(p, "p"),
        _normalise_nonnegative_integer(d, "d"),
        _normalise_lag_order(q, "q"),
    )


def _normalise_seasonal_order(seasonal_order):
    if not isinstance(seasonal_order, (tuple, list)) or len(seasonal_order) != 4:
        raise ValueError(
            f"seasonal_order must be a tuple of (P, D, Q, s), got {seasonal_order}"
        )
    P, D, Q, s = seasonal_order
    return (
        _normalise_lag_order(P, "P"),
        _normalise_nonnegative_integer(D, "D"),
        _normalise_lag_order(Q, "Q"),
        _normalise_nonnegative_integer(s, "s"),
    )


def _normalise_log(log):
    """Return an explicit boolean log-transform flag."""
    if not isinstance(log, (bool, np.bool_)):
        raise TypeError("log must be a boolean")
    return bool(log)


def _normalise_fit_method(method):
    """Return a supported statsmodels optimizer name."""
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    method = method.strip().lower()
    if not method:
        raise ValueError("method must be a non-empty string")
    if method not in _SARIMAX_OPTIMIZERS:
        supported = ", ".join(sorted(_SARIMAX_OPTIMIZERS))
        raise ValueError(f"method must be one of: {supported}")
    return method


def _normalise_maxiter(maxiter):
    """Return a strictly positive optimizer iteration limit."""
    if isinstance(maxiter, (bool, np.bool_)) or not isinstance(
        maxiter,
        (int, np.integer),
    ):
        raise TypeError("maxiter must be a positive integer")
    maxiter = int(maxiter)
    if maxiter <= 0:
        raise ValueError("maxiter must be a positive integer")
    return maxiter


def _normalise_start_params(start_params):
    """Return a finite copied one-dimensional starting vector."""
    if start_params is None:
        return None
    values = _numeric_array(start_params, "start_params")
    if values.ndim != 1:
        raise ValueError("start_params must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("start_params must contain only finite values")
    return values.copy()


def _normalise_cov_type(cov_type):
    if not isinstance(cov_type, str):
        raise TypeError("cov_type must be a string")
    cov_type = cov_type.strip().lower()
    if cov_type not in _SARIMAX_COVARIANCE_TYPES:
        allowed = ", ".join(sorted(_SARIMAX_COVARIANCE_TYPES))
        raise ValueError(f"cov_type must be one of: {allowed}")
    return cov_type


def _normalise_require_convergence(require_convergence):
    """Return an explicit boolean convergence requirement."""
    if not isinstance(require_convergence, (bool, np.bool_)):
        raise TypeError("require_convergence must be a boolean")
    return bool(require_convergence)


def _prediction_arrays(prediction, alpha, *, log):
    """Return prediction arrays, optionally on the original response scale."""
    frame = prediction.summary_frame(alpha=alpha)
    mean = np.asarray(frame["mean"], dtype=float)
    lower = np.asarray(frame["mean_ci_lower"], dtype=float)
    upper = np.asarray(frame["mean_ci_upper"], dtype=float)
    if not log:
        return mean, lower, upper

    variance = np.asarray(prediction.var_pred_mean, dtype=float)
    variance = np.maximum(variance, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.exp(mean + 0.5 * variance)
        lower = np.exp(lower)
        upper = np.exp(upper)
    return mean, lower, upper


def _active_lags(order_component):
    if isinstance(order_component, (int, np.integer)):
        return tuple(range(1, int(order_component) + 1))
    return tuple(order_component)


def _maximum_lag(order_component):
    return max(_active_lags(order_component), default=0)


def _automatic_rdl_likelihood_burn(distributed_lags, order, seasonal_order):
    """Return burn needed to flush incomplete finite input history."""
    histories = [
        spec.delay + max(spec.numerator_lags)
        for spec in distributed_lags.values()
        if spec.initialization == "auto" and not spec.denominator_lags
    ]
    input_history = max(histories, default=0)
    if input_history == 0:
        return 0

    p, d, q = order
    P, D, Q, s = seasonal_order
    ar_depth = _maximum_lag(p) + _maximum_lag(P) * s
    ma_depth = _maximum_lag(q) + _maximum_lag(Q) * s
    integration_depth = d + D * s
    return input_history + max(ar_depth, ma_depth, integration_depth)


def _display_order(order):
    p, d, q = order
    return (_maximum_lag(p), d, _maximum_lag(q))


def _normalise_exog_names(names, width):
    if names is None:
        raise ValueError("exog_names is required for array exog")
    names = tuple(names)
    if len(names) != width:
        raise ValueError(f"exog_names must contain one name per exog column ({width})")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("exog_names must contain non-empty strings")
    names = tuple(name.strip() for name in names)
    if len(set(names)) != len(names):
        raise ValueError("exog_names must be unique")
    if any(name.startswith("event__") for name in names):
        raise ValueError("ordinary exog names must not use the event__ namespace")
    return names


def _normalise_data_and_dates(data, dates):
    if isinstance(data, pd.Series):
        if dates is not None:
            raise ValueError("dates must not be provided when data is a pandas Series")
        values = _numeric_array(data.to_numpy(), "data")
        data_dates = (
            _validate_datetime_index(data.index, "data dates")
            if isinstance(data.index, pd.DatetimeIndex)
            else None
        )
    else:
        values = _numeric_array(data, "data")
        data_dates = None if dates is None else _validate_datetime_index(dates, "dates")
    if values.ndim != 1:
        raise ValueError(f"data must be one-dimensional, got shape {values.shape}")
    if data_dates is not None and len(data_dates) != len(values):
        raise ValueError("dates must contain exactly one value per data observation")
    return values.copy(), data_dates


def _normalise_dataframe_exog(exog, dates, exog_names, endog_length):
    if exog_names is not None:
        raise ValueError("exog_names must not be provided when exog is a DataFrame")
    names = _normalise_exog_names(tuple(exog.columns), exog.shape[1])
    if dates is None:
        if len(exog) != endog_length:
            raise ValueError(
                f"exog has {len(exog)} observations; expected "
                f"{endog_length} observations"
            )
        historical = _numeric_array(exog.loc[:, list(names)], "exog")
        return historical.copy(), names, None
    index = _validate_datetime_index(exog.index, "exog dates")
    if str(index.tz) != str(dates.tz):
        raise ValueError("exog dates timezone must match data dates timezone")
    missing_dates = dates.difference(index)
    if len(missing_dates):
        raise ValueError(
            f"exog is missing historical date {missing_dates[0].isoformat()}"
        )
    historical_mask = index <= dates[-1]
    extra_historical = index[historical_mask & ~index.isin(dates)]
    if len(extra_historical):
        raise ValueError(
            f"exog contains extra historical date {extra_historical[0].isoformat()}"
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


def _normalise_series_exog(exog, dates, exog_names, endog_length):
    name = exog.name
    has_name = isinstance(name, str) and bool(name.strip())
    if has_name:
        if exog_names is not None:
            raise ValueError(
                "exog_names must not be provided when exog is a named Series"
            )
        resolved_name = _normalise_exog_names((name,), 1)[0]
    else:
        if exog_names is None:
            raise ValueError(
                "an unnamed exog Series requires exog_names with exactly one name"
            )
        resolved_name = _normalise_exog_names(exog_names, 1)[0]

    frame = exog.rename(resolved_name).to_frame()
    return _normalise_dataframe_exog(
        frame,
        dates,
        None,
        endog_length,
    )


def _normalise_array_exog(exog, endog_length, exog_names):
    values = _numeric_array(exog, "exog")
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    elif values.ndim != 2:
        raise ValueError(
            f"exog must be one- or two-dimensional, got shape {values.shape}"
        )
    if len(values) != endog_length:
        raise ValueError(
            f"exog has {len(values)} observations; expected {endog_length} observations"
        )
    names = _normalise_exog_names(exog_names, values.shape[1])
    return values.copy(), names


def _normalise_sarimax_inputs(
    data,
    *,
    dates=None,
    exog=None,
    exog_names=None,
    missing="drop",
):

    endog, data_dates = _normalise_data_and_dates(data, dates)

    future_exog = None
    if exog is None:
        if exog_names is not None:
            raise ValueError("exog_names requires exog")
        historical_exog = None
        names = ()
    elif isinstance(exog, pd.Series):
        historical_exog, names, future_exog = _normalise_series_exog(
            exog,
            data_dates,
            exog_names,
            len(endog),
        )
    elif isinstance(exog, pd.DataFrame):
        historical_exog, names, future_exog = _normalise_dataframe_exog(
            exog,
            data_dates,
            exog_names,
            len(endog),
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
    dropped_positions = _resolve_missing_rows(
        finite,
        missing,
        name="data or historical exog",
    )
    if missing == "drop":
        endog = endog[finite]
        if data_dates is not None:
            data_dates = data_dates[finite].copy()
        if historical_exog is not None:
            historical_exog = historical_exog[finite]

    return _SARIMAXInputs(
        endog=endog.copy(),
        dates=data_dates,
        exog=(
            None
            if historical_exog is None
            else np.array(historical_exog, dtype=float, copy=True)
        ),
        dropped_positions=dropped_positions,
        exog_names=names,
        future_exog=future_exog,
    )


def _validate_events(events, dates):
    event_specs = tuple(events or ())
    if any(not isinstance(event, EventSpec) for event in event_specs):
        raise TypeError("events must contain only EventSpec instances")
    if event_specs and dates is None:
        raise ValueError("events require dated data")
    return event_specs


def _normalise_distributed_lags(distributed_lags, inputs):
    if distributed_lags is None:
        return {}
    if not isinstance(distributed_lags, Mapping):
        raise TypeError(
            "distributed_lags must be a mapping from input name to RationalLagSpec"
        )
    if not distributed_lags:
        return {}
    if inputs.exog is None:
        raise ValueError("distributed_lags requires exog input data")

    unknown = [name for name in distributed_lags if name not in inputs.exog_names]
    if unknown:
        raise ValueError(f"distributed_lags contains unknown exog input {unknown[0]!r}")
    for name, spec in distributed_lags.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("distributed_lags names must be non-empty strings")
        if not isinstance(spec, RationalLagSpec):
            raise TypeError(f"distributed_lags[{name!r}] must be a RationalLagSpec")
    if inputs.dropped_positions:
        raise ValueError(
            "rational distributed lags require consecutive observations; "
            "missing='drop' removed rows"
        )
    if inputs.dates is not None and len(inputs.dates) >= 3:
        frequency = inputs.dates.freq or pd.infer_freq(inputs.dates)
        if frequency is None:
            raise ValueError("rational distributed lags require regularly spaced dates")

    return {
        name: distributed_lags[name]
        for name in inputs.exog_names
        if name in distributed_lags
    }


def _validate_design_matrix(frame, trend):
    if frame is None:
        return
    values = frame.to_numpy(dtype=float)
    for index, name in enumerate(frame.columns):
        column = values[:, index]
        if np.all(column == 0.0):
            raise ValueError(f"all-zero design column is not identified: {name}")
        if np.all(column == column[0]) and "c" in trend:
            raise ValueError(
                f"constant exog column {name!r} conflicts with trend={trend!r}"
            )
    rank = np.linalg.matrix_rank(values)
    if rank < values.shape[1]:
        raise ValueError(
            "combined exogenous design matrix is rank deficient: "
            f"rank {rank} for {values.shape[1]} columns"
        )


def _combined_design(inputs, events, trend, *, distributed_lag_names=()):
    index = (
        inputs.dates if inputs.dates is not None else pd.RangeIndex(len(inputs.endog))
    )
    frames = []
    if inputs.exog is not None:
        exog_frame = pd.DataFrame(
            inputs.exog,
            index=index,
            columns=inputs.exog_names,
        )
        ordinary_names = [
            name for name in inputs.exog_names if name not in distributed_lag_names
        ]
        if ordinary_names:
            frames.append(exog_frame.loc[:, ordinary_names])
    event_metadata: dict[str, EventColumns] = {}
    event_frame = None
    if events:
        event_frame, event_metadata = build_event_matrix(
            inputs.dates,
            events,
            reserved_names=inputs.exog_names,
        )
        frames.append(event_frame)
    design = pd.concat(frames, axis=1) if frames else None
    _validate_design_matrix(design, trend)
    return design, event_frame, event_metadata


@dataclass
class ScenarioForecastResult:
    """Forecasts produced from multiple future-exogenous scenarios.

    Parameters
    ----------
    scenarios : dict of str to PredictResult
        One forecast per named future-input scenario.
    default_name : str or None
        Scenario selected by convenience accessors, when defined.
    dates : pandas.DatetimeIndex or None
        Shared forecast dates.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsModels import PredictResult, ScenarioForecastResult
    >>> forecast = PredictResult(np.array([1.0]), None, None, np.array([True]))
    >>> result = ScenarioForecastResult({"baseline": forecast}, "baseline", None)
    >>> result["baseline"].mean.tolist()
    [1.0]
    """

    scenarios: dict[str, PredictResult]
    default_name: str | None
    dates: pd.DatetimeIndex | None

    def __post_init__(self):
        if not isinstance(self.scenarios, dict) or not self.scenarios:
            raise ValueError("scenarios must be a non-empty dictionary")
        if any(
            not isinstance(name, str) or not name.strip() for name in self.scenarios
        ):
            raise ValueError("scenario names must be non-empty strings")
        lengths = {
            len(np.asarray(prediction.mean)) for prediction in self.scenarios.values()
        }
        if len(lengths) != 1:
            raise ValueError("all scenario predictions must have equal length")
        if self.default_name is not None and self.default_name not in self.scenarios:
            raise ValueError("default_name must identify an existing scenario")
        if self.dates is not None:
            self.dates = _validate_datetime_index(
                self.dates,
                "scenario dates",
            ).copy()
            if len(self.dates) != next(iter(lengths)):
                raise ValueError("scenario dates length must match prediction length")
        self.scenarios = dict(self.scenarios)

    def __getitem__(self, name):
        return self.scenarios[name]

    def summary(self):
        """Return a compact scenario inventory.

        Returns
        -------
        str

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsModels import PredictResult, ScenarioForecastResult
        >>> forecast = PredictResult(np.array([1.0]), None, None, np.array([True]))
        >>> result = ScenarioForecastResult({"baseline": forecast}, "baseline", None)
        >>> "baseline" in result.summary()
        True
        """
        default = self.default_name or "none"
        names = ", ".join(self.scenarios)
        return (
            f"Forecast scenarios: {names}\n"
            f"Default scenario: {default}\n"
            f"Periods: {len(next(iter(self.scenarios.values())).mean)}"
        )

    def plot(self, title=None):
        """Plot all scenario means on shared axes.

        Parameters
        ----------
        title : str, optional
            Chart title.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsModels import PredictResult, ScenarioForecastResult
        >>> forecast = PredictResult(np.array([1.0]), None, None, np.array([True]))
        >>> result = ScenarioForecastResult({"baseline": forecast}, "baseline", None)
        >>> fig, ax = result.plot()
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        first = next(iter(self.scenarios.values()))
        x = self.dates if self.dates is not None else np.arange(len(first.mean))
        for name, prediction in self.scenarios.items():
            ax.plot(x, prediction.mean, label=name)
        ax.set_title(title or "Forecast Scenarios")
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig, ax


def _coerce_future_frame(
    values,
    *,
    name,
    expected_dates,
    exog_names,
    array_dates_provided,
):
    if isinstance(values, pd.Series):
        if len(exog_names) != 1:
            raise ValueError(
                f"scenario {name!r} Series requires exactly one exog column"
            )
        if values.name is not None:
            series_name = _normalise_exog_names((values.name,), 1)[0]
            if series_name != exog_names[0]:
                raise ValueError(
                    f"scenario {name!r} Series name must be "
                    f"{exog_names[0]!r}, got {series_name!r}"
                )
        if isinstance(expected_dates, pd.DatetimeIndex):
            index = _validate_datetime_index(
                values.index,
                f"scenario {name!r} dates",
            )
        else:
            index = values.index
        if len(index) != len(expected_dates) or not index.equals(expected_dates):
            raise ValueError(
                f"scenario {name!r} dates/rows must exactly match "
                "the requested future dates"
            )
        array = _numeric_array(values, f"scenario {name!r}").reshape(-1, 1)
    elif isinstance(values, pd.DataFrame):
        actual_columns = tuple(values.columns)
        if actual_columns != exog_names:
            raise ValueError(
                f"scenario {name!r} columns must be exactly {exog_names}, "
                f"got {actual_columns}"
            )
        if isinstance(expected_dates, pd.DatetimeIndex):
            index = _validate_datetime_index(
                values.index,
                f"scenario {name!r} dates",
            )
        else:
            index = values.index
        if len(index) != len(expected_dates) or not index.equals(expected_dates):
            raise ValueError(
                f"scenario {name!r} dates/rows must exactly match "
                "the requested future dates"
            )
        array = _numeric_array(values, f"scenario {name!r}")
    else:
        if not array_dates_provided:
            raise ValueError("array future_exog requires future_dates")
        array = _numeric_array(values, f"scenario {name!r}")
        if array.ndim == 1:
            if len(exog_names) != 1:
                raise ValueError(
                    f"scenario {name!r} one-dimensional array requires "
                    "exactly one exog column"
                )
            array = array.reshape(-1, 1)
        elif array.ndim != 2:
            raise ValueError(f"scenario {name!r} must be one- or two-dimensional")
        if array.shape != (len(expected_dates), len(exog_names)):
            raise ValueError(
                f"scenario {name!r} has shape {array.shape}; expected "
                f"{(len(expected_dates), len(exog_names))}"
            )
    if array.ndim != 2 or array.shape != (
        len(expected_dates),
        len(exog_names),
    ):
        raise ValueError(f"scenario {name!r} rows/columns do not match the forecast")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"scenario {name!r} contains non-finite values")
    return pd.DataFrame(
        np.array(array, dtype=float, copy=True),
        index=expected_dates.copy(),
        columns=exog_names,
    )


def _normalise_future_scenarios(
    future_exog,
    *,
    future_dates,
    expected_dates,
    exog_names,
    default_future_exog,
):
    scenarios = {}
    default_name = None
    if default_future_exog is not None:
        missing = expected_dates.difference(default_future_exog.index)
        if len(missing):
            missing_label = (
                missing[0].isoformat()
                if hasattr(missing[0], "isoformat")
                else str(missing[0])
            )
            raise ValueError(f"default future exog is missing date {missing_label}")
        scenarios["default"] = _coerce_future_frame(
            default_future_exog.loc[expected_dates],
            name="default",
            expected_dates=expected_dates,
            exog_names=exog_names,
            array_dates_provided=True,
        )
        default_name = "default"

    if future_exog is None:
        if not scenarios:
            missing_date = (
                expected_dates[0].isoformat()
                if hasattr(expected_dates[0], "isoformat")
                else str(expected_dates[0])
            )
            raise ValueError(f"future exog is required starting at {missing_date}")
        return scenarios, default_name

    if isinstance(future_exog, Mapping):
        if not future_exog:
            raise ValueError("future_exog scenario mapping must not be empty")
        items = tuple(future_exog.items())
    else:
        items = (("custom", future_exog),)

    for name, values in items:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("scenario names must be non-empty strings")
        if name == "default":
            raise ValueError("'default' is a reserved scenario name")
        scenarios[name] = _coerce_future_frame(
            values,
            name=name,
            expected_dates=expected_dates,
            exog_names=exog_names,
            array_dates_provided=(
                future_dates is not None
                or not isinstance(expected_dates, pd.DatetimeIndex)
            ),
        )
    return scenarios, default_name


@dataclass(frozen=True)
class ARCycleResult:
    """Algebraic damped-cycle diagnostic for an AR(2) component.

    The period is measured in original observation intervals. It is available
    only when the selected AR(2) component has complex conjugate roots and the
    fitted model's complete AR polynomial is stationary.

    Parameters
    ----------
    component : str
        Non-seasonal or seasonal AR component label.
    phi1, phi2 : float or None
        Selected AR(2) coefficients.
    discriminant : float or None
        Quadratic discriminant determining real versus complex roots.
    has_complex_roots : bool
        Whether the selected component implies a damped oscillation.
    is_stationary : bool
        Stationarity verdict for the complete fitted AR polynomial.
    period : float or None
        Algebraic cycle period in observation intervals.
    lag_scale : int
        Original-time multiplier, equal to one for non-seasonal cycles.

    Attributes
    ----------
    identified : bool
        Whether a stationary complex-root cycle period is available.

    Examples
    --------
    >>> from Ts.TsModels import ARCycleResult
    >>> result = ARCycleResult(
    ...     component="nonseasonal", lag_scale=1, phi1=1.2, phi2=-0.5,
    ...     discriminant=-0.56, has_complex_roots=True,
    ...     is_stationary=True, period=8.0,
    ... )
    >>> result.identified
    True
    """

    component: str
    lag_scale: int
    phi1: float
    phi2: float
    discriminant: float
    has_complex_roots: bool
    is_stationary: bool
    period: float | None

    @property
    def identified(self):
        """Whether the fitted AR(2) component identifies a damped cycle."""
        return self.period is not None


@dataclass
class SARIMAXResult(BaseModelResult):
    """Result container for SARIMAX model estimation.

    Inherits all fields from :class:`BaseModelResult` and adds SARIMAX-specific
    prediction and forecasting methods.

    Parameters
    ----------
    _order : tuple or None
        SARIMAX order (p, d, q).
    _seasonal_order : tuple or None
        Seasonal order (P, D, Q, s).
    _statsmodels_result : object
        Raw statsmodels SARIMAXResultsWrapper, stored for internal
        predict / forecast delegation.
    model_type, params, std_errors, p_values : see BaseModelResult
    aic, bic, log_likelihood, residuals, fitted_values, nobs, data : see BaseModelResult
        Shared estimation output inherited from ``BaseModelResult``.

    Examples
    --------
    >>> from Ts.TsModels import SARIMAX, SARIMAXResult
    >>> from Ts.TsSims import simulate_sarima
    >>> data = simulate_sarima(n=60, order=(1, 0, 0), seed=42).data
    >>> result = SARIMAX(data, order=(1, 0, 0)).fit()
    >>> isinstance(result, SARIMAXResult)
    True
    >>> result.converged
    True
    """

    _order: tuple | None = None
    _seasonal_order: tuple | None = None
    _statsmodels_result: object = None
    _trend: str = "c"
    _log_transform: bool = False

    @property
    def log(self):
        """Whether the response was fitted on the natural-log scale."""
        return bool(self._log_transform)

    @property
    def converged(self):
        """Whether the maximum-likelihood optimizer reported convergence."""
        details = getattr(self._statsmodels_result, "mle_retvals", None)
        return bool(details and details.get("converged", False))

    @property
    def optimizer(self):
        """Return the optimizer used by statsmodels, if recorded."""
        settings = getattr(self._statsmodels_result, "mle_settings", None)
        if not settings:
            return None
        optimizer = settings.get("optimizer")
        return None if optimizer is None else str(optimizer)

    @property
    def optimization_details(self):
        """Return a defensive copy of statsmodels optimizer diagnostics."""
        details = getattr(self._statsmodels_result, "mle_retvals", None)
        return {} if details is None else copy.deepcopy(dict(details))

    @property
    def likelihood_burn(self):
        """Number of leading observations excluded from the likelihood."""
        if self._statsmodels_result is None:
            return 0
        return int(self._statsmodels_result.loglikelihood_burn)

    @property
    def effective_nobs(self):
        """Number of observations contributing to the fitted likelihood."""
        return self.nobs - self.likelihood_burn

    def _mask_state_initialization(self, values):
        """Mask state-space burn-in values that are not valid fitted data."""
        if values is None:
            return None
        masked = np.asarray(values, dtype=float).copy()
        if self._statsmodels_result is None:
            return masked
        burn = int(self._statsmodels_result.loglikelihood_burn)
        if burn < 0 or burn > len(masked):
            raise RuntimeError(
                "statsmodels returned an invalid loglikelihood_burn value: "
                f"{burn} for {len(masked)} observations"
            )
        masked[:burn] = np.nan
        return masked

    def _fitted_values_for_plot(self):
        """Return fitted values after the state-space initialization period."""
        return self._mask_state_initialization(self.fitted_values)

    _dates: pd.DatetimeIndex | None = None
    _ordinary_exog: np.ndarray | None = None
    _ordinary_exog_names: tuple[str, ...] = ()
    _static_exog_names: tuple[str, ...] = ()
    _event_specs: tuple[EventSpec, ...] = ()
    _event_metadata: dict[str, EventColumns] | None = None
    _design_columns: tuple[str, ...] = ()
    _design_matrix: np.ndarray | None = None
    _default_future_exog: pd.DataFrame | None = None
    _model_kwargs: dict | None = None
    _distributed_lag_results: dict[str, RationalLagResult] | None = None
    _enforce_distributed_lag_stability: bool = True

    @property
    def distributed_lags(self):
        """Return structured rational distributed-lag results by input."""
        return dict(self._distributed_lag_results or {})

    @property
    def distributed_lag_names(self):
        """Return distributed-lag input names in fitted parameter order."""
        return tuple((self._distributed_lag_results or {}).keys())

    @property
    def distributed_lag_coefficients(self):
        """Return estimated and fixed transfer-polynomial coefficients."""
        results = self._distributed_lag_results or {}
        if not results:
            return pd.DataFrame(
                columns=[
                    "input",
                    "component",
                    "lag",
                    "parameter",
                    "estimate",
                    "standard_error",
                    "p_value",
                    "fixed",
                ]
            )
        return pd.concat(
            [result.coefficients for result in results.values()],
            ignore_index=True,
        )

    @property
    def steady_state_gains(self):
        """Return per-input steady-state gains with delta-method intervals."""
        rows = [
            result.gain() for result in (self._distributed_lag_results or {}).values()
        ]
        return pd.DataFrame(
            rows,
            columns=[
                "input",
                "estimate",
                "standard_error",
                "lower",
                "upper",
                "stable",
            ],
        )

    def weights(self, steps):
        """Return one impulse-weight column per distributed-lag input.

        Parameters
        ----------
        steps : int
            Strictly positive response horizon.

        Returns
        -------
        pandas.DataFrame
            Dynamic weights with one column per RDL input.

        Examples
        --------
        >>> from Ts.TsModels import RationalLagSpec, SARIMAX
        >>> from Ts.TsSims import RDLInputSpec, simulate_rdl
        >>> simulated = simulate_rdl(
        ...     n=100,
        ...     distributed_lags={"x": RDLInputSpec({0: 1.0}, {1: 0.4})},
        ...     seed=42,
        ... )
        >>> result = SARIMAX(
        ...     simulated.data,
        ...     exog=simulated.exog,
        ...     distributed_lags={"x": RationalLagSpec(0, 1)},
        ... ).fit()
        >>> result.weights(4).shape
        (4, 1)
        """
        results = self._distributed_lag_results or {}
        if not results:
            raise ValueError("model has no rational distributed-lag inputs")
        return pd.concat(
            [result.weights(steps) for result in results.values()],
            axis=1,
        )

    def feedback_test(
        self,
        lags,
        inputs=None,
        *,
        trend="c",
        missing="raise",
        alpha=0.05,
    ):
        """Test whether past model output predicts current exogenous inputs.

        The method composes :class:`Ts.TsTests.FeedbackTest` using the
        original historical exogenous columns retained by this fitted model.
        Every exogenous input remains in the lagged control set even when
        ``inputs`` selects only a subset of current-input equations.

        Parameters
        ----------
        lags : int
            Positive common lag order.
        inputs : str or sequence of str, optional
            Inputs to test. The default tests every original exogenous input.
        trend : {"n", "c", "t", "ct"}, default "c"
            Deterministic terms in each feedback regression.
        missing : {"raise", "drop"}, default "raise"
            Missing-data policy applied by the feedback test.
        alpha : float, default 0.05
            Significance level for feedback decisions.

        Returns
        -------
        FeedbackTestResult
            Full OLS regressions and joint feedback F tests.

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsModels import SARIMAX
        >>> rng = np.random.default_rng(42)
        >>> result = SARIMAX(rng.normal(size=80), exog=rng.normal(size=(80, 1)), exog_names=["x"]).fit()
        >>> result.feedback_test(lags=1).input_names
        ('x',)
        """
        if self._ordinary_exog is None or not self._ordinary_exog_names:
            raise ValueError("feedback_test requires fitted exogenous inputs")

        from Ts.TsTests import FeedbackTest

        if self._dates is None:
            response = np.asarray(self.data, dtype=float)
            exog = pd.DataFrame(
                self._ordinary_exog,
                columns=self._ordinary_exog_names,
            )
        else:
            response = pd.Series(
                np.asarray(self.data, dtype=float),
                index=self._dates,
                name="y",
            )
            exog = pd.DataFrame(
                self._ordinary_exog,
                index=self._dates,
                columns=self._ordinary_exog_names,
            )
        return FeedbackTest(
            response,
            exog,
            lags,
            tested_inputs=inputs,
            trend=trend,
            missing=missing,
            alpha=alpha,
        ).fit()

    def residual_ccf_test(
        self,
        input_models,
        lags=12,
        inputs=None,
        *,
        alpha=0.05,
    ):
        """Test RDL residuals against explicitly prewhitened input innovations.

        Each input model must be a converged, univariate, input-only
        :class:`SARIMAXResult` (or an :class:`AutoModelResult` whose best model
        is such a result) fitted to the exact historical RDL input path. The
        method aligns post-burn innovations at the common sample end and
        deducts only that input's fitted transfer-polynomial parameters from
        the joint S* degrees of freedom.

        Parameters
        ----------
        input_models : mapping of str to SARIMAXResult or AutoModelResult
            Explicit prewhitening model for every selected RDL input.
        lags : int, default 12
            Positive maximum lag K; the test includes lags 0 through K.
        inputs : str or sequence of str, optional
            RDL inputs to test. The default tests every fitted RDL input.
        alpha : float, default 0.05
            Significance level for pointwise bands and joint decisions.

        Returns
        -------
        ResidualCCFTestResult
            Per-input residual CCFs and joint Box-Jenkins S* tests.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from Ts.TsModels import RationalLagSpec, SARIMAX
        >>> rng = np.random.default_rng(42)
        >>> x = pd.Series(rng.normal(size=80), name="x")
        >>> y = 0.8 * x.to_numpy() + rng.normal(scale=0.3, size=80)
        >>> fitted = SARIMAX(y, exog=x, trend="n", distributed_lags={"x": RationalLagSpec()}).fit()
        >>> input_model = SARIMAX(x, trend="n").fit()
        >>> fitted.residual_ccf_test({"x": input_model}, lags=4).input_names
        ('x',)
        """
        results = self._distributed_lag_results or {}
        if not results:
            raise ValueError(
                "residual_ccf_test requires a fitted rational distributed-lag model"
            )
        if not self.converged:
            raise RuntimeError("fitted RDL model did not converge")
        if not isinstance(input_models, Mapping):
            raise TypeError("input_models must be a mapping keyed by RDL input name")

        if inputs is None:
            selected = tuple(results)
        elif isinstance(inputs, str):
            selected = (inputs,)
        else:
            try:
                selected = tuple(inputs)
            except TypeError as error:
                raise TypeError(
                    "inputs must be a name or an iterable of names"
                ) from error
        if not selected:
            raise ValueError("inputs must contain at least one RDL input")
        if len(set(selected)) != len(selected):
            raise ValueError("inputs must be unique")
        unknown = [name for name in selected if name not in results]
        if unknown:
            raise ValueError(f"inputs contains unknown RDL input {unknown[0]!r}")
        unknown_models = [name for name in input_models if name not in results]
        if unknown_models:
            raise ValueError(
                f"input_models contains unknown RDL input {unknown_models[0]!r}"
            )
        missing_models = [name for name in selected if name not in input_models]
        if missing_models:
            raise ValueError(f"missing fitted input model for {missing_models[0]!r}")

        innovations = {}
        parameter_counts = {}
        for name in selected:
            supplied = input_models[name]
            candidate = getattr(supplied, "best_result", supplied)
            if not isinstance(candidate, SARIMAXResult):
                raise TypeError(
                    f"input_models[{name!r}] must be a SARIMAXResult or an "
                    "AutoModelResult with a SARIMAXResult best model"
                )
            if not candidate.converged:
                raise RuntimeError(
                    f"input prewhitening model for {name!r} did not converge"
                )
            if (
                candidate._ordinary_exog is not None
                or candidate._event_specs
                or candidate.distributed_lag_names
            ):
                raise ValueError(
                    f"input prewhitening model for {name!r} must be an input-only "
                    "univariate SARIMAX without exog, events, or distributed lags"
                )
            if candidate.log:
                raise ValueError(
                    f"input prewhitening model for {name!r} must use the same "
                    "untransformed input scale as the fitted RDL model"
                )

            position = self._ordinary_exog_names.index(name)
            expected = np.asarray(self._ordinary_exog[:, position], dtype=float)
            observed = np.asarray(candidate.data, dtype=float)
            if observed.ndim != 1 or not np.array_equal(observed, expected):
                raise ValueError(
                    f"input prewhitening model for {name!r} was not fitted to "
                    "the exact historical input used by this RDL model"
                )
            target_dates = self._dates
            input_dates = candidate._dates
            if (target_dates is None) != (input_dates is None) or (
                target_dates is not None and not target_dates.equals(input_dates)
            ):
                raise ValueError(
                    f"input prewhitening model for {name!r} must use the exact "
                    "RDL observation calendar"
                )
            innovations[name] = np.asarray(candidate.residuals, dtype=float)
            parameter_counts[name] = len(results[name].spec.parameter_names(name))

        from Ts.TsTests import ResidualCCFTest

        return ResidualCCFTest(
            self.residuals,
            innovations,
            lags,
            transfer_params=parameter_counts,
            alpha=alpha,
        ).fit()

    def plot_impulse_response(
        self,
        steps=20,
        inputs=None,
        sample_weights=None,
        **kwargs,
    ):
        """Plot fitted RDL weights, optionally over sample weights.

        Multiple selected inputs are shown as facets in fitted input order.
        Without ``sample_weights``, bars reuse :meth:`weights`. When sample
        weights from a preliminary finite-lag model are supplied, those values
        are plotted as bars and the fitted rational transfer-function weights
        are overlaid as solid lines.

        Parameters
        ----------
        steps : int, default 20
            Strictly positive response horizon.
        inputs : str or sequence of str, optional
            RDL inputs to plot. The default plots every fitted RDL input.
        sample_weights : pandas.Series or pandas.DataFrame, optional
            Preliminary finite-lag estimates to plot as bars. Their lag index
            and selected response names must match the fitted weights.
        **kwargs
            Additional options forwarded to
            :func:`Ts.TsPlots.plot_lag_response`.

        Returns
        -------
        tuple
            ``(fig, ax)`` for one input or ``(fig, axes)`` for facets.

        Examples
        --------
        >>> from Ts.TsModels import RationalLagSpec, SARIMAX
        >>> from Ts.TsSims import RDLInputSpec, simulate_rdl
        >>> simulated = simulate_rdl(n=80, distributed_lags={"x": RDLInputSpec({0: 1.0}, {1: 0.4})}, seed=42)
        >>> result = SARIMAX(simulated.data, exog=simulated.exog, distributed_lags={"x": RationalLagSpec(0, 1)}).fit()
        >>> fig, ax = result.plot_impulse_response(5)
        >>> len(ax.patches)
        5
        """
        results = self._distributed_lag_results or {}
        if not results:
            raise ValueError("model has no rational distributed-lag inputs")
        if inputs is None:
            selected = tuple(results)
        elif isinstance(inputs, str):
            selected = (inputs,)
        else:
            try:
                selected = tuple(inputs)
            except TypeError as error:
                raise TypeError(
                    "inputs must be a name or an iterable of names"
                ) from error
        if not selected:
            raise ValueError("inputs must contain at least one RDL input")
        if len(set(selected)) != len(selected):
            raise ValueError("inputs must be unique")
        unknown = [name for name in selected if name not in results]
        if unknown:
            raise ValueError(f"inputs contains unknown RDL input {unknown[0]!r}")

        from Ts.TsPlots import plot_lag_response

        weights = pd.concat([results[name].weights(steps) for name in selected], axis=1)
        if sample_weights is None:
            return plot_lag_response(weights, **kwargs)
        if isinstance(sample_weights, pd.DataFrame):
            missing = [name for name in selected if name not in sample_weights.columns]
            if missing:
                raise ValueError(
                    "sample_weights is missing selected input " f"{missing[0]!r}"
                )
            sample_weights = sample_weights.loc[:, list(selected)]
        return plot_lag_response(
            sample_weights,
            line_data=weights,
            **kwargs,
        )

    @property
    def dates(self):
        """Return a copy of the fitted observation dates."""
        return None if self._dates is None else self._dates.copy()

    @property
    def exog_names(self):
        """Return ordinary exogenous-variable names."""
        return tuple(self._ordinary_exog_names)

    @property
    def event_names(self):
        """Return event names in model-column order."""
        return tuple(event.name for event in self._event_specs)

    @property
    def design_columns(self):
        """Return the fitted combined-design column names."""
        return tuple(self._design_columns)

    @property
    def ar_lags(self):
        """Active non-seasonal autoregressive lags."""
        if self._order is None:
            return ()
        return _active_lags(self._order[0])

    @property
    def ma_lags(self):
        """Active non-seasonal moving-average lags."""
        if self._order is None:
            return ()
        return _active_lags(self._order[2])

    @property
    def fixed_params(self):
        """Sparse AR/MA and transfer coefficients excluded and fixed at zero."""
        fixed = {}
        if self.ar_lags:
            for lag in range(1, max(self.ar_lags) + 1):
                if lag not in self.ar_lags:
                    fixed[f"ar.L{lag}"] = 0.0
        if self.ma_lags:
            for lag in range(1, max(self.ma_lags) + 1):
                if lag not in self.ma_lags:
                    fixed[f"ma.L{lag}"] = 0.0
        for result in (self._distributed_lag_results or {}).values():
            fixed.update(result.fixed_params)
        return fixed

    @property
    def is_stationary(self):
        """Whether all roots of the fitted AR polynomial lie outside one."""
        return bool(np.all(np.abs(self.arroots) > 1.0))

    @property
    def is_invertible(self):
        """Whether all roots of the fitted MA polynomial lie outside one."""
        return bool(np.all(np.abs(self.maroots) > 1.0))

    @property
    def stationarity_enforced(self):
        """Whether stationarity was enforced during maximum likelihood."""
        if self._model_kwargs is not None:
            return bool(self._model_kwargs.get("enforce_stationarity", True))
        return bool(
            getattr(
                getattr(self._statsmodels_result, "model", None),
                "enforce_stationarity",
                True,
            )
        )

    @property
    def invertibility_enforced(self):
        """Whether invertibility was enforced during maximum likelihood."""
        if self._model_kwargs is not None:
            return bool(self._model_kwargs.get("enforce_invertibility", True))
        return bool(
            getattr(
                getattr(self._statsmodels_result, "model", None),
                "enforce_invertibility",
                True,
            )
        )

    @staticmethod
    def _format_root_diagnostic(roots, passed, *, absent):
        if len(roots) == 0:
            return f"Not applicable ({absent})"
        minimum = float(np.min(np.abs(roots)))
        conclusion = "Passed" if passed else "Failed"
        return f"{conclusion} (minimum |root| = {minimum:.4f})"

    def summary(self) -> str:
        """Return a formatted parameter summary string.

        Overrides BaseModelResult to add SARIMAX-specific details (order,
        seasonal_order), sparse-lag constraints, and AR/MA root conditions.

        Returns
        -------
        str

        Examples
        --------
        >>> from Ts.TsModels import SARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(n=80, order=(1, 0, 0), ar=[0.5], seed=42).data
        >>> result = SARIMAX(data, order=(1, 0, 0)).fit()
        >>> "SARIMAX" in result.summary()
        True
        """
        base = super().summary()
        lines = base.split("\n")
        header_lines = [lines[0]]
        if self._order:
            header_lines.append(f"Order: SARIMAX{_display_order(self._order)}")
        if self._seasonal_order and self._seasonal_order != (0, 0, 0, 0):
            header_lines.append(f"Seasonal Order: {self._seasonal_order}")
        for name, result in (self._distributed_lag_results or {}).items():
            numerator = ", ".join(str(lag) for lag in result.spec.numerator_lags)
            denominator = (
                ", ".join(str(lag) for lag in result.spec.denominator_lags) or "none"
            )
            initialization = result.spec.initialization
            if initialization == "auto":
                initialization += f" -> {result.spec.resolved_initialization}"
            header_lines.append(
                f"RDL {name}          : numerator [{numerator}]; "
                f"denominator [{denominator}]; delay {result.spec.delay}; "
                f"initialization {initialization}"
            )
            gain = result.steady_state_gain
            gain_text = "undefined" if not np.isfinite(gain) else f"{gain:.4f}"
            scale = " (log-response scale)" if self.log else ""
            header_lines.append(
                f"RDL {name} gain     : {gain_text}{scale}; "
                f"stable={'Yes' if result.is_stable else 'No'}"
            )
        if self.ar_lags:
            header_lines.append(
                "Active AR Lags     : " + ", ".join(str(lag) for lag in self.ar_lags)
            )
        if self.ma_lags:
            header_lines.append(
                "Active MA Lags     : " + ", ".join(str(lag) for lag in self.ma_lags)
            )
        if self.log:
            header_lines.append(
                "Response Scale     : original (log fit; bias-adjusted mean)"
            )
        level_inference = self.level_intercept_inference()
        if level_inference is not None:
            level_label = (
                "Log-response Intercept C" if self.log else "Level Intercept C"
            )
            header_lines.append(
                f"{level_label:<21s}: {level_inference['estimate']:.4f} "
                f"(SE={level_inference['standard_error']:.4f}, "
                f"|t|={abs(level_inference['statistic']):.2f})"
            )
            if "intercept" in self.params:
                header_lines.append(
                    f"State Intercept c    : {self.params['intercept']:.4f}"
                )
        if self.log:
            log_variance = self.unconditional_log_variance
            if log_variance is not None:
                header_lines.append(f"Log-response Variance: {log_variance:.4f}")
                header_lines.append(
                    f"Original-scale Mean  : {self.long_run_equilibrium():.4f}"
                )
        if self.likelihood_burn:
            header_lines.append(
                f"Likelihood Burn      : {self.likelihood_burn} "
                f"(effective n={self.effective_nobs})"
            )
        if self.fixed_params:
            header_lines.append("Fixed at Zero      : " + ", ".join(self.fixed_params))
        header_lines.extend(
            [
                "AR Stationarity    : "
                + self._format_root_diagnostic(
                    self.arroots,
                    self.is_stationary,
                    absent="no AR terms",
                ),
                "MA Invertibility   : "
                + self._format_root_diagnostic(
                    self.maroots,
                    self.is_invertible,
                    absent="no MA terms",
                ),
                "Stationarity Enforced  : "
                + ("Yes" if self.stationarity_enforced else "No"),
                "Invertibility Enforced : "
                + ("Yes" if self.invertibility_enforced else "No"),
                "Optimizer              : " + (self.optimizer or "Unknown"),
                "Converged              : " + ("Yes" if self.converged else "No"),
            ]
        )
        return "\n".join(header_lines + lines[1:])

    def _resolve_prediction_bounds(self, start, end, future_dates):
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

        future_index = None
        future_bound = max(
            (timestamp for timestamp in parsed_bounds if timestamp > self._dates[-1]),
            default=None,
        )
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
                        "future_dates is required when date frequency "
                        "cannot be inferred"
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
                    f"prediction date {timestamp.isoformat()} for {name} "
                    "is absent from the prediction calendar"
                )
            return location

        return position(start, "start"), position(end, "end")

    def _prediction_dates(self, window, future_dates):
        if self._dates is None:
            return None
        parts = []
        if window.in_sample_size:
            parts.append(
                self._dates[window.start : window.start + window.in_sample_size]
            )
        if window.has_forecast:
            parts.append(future_dates[window.forecast_skip :])
        if not parts:
            return None
        result = parts[0]
        for part in parts[1:]:
            result = result.append(part)
        return result

    def _resolve_future_dates(self, steps, future_dates):
        if future_dates is not None:
            dates = _validate_datetime_index(future_dates, "future_dates")
            if len(dates) != steps:
                raise ValueError(
                    f"future_dates has {len(dates)} rows; expected {steps}"
                )
            if self._dates is not None and dates[0] <= self._dates[-1]:
                raise ValueError("future_dates must begin after the sample")
            return dates.copy()
        if self._dates is None:
            return None
        frequency = self._dates.freq or pd.infer_freq(self._dates)
        if frequency is None:
            raise ValueError(
                "future_dates is required when date frequency cannot be inferred"
            )
        return pd.date_range(
            start=self._dates[-1],
            periods=steps + 1,
            freq=frequency,
        )[1:]

    def _future_design(self, ordinary_exog, future_dates):
        frames = []
        if ordinary_exog is not None and self._static_exog_names:
            frames.append(ordinary_exog.loc[:, list(self._static_exog_names)])
        if self._event_specs:
            if future_dates is None or self._dates is None:
                raise ValueError("future event generation requires future_dates")
            calendar = self._dates.append(future_dates)
            event_frame, _ = build_event_matrix(
                future_dates,
                self._event_specs,
                calendar=calendar,
                reserved_names=self._ordinary_exog_names,
            )
            frames.append(event_frame)
        if not frames:
            return None
        design = pd.concat(frames, axis=1)
        if tuple(design.columns) != self._design_columns:
            raise RuntimeError("future design columns do not match the fitted design")
        return design

    def _future_observation_intercept(self, future_inputs, future_design):
        if future_inputs is None:
            raise ValueError(
                "future exog is required for rational distributed-lag forecasting"
            )
        steps = len(future_inputs)
        intercept = np.zeros(steps)
        if future_design is not None:
            coefficients = np.array(
                [self.params[name] for name in future_design.columns],
                dtype=float,
            )
            intercept += future_design.to_numpy(dtype=float) @ coefficients

        if self._ordinary_exog is None:
            raise RuntimeError("fitted distributed-lag inputs are unavailable")
        history = pd.DataFrame(
            self._ordinary_exog,
            columns=self._ordinary_exog_names,
        )
        combined = pd.concat(
            [history, future_inputs.reset_index(drop=True)],
            ignore_index=True,
        )
        for name, result in (self._distributed_lag_results or {}).items():
            intercept += result.filter(combined[name].to_numpy())[-steps:]
        return intercept[None, :]

    def _predict_one(
        self,
        window,
        *,
        dynamic,
        alpha,
        future_design,
        future_inputs=None,
    ):
        start, end = window.start, window.end
        mean = np.full(window.size, np.nan)
        lower = np.full(window.size, np.nan)
        upper = np.full(window.size, np.nan)
        is_oos = np.zeros(window.size, dtype=bool)

        if window.has_forecast:
            n_in = window.in_sample_size
            if n_in > 0:
                pred_in = self._statsmodels_result.get_prediction(
                    start=start, end=self.nobs - 1, dynamic=dynamic
                )
                in_mean, in_lower, in_upper = _prediction_arrays(
                    pred_in,
                    alpha,
                    log=self.log,
                )
                mean[:n_in] = in_mean
                lower[:n_in] = in_lower
                upper[:n_in] = in_upper

            forecast_kwargs = {}
            if future_design is not None:
                forecast_kwargs["exog"] = future_design
            if self._distributed_lag_results:
                forecast_kwargs["obs_intercept"] = self._future_observation_intercept(
                    future_inputs,
                    future_design,
                )
            fc = self._statsmodels_result.get_forecast(
                steps=window.forecast_steps,
                **forecast_kwargs,
            )
            fc_mean, fc_lower, fc_upper = _prediction_arrays(
                fc,
                alpha,
                log=self.log,
            )
            forecast_slice = slice(window.forecast_skip, None)
            mean[n_in:] = fc_mean[forecast_slice]
            lower[n_in:] = fc_lower[forecast_slice]
            upper[n_in:] = fc_upper[forecast_slice]
            is_oos[n_in:] = True

        else:
            pred = self._statsmodels_result.get_prediction(
                start=start, end=end, dynamic=dynamic
            )
            mean, lower, upper = _prediction_arrays(
                pred,
                alpha,
                log=self.log,
            )

        _full_lower = None
        _full_upper = None
        if self._statsmodels_result is not None and self.fitted_values is not None:
            full_pred = self._statsmodels_result.get_prediction(
                start=0, end=self.nobs - 1
            )
            _, full_lower, full_upper = _prediction_arrays(
                full_pred,
                alpha,
                log=self.log,
            )
            _full_lower = self._mask_state_initialization(full_lower)
            _full_upper = self._mask_state_initialization(full_upper)

        return PredictResult(
            mean=mean,
            lower=lower,
            upper=upper,
            is_oos=is_oos,
            _full_data=self.data,
            _full_fitted=self._fitted_values_for_plot(),
            _full_lower=_full_lower,
            _full_upper=_full_upper,
            _start=start,
        )

    def predict(
        self,
        start=0,
        end=None,
        dynamic=False,
        alpha=0.05,
        *,
        future_exog=None,
        future_dates=None,
    ):
        """Return predictions under one or more exogenous scenarios.

        When the model was created with ``log=True``, all returned prediction
        arrays are on the original response scale. Point predictions are
        lognormal means using each prediction's own variance; interval bounds
        are the exponentiated Gaussian bounds from the log scale.

        Parameters
        ----------
        start : int or datetime-like, default 0
            First requested in-sample or forecast position.
        end : int or datetime-like, optional
            Inclusive last position; defaults to the fitted sample end.
        dynamic : bool or int or datetime-like, default False
            Statsmodels dynamic-prediction control.
        alpha : float, default 0.05
            Significance level for prediction intervals.
        future_exog : array-like, pandas object, or mapping, optional
            Complete future input path or named scenario paths.
        future_dates : datetime-like sequence, optional
            Exact forecast dates, required for array future inputs on dated
            models when they cannot be inferred.

        Returns
        -------
        PredictResult or ScenarioForecastResult
            One prediction path, or one path per named input scenario.

        Examples
        --------
        >>> from Ts.TsModels import SARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> fitted = SARIMAX(
        ...     simulate_sarima(n=60, order=(1, 0, 0), seed=42).data,
        ...     order=(1, 0, 0),
        ... ).fit()
        >>> forecast = fitted.predict(start=60, end=64)
        >>> forecast.mean.shape
        (5,)
        """
        if self._statsmodels_result is None:
            raise RuntimeError("No fitted statsmodels result available")

        start, end = self._resolve_prediction_bounds(
            start,
            end,
            future_dates,
        )
        window = _resolve_prediction_window(self.nobs, start, end)
        alpha = _validate_prediction_alpha(alpha)
        if not window.has_forecast:
            if future_exog is not None or future_dates is not None:
                raise ValueError(
                    "future_exog and future_dates require an out-of-sample range"
                )
            return self._predict_one(
                window,
                dynamic=dynamic,
                alpha=alpha,
                future_design=None,
            )

        resolved_dates = self._resolve_future_dates(
            window.forecast_steps,
            future_dates,
        )
        expected_future_index = (
            resolved_dates
            if resolved_dates is not None
            else pd.RangeIndex(window.forecast_steps)
        )
        if self._ordinary_exog_names:
            ordinary_scenarios, default_name = _normalise_future_scenarios(
                future_exog,
                future_dates=future_dates,
                expected_dates=expected_future_index,
                exog_names=self._ordinary_exog_names,
                default_future_exog=self._default_future_exog,
            )
        else:
            if future_exog is not None:
                raise ValueError(
                    "future_exog is invalid because the model has no "
                    "ordinary exogenous variables"
                )
            ordinary_scenarios = {"default": None}
            default_name = "default"

        predictions = {}
        for name, ordinary_exog in ordinary_scenarios.items():
            predictions[name] = self._predict_one(
                window,
                dynamic=dynamic,
                alpha=alpha,
                future_design=self._future_design(
                    ordinary_exog,
                    expected_future_index,
                ),
                future_inputs=ordinary_exog,
            )
        if len(predictions) == 1:
            return next(iter(predictions.values()))
        scenario_dates = self._prediction_dates(window, resolved_dates)
        return ScenarioForecastResult(
            scenarios=predictions,
            default_name=default_name,
            dates=scenario_dates,
        )

    def policy_effect(
        self,
        events,
        *,
        start=0,
        end=None,
        method="simulation",
        alpha=0.05,
        n_draws=2000,
        seed=None,
    ):
        """Estimate conditional effects for selected fitted events.

        Parameters
        ----------
        events : str or sequence of str
            Fitted event names included in the contrast.
        start, end : int or datetime-like, optional
            Inclusive effect window; defaults to the fitted sample.
        method : {"delta", "simulation", "bootstrap"}, default "simulation"
            Interval construction method.
        alpha : float, default 0.05
            Significance level.
        n_draws : int, default 2000
            Simulation or bootstrap draw count.
        seed : int, optional
            Reproducibility seed for stochastic inference.

        Returns
        -------
        PolicyEffectResult
            Coefficients, factual/counterfactual paths, contrasts, intervals,
            cumulative effects, and identification note.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from Ts.TsModels import EventSpec, SARIMAX
        >>> dates = pd.date_range("2020-01-01", periods=60, freq="MS")
        >>> data = np.random.default_rng(42).normal(size=60)
        >>> event = EventSpec("policy", [dates[35]], kind="step")
        >>> fitted = SARIMAX(data, dates=dates, events=[event]).fit()
        >>> effect = fitted.policy_effect("policy", method="delta")
        >>> effect.method
        'delta'
        """
        if self.log:
            raise NotImplementedError(
                "policy_effect is not available for log=True because log-scale "
                "event coefficients require an explicit multiplicative effect "
                "definition"
            )
        from Ts.TsModels._intervention import estimate_policy_effect

        return estimate_policy_effect(
            self,
            events=events,
            start=start,
            end=end,
            method=method,
            alpha=alpha,
            n_draws=n_draws,
            seed=seed,
        )

    def cycle_period(self, *, seasonal=False):
        """Test an AR(2) component and return its damped-cycle period.

        Complex conjugate roots require phi1 squared plus four times phi2 to
        be negative. When that condition and AR stationarity both hold, the
        period is 2*pi divided by the root angle. Seasonal results are scaled
        by the SARIMAX seasonal period so they remain in observation intervals.

        Parameters
        ----------
        seasonal : bool, default False
            If False, inspect non-seasonal lags 1 and 2. If True, inspect
            seasonal lags s and 2s.

        Returns
        -------
        ARCycleResult
            Coefficients, condition diagnostics, and the period in original
            observation intervals. The period is None when the complex-root
            or stationarity condition fails.

        Raises
        ------
        TypeError
            If seasonal is not boolean.
        ValueError
            If the selected component does not contain exactly its first and
            second AR lags.
        RuntimeError
            If no fitted statsmodels result or finite AR coefficients exist.

        Examples
        --------
        >>> from Ts.TsModels import ARCycleResult, SARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(
        ...     n=300, order=(2, 0, 0), ar=[1.1, -0.5], seed=42
        ... ).data
        >>> cycle = SARIMAX(data, order=(2, 0, 0)).fit().cycle_period()
        >>> isinstance(cycle, ARCycleResult)
        True
        """
        if not isinstance(seasonal, (bool, np.bool_)):
            raise TypeError("seasonal must be a boolean")
        if self._statsmodels_result is None:
            raise RuntimeError("No fitted statsmodels result available")

        if seasonal:
            active_lags = _active_lags(self._seasonal_order[0])
            lag_scale = int(self._seasonal_order[3])
            component = "seasonal"
            polynomial = np.asarray(
                self._statsmodels_result.polynomial_seasonal_ar,
                dtype=float,
            )
        else:
            active_lags = self.ar_lags
            lag_scale = 1
            component = "nonseasonal"
            polynomial = np.asarray(
                self._statsmodels_result.polynomial_ar,
                dtype=float,
            )

        if active_lags != (1, 2):
            label = "seasonal AR lags" if seasonal else "nonseasonal AR lags"
            raise ValueError(f"cycle_period requires {label} (1, 2); got {active_lags}")

        second_position = 2 * lag_scale
        if len(polynomial) <= second_position:
            raise RuntimeError("Fitted AR polynomial does not contain two coefficients")
        phi1 = -float(polynomial[lag_scale])
        phi2 = -float(polynomial[second_position])
        if not np.all(np.isfinite([phi1, phi2])):
            raise RuntimeError("Fitted AR(2) coefficients must be finite")

        discriminant = float(phi1**2 + 4.0 * phi2)
        has_complex_roots = discriminant < 0.0
        is_stationary = self.is_stationary
        period = None
        if has_complex_roots and is_stationary:
            cosine = phi1 / (2.0 * np.sqrt(-phi2))
            angular_frequency = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
            if 0.0 < angular_frequency < np.pi:
                period = float(lag_scale * 2.0 * np.pi / angular_frequency)

        return ARCycleResult(
            component=component,
            lag_scale=lag_scale,
            phi1=phi1,
            phi2=phi2,
            discriminant=discriminant,
            has_complex_roots=has_complex_roots,
            is_stationary=is_stationary,
            period=period,
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

        Examples
        --------
        >>> from Ts.TsModels import SARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(n=60, order=(1, 0, 0), seed=42).data
        >>> result = SARIMAX(data, order=(1, 0, 0)).fit()
        >>> fig, ax = result.plot_roots()
        >>> ax.get_aspect()
        1.0
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
            np.cos(theta),
            np.sin(theta),
            color=DEFAULT_PALETTE[1],
            linewidth=1.0,
            linestyle="--",
        )

        ax.axhline(0, color=DEFAULT_PALETTE[1], linewidth=0.5, alpha=0.5)
        ax.axvline(0, color=DEFAULT_PALETTE[1], linewidth=0.5, alpha=0.5)

        if len(ar_roots) > 0:
            inv_ar = 1.0 / ar_roots
            ax.scatter(
                inv_ar.real,
                inv_ar.imag,
                color=DEFAULT_PALETTE[0],
                marker="o",
                s=50,
                edgecolors=DEFAULT_PALETTE[7],
                linewidth=0.5,
                zorder=5,
                label="AR roots",
            )

        if len(ma_roots) > 0:
            inv_ma = 1.0 / ma_roots
            ax.scatter(
                inv_ma.real,
                inv_ma.imag,
                color=DEFAULT_PALETTE[4],
                marker="^",
                s=50,
                edgecolors=DEFAULT_PALETTE[7],
                linewidth=0.5,
                zorder=5,
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
            order_str = f"SARIMAX{_display_order(self._order)}"
            if self._seasonal_order and self._seasonal_order != (0, 0, 0, 0):
                order_str += str(self._seasonal_order)
            title = f"{order_str}: Inverse AR and MA Roots"
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")

        fig.tight_layout(pad=1.5)
        return fig, ax

    @property
    def level_intercept(self):
        """Return the constant on the undifferenced fitted-response level.

        Statsmodels' SARIMAX ``trend="c"`` parameter is the state-equation
        intercept.  For an undifferenced stationary model, the corresponding
        regression-level constant is ``intercept / A(1)``, where ``A`` is the
        complete reduced AR polynomial.  With ``log=True``, this value remains
        on the natural-log response scale; it is not an original-scale mean
        and must not be exponentiated as though it were a point forecast.
        """
        if self._statsmodels_result is None:
            raise RuntimeError("No fitted statsmodels result available")

        _p, d, _q = self._order
        _P, D, _Q, _s = self._seasonal_order
        if (d or 0) + (D or 0) > 0:
            return None
        if self._trend in ("t", "ct", "ctt"):
            return None
        if not self.is_stationary:
            return None
        if self._trend == "n":
            return 0.0

        reduced_ar = np.asarray(
            self._statsmodels_result.polynomial_reduced_ar,
            dtype=float,
        )
        denominator = float(np.sum(reduced_ar))
        if abs(denominator) < 1e-10:
            return None
        return self.params.get("intercept", 0.0) / denominator

    def _stationary_response_variance(self):
        """Return the exact stationary variance from fitted state matrices."""
        if self.level_intercept is None:
            return None

        state_space = self._statsmodels_result.model.ssm

        def time_invariant_matrix(values):
            matrix = np.asarray(values, dtype=float)
            if matrix.ndim == 2:
                return matrix
            if matrix.ndim == 3 and matrix.shape[-1] == 1:
                return matrix[..., 0]
            return None

        transition = time_invariant_matrix(state_space.transition)
        selection = time_invariant_matrix(state_space.selection)
        state_cov = time_invariant_matrix(state_space.state_cov)
        design = time_invariant_matrix(state_space.design)
        obs_cov = time_invariant_matrix(state_space.obs_cov)
        if any(
            matrix is None
            for matrix in (transition, selection, state_cov, design, obs_cov)
        ):
            return None

        innovation_cov = selection @ state_cov @ selection.T
        try:
            stationary_state_cov = solve_discrete_lyapunov(
                transition,
                innovation_cov,
            )
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "Unable to solve the stationary SARIMAX state covariance"
            ) from error

        response_cov = design @ stationary_state_cov @ design.T + obs_cov
        variance = float(response_cov[0, 0])
        if not np.isfinite(variance):
            raise RuntimeError("Stationary SARIMAX response variance is not finite")
        tolerance = 1e-10 * max(1.0, abs(variance))
        if variance < -tolerance:
            raise RuntimeError("Stationary SARIMAX response variance is negative")
        return max(variance, 0.0)

    @property
    def unconditional_log_variance(self):
        """Return the stationary variance of the natural-log response.

        The variance is obtained exactly from the fitted time-invariant state
        matrices by solving the discrete Lyapunov equation. It is available
        only for stationary, undifferenced ``log=True`` models without a time
        trend. Deterministic exogenous, event, and distributed-lag inputs are
        held at the zero-input baseline.
        """
        if not self.log:
            return None
        return self._stationary_response_variance()

    def long_run_equilibrium(self):
        """Return the unconditional mean (long-run equilibrium) of the
        estimated SARIMAX process.

        For a stationary ARMA/SARMA process with reduced AR polynomial
        :math:`A(B)` and constant :math:`c`, the fitted-scale mean is:

        .. math::

            \\mu = \\frac{c}{A(1)}

        This reduced-polynomial form handles sparse and multiplicative
        seasonal AR terms without assuming contiguous lag coefficients.

        With ``log=True``, the original-scale Gaussian mean is bias-adjusted
        with the exact stationary log-response variance. Deterministic
        exogenous, event, and distributed-lag inputs are held at zero.

        Returns ``None`` when the concept is not applicable:

        - ``d > 0`` or ``D > 0`` — differencing removes the mean.
        - ``trend`` includes a time component (``"t"``, ``"ct"``, ``"ctt"``)
          — trend-stationary series have no constant equilibrium.
        - AR polynomial is non-stationary (inverse roots on or outside the
          unit circle) or :math:`A(1) \\approx 0`.

        Returns
        -------
        float or None

        Examples
        --------
        >>> from Ts.TsModels import SARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(
        ...     n=200, order=(1, 0, 0), ar=[0.5], const=1.0, seed=42
        ... ).data
        >>> equilibrium = SARIMAX(data, order=(1, 0, 0), trend="c").fit().long_run_equilibrium()
        >>> equilibrium is None
        False
        """
        level_intercept = self.level_intercept
        if level_intercept is None or not self.log:
            return level_intercept

        log_variance = self.unconditional_log_variance
        if log_variance is None:
            return None
        with np.errstate(over="ignore"):
            return float(np.exp(level_intercept + 0.5 * log_variance))

    def level_intercept_inference(self, alpha=0.05):
        """Return delta-method inference for the fitted-response constant.

        With ``log=True``, the estimate, standard error, statistic, and
        confidence limits are all reported on the natural-log response scale.

        Parameters
        ----------
        alpha : float, default 0.05
            Two-sided significance level for the confidence interval.

        Returns
        -------
        dict or None
            Estimate, standard error, z statistic, p value, and confidence
            limits, or ``None`` when :attr:`level_intercept` is undefined.

        Examples
        --------
        >>> from Ts.TsModels import SARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(
        ...     n=120, order=(1, 0, 0), ar=[0.5], const=2.0, seed=42
        ... ).data
        >>> result = SARIMAX(data, order=(1, 0, 0), trend="c").fit()
        >>> inference = result.level_intercept_inference()
        >>> inference["standard_error"] > 0.0
        True
        """
        alpha = _validate_prediction_alpha(alpha)
        estimate = self.level_intercept
        if estimate is None or self._trend == "n":
            return None

        fitted = self._statsmodels_result
        names = tuple(fitted.param_names)
        parameter_values = np.asarray(fitted.params, dtype=float)
        gradient = np.zeros(len(names), dtype=float)
        intercept_position = names.index("intercept")
        raw_intercept = float(parameter_values[intercept_position])
        nonseasonal_ar = sum(
            float(parameter_values[position])
            for position, name in enumerate(names)
            if name.startswith("ar.L")
        )
        seasonal_ar = sum(
            float(parameter_values[position])
            for position, name in enumerate(names)
            if name.startswith("ar.S.L")
        )
        nonseasonal_at_one = 1.0 - nonseasonal_ar
        seasonal_at_one = 1.0 - seasonal_ar
        reduced_at_one = nonseasonal_at_one * seasonal_at_one
        gradient[intercept_position] = 1.0 / reduced_at_one
        for position, name in enumerate(names):
            if name.startswith("ar.L"):
                gradient[position] = raw_intercept * seasonal_at_one / reduced_at_one**2
            elif name.startswith("ar.S.L"):
                gradient[position] = (
                    raw_intercept * nonseasonal_at_one / reduced_at_one**2
                )
        covariance = np.asarray(fitted.cov_params(), dtype=float)
        variance = float(gradient @ covariance @ gradient)
        standard_error = np.sqrt(max(variance, 0.0))

        statistic = (
            np.nan if standard_error == 0.0 else float(estimate / standard_error)
        )
        p_value = (
            np.nan
            if not np.isfinite(statistic)
            else float(2.0 * norm.sf(abs(statistic)))
        )
        critical = float(norm.ppf(1.0 - alpha / 2.0))
        return {
            "estimate": float(estimate),
            "standard_error": float(standard_error),
            "statistic": statistic,
            "p_value": p_value,
            "lower": float(estimate - critical * standard_error),
            "upper": float(estimate + critical * standard_error),
        }


class SARIMAX(BaseModel):
    """SARIMAX model estimation via statsmodels SARIMAX.

    Parameters
    ----------
    data : array-like
        Time series data (1-D).
    order : tuple
        ``(p, d, q)`` non-seasonal order. ``p`` and ``q`` may be
        non-negative integers or iterables containing the exact positive
        lags to estimate. For example, ``([1, 3], 0, 0)`` estimates AR
        lags 1 and 3 while fixing lag 2 at zero. Default ``(0, 0, 0)``.
    seasonal_order : tuple
        ``(P, D, Q, s)`` seasonal order. ``P`` and ``Q`` accept the same
        integer-or-active-lags form. Default ``(0, 0, 0, 0)``.
    trend : str
        Trend specification: ``"n"`` (none), ``"c"`` (constant),
        ``"t"`` (linear), ``"ct"`` (both). Default ``"c"``.
    enforce_stationarity : bool
        Whether to enforce stationarity of the AR polynomial. Default ``True``.
    enforce_invertibility : bool
        Whether to enforce invertibility of the MA polynomial. Default ``True``.
    dates : datetime-like sequence, optional
        Strict sample dates. A Series DatetimeIndex is inferred automatically.
        Array inputs may provide dates explicitly.
    exog : pandas.Series, pandas.DataFrame, or array-like, optional
        Exogenous inputs. A named Series represents one input and preserves
        its name and optional DatetimeIndex. One-dimensional arrays represent
        one input and require exactly one ``exog_names`` value. A dated Series
        or DataFrame may also include future rows for default forecasting.
    exog_names : sequence[str], optional
        Required for array exog and for an unnamed Series. Named Series and
        DataFrame labels are authoritative and must not be overridden.
    events : sequence of EventSpec, optional
        Date-mapped pulse, step, or event-study intervention designs.
    missing : {"raise", "drop"}
        Non-finite input policy. ``"drop"`` records removed zero-based rows
        in :attr:`dropped_positions`. Default ``"drop"``; use ``"raise"``
        to reject any sample change.
    log : bool
        Fit the model to the natural logarithm of the response. The input must
        be on its original positive scale. Fitted values, predictions, and
        intervals are returned on the original scale; prediction means use
        the horizon-specific lognormal bias correction. Default ``False``.
    distributed_lags : mapping[str, RationalLagSpec], optional
        Rational transfer-function specifications keyed by exogenous input
        name. Selected columns are excluded from the ordinary static design.
    enforce_distributed_lag_stability : bool
        Whether denominator parameters are transformed toward stable values
        and the complete fitted denominator polynomial must be stable.

    Examples
    --------
    Fit a non-seasonal AR model and inspect the returned result.

    >>> from Ts.TsModels import SARIMAX
    >>> from Ts.TsSims import simulate_sarima
    >>> data = simulate_sarima(n=80, order=(1, 0, 0), ar=[0.6], seed=42).data
    >>> result = SARIMAX(data, order=(1, 0, 0), trend="c").fit()
    >>> result.nobs
    80

    A named Series is a one-column exogenous input and preserves its name.

    >>> import pandas as pd
    >>> exog = pd.Series(range(80), dtype=float, name="policy")
    >>> result = SARIMAX(data, exog=exog).fit()
    >>> result.exog_names
    ('policy',)
    """

    def __init__(
        self,
        data,
        order=(0, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
        *,
        dates=None,
        exog=None,
        exog_names=None,
        events=None,
        missing="drop",
        log=False,
        distributed_lags=None,
        enforce_distributed_lag_stability=True,
    ):
        inputs = _normalise_sarimax_inputs(
            data,
            dates=dates,
            exog=exog,
            exog_names=exog_names,
            missing=missing,
        )

        order = _normalise_order(order)
        seasonal_order = _normalise_seasonal_order(seasonal_order)
        log = _normalise_log(log)
        if len(inputs.endog) < 10:
            raise ValueError(f"Need at least 10 observations, got {len(inputs.endog)}")
        if log and np.any(inputs.endog <= 0.0):
            raise ValueError("log transformation requires strictly positive data")
        event_specs = _validate_events(events, inputs.dates)
        distributed_lags = _normalise_distributed_lags(distributed_lags, inputs)
        if not isinstance(
            enforce_distributed_lag_stability,
            (bool, np.bool_),
        ):
            raise TypeError("enforce_distributed_lag_stability must be boolean")
        design, event_frame, event_metadata = _combined_design(
            inputs,
            event_specs,
            trend,
            distributed_lag_names=tuple(distributed_lags),
        )

        self.data = inputs.endog
        self._model_data = np.log(inputs.endog) if log else inputs.endog.copy()
        self.log = log
        self.dates = inputs.dates
        self.exog = inputs.exog
        self.exog_names = inputs.exog_names
        self.distributed_lags = distributed_lags
        self.distributed_lag_names = tuple(distributed_lags)
        self.ordinary_exog_names = tuple(
            name for name in inputs.exog_names if name not in distributed_lags
        )
        if inputs.exog is None:
            self.distributed_inputs = None
        else:
            positions = [
                inputs.exog_names.index(name) for name in self.distributed_lag_names
            ]
            self.distributed_inputs = (
                None
                if not positions
                else np.array(inputs.exog[:, positions], dtype=float, copy=True)
            )
        self.future_exog = inputs.future_exog
        self.events = event_specs
        self._event_frame = event_frame
        self._event_metadata = event_metadata
        self._design_frame = design
        self.design_matrix = (
            None if design is None else design.to_numpy(dtype=float, copy=True)
        )
        self.design_columns = () if design is None else tuple(design.columns)
        self.missing = missing
        self.dropped_positions = inputs.dropped_positions
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.enforce_distributed_lag_stability = bool(enforce_distributed_lag_stability)

    def _clone_for_evaluation(self, data, exog=None, *, dates=None):
        """Rebuild all derived design state for an evaluation window."""
        return type(self)(
            data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            dates=dates,
            exog=exog,
            exog_names=self.exog_names if exog is not None else None,
            events=self.events,
            missing="raise",
            log=self.log,
            distributed_lags=self.distributed_lags,
            enforce_distributed_lag_stability=(self.enforce_distributed_lag_stability),
        )

    def _evaluation_predict_kwargs(self, start, stop):
        """Return future exogenous values and dates for one evaluation."""
        kwargs = {}
        if self.exog is not None:
            expected = stop - start
            future_exog = self.exog[start:stop]
            if len(future_exog) != expected:
                missing_start = max(start, len(self.exog))
                if self.dates is None:
                    missing = f"positions {missing_start} through {stop - 1}"
                else:
                    missing = ", ".join(
                        timestamp.isoformat()
                        for timestamp in self.dates[missing_start:stop]
                    )
                raise ValueError(f"future exog is missing dates: {missing}")
            kwargs["future_exog"] = np.array(
                future_exog,
                dtype=float,
                copy=True,
            )
        if self.dates is not None:
            future_dates = self.dates[start:stop]
            if len(future_dates) != stop - start:
                raise ValueError("future dates do not cover the evaluation window")
            kwargs["future_dates"] = future_dates.copy()
        return kwargs

    def fit(
        self,
        *,
        start_params=None,
        method="bfgs",
        maxiter=500,
        cov_type="oim",
        require_convergence=True,
    ):
        """Estimate the SARIMAX model via maximum likelihood.

        Parameters
        ----------
        start_params : array-like, optional
            Finite initial parameter vector in statsmodels parameter order.
            It must contain exactly one value per fitted parameter.
        method : str, default ``"bfgs"``
            Optimizer passed to statsmodels. Supported values are ``"newton"``,
            ``"nm"``, ``"bfgs"``, ``"lbfgs"``, ``"powell"``, ``"cg"``,
            ``"ncg"``, and ``"basinhopping"``.
        maxiter : int, default 500
            Strictly positive maximum number of optimizer iterations.
        cov_type : str, default ``"oim"``
            Parameter covariance estimator passed to statsmodels. Use
            ``"oim"`` for observed-information standard errors such as the
            preliminary LTF table in the textbook.
        require_convergence : bool, default True
            Raise :class:`RuntimeError` instead of returning a result when the
            optimizer does not report convergence. Pass ``False`` explicitly
            to inspect a non-converged result.

        Returns
        -------
        SARIMAXResult

        Examples
        --------
        >>> from Ts.TsModels import SARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(n=80, order=(1, 0, 0), ar=[0.5], seed=42).data
        >>> result = SARIMAX(data, order=(1, 0, 0)).fit(maxiter=100)
        >>> result.nobs
        80
        """
        method = _normalise_fit_method(method)
        maxiter = _normalise_maxiter(maxiter)
        cov_type = _normalise_cov_type(cov_type)
        start_params = _normalise_start_params(start_params)
        require_convergence = _normalise_require_convergence(require_convergence)

        p, d, q = self.order
        P, D, Q, s = self.seasonal_order

        model_dates = self.dates
        if model_dates is not None and model_dates.freq is None:
            inferred_frequency = pd.infer_freq(model_dates)
            model_dates = (
                None
                if inferred_frequency is None
                else pd.DatetimeIndex(model_dates, freq=inferred_frequency)
            )
        model_exog = self._design_frame
        if model_exog is not None:
            if model_dates is None:
                model_exog = model_exog.reset_index(drop=True)
            else:
                model_exog = model_exog.copy()
                model_exog.index = model_dates

        model_kwargs = {
            "exog": model_exog,
            "dates": model_dates,
            "order": (p, d, q),
            "seasonal_order": (P, D, Q, s),
            "trend": self.trend,
            "enforce_stationarity": self.enforce_stationarity,
            "enforce_invertibility": self.enforce_invertibility,
        }
        if self.distributed_lags:
            rdl_likelihood_burn = _automatic_rdl_likelihood_burn(
                self.distributed_lags,
                self.order,
                self.seasonal_order,
            )
            if rdl_likelihood_burn >= len(self._model_data):
                raise ValueError(
                    "automatic RDL initialization leaves no observations for "
                    "likelihood estimation"
                )
            model = _RationalLagSARIMAX(
                self._model_data,
                distributed_inputs=self.distributed_inputs,
                distributed_lags=self.distributed_lags,
                enforce_distributed_lag_stability=(
                    self.enforce_distributed_lag_stability
                ),
                rdl_loglikelihood_burn=rdl_likelihood_burn,
                **model_kwargs,
            )
        else:
            model = StatsmodelsSARIMAX(self._model_data, **model_kwargs)
        if start_params is not None and len(start_params) != model.k_params:
            raise ValueError(
                "start_params must contain exactly "
                f"{model.k_params} parameters, got {len(start_params)}"
            )
        fitted = model.fit(
            start_params=start_params,
            method=method,
            maxiter=maxiter,
            cov_type=cov_type,
            disp=False,
        )
        converged = bool(fitted.mle_retvals.get("converged", False))
        if require_convergence and not converged:
            raise RuntimeError(
                "SARIMAX optimization failed to converge with "
                f"method={method!r} within maxiter={maxiter}; "
                f"optimizer details: {fitted.mle_retvals}"
            )

        distributed_lag_results = _make_rational_lag_results(
            fitted,
            self.distributed_lags,
        )
        if self.enforce_distributed_lag_stability:
            unstable = [
                name
                for name, result in distributed_lag_results.items()
                if not result.is_stable
            ]
            if unstable:
                raise RuntimeError(
                    "rational distributed-lag denominator is unstable for "
                    + ", ".join(unstable)
                )

        params = {}
        std_errors = {}
        p_values = {}
        for name, param, bse_val, pval in zip(
            fitted.param_names, fitted.params, fitted.bse, fitted.pvalues, strict=False
        ):
            params[name] = float(param)
            std_errors[name] = float(bse_val)
            p_values[name] = float(pval)

        parameter_names = tuple(fitted.param_names)
        parameter_covariance = np.asarray(fitted.cov_params(), dtype=float)

        burn = int(fitted.loglikelihood_burn)
        resid = np.asarray(fitted.resid)[burn:].copy()
        if self.log:
            fitted_prediction = fitted.get_prediction(start=0, end=len(self.data) - 1)
            fitted_vals, _, _ = _prediction_arrays(
                fitted_prediction,
                0.05,
                log=True,
            )
            fitted_vals[:burn] = np.nan
        else:
            fitted_vals = np.asarray(fitted.fittedvalues)

        result = SARIMAXResult(
            model_type="SARIMAX",
            params=params,
            std_errors=std_errors,
            p_values=p_values,
            aic=float(fitted.aic),
            bic=float(fitted.bic),
            log_likelihood=float(fitted.llf),
            residuals=resid,
            fitted_values=fitted_vals,
            nobs=int(fitted.nobs),
            data=self.data.copy(),
            _parameter_covariance=parameter_covariance,
            _parameter_names=parameter_names,
            _order=self.order,
            _seasonal_order=self.seasonal_order,
            _statsmodels_result=fitted,
            _trend=self.trend,
            _log_transform=self.log,
            _dates=None if self.dates is None else self.dates.copy(),
            _ordinary_exog=(None if self.exog is None else self.exog.copy()),
            _ordinary_exog_names=self.exog_names,
            _static_exog_names=self.ordinary_exog_names,
            _event_specs=self.events,
            _event_metadata=dict(self._event_metadata),
            _design_columns=self.design_columns,
            _design_matrix=(
                None if self.design_matrix is None else self.design_matrix.copy()
            ),
            _default_future_exog=(
                None if self.future_exog is None else self.future_exog.copy()
            ),
            _model_kwargs={
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "trend": self.trend,
                "enforce_stationarity": self.enforce_stationarity,
                "enforce_invertibility": self.enforce_invertibility,
            },
            _distributed_lag_results=distributed_lag_results,
            _enforce_distributed_lag_stability=(self.enforce_distributed_lag_stability),
        )

        self.result_ = result
        return result
