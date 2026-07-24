"""Shared validation and aggregation helpers for evaluation methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._metrics import compute_metrics


@dataclass(frozen=True)
class EvaluationPeriods:
    """Resolved inclusive estimation and validation periods."""

    estimation_start: int
    estimation_stop: int
    validation_start: int
    validation_stop: int
    dates: pd.DatetimeIndex | None

    @property
    def estimation_indices(self):
        """Return zero-based positions used for parameter estimation."""
        return np.arange(
            self.estimation_start,
            self.estimation_stop,
            dtype=int,
        )

    @property
    def validation_indices(self):
        """Return zero-based positions scored during validation."""
        return np.arange(
            self.validation_start,
            self.validation_stop,
            dtype=int,
        )

    @property
    def estimation_dates(self):
        """Return estimation dates when the model has a date index."""
        if self.dates is None:
            return None
        return self.dates[self.estimation_start:self.estimation_stop].copy()

    @property
    def validation_dates(self):
        """Return validation dates when the model has a date index."""
        if self.dates is None:
            return None
        return self.dates[self.validation_start:self.validation_stop].copy()


def _validated_model_dates(model, data):
    """Return a strict model calendar or None for position-based data."""
    values = getattr(model, "dates", None)
    if values is None:
        return None
    try:
        dates = pd.DatetimeIndex(values)
    except (TypeError, ValueError) as error:
        raise TypeError("model.dates must be datetime-like") from error
    if len(dates) != len(data):
        raise ValueError(
            "model.dates must contain one date per observation"
        )
    if dates.hasnans:
        raise ValueError("model.dates must not contain missing dates")
    if not dates.is_unique:
        raise ValueError("model.dates must be unique")
    if not dates.is_monotonic_increasing:
        raise ValueError("model.dates must be strictly increasing")
    return dates.copy()


def _period_pair(name, period):
    """Require one public period to contain exactly two inclusive bounds."""
    if not isinstance(period, (tuple, list)) or len(period) != 2:
        raise TypeError(f"{name} must be a (start, end) pair")
    return period[0], period[1]


def _position_bound(name, bound, nobs):
    """Resolve one zero-based inclusive position bound."""
    if isinstance(bound, (bool, np.bool_)) or not isinstance(
        bound,
        (int, np.integer),
    ):
        raise TypeError(
            f"{name} must be an integer position because the model has no dates"
        )
    position = int(bound)
    if position < 0 or position >= nobs:
        raise ValueError(
            f"{name}={position} is outside the observed positions 0..{nobs - 1}"
        )
    return position


def _date_bound(name, bound, dates):
    """Resolve one exact date bound without nearest-date coercion."""
    try:
        timestamp = pd.Timestamp(bound)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be datetime-like") from error
    if str(timestamp.tz) != str(dates.tz):
        raise ValueError(f"{name} timezone must match model.dates")
    position = int(dates.get_indexer([timestamp])[0])
    if position < 0:
        raise ValueError(
            f"{name}={timestamp.isoformat()} does not exist in model.dates"
        )
    return position


def _resolve_period(name, period, data, dates):
    """Resolve inclusive public bounds to a half-open positional slice."""
    start_bound, end_bound = _period_pair(name, period)
    if dates is None:
        start = _position_bound(f"{name} start", start_bound, len(data))
        end = _position_bound(f"{name} end", end_bound, len(data))
    else:
        start = _date_bound(f"{name} start", start_bound, dates)
        end = _date_bound(f"{name} end", end_bound, dates)
    if start > end:
        raise ValueError(f"{name} start must not be later than its end")
    return start, end + 1


def resolve_evaluation_periods(
    model,
    data,
    estimation_period,
    validation_period,
):
    """Validate and resolve explicit estimation and validation periods."""
    dates = _validated_model_dates(model, data)
    estimation_start, estimation_stop = _resolve_period(
        "estimation_period",
        estimation_period,
        data,
        dates,
    )
    validation_start, validation_stop = _resolve_period(
        "validation_period",
        validation_period,
        data,
        dates,
    )
    if estimation_stop - estimation_start < 10:
        raise ValueError("estimation_period must contain at least 10 observations")
    if validation_start < estimation_stop:
        raise ValueError(
            "validation_period must start strictly later than "
            "estimation_period ends"
        )
    return EvaluationPeriods(
        estimation_start=estimation_start,
        estimation_stop=estimation_stop,
        validation_start=validation_start,
        validation_stop=validation_stop,
        dates=dates,
    )


def validate_positive_int(name, value, minimum=1):
    """Return an integer argument after rejecting booleans and small values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer >= {minimum}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def validate_alpha(alpha):
    """Return a finite significance level strictly between zero and one."""
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("alpha must be between 0 and 1") from error
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    return alpha


def model_data(model):
    """Return validated one- or two-dimensional model data."""
    if not hasattr(model, "data"):
        raise TypeError("model must expose a data array")
    data = np.asarray(model.data, dtype=float)
    if data.ndim not in (1, 2):
        raise ValueError(
            f"model.data must be one- or two-dimensional, got {data.shape}"
        )
    return data


def expected_forecast_shape(data, horizon):
    """Return one forecast origin's required output shape."""
    if data.ndim == 1:
        return (horizon,)
    return (horizon, data.shape[1])


def normalise_forecast(values, expected_shape, name):
    """Copy a forecast array and require the documented horizon shape."""
    array = np.asarray(values, dtype=float)
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} forecast has shape {array.shape}, expected "
            f"{expected_shape}"
        )
    return array.copy()


def prediction_arrays(prediction, expected_shape):
    """Extract and validate point and interval forecasts."""
    mean = normalise_forecast(prediction.mean, expected_shape, "mean")
    lower = prediction.lower
    upper = prediction.upper
    if lower is None and upper is None:
        return mean, None, None
    if lower is None or upper is None:
        raise ValueError(
            "prediction lower and upper bounds must both be set"
        )
    return (
        mean,
        normalise_forecast(lower, expected_shape, "lower"),
        normalise_forecast(upper, expected_shape, "upper"),
    )


def fit_and_forecast(
    model,
    train_data,
    exog,
    dates,
    predict_kwargs,
    horizon,
    alpha,
    expected_shape,
):
    """Fit an isolated model window and return validated forecasts."""
    cloned = model._clone_for_evaluation(
        train_data,
        exog=exog,
        dates=dates,
    )
    fitted = cloned.fit()
    predict = getattr(fitted, "predict", None)
    if not callable(predict):
        raise NotImplementedError(
            f"{type(model).__name__} does not provide predict()"
        )
    prediction = predict(
        start=fitted.nobs,
        end=fitted.nobs + horizon - 1,
        alpha=alpha,
        **predict_kwargs,
    )
    return fitted, prediction_arrays(prediction, expected_shape)


def evaluation_actual(model, observed, train_data, expected_shape):
    """Return and validate one evaluation target window."""
    actual = np.asarray(
        model._evaluation_actual(observed, train_data),
        dtype=float,
    )
    if actual.shape != expected_shape:
        raise ValueError(
            "evaluation target has shape "
            f"{actual.shape}, expected {expected_shape}"
        )
    return actual


def metrics_by_horizon(actual, predicted):
    """Compute one metric dictionary per forecast horizon."""
    if actual.shape != predicted.shape or predicted.ndim not in (2, 3):
        raise ValueError(
            "backtest actual and predicted must share a 2D or 3D shape"
        )
    return [
        compute_metrics(actual[:, index], predicted[:, index])
        for index in range(predicted.shape[1])
    ]


def oos_metrics_by_series(actual, predicted):
    """Compute one metric dictionary per endogenous series."""
    if actual.shape != predicted.shape:
        raise ValueError("OOS actual and predicted must have the same shape")
    if predicted.ndim == 1:
        return [compute_metrics(actual, predicted)]
    if predicted.ndim == 2:
        return [
            compute_metrics(actual[:, index], predicted[:, index])
            for index in range(predicted.shape[1])
        ]
    raise ValueError(
        f"unsupported evaluation shape for series metrics: {predicted.shape}"
    )


def backtest_metrics_by_series(actual, predicted):
    '''Compute rolling-origin metrics for each endogenous series.'''
    if actual.shape != predicted.shape:
        raise ValueError(
            'backtest actual and predicted must have the same shape'
        )
    if predicted.ndim == 2:
        return [compute_metrics(actual, predicted)]
    if predicted.ndim == 3:
        return [
            compute_metrics(actual[:, :, index], predicted[:, :, index])
            for index in range(predicted.shape[2])
        ]
    raise ValueError(
        f'unsupported backtest shape for series metrics: {predicted.shape}'
    )


def training_exog(model, start, stop):
    """Return a copied exogenous training window when the model has one."""
    exog = getattr(model, "exog", None)
    if exog is None:
        return None
    return np.array(np.asarray(exog)[start:stop], dtype=float, copy=True)


def training_dates(model, start, stop):
    """Return a copied date-index training window when available."""
    dates = getattr(model, "dates", None)
    if dates is None:
        return None
    return dates[start:stop].copy()


def validate_model_protocol(model, context):
    """Require the evaluation hooks supplied by TsModels.BaseModel."""
    required = (
        "_clone_for_evaluation",
        "_evaluation_actual",
        "_evaluation_predict_kwargs",
        "_validate_evaluation",
    )
    missing = [name for name in required if not callable(getattr(model, name, None))]
    if missing:
        raise TypeError(
            "model does not implement the evaluation protocol: "
            + ", ".join(missing)
        )
    model._validate_evaluation(context)
