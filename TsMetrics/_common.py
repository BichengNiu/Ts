"""Shared validation and aggregation helpers for evaluation methods."""

from __future__ import annotations

import numpy as np

from ._metrics import compute_metrics


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
