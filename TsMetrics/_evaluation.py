"""Model validation and isolated forecast execution."""

from __future__ import annotations

from inspect import Parameter, signature
from collections.abc import Mapping

import numpy as np


from ._protocols import EvaluationModelProtocol
from ._results import _normalise_names


def validate_fit_kwargs(model, fit_kwargs, *, model_name=None):
    """Copy fit keywords after validating support without fitting a model."""
    if fit_kwargs is None:
        return {}
    if not isinstance(fit_kwargs, Mapping):
        raise TypeError("fit_kwargs must be a mapping or None")
    if not all(isinstance(name, str) and name for name in fit_kwargs):
        raise TypeError("fit_kwargs names must be non-empty strings")
    fit = getattr(model, "fit", None)
    if not callable(fit):
        raise TypeError("model must provide fit()")
    try:
        parameters = signature(fit).parameters.values()
    except (TypeError, ValueError) as error:
        label = type(model).__name__ if model_name is None else repr(model_name)
        raise TypeError(
            f"model {label} does not expose an inspectable fit() supporting method"
        ) from error

    parameters = tuple(parameters)
    accepts_kwargs = any(
        parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters
    )
    accepted_names = {parameter.name for parameter in parameters}
    unsupported = [
        name for name in fit_kwargs if name not in accepted_names and not accepts_kwargs
    ]
    if unsupported:
        label = type(model).__name__ if model_name is None else repr(model_name)
        names = ", ".join(repr(name) for name in unsupported)
        raise TypeError(f"model {label} does not support fit keyword(s): {names}")
    return dict(fit_kwargs)


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


def model_series_names(model, data):
    """Return validated display names for a multivariate model, when available."""
    if data.ndim == 1:
        return None
    names = getattr(model, "data_names", None)
    if names is None:
        return None
    return _normalise_names(names, data.shape[1], "model.data_names")


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
            f"{name} forecast has shape {array.shape}, expected {expected_shape}"
        )
    return array.copy()


def prediction_arrays(prediction, expected_shape):
    """Extract and validate point and interval forecasts."""
    try:
        mean_values = prediction.mean
        lower = prediction.lower
        upper = prediction.upper
    except AttributeError as error:
        raise TypeError(
            "predict() must return mean, lower, and upper attributes"
        ) from error
    mean = normalise_forecast(mean_values, expected_shape, "mean")
    if lower is None and upper is None:
        return mean, None, None
    if lower is None or upper is None:
        raise ValueError("prediction lower and upper bounds must both be set")
    return (
        mean,
        normalise_forecast(lower, expected_shape, "lower"),
        normalise_forecast(upper, expected_shape, "upper"),
    )


def _optional_parameter_value(values, name):
    """Return one optional scalar diagnostic without tightening the protocol."""
    get = getattr(values, "get", None)
    if not callable(get):
        return np.nan
    try:
        value = get(name, np.nan)
        array = np.asarray(value)
        if array.size != 1:
            return np.nan
        return float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        return np.nan


def parameter_snapshot(fitted):
    """Extract scalar parameter diagnostics from a fitted result when exposed."""
    try:
        params = getattr(fitted, "params", None)
        items = getattr(params, "items", None)
        if not callable(items):
            return ()
        items = tuple(items())
    except Exception:
        return ()
    try:
        std_errors = getattr(fitted, "std_errors", None)
    except Exception:
        std_errors = None
    try:
        p_values = getattr(fitted, "p_values", None)
    except Exception:
        p_values = None
    records = []
    for name, value in items:
        if not isinstance(name, str) or not name:
            continue
        try:
            array = np.asarray(value)
            if array.size != 1:
                continue
            estimate = float(array.reshape(-1)[0])
        except (TypeError, ValueError):
            continue
        records.append(
            {
                "parameter": name,
                "estimate": estimate,
                "std_error": _optional_parameter_value(std_errors, name),
                "p_value": _optional_parameter_value(p_values, name),
            }
        )
    return tuple(records)


def fit_and_forecast(
    model: EvaluationModelProtocol,
    train_data,
    exog,
    dates,
    predict_kwargs,
    horizon,
    alpha,
    expected_shape,
    fit_kwargs=None,
):
    """Fit an isolated model window and return metadata and forecasts."""
    cloned = model._clone_for_evaluation(
        train_data,
        exog=exog,
        dates=dates,
    )
    fit = getattr(cloned, "fit", None)
    if not callable(fit):
        raise TypeError("_clone_for_evaluation() must return an object with fit()")
    fitted = fit(**({} if fit_kwargs is None else fit_kwargs))
    nobs = getattr(fitted, "nobs", None)
    if (
        isinstance(nobs, (bool, np.bool_))
        or not isinstance(nobs, (int, np.integer))
        or nobs < 1
    ):
        raise TypeError("fit() must return a result with a positive integer nobs")
    model_type = getattr(fitted, "model_type", None)
    if not isinstance(model_type, str) or not model_type:
        raise TypeError("fit() must return a result with a non-empty model_type")
    predict = getattr(fitted, "predict", None)
    if not callable(predict):
        raise NotImplementedError(f"{type(model).__name__} does not provide predict()")
    prediction = predict(
        start=int(nobs),
        end=int(nobs) + horizon - 1,
        alpha=alpha,
        **predict_kwargs,
    )
    return (
        model_type,
        parameter_snapshot(fitted),
        prediction_arrays(prediction, expected_shape),
    )


def evaluation_actual(model, observed, train_data, expected_shape):
    """Return and validate one evaluation target window."""
    actual = np.asarray(
        model._evaluation_actual(observed, train_data),
        dtype=float,
    )
    if actual.shape != expected_shape:
        raise ValueError(
            f"evaluation target has shape {actual.shape}, expected {expected_shape}"
        )
    return actual


def training_exog(model, start, stop):
    """Return a copied exogenous training window when the model has one."""
    exog = getattr(model, "exog", None)
    if exog is None:
        return None
    return np.array(np.asarray(exog)[start:stop], dtype=float, copy=True)


def training_dates(dates, start, stop):
    """Return a copied date-index training window when available."""
    if dates is None:
        return None
    return dates[start:stop].copy()


def validate_model_protocol(
    model: EvaluationModelProtocol,
    context: str,
) -> str:
    """Validate the evaluation contract and return its target name."""
    required = (
        "_clone_for_evaluation",
        "_evaluation_actual",
        "_evaluation_predict_kwargs",
        "_validate_evaluation",
    )
    missing = [name for name in required if not callable(getattr(model, name, None))]
    if missing:
        raise TypeError(
            "model does not implement the evaluation protocol: " + ", ".join(missing)
        )
    target = getattr(model, "_evaluation_target_name", None)
    if not isinstance(target, str) or not target:
        raise TypeError(
            "model does not implement the evaluation protocol: _evaluation_target_name"
        )
    model._validate_evaluation(context)
    return target
