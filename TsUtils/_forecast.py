"""Shared forecast-shape and fitted-parameter contracts."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


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
            f"{name} forecast has shape {array.shape}, expected {expected_shape}"
        )
    return array.copy()


def forecast_arrays(values, expected_shape):
    """Extract and validate point and interval arrays from a forecast tuple."""
    try:
        mean_values, lower, upper = values
    except (TypeError, ValueError) as error:
        raise TypeError("forecast must contain mean, lower, and upper values") from error
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


def prediction_arrays(prediction, expected_shape):
    """Extract and validate point and interval forecasts."""
    try:
        values = (prediction.mean, prediction.lower, prediction.upper)
    except AttributeError as error:
        raise TypeError(
            "predict() must return mean, lower, and upper attributes"
        ) from error
    return forecast_arrays(values, expected_shape)


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


def normalise_forecast_result(result, expected_shape):
    """Validate the model-facing evaluation result contract."""
    try:
        model_type, parameters, forecast = result
    except (TypeError, ValueError) as error:
        raise TypeError(
            "evaluation forecast must contain model_type, parameters, and forecast"
        ) from error
    if not isinstance(model_type, str) or not model_type:
        raise TypeError("evaluation forecast model_type must be a non-empty string")
    try:
        parameters = tuple(parameters)
    except TypeError as error:
        raise TypeError("evaluation forecast parameters must be iterable") from error
    if not all(isinstance(parameter, Mapping) for parameter in parameters):
        raise TypeError("evaluation forecast parameters must contain mappings")
    return (
        model_type,
        tuple(dict(parameter) for parameter in parameters),
        forecast_arrays(forecast, expected_shape),
    )
