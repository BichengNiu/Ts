"""Shared fitted-window execution for model-side forecast consumers."""

from __future__ import annotations

from typing import Any

import numpy as np

from Ts.TsUtils._validation import validate_fit_kwargs
from Ts.TsUtils._forecast import expected_forecast_shape
from Ts.TsUtils._forecast import model_data
from Ts.TsUtils._forecast import normalise_forecast
from Ts.TsUtils._forecast import normalise_forecast_result
from Ts.TsUtils._forecast import parameter_snapshot
from Ts.TsUtils._forecast import prediction_arrays


def fit_and_forecast(
    model: Any,
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
    return normalise_forecast_result(
        (
            model_type,
            parameter_snapshot(fitted),
            prediction_arrays(prediction, expected_shape),
        ),
        expected_shape,
    )


__all__ = [
    "expected_forecast_shape",
    "fit_and_forecast",
    "model_data",
    "normalise_forecast",
    "parameter_snapshot",
    "prediction_arrays",
    "validate_fit_kwargs",
]
