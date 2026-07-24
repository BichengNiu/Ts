"""Leakage-free single-split out-of-sample evaluation."""

from __future__ import annotations

import numpy as np

from ._common import (
    evaluation_actual,
    expected_forecast_shape,
    fit_and_forecast,
    model_data,
    oos_metrics_by_series,
    training_exog,
    validate_alpha,
    validate_model_protocol,
    validate_positive_int,
)
from ._metrics import compute_metrics
from ._results import OOSResult


def oos(model, split, *, alpha=0.05):
    """Evaluate a model on a held-out suffix without estimation leakage.

    The model is cloned and fitted using exactly ``model.data[:split]``.
    It then forecasts every observation from ``split`` to the end of the
    original data. The caller and any existing fitted result are unchanged.
    """
    data = model_data(model)
    validate_model_protocol(model, "oos")
    split = validate_positive_int("split", split, minimum=10)
    alpha = validate_alpha(alpha)
    if split >= len(data):
        raise ValueError(
            f"split must be smaller than the number of observations ({len(data)})"
        )

    train_data = data[:split]
    horizon = len(data) - split
    expected_shape = expected_forecast_shape(data, horizon)
    fitted, (mean, lower, upper) = fit_and_forecast(
        model,
        train_data,
        training_exog(model, 0, split),
        horizon,
        alpha,
        expected_shape,
    )
    actual = evaluation_actual(
        model,
        data[split:],
        train_data,
        expected_shape,
    )

    return OOSResult(
        mean=mean,
        actual=actual,
        lower=lower,
        upper=upper,
        target_indices=np.arange(split, len(data), dtype=int),
        metrics=compute_metrics(actual, mean),
        metrics_by_series=oos_metrics_by_series(actual, mean),
        model_type=fitted.model_type,
        target=model._evaluation_target_name,
        split=split,
    )
