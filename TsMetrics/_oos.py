"""Leakage-free validation over explicit estimation and validation periods."""

from __future__ import annotations

import numpy as np

from Ts.TsUtils._validation import validate_alpha

from ._evaluation import (
    evaluation_actual,
    expected_forecast_shape,
    fit_and_forecast,
    model_data,
    model_series_names,
    training_dates,
    training_exog,
    validate_fit_method,
    validate_model_protocol,
)
from ._periods import resolve_evaluation_periods
from ._results import OOSResult


def _validation_slice(values, offset):
    """Return the scored suffix while preserving an absent interval."""
    if values is None:
        return None
    return np.array(values[offset:], dtype=float, copy=True)


def oos(
    model,
    estimation_period,
    validation_period,
    *,
    alpha=0.05,
    method=None,
):
    """Evaluate a model without exposing validation targets to estimation.

    Both public periods use inclusive bounds. Date-aware models require exact
    date labels; position-based models require zero-based integer positions.
    A gap between the periods is allowed and forecast through, but only the
    validation period is scored.

    Parameters
    ----------
    model : estimator
        Unfitted Ts model implementing the evaluation protocol. The evaluator
        clones and refits it on the estimation period.
    estimation_period : tuple of int or datetime-like
        Inclusive training bounds. Use positions for undated models and exact
        dates for date-aware models.
    validation_period : tuple of int or datetime-like
        Inclusive scored bounds strictly after the estimation period.
    alpha : float, default 0.05
        Significance level for forecast intervals.
    method : str, optional
        Optimizer forwarded to the cloned model's ``fit()`` method. ``None``
        preserves that model's default fitting behavior.

    Returns
    -------
    OOSResult
        Forecasts, actuals, period metadata, intervals, and metrics.

    Examples
    --------
    >>> from Ts.TsMetrics import oos
    >>> from Ts.TsModels import SARIMAX
    >>> from Ts.TsSims import simulate_sarima
    >>> data = simulate_sarima(n=60, order=(1, 0, 0), seed=42).data
    >>> result = oos(SARIMAX(data, order=(1, 0, 0)), (0, 39), (40, 49))
    >>> result.mean.shape
    (10,)
    >>> result.validation_indices[[0, -1]].tolist()
    [40, 49]
    """
    data = model_data(model)
    series_names = model_series_names(model, data)
    target = validate_model_protocol(model, "oos")
    fit_kwargs = validate_fit_method(model, method)
    periods = resolve_evaluation_periods(
        model,
        data,
        estimation_period,
        validation_period,
    )
    alpha = validate_alpha(alpha)

    train_data = data[periods.estimation_start : periods.estimation_stop]
    bridge_horizon = periods.validation_stop - periods.estimation_stop
    bridge_shape = expected_forecast_shape(data, bridge_horizon)
    model_type, (bridge_mean, bridge_lower, bridge_upper) = fit_and_forecast(
        model,
        train_data,
        training_exog(
            model,
            periods.estimation_start,
            periods.estimation_stop,
        ),
        training_dates(
            periods.dates,
            periods.estimation_start,
            periods.estimation_stop,
        ),
        model._evaluation_predict_kwargs(
            periods.estimation_stop,
            periods.validation_stop,
        ),
        bridge_horizon,
        alpha,
        bridge_shape,
        fit_kwargs,
    )

    validation_offset = periods.validation_start - periods.estimation_stop
    mean = _validation_slice(bridge_mean, validation_offset)
    lower = _validation_slice(bridge_lower, validation_offset)
    upper = _validation_slice(bridge_upper, validation_offset)
    validation_shape = expected_forecast_shape(
        data,
        periods.validation_stop - periods.validation_start,
    )
    if mean.shape != validation_shape:
        raise ValueError(
            f"validation forecast has shape {mean.shape}, expected {validation_shape}"
        )
    actual = evaluation_actual(
        model,
        data[periods.validation_start : periods.validation_stop],
        train_data,
        validation_shape,
    )

    return OOSResult(
        mean=mean,
        actual=actual,
        lower=lower,
        upper=upper,
        estimation_indices=periods.estimation_indices,
        validation_indices=periods.validation_indices,
        estimation_dates=periods.estimation_dates,
        validation_dates=periods.validation_dates,
        model_type=model_type,
        target=target,
        series_names=series_names,
        alpha=alpha,
    )
