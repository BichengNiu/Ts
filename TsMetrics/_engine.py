"""One leakage-free engine for all historical forecast evaluation schemes."""

from __future__ import annotations

from collections.abc import Mapping

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
    validate_fit_kwargs,
    validate_model_protocol,
)
from ._metrics import ERROR_METRIC_NAMES
from ._periods import validated_model_dates
from ._results import ForecastComparisonResult, ForecastEvaluationResult
from ._schemes import Holdout, RollingOrigin


def _validated_models(models):
    """Validate named estimators and return shared data/calendar metadata."""
    if not isinstance(models, Mapping):
        raise TypeError("models must be a mapping of names to estimators")
    if not models:
        raise ValueError("models must not be empty")
    if not all(isinstance(name, str) and name for name in models):
        raise TypeError("model names must be non-empty strings")

    items = list(models.items())
    metadata = {}
    for name, model in items:
        data = model_data(model)
        target = validate_model_protocol(model, "evaluate_forecasts")
        dates = validated_model_dates(model, data)
        series_names = model_series_names(model, data)
        metadata[name] = (data, target, dates, series_names)

    reference_name = items[0][0]
    reference_data, reference_target, reference_dates, reference_series = metadata[
        reference_name
    ]
    for name, _model in items[1:]:
        data, target, dates, series_names = metadata[name]
        if data.shape != reference_data.shape or not np.array_equal(
            data,
            reference_data,
            equal_nan=True,
        ):
            raise ValueError("all models must use the same target values")
        if target != reference_target:
            raise ValueError("all models must use the same evaluation target")
        if (dates is None) != (reference_dates is None):
            raise ValueError("all models must use the same date metadata")
        if dates is not None and not dates.equals(reference_dates):
            raise ValueError("all models must use the same dates")
        if series_names != reference_series:
            raise ValueError("all models must use the same series names")
    return items, metadata, reference_data, reference_dates


def _validated_request(
    items,
    scheme,
    rank_by,
    alpha,
    fit_kwargs,
    on_error,
    future_exog,
):
    """Validate every batch option before the first model fit."""
    if not isinstance(scheme, (Holdout, RollingOrigin)):
        raise TypeError("scheme must be a Holdout or RollingOrigin")
    if rank_by not in ERROR_METRIC_NAMES:
        raise ValueError(
            f"rank_by must be one of {list(ERROR_METRIC_NAMES)}, got {rank_by!r}"
        )
    alpha = validate_alpha(alpha)
    if on_error not in {"raise", "record"}:
        raise ValueError("on_error must be either 'raise' or 'record'")
    validated_fit_kwargs = {}
    for name, model in items:
        validated_fit_kwargs[name] = validate_fit_kwargs(
            model,
            fit_kwargs,
            model_name=name,
        )

    exogenous = {
        name: getattr(model, "exog", None) is not None for name, model in items
    }
    if any(exogenous.values()):
        if future_exog != "observed":
            names = ", ".join(name for name, has_exog in exogenous.items() if has_exog)
            raise ValueError(
                "models with exogenous inputs require "
                f"future_exog='observed' for conditional evaluation: {names}"
            )
    elif future_exog is not None:
        raise ValueError("future_exog is only valid for models with exogenous inputs")
    return alpha, validated_fit_kwargs, exogenous


def _evaluate_one_model(
    model,
    data,
    target,
    dates,
    series_names,
    splits,
    *,
    alpha,
    fit_kwargs,
    on_error,
    uses_observed_future_exog,
):
    """Evaluate one estimator over already-resolved shared splits."""
    horizon = len(splits[0].target_indices)
    forecast_shape = expected_forecast_shape(data, horizon)
    output_shape = (len(splits), *forecast_shape)
    mean = np.full(output_shape, np.nan)
    actual = np.full(output_shape, np.nan)
    lower = np.full(output_shape, np.nan)
    upper = np.full(output_shape, np.nan)
    failures = []
    parameter_estimates = []
    has_interval = False
    model_type = type(model).__name__

    for row, split in enumerate(splits):
        train_start = int(split.train_indices[0])
        train_stop = int(split.train_indices[-1]) + 1
        target_start = int(split.target_indices[0])
        target_stop = int(split.target_indices[-1]) + 1
        train_data = data[split.train_indices]
        bridge_horizon = target_stop - train_stop
        bridge_shape = expected_forecast_shape(data, bridge_horizon)
        try:
            model_type, parameters, forecast = fit_and_forecast(
                model,
                train_data,
                training_exog(model, train_start, train_stop),
                training_dates(dates, train_start, train_stop),
                model._evaluation_predict_kwargs(train_stop, target_stop),
                bridge_horizon,
                alpha,
                bridge_shape,
                fit_kwargs,
            )
            bridge_mean, bridge_lower, bridge_upper = forecast
            offset = target_start - train_stop
            forecast_mean = np.array(bridge_mean[offset:], dtype=float, copy=True)
            if forecast_mean.shape != forecast_shape:
                raise ValueError(
                    f"scored forecast has shape {forecast_mean.shape}, "
                    f"expected {forecast_shape}"
                )
            target_values = evaluation_actual(
                model,
                data[split.target_indices],
                train_data,
                forecast_shape,
            )
            mean[row] = forecast_mean
            actual[row] = target_values
            if bridge_lower is not None:
                lower[row] = bridge_lower[offset:]
                upper[row] = bridge_upper[offset:]
                has_interval = True
            parameter_estimates.extend(
                {"split": split.split, **parameter} for parameter in parameters
            )
        except Exception as error:
            if on_error == "raise":
                raise
            failures.append(
                {
                    "split": split.split,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )

    return ForecastEvaluationResult(
        mean=mean,
        actual=actual,
        lower=lower if has_interval else None,
        upper=upper if has_interval else None,
        splits=splits,
        failures=failures,
        model_type=model_type,
        target=target,
        dates=dates,
        series_names=series_names,
        alpha=alpha,
        uses_observed_future_exog=uses_observed_future_exog,
        parameter_estimates=parameter_estimates,
    )


def evaluate_forecasts(
    models,
    *,
    scheme,
    rank_by="rmse",
    alpha=0.05,
    fit_kwargs=None,
    on_error="raise",
    future_exog=None,
):
    """Evaluate and compare named estimators under one time-ordered scheme.

    Every estimator is cloned and re-fitted independently on every training
    split. Fixed holdout and rolling-origin evaluation use the same execution,
    result, metric, and ranking contracts.

    Parameters
    ----------
    models : mapping of str to estimator
        Named unfitted estimators sharing target values and calendar metadata.
    scheme : Holdout or RollingOrigin
        Time-ordered training and scoring design.
    rank_by : str, default "rmse"
        Canonical error metric used for ascending model ranking.
    alpha : float, default 0.05
        Significance level requested for forecast intervals.
    fit_kwargs : mapping, optional
        Keywords forwarded to every estimator ``fit()`` call.
    on_error : {"raise", "record"}, default "raise"
        Whether a failed split raises or is retained as an all-NaN row.
    future_exog : {"observed"}, optional
        Explicit conditional-evaluation policy required by exogenous models.

    Returns
    -------
    ForecastComparisonResult
        Aligned forecasts, common-sample metrics, rankings, and metadata.

    Examples
    --------
    >>> from Ts.TsMetrics import Holdout, evaluate_forecasts
    >>> report = evaluate_forecasts(
    ...     {"model": estimator},
    ...     scheme=Holdout(train=(0, 19), test=(20, 24)),
    ... )
    >>> report.table.index.tolist()
    ['model']
    """
    items, metadata, reference_data, reference_dates = _validated_models(models)
    alpha, fit_kwargs_by_model, exogenous = _validated_request(
        items,
        scheme,
        rank_by,
        alpha,
        fit_kwargs,
        on_error,
        future_exog,
    )
    splits = scheme.split(len(reference_data), reference_dates)
    results = {}
    for name, model in items:
        data, target, dates, series_names = metadata[name]
        results[name] = _evaluate_one_model(
            model,
            data,
            target,
            dates,
            series_names,
            splits,
            alpha=alpha,
            fit_kwargs=fit_kwargs_by_model[name],
            on_error=on_error,
            uses_observed_future_exog=exogenous[name],
        )
    return ForecastComparisonResult(results=results, rank_by=rank_by)


__all__ = ["evaluate_forecasts"]
