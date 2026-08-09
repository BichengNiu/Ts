"""Comparison of forecast evaluation results."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ._evaluation import validate_fit_method
from ._metrics import ERROR_METRIC_NAMES
from ._oos import oos
from ._results import (
    BacktestResult,
    ComparisonResult,
    OOSComparisonResult,
    OOSResult,
)

_ERROR_METRICS = frozenset(ERROR_METRIC_NAMES)


def _validate_metric(metric, name="metric"):
    """Return a supported error metric before expensive evaluation work."""
    if metric not in _ERROR_METRICS:
        raise ValueError(
            f"{name} must be one of {sorted(_ERROR_METRICS)}, got {metric!r}"
        )
    return metric


def _comparison_items(results, metric):
    """Validate the comparison request and return its ordered items."""
    if not isinstance(results, Mapping):
        raise TypeError("results must be a mapping of names to results")
    if not results:
        raise ValueError("results must not be empty")
    _validate_metric(metric)
    return list(results.items())


def _comparison_reference(items):
    """Return the first valid result as the comparison reference."""
    first_name, first = items[0]
    if not isinstance(first, (OOSResult, BacktestResult)):
        raise TypeError(f"result {first_name!r} is not an OOSResult or BacktestResult")
    return first


def _comparable_score(name, result, reference, metric):
    """Validate and score one result against shared reference data."""
    if not isinstance(name, str):
        raise TypeError("result names must be strings")
    if not isinstance(result, type(reference)):
        raise TypeError("all compared results must use the same evaluation method")
    if result.target != reference.target:
        raise ValueError("all compared results must use the same target")
    if isinstance(result, OOSResult):
        for attribute in ("estimation_indices", "validation_indices"):
            if not np.array_equal(
                np.asarray(getattr(result, attribute)),
                np.asarray(getattr(reference, attribute)),
            ):
                raise ValueError(
                    f"all compared OOS results must use the same {attribute}"
                )
        for attribute in ("estimation_dates", "validation_dates"):
            result_dates = getattr(result, attribute)
            reference_dates = getattr(reference, attribute)
            if (result_dates is None) != (reference_dates is None):
                raise ValueError(
                    "all compared OOS results must use the same date metadata"
                )
            if result_dates is not None and not result_dates.equals(reference_dates):
                raise ValueError(
                    f"all compared OOS results must use the same {attribute}"
                )
    elif not np.array_equal(
        np.asarray(result.target_indices),
        np.asarray(reference.target_indices),
    ):
        raise ValueError("all compared backtests must use the same target indices")
    result_actual = np.asarray(result.actual)
    reference_actual = np.asarray(reference.actual)
    if result_actual.shape != reference_actual.shape or not np.array_equal(
        result_actual,
        reference_actual,
        equal_nan=True,
    ):
        raise ValueError("all compared results must use the same actual values")
    if metric not in result.metrics:
        raise ValueError(f"result {name!r} does not contain metric {metric!r}")
    return float(result.metrics[metric])


def compare_forecasts(results, *, metric="rmse"):
    """Rank comparable OOS or backtest results by an error metric.

    All results must use the same target, target indices, and actual values.
    Lower metric values rank first; non-finite scores rank last.

    Parameters
    ----------
    results : mapping or sequence of OOSResult or BacktestResult
        Comparable evaluation results. Mapping keys become model names.
    metric : str, default "rmse"
        Canonical error metric used for ranking.

    Returns
    -------
    ComparisonResult
        Scores, shared target, and ascending ranking.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsMetrics import OOSResult, compare_forecasts
    >>> def result(mean):
    ...     return OOSResult(
    ...         np.array(mean), np.array([1.0, 2.0]), None, None,
    ...         np.arange(3), np.array([3, 4]), None, None, "Example", "level"
    ...     )
    >>> comparison = compare_forecasts(
    ...     {"perfect": result([1.0, 2.0]), "shifted": result([2.0, 3.0])}
    ... )
    >>> comparison.ranking
    ['perfect', 'shifted']
    """
    items = _comparison_items(results, metric)
    reference = _comparison_reference(items)
    scores = {
        name: _comparable_score(name, result, reference, metric)
        for name, result in items
    }

    return ComparisonResult(
        metric=metric,
        scores=scores,
        target=reference.target,
    )


def _require_complete_scoring_sample(results):
    """Reject per-model omission of non-finite validation pairs."""
    for name, result in results.items():
        if not np.all(np.isfinite(result.actual)) or not np.all(
            np.isfinite(result.mean)
        ):
            raise ValueError(
                f"result {name!r} contains non-finite actual or forecast values; "
                "multi-model OOS comparison requires a complete shared "
                "validation sample"
            )


def evaluate_models_oos(
    models,
    estimation_period,
    validation_period,
    *,
    alpha=0.05,
    rank_by="rmse",
    method=None,
):
    """Evaluate multiple models on one explicit, complete OOS sample.

    Every model receives the same inclusive estimation and validation bounds.
    The returned report retains each :class:`OOSResult`, exposes all canonical
    error metrics, and ranks models by ``rank_by``.

    Parameters
    ----------
    models : mapping of str to estimator
        Named unfitted Ts estimators evaluated on the shared split.
    estimation_period : tuple of int or datetime-like
        Inclusive shared estimation bounds.
    validation_period : tuple of int or datetime-like
        Inclusive shared validation bounds.
    alpha : float, default 0.05
        Significance level for each forecast interval.
    rank_by : str, default "rmse"
        Canonical error metric used for ordering models.
    method : str, optional
        Optimizer forwarded to every model's ``fit()`` method. ``None`` keeps
        each model's default. Every named model must support this argument.

    Returns
    -------
    OOSComparisonResult
        Individual evaluations plus the complete ranking table.

    Examples
    --------
    >>> from Ts.TsMetrics import evaluate_models_oos
    >>> from Ts.TsModels import SARIMAX
    >>> from Ts.TsSims import simulate_sarima
    >>> data = simulate_sarima(n=60, order=(1, 0, 0), seed=42).data
    >>> report = evaluate_models_oos(
    ...     {
    ...         "mean": SARIMAX(data, order=(0, 0, 0)),
    ...         "ar1": SARIMAX(data, order=(1, 0, 0)),
    ...     },
    ...     estimation_period=(0, 39),
    ...     validation_period=(40, 49),
    ... )
    >>> report.table.shape[0]
    2
    """
    _validate_metric(rank_by, "rank_by")
    if not isinstance(models, Mapping):
        raise TypeError("models must be a mapping of names to models")
    if not models:
        raise ValueError("models must not be empty")
    if not all(isinstance(name, str) for name in models):
        raise TypeError("model names must be strings")

    for name, model in models.items():
        validate_fit_method(model, method, model_name=name)

    evaluations = {
        name: oos(
            model,
            estimation_period=estimation_period,
            validation_period=validation_period,
            alpha=alpha,
            method=method,
        )
        for name, model in models.items()
    }

    compare_forecasts(evaluations, metric=rank_by)
    _require_complete_scoring_sample(evaluations)
    return OOSComparisonResult(
        evaluations=evaluations,
        rank_by=rank_by,
    )
