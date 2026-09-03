"""Point-forecast performance metrics."""

from __future__ import annotations

import numpy as np


ERROR_METRIC_NAMES = ("mae", "mse", "rmse", "mape", "smape", "theil_u1")
AUXILIARY_METRIC_NAMES = ("mpe",)


def _validate_nan_policy(nan_policy):
    """Validate the common finite-pair policy."""
    if nan_policy not in {"omit", "raise"}:
        raise ValueError(
            f"nan_policy must be either 'omit' or 'raise', got {nan_policy!r}"
        )


def _validate_rank_metric(rank_by):
    """Return *rank_by* after requiring one canonical error metric name."""
    if rank_by not in ERROR_METRIC_NAMES:
        raise ValueError(
            f"rank_by must be one of {list(ERROR_METRIC_NAMES)}, got {rank_by!r}"
        )
    return rank_by


def _paired_values(actual, predicted, nan_policy):
    """Return aligned finite float vectors under the requested NaN policy."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape:
        raise ValueError(
            "actual and predicted must have the same shape, got "
            f"{actual_array.shape} and {predicted_array.shape}"
        )
    _validate_nan_policy(nan_policy)

    actual_flat = actual_array.ravel()
    predicted_flat = predicted_array.ravel()
    finite = np.isfinite(actual_flat) & np.isfinite(predicted_flat)
    if nan_policy == "raise" and not np.all(finite):
        raise ValueError("actual and predicted contain non-finite pairs")
    return actual_flat[finite], predicted_flat[finite]


def _mean_or_nan(values):
    """Return a float mean without warning for an empty vector."""
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _errors(actual, predicted):
    """Subtract finite vectors while allowing a mathematically infinite error."""
    with np.errstate(over="ignore", invalid="ignore"):
        return predicted - actual


def _root_mean_square(values):
    """Return RMS using scaling that avoids avoidable square overflow."""
    if values.size == 0:
        return float("nan")
    absolute = np.abs(values)
    scale = float(np.max(absolute))
    if scale == 0.0:
        return 0.0
    if not np.isfinite(scale):
        return float("inf")
    scaled = absolute / scale
    return float(scale * np.sqrt(np.mean(scaled * scaled)))


def _mape_values(actual, predicted):
    """Compute MAPE from already paired finite vectors."""
    nonzero = actual != 0.0
    if not np.any(nonzero):
        return float("nan")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative_errors = np.abs(predicted[nonzero] / actual[nonzero] - 1.0)
    return float(np.mean(relative_errors) * 100.0)


def _smape_values(actual, predicted):
    """Compute sMAPE without overflowing pairwise sums."""
    if actual.size == 0:
        return float("nan")
    scale = np.maximum(np.abs(actual), np.abs(predicted))
    terms = np.zeros_like(scale)
    positive = scale > 0.0
    scaled_actual = actual[positive] / scale[positive]
    scaled_predicted = predicted[positive] / scale[positive]
    terms[positive] = (
        2.0
        * np.abs(scaled_predicted - scaled_actual)
        / (np.abs(scaled_actual) + np.abs(scaled_predicted))
    )
    return float(np.mean(terms) * 100.0)


def _theil_u1_values(actual, predicted):
    """Compute Theil U1 after cancelling a common stable scale."""
    if actual.size == 0:
        return float("nan")
    scale = float(max(np.max(np.abs(actual)), np.max(np.abs(predicted))))
    if scale == 0.0:
        return 0.0
    scaled_actual = actual / scale
    scaled_predicted = predicted / scale
    numerator = _root_mean_square(scaled_predicted - scaled_actual)
    denominator = _root_mean_square(scaled_actual) + _root_mean_square(scaled_predicted)
    return float(numerator / denominator)


def mae(actual, predicted, *, nan_policy="omit"):
    """Return mean absolute error.

    Parameters
    ----------
    actual, predicted : array-like
        Aligned observed and forecast values with identical shapes.
    nan_policy : {"omit", "raise"}, default "omit"
        Drop paired non-finite values or reject them.

    Returns
    -------
    float
        Mean absolute paired error, or NaN when no valid pair remains.

    Examples
    --------
    >>> from Ts.TsMetrics import mae
    >>> mae([1, 2, 3], [1, 4, 2])
    1.0
    """
    actual_values, predicted_values = _paired_values(
        actual,
        predicted,
        nan_policy,
    )
    return _mean_or_nan(np.abs(_errors(actual_values, predicted_values)))


def mse(actual, predicted, *, nan_policy="omit"):
    """Return mean squared error.

    Parameters
    ----------
    actual, predicted : array-like
        Aligned observed and forecast values with identical shapes.
    nan_policy : {"omit", "raise"}, default "omit"
        Drop paired non-finite values or reject them.

    Returns
    -------
    float
        Mean squared paired error, or NaN when no valid pair remains.

    Examples
    --------
    >>> from Ts.TsMetrics import mse
    >>> mse([1, 2, 3], [1, 4, 2])
    1.6666666666666667
    """
    actual_values, predicted_values = _paired_values(
        actual,
        predicted,
        nan_policy,
    )
    with np.errstate(over="ignore"):
        squared_errors = np.square(_errors(actual_values, predicted_values))
    return _mean_or_nan(squared_errors)


def rmse(actual, predicted, *, nan_policy="omit"):
    """Return root mean squared error.

    Parameters
    ----------
    actual, predicted : array-like
        Aligned observed and forecast values with identical shapes.
    nan_policy : {"omit", "raise"}, default "omit"
        Drop paired non-finite values or reject them.

    Returns
    -------
    float
        Square root of mean squared error.

    Examples
    --------
    >>> from Ts.TsMetrics import rmse
    >>> round(rmse([1, 2, 3], [1, 4, 2]), 6)
    1.290994
    """
    actual_values, predicted_values = _paired_values(
        actual,
        predicted,
        nan_policy,
    )
    return _root_mean_square(_errors(actual_values, predicted_values))


def mape(actual, predicted, *, nan_policy="omit"):
    """Return mean absolute percentage error in percentage points.

    Pairs whose actual value is zero are excluded because their percentage
    error is undefined. The result is NaN when no non-zero actual remains.

    Parameters
    ----------
    actual, predicted : array-like
        Aligned observed and forecast values with identical shapes.
    nan_policy : {"omit", "raise"}, default "omit"
        Drop paired non-finite values or reject them.

    Returns
    -------
    float
        Mean absolute percentage error in percentage points.

    Examples
    --------
    >>> from Ts.TsMetrics import mape
    >>> round(mape([100, 200], [90, 220]), 10)
    10.0
    """
    actual_values, predicted_values = _paired_values(
        actual,
        predicted,
        nan_policy,
    )
    return _mape_values(actual_values, predicted_values)


def mpe(actual, predicted, *, nan_policy="omit"):
    """Return mean percentage error in percentage points.

    The signed error is ``(predicted - actual) / actual``. Positive values
    indicate systematic overprediction. Pairs whose actual value is zero are
    excluded because their percentage error is undefined.

    Parameters
    ----------
    actual, predicted : array-like
        Aligned observed and forecast values with identical shapes.
    nan_policy : {"omit", "raise"}, default "omit"
        Drop paired non-finite values or reject them.

    Returns
    -------
    float
        Mean signed percentage error in percentage points, or NaN when no
        non-zero actual remains.

    Examples
    --------
    >>> from Ts.TsMetrics import mpe
    >>> round(mpe([100, 200], [110, 170]), 6)
    -2.5
    """
    actual_values, predicted_values = _paired_values(
        actual,
        predicted,
        nan_policy,
    )
    nonzero = actual_values != 0.0
    if not np.any(nonzero):
        return float("nan")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative_errors = predicted_values[nonzero] / actual_values[nonzero] - 1.0
    return float(np.mean(relative_errors) * 100.0)


def smape(actual, predicted, *, nan_policy="omit"):
    """Return symmetric mean absolute percentage error.

    Terms where both actual and prediction are zero contribute zero.

    Parameters
    ----------
    actual, predicted : array-like
        Aligned observed and forecast values with identical shapes.
    nan_policy : {"omit", "raise"}, default "omit"
        Drop paired non-finite values or reject them.

    Returns
    -------
    float
        Symmetric percentage error in percentage points.

    Examples
    --------
    >>> from Ts.TsMetrics import smape
    >>> round(smape([100, 200], [90, 220]), 6)
    10.025063
    """
    actual_values, predicted_values = _paired_values(
        actual,
        predicted,
        nan_policy,
    )
    return _smape_values(actual_values, predicted_values)


def theil_u1(actual, predicted, *, nan_policy="omit"):
    """Return Theil's U1 inequality coefficient.

    U1 is bounded between zero and one for finite real-valued inputs. It is
    zero for a perfect all-zero forecast.

    Parameters
    ----------
    actual, predicted : array-like
        Aligned observed and forecast values with identical shapes.
    nan_policy : {"omit", "raise"}, default "omit"
        Drop paired non-finite values or reject them.

    Returns
    -------
    float
        Theil's U1 inequality coefficient.

    Examples
    --------
    >>> from Ts.TsMetrics import theil_u1
    >>> theil_u1([1, 2, 3], [1, 2, 3])
    0.0
    """
    actual_values, predicted_values = _paired_values(
        actual,
        predicted,
        nan_policy,
    )
    return _theil_u1_values(actual_values, predicted_values)


def directional_accuracy(
    actual,
    predicted,
    reference=None,
    *,
    nan_policy="omit",
):
    """Return the proportion of matching forecast and actual directions.

    When ``reference`` is omitted, *actual* and *predicted* are interpreted as
    already-computed changes. When it is provided, the directions are
    ``actual - reference`` and ``predicted - reference``. Equal zero changes
    count as a matching direction; a zero change matches a non-zero change
    only when both signs are equal.

    Parameters
    ----------
    actual, predicted : array-like
        Aligned actual changes and predicted changes, or aligned levels when
        ``reference`` is supplied.
    reference : array-like, optional
        Common level from which actual and predicted changes are measured.
        It must have the same shape as *actual* and *predicted*.
    nan_policy : {"omit", "raise"}, default "omit"
        Drop non-finite direction pairs or reject them.

    Returns
    -------
    float
        Direction hit rate in the closed interval [0, 1], or NaN when no
        valid pair remains.

    Examples
    --------
    >>> from Ts.TsMetrics import directional_accuracy
    >>> directional_accuracy([1, -1, 0], [2, -3, 0])
    1.0
    """
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape:
        raise ValueError(
            "actual and predicted must have the same shape, got "
            f"{actual_array.shape} and {predicted_array.shape}"
        )
    _validate_nan_policy(nan_policy)
    if reference is None:
        actual_change = actual_array.ravel()
        predicted_change = predicted_array.ravel()
        finite = np.isfinite(actual_change) & np.isfinite(predicted_change)
    else:
        reference_array = np.asarray(reference, dtype=float)
        if reference_array.shape != actual_array.shape:
            raise ValueError(
                "reference must have the same shape as actual and predicted, "
                f"got {reference_array.shape} and {actual_array.shape}"
            )
        actual_flat = actual_array.ravel()
        predicted_flat = predicted_array.ravel()
        reference_flat = reference_array.ravel()
        actual_change = actual_flat - reference_flat
        predicted_change = predicted_flat - reference_flat
        finite = (
            np.isfinite(actual_flat)
            & np.isfinite(predicted_flat)
            & np.isfinite(reference_flat)
        )
    if nan_policy == "raise" and not np.all(finite):
        raise ValueError("direction inputs contain non-finite values")
    if not np.any(finite):
        return float("nan")
    return float(
        np.mean(
            np.sign(actual_change[finite])
            == np.sign(predicted_change[finite])
        )
    )


def relative_win_rate(actual, predicted, baseline, *, nan_policy="omit"):
    """Return the share of periods where a forecast beats a baseline.

    A win is a strict reduction in absolute error compared with *baseline*.
    Ties remain in the denominator but are not counted as wins.

    Parameters
    ----------
    actual, predicted, baseline : array-like
        Aligned observed values, model forecasts, and baseline forecasts with
        identical shapes.
    nan_policy : {"omit", "raise"}, default "omit"
        Drop non-finite triples or reject them.

    Returns
    -------
    float
        Strict win rate in [0, 1], or NaN when no valid triple remains.

    Examples
    --------
    >>> from Ts.TsMetrics import relative_win_rate
    >>> relative_win_rate([10, 10], [9, 12], [8, 13])
    1.0
    """
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    baseline_array = np.asarray(baseline, dtype=float)
    if not (
        actual_array.shape == predicted_array.shape == baseline_array.shape
    ):
        raise ValueError(
            "actual, predicted, and baseline must have the same shape"
        )
    _validate_nan_policy(nan_policy)
    actual_flat = actual_array.ravel()
    predicted_flat = predicted_array.ravel()
    baseline_flat = baseline_array.ravel()
    finite = (
        np.isfinite(actual_flat)
        & np.isfinite(predicted_flat)
        & np.isfinite(baseline_flat)
    )
    if nan_policy == "raise" and not np.all(finite):
        raise ValueError("actual, predicted, and baseline contain non-finite values")
    if not np.any(finite):
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore"):
        model_error = np.abs(predicted_flat[finite] - actual_flat[finite])
        baseline_error = np.abs(baseline_flat[finite] - actual_flat[finite])
    return float(np.mean(model_error < baseline_error))


def trend_correlation(actual, predicted, *, nan_policy="omit"):
    """Return Pearson correlation for paired actual and forecast paths.

    The result describes co-movement, not absolute accuracy. A constant path
    has undefined correlation and returns NaN.

    Parameters
    ----------
    actual, predicted : array-like
        Aligned observed and forecast values with identical shapes.
    nan_policy : {"omit", "raise"}, default "omit"
        Drop paired non-finite values or reject them.

    Returns
    -------
    float
        Pearson correlation in [-1, 1], or NaN when fewer than two valid
        variable pairs remain.

    Examples
    --------
    >>> from Ts.TsMetrics import trend_correlation
    >>> trend_correlation([1, 2, 3], [2, 4, 6])
    1.0
    """
    actual_values, predicted_values = _paired_values(
        actual,
        predicted,
        nan_policy,
    )
    if actual_values.size < 2:
        return float("nan")
    actual_centered = actual_values - np.mean(actual_values)
    predicted_centered = predicted_values - np.mean(predicted_values)
    denominator = np.sqrt(
        np.sum(actual_centered**2) * np.sum(predicted_centered**2)
    )
    if denominator == 0.0 or not np.isfinite(denominator):
        return float("nan")
    return float(np.sum(actual_centered * predicted_centered) / denominator)


def compute_metrics(actual, predicted, *, nan_policy="omit"):
    """Compute the canonical point-forecast metric set.

    Returns MAE, MSE, RMSE, MPE, MAPE, sMAPE, Theil U1, and the number of
    finite actual/prediction pairs used by the common error metrics.

    Parameters
    ----------
    actual, predicted : array-like
        Aligned observed and forecast values with identical shapes.
    nan_policy : {"omit", "raise"}, default "omit"
        Drop paired non-finite values or reject them.

    Returns
    -------
    dict
        Keys ``mae``, ``mse``, ``rmse``, ``mpe``, ``mape``, ``smape``,
        ``theil_u1``, and ``n``.

    Examples
    --------
    >>> from Ts.TsMetrics import compute_metrics
    >>> metrics = compute_metrics([1, 2, 3], [1, 4, 2])
    >>> (metrics["mae"], metrics["n"])
    (1.0, 3)
    """
    actual_values, predicted_values = _paired_values(
        actual,
        predicted,
        nan_policy,
    )
    n_valid = int(actual_values.size)
    if n_valid == 0:
        return {
            "mae": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mpe": float("nan"),
            "mape": float("nan"),
            "smape": float("nan"),
            "theil_u1": float("nan"),
            "n": 0,
        }

    errors = _errors(actual_values, predicted_values)
    absolute_errors = np.abs(errors)
    with np.errstate(over="ignore"):
        squared_errors = np.square(errors)

    return {
        "mae": _mean_or_nan(absolute_errors),
        "mse": _mean_or_nan(squared_errors),
        "rmse": _root_mean_square(errors),
        "mpe": mpe(actual_values, predicted_values, nan_policy="raise"),
        "mape": _mape_values(actual_values, predicted_values),
        "smape": _smape_values(actual_values, predicted_values),
        "theil_u1": _theil_u1_values(actual_values, predicted_values),
        "n": n_valid,
    }
