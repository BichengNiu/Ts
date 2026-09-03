"""Tests for point-forecast performance metrics."""

import numpy as np
import pytest

from Ts.TsMetrics import (
    compute_metrics,
    directional_accuracy,
    mae,
    mape,
    mse,
    mpe,
    relative_win_rate,
    rmse,
    smape,
    theil_u1,
    trend_correlation,
)


def test_point_metrics_have_expected_values():
    """Core metrics use one shared, documented error convention."""
    actual = np.array([1.0, 2.0, 4.0])
    predicted = np.array([2.0, 2.0, 1.0])

    assert mae(actual, predicted) == pytest.approx(4.0 / 3.0)
    assert mse(actual, predicted) == pytest.approx(10.0 / 3.0)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt(10.0 / 3.0))
    assert mpe(actual, predicted) == pytest.approx((100.0 + 0.0 - 75.0) / 3.0)
    assert mape(actual, predicted) == pytest.approx(np.mean([1.0, 0.0, 0.75]) * 100.0)
    assert smape(actual, predicted) == pytest.approx(
        np.mean([2.0 / 3.0, 0.0, 6.0 / 5.0]) * 100.0
    )
    assert 0.0 <= theil_u1(actual, predicted) <= 1.0


def test_compute_metrics_omits_only_nonfinite_pairs():
    """The default missing-value policy scores every finite pair."""
    metrics = compute_metrics(
        np.array([1.0, np.nan, 3.0, np.inf]),
        np.array([1.0, 2.0, 5.0, 4.0]),
    )

    assert metrics["n"] == 2
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.0))
    assert metrics["mae"] == pytest.approx(1.0)
    assert set(metrics) == {
        "mae",
        "mse",
        "rmse",
        "mpe",
        "mape",
        "smape",
        "theil_u1",
        "n",
    }


def test_compute_metrics_can_reject_nonfinite_pairs():
    """nan_policy=raise fails instead of silently filtering."""
    with pytest.raises(ValueError, match="non-finite"):
        compute_metrics(
            np.array([1.0, np.nan]),
            np.array([1.0, 2.0]),
            nan_policy="raise",
        )


def test_metrics_reject_mismatched_shapes():
    """Actual and predicted arrays must describe identical observations."""
    with pytest.raises(ValueError, match="same shape"):
        compute_metrics(np.ones(2), np.ones((2, 1)))


def test_zero_actuals_have_explicit_percentage_semantics():
    """MAPE excludes zero actuals while all-zero perfect forecasts score zero."""
    assert np.isnan(mpe(np.zeros(2), np.array([0.0, 1.0])))
    assert np.isnan(mape(np.zeros(2), np.array([0.0, 1.0])))
    assert smape(np.zeros(2), np.zeros(2)) == 0.0
    assert theil_u1(np.zeros(2), np.zeros(2)) == 0.0


def test_all_missing_pairs_return_nan_metrics_and_zero_count():
    """An empty finite intersection remains observable through n=0."""
    metrics = compute_metrics(
        np.array([np.nan, np.inf]),
        np.array([1.0, 2.0]),
    )

    assert metrics["n"] == 0
    assert all(np.isnan(value) for key, value in metrics.items() if key != "n")


def test_rmse_remains_finite_when_squaring_would_overflow():
    """RMSE uses stable scaling instead of first squaring large errors."""
    actual = np.array([1e200, -1e200])
    predicted = np.zeros(2)

    assert rmse(actual, predicted) == pytest.approx(1e200)
    assert compute_metrics(actual, predicted)["rmse"] == pytest.approx(1e200)


def test_scale_metrics_are_stable_at_extreme_magnitudes():
    """Percentage metrics and Theil U1 avoid overflow and underflow."""
    large = np.array([1e308])
    opposite = np.array([-1e308])
    tiny = np.array([1e-200])
    zero = np.zeros(1)

    assert mape(large, opposite) == pytest.approx(200.0)
    assert smape(large, opposite) == pytest.approx(200.0)
    assert theil_u1(large, opposite) == pytest.approx(1.0)
    assert theil_u1(tiny, zero) == pytest.approx(1.0)
    metrics = compute_metrics(tiny, zero)
    assert metrics["theil_u1"] == pytest.approx(1.0)


def test_directional_accuracy_supports_changes_and_common_reference():
    """Direction scores compare signs without confusing levels and changes."""
    assert directional_accuracy([1.0, -2.0, 0.0], [2.0, -1.0, 0.0]) == 1.0
    assert directional_accuracy(
        [11.0, 8.0, 10.0],
        [12.0, 9.0, 9.0],
        reference=[10.0, 10.0, 10.0],
    ) == pytest.approx(2.0 / 3.0)


def test_relative_win_rate_counts_strict_model_wins():
    """Ties are not wins but remain in the valid comparison denominator."""
    assert relative_win_rate(
        [10.0, 10.0, 10.0],
        [9.0, 8.0, 12.0],
        [8.0, 12.0, 12.0],
    ) == pytest.approx(1.0 / 3.0)


def test_trend_correlation_describes_co_movement():
    """Correlation is undefined for constant or undersized paths."""
    assert trend_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert np.isnan(trend_correlation([1.0], [2.0]))
    assert np.isnan(trend_correlation([1.0, 1.0], [2.0, 3.0]))
