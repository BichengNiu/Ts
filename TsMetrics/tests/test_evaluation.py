"""Tests for leakage-free OOS evaluation, backtesting, and comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from Ts.TsMetrics import (
    BacktestResult,
    ComparisonResult,
    OOSResult,
    backtest,
    compare_forecasts,
    oos,
)
from Ts.TsModels._base import BaseModel, BaseModelResult, PredictResult


@dataclass
class _MeanResult(BaseModelResult):
    """Deterministic result that forecasts its training mean."""

    training_mean: float = 0.0

    def predict(self, start=0, end=None, dynamic=False, alpha=0.05):
        """Forecast the stored training mean."""
        del dynamic, alpha
        if end is None:
            end = start
        horizon = end - start + 1
        mean = np.full(horizon, self.training_mean)
        return PredictResult(
            mean=mean,
            lower=mean - 0.5,
            upper=mean + 0.5,
            is_oos=np.ones(horizon, dtype=bool),
        )


class _MeanModel(BaseModel):
    """Model double whose fitted value reveals the exact training window."""

    def __init__(self, data, dates=None):
        self.data = np.asarray(data, dtype=float)
        self.dates = dates
        self.fit_windows = []
        self.result_ = None

    def fit(self):
        """Record the fit window and return a deterministic forecaster."""
        self.fit_windows.append(self.data.copy())
        training_mean = float(np.mean(self.data))
        result = _MeanResult(
            model_type="MEAN",
            params={},
            std_errors={},
            p_values={},
            aic=0.0,
            bic=0.0,
            log_likelihood=0.0,
            residuals=self.data - training_mean,
            fitted_values=np.full_like(self.data, training_mean),
            nobs=len(self.data),
            data=self.data.copy(),
            training_mean=training_mean,
        )
        self.result_ = result
        return result


class _InvalidTargetModel(_MeanModel):
    """Model double whose target validation fails after forecasting."""

    def _evaluation_actual(self, observed, train_data):
        """Return a deliberately invalid target shape."""
        del train_data
        return np.asarray(observed)[:-1]


def test_oos_fits_only_the_estimation_period():
    """The holdout API cannot estimate parameters from evaluation data."""
    model = _MeanModel(np.arange(15.0))

    result = oos(
        model,
        estimation_period=(0, 9),
        validation_period=(10, 14),
    )

    assert len(model.fit_windows) == 1
    np.testing.assert_array_equal(model.fit_windows[0], np.arange(10.0))
    np.testing.assert_array_equal(result.estimation_indices, np.arange(10))
    np.testing.assert_array_equal(
        result.validation_indices,
        np.arange(10, 15),
    )
    np.testing.assert_allclose(result.mean, 4.5)
    np.testing.assert_array_equal(result.actual, np.arange(10.0, 15.0))
    assert result.metrics["n"] == 5
    assert result.target == "observed"
    assert model.result_ is None
    assert not hasattr(result, "split")


def test_oos_allows_a_gap_and_scores_only_the_validation_period():
    """A later validation period may follow an unscored forecast gap."""
    model = _MeanModel(np.arange(20.0))

    result = model.oos(
        estimation_period=(2, 11),
        validation_period=(15, 18),
    )

    np.testing.assert_array_equal(model.fit_windows[0], np.arange(2.0, 12.0))
    np.testing.assert_array_equal(
        result.estimation_indices,
        np.arange(2, 12),
    )
    np.testing.assert_array_equal(
        result.validation_indices,
        np.arange(15, 19),
    )
    np.testing.assert_array_equal(result.actual, np.arange(15.0, 19.0))
    np.testing.assert_allclose(result.mean, 6.5)


def test_oos_resolves_exact_date_periods():
    """Date bounds are inclusive and must exist in the model calendar."""
    dates = pd.date_range("2020-01-01", periods=20, freq="MS")
    model = _MeanModel(np.arange(20.0), dates=dates)

    result = model.oos(
        estimation_period=("2020-03-01", "2020-12-01"),
        validation_period=("2021-03-01", "2021-06-01"),
    )

    assert result.estimation_dates.equals(dates[2:12])
    assert result.validation_dates.equals(dates[14:18])
    np.testing.assert_array_equal(
        result.validation_indices,
        np.arange(14, 18),
    )


@pytest.mark.parametrize(
    ("estimation_period", "validation_period", "message"),
    [
        ((0, 9), (9, 12), "strictly later"),
        ((5, 4), (10, 12), "estimation_period start"),
        ((0, 9), (12, 10), "validation_period start"),
        ((0, 9), (10, 20), "outside"),
    ],
)
def test_oos_rejects_invalid_period_relationships(
    estimation_period,
    validation_period,
    message,
):
    """Period ordering, overlap, and coverage are hard failures."""
    model = _MeanModel(np.arange(20.0))

    with pytest.raises(ValueError, match=message):
        model.oos(
            estimation_period=estimation_period,
            validation_period=validation_period,
        )


def test_oos_rejects_missing_date_bound():
    """Date selection never silently snaps to a nearby observation."""
    dates = pd.date_range("2020-01-01", periods=20, freq="MS")
    model = _MeanModel(np.arange(20.0), dates=dates)

    with pytest.raises(ValueError, match="does not exist"):
        model.oos(
            estimation_period=("2020-01-01", "2020-10-15"),
            validation_period=("2020-11-01", "2021-08-01"),
        )


def test_oos_has_no_split_compatibility_path():
    """The removed split API fails instead of entering a compatibility path."""
    from inspect import signature

    model = _MeanModel(np.arange(15.0))

    assert "split" not in signature(model.oos).parameters


def test_public_backtest_uses_each_origin_without_future_leakage():
    """Rolling-origin evaluation refits on data strictly before each origin."""
    model = _MeanModel(np.arange(20.0))

    result = backtest(
        model,
        initial_window=10,
        horizon=2,
        step=3,
    )

    assert result.origins.tolist() == [10, 13, 16]
    assert [window[-1] for window in model.fit_windows] == [9.0, 12.0, 15.0]
    assert result.target_indices.tolist() == [
        [10, 11],
        [13, 14],
        [16, 17],
    ]
    assert result.metrics["n"] == 6


def test_expanding_backtest_rejects_ignored_window_size():
    """An irrelevant rolling-window argument is not silently ignored."""
    model = _MeanModel(np.arange(20.0))

    with pytest.raises(ValueError, match="only valid"):
        backtest(model, initial_window=10, window_size=10)


def test_recorded_failure_does_not_leave_partial_forecast_values():
    """A failed origin is committed atomically as an all-NaN row."""
    model = _InvalidTargetModel(np.arange(15.0))

    result = backtest(
        model,
        initial_window=10,
        horizon=2,
        step=2,
        on_error="record",
    )

    assert np.isnan(result.mean).all()
    assert np.isnan(result.actual).all()
    assert len(result.failures) == 2
    assert result.metrics["n"] == 0


def test_base_model_methods_delegate_to_tsmetrics():
    """Model convenience methods expose the canonical evaluation engine."""
    model = _MeanModel(np.arange(16.0))

    holdout = model.oos(
        estimation_period=(0, 11),
        validation_period=(12, 15),
    )
    rolling = model.backtest(initial_window=10, horizon=1, step=3)

    assert isinstance(holdout, OOSResult)
    assert isinstance(rolling, BacktestResult)


def _oos_result(mean, actual, target="observed"):
    """Build a minimal comparable result."""
    mean = np.asarray(mean, dtype=float)
    actual = np.asarray(actual, dtype=float)

    return OOSResult(
        mean=mean,
        actual=actual,
        lower=None,
        upper=None,
        estimation_indices=np.arange(10),
        validation_indices=np.arange(10, 10 + len(mean)),
        estimation_dates=None,
        validation_dates=None,
        model_type="TEST",
        target=target,
    )


def test_compare_forecasts_ranks_lower_error_first():
    """Forecast comparison returns deterministic ascending error ranking."""
    actual = np.array([1.0, 2.0, 3.0])
    results = {
        "weak": _oos_result([2.0, 3.0, 4.0], actual),
        "strong": _oos_result([1.0, 2.0, 3.2], actual),
    }

    comparison = compare_forecasts(results, metric="rmse")

    assert comparison.ranking == ["strong", "weak"]
    assert comparison.scores["strong"] < comparison.scores["weak"]
    assert comparison.target == "observed"


def test_compare_forecasts_rejects_incomparable_targets():
    """Mean and volatility forecasts cannot share one performance ranking."""
    actual = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="target"):
        compare_forecasts(
            {
                "mean": _oos_result(actual, actual, target="observed"),
                "vol": _oos_result(
                    actual,
                    actual,
                    target="absolute_demeaned_return_proxy",
                ),
            }
        )


def test_compare_forecasts_rejects_different_actual_values():
    """Models must be evaluated on the same observations."""
    with pytest.raises(ValueError, match="actual"):
        compare_forecasts(
            {
                "a": _oos_result([1.0, 2.0], [1.0, 2.0]),
                "b": _oos_result([1.0, 2.0], [1.0, 3.0]),
            }
        )


def test_compare_forecasts_rejects_different_actual_shapes():
    """Broadcast-compatible actual arrays are still incomparable."""
    actual = np.array([1.0, 2.0])
    first = _oos_result(actual, actual)
    second = _oos_result(actual, actual)
    second.actual = second.actual[np.newaxis, :]

    with pytest.raises(ValueError, match="actual"):
        compare_forecasts({"a": first, "b": second})


def test_compare_forecasts_requires_string_model_names():
    """String conversion cannot silently merge distinct model labels."""
    result = _oos_result([1.0], [1.0])

    with pytest.raises(TypeError, match="names must be strings"):
        compare_forecasts({1: result})


def test_compare_forecasts_requires_exact_actual_values():
    """Numerically close but different observations are not comparable."""
    first = _oos_result([1.0, 2.0], [1.0, 2.0])
    second = _oos_result([1.0, 2.0], [1.0, 2.0])
    second.actual[0] += 1e-12

    with pytest.raises(ValueError, match="actual"):
        compare_forecasts({"a": first, "b": second})


def test_comparison_result_rejects_negative_error_scores():
    """Finite error metrics cannot encode impossible negative values."""
    with pytest.raises(ValueError, match="non-negative"):
        ComparisonResult(
            metric="rmse",
            scores={"a": -1.0, "b": 2.0},
            target="observed",
        )


def test_predict_rejects_removed_pseudo_oos_argument():
    """Pseudo-OOS is removed instead of retained as a compatibility path."""
    from Ts.TsModels import SARIMA
    from Ts.TsSims import simulate_sarima

    data = simulate_sarima(
        n=40,
        order=(1, 0, 0),
        ar=[0.5],
        seed=42,
        burn=50,
    ).data
    fitted = SARIMA(data, order=(1, 0, 0)).fit()

    with pytest.raises(TypeError, match="oos_start"):
        fitted.predict(oos_start=25)


def test_oos_passes_holdout_exog_without_holdout_y():
    from Ts.TsModels import SARIMA

    dates = pd.date_range("2020-01-01", periods=30, freq="MS")
    exog = pd.DataFrame({"x": np.arange(30.0)}, index=dates)
    data = pd.Series(2.0 * exog["x"].to_numpy(), index=dates)
    model = SARIMA(
        data,
        exog=exog,
        order=(0, 0, 0),
        trend="n",
    )

    result = model.oos(
        estimation_period=(dates[0], dates[19]),
        validation_period=(dates[20], dates[29]),
    )

    assert result.mean.shape == (10,)
    assert result.metrics["rmse"] < 1e-5
    assert model.result_ is None


def test_record_mode_reports_missing_future_exog_dates():
    from Ts.TsModels import SARIMA

    dates = pd.date_range("2022-01-01", periods=23, freq="MS")
    exog = pd.DataFrame(
        {"x": np.arange(23.0)},
        index=dates,
    )
    model = SARIMA(
        pd.Series(1.5 * exog["x"].to_numpy(), index=dates),
        exog=exog,
        order=(0, 0, 0),
        trend="n",
    )
    model.exog = model.exog[:20].copy()

    result = model.backtest(
        initial_window=20,
        horizon=3,
        on_error="record",
    )

    assert np.isnan(result.mean).all()
    assert len(result.failures) == 1
    message = result.failures[0]["message"]
    assert dates[20].isoformat() in message
    assert dates[22].isoformat() in message
