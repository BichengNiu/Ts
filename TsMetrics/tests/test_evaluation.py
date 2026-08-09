"""Tests for leakage-free OOS evaluation, backtesting, and comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from Ts.TsMetrics import (
    BacktestResult,
    ComparisonResult,
    OOSComparisonResult,
    OOSResult,
    backtest,
    compare_forecasts,
    evaluate_models_oos,
    oos,
)
from Ts.TsModels._base import BaseModel, BaseModelResult, PredictResult
from Ts.TsMetrics._results import (
    ForecastComparisonResult,
    ForecastEvaluationResult,
)
from Ts.TsMetrics._engine import evaluate_forecasts
from Ts.TsMetrics._schemes import Holdout, RollingOrigin


@dataclass
class _MeanResult(BaseModelResult):
    """Deterministic result that forecasts its training mean."""

    training_mean: float = 0.0
    forecast_bias: float = 0.0
    missing_forecast: bool = False

    def predict(self, start=0, end=None, dynamic=False, alpha=0.05):
        """Forecast the stored training mean."""
        del dynamic, alpha
        if end is None:
            end = start
        horizon = end - start + 1
        mean = np.full(horizon, self.training_mean + self.forecast_bias)
        if self.missing_forecast:
            mean[0] = np.nan
        return PredictResult(
            mean=mean,
            lower=mean - 0.5,
            upper=mean + 0.5,
            is_oos=np.ones(horizon, dtype=bool),
        )


class _MeanModel(BaseModel):
    """Model double whose fitted value reveals the exact training window."""

    def __init__(
        self,
        data,
        dates=None,
        *,
        forecast_bias=0.0,
        missing_forecast=False,
    ):
        self.data = np.asarray(data, dtype=float)
        self.dates = dates
        self.forecast_bias = forecast_bias
        self.missing_forecast = missing_forecast
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
            forecast_bias=self.forecast_bias,
            missing_forecast=self.missing_forecast,
        )
        self.result_ = result
        return result


class _OptimizerMeanModel(_MeanModel):
    """Mean model that records the optimizer selected for its clone."""

    def __init__(self, data, dates=None, *, methods=None, **kwargs):
        super().__init__(data, dates=dates, **kwargs)
        self.methods = [] if methods is None else methods

    def fit(self, *, method="bfgs"):
        """Record the requested optimizer before fitting the mean model."""
        self.methods.append(method)
        return super().fit()


class _FitOptionsMeanModel(_MeanModel):
    """Mean model that records general fit keyword forwarding."""

    def __init__(self, data, dates=None, *, options=None, **kwargs):
        super().__init__(data, dates=dates, **kwargs)
        self.options = [] if options is None else options

    def fit(self, *, method="bfgs", maxiter=100):
        self.options.append((method, maxiter))
        return super().fit()


class _FailingMeanModel(_MeanModel):
    """Mean model that raises for one selected training length."""

    def __init__(self, data, dates=None, *, fail_length, **kwargs):
        super().__init__(data, dates=dates, **kwargs)
        self.fail_length = fail_length

    def fit(self):
        if len(self.data) == self.fail_length:
            raise RuntimeError("planned fit failure")
        return super().fit()


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


def test_overlapping_dated_windows_run_through_latest_observation():
    """One-period overlap ends the final complete window at the latest date."""
    dates = pd.date_range("2020-01-01", periods=20, freq="MS")
    model = _MeanModel(np.arange(20.0), dates=dates)

    result = model.backtest(
        initial_window=10,
        horizon=3,
        step=1,
        window="expanding",
    )
    frame = result.metrics_by_window

    assert len(frame) == 20 - 10 - 3 + 1
    assert frame["window_start"].tolist()[:2] == [dates[10], dates[11]]
    assert frame["window_end"].tolist()[:2] == [dates[12], dates[13]]
    assert frame.iloc[-1]["window_end"] == dates[-1]
    assert [window[-1] for window in model.fit_windows] == list(
        np.arange(9.0, 17.0)
    )
    assert model.result_ is None


def test_backtest_metrics_by_window_are_exact_and_position_labelled():
    """Each row scores one complete forecast window at its target positions."""
    result = BacktestResult(
        mean=np.array([[2.0, 4.0], [3.0, 8.0]]),
        actual=np.array([[1.0, 5.0], [3.0, 4.0]]),
        lower=None,
        upper=None,
        origins=np.array([10, 11]),
        failures=[],
        model_type="TEST",
        window="expanding",
        target="observed",
    )

    frame = result.metrics_by_window

    assert frame.columns.tolist() == [
        "window_start",
        "window_end",
        "mae",
        "mse",
        "rmse",
        "mape",
        "smape",
        "theil_u1",
        "n",
    ]
    assert frame["window_start"].tolist() == [10, 11]
    assert frame["window_end"].tolist() == [11, 12]
    assert frame["rmse"].tolist() == pytest.approx([1.0, np.sqrt(8.0)])
    assert frame["mape"].tolist() == pytest.approx([60.0, 50.0])
    assert frame["n"].tolist() == [2, 2]


def test_backtest_metrics_by_window_use_calendar_labels():
    """Date-aware results label every complete target window with real dates."""
    dates = pd.date_range("2020-01-01", periods=13, freq="MS")
    result = BacktestResult(
        mean=np.ones((2, 2)),
        actual=np.ones((2, 2)),
        lower=None,
        upper=None,
        origins=np.array([10, 11]),
        failures=[],
        model_type="TEST",
        window="expanding",
        target="observed",
        dates=dates,
    )

    frame = result.metrics_by_window

    assert frame["window_start"].tolist() == [dates[10], dates[11]]
    assert frame["window_end"].tolist() == [dates[11], dates[12]]


def test_backtest_metrics_by_window_separate_multivariate_series():
    """Window RMSE is reported per series instead of mixing data scales."""
    actual = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ]
    )
    mean = actual + np.array([[[1.0, 2.0], [1.0, 2.0]]])
    result = BacktestResult(
        mean=mean,
        actual=actual,
        lower=None,
        upper=None,
        origins=np.array([10, 11]),
        failures=[],
        model_type="TEST",
        window="expanding",
        target="observed",
        series_names=("output", "prices"),
    )

    frame = result.metrics_by_window

    assert frame.columns.tolist() == [
        "window_start",
        "window_end",
        "series",
        "mae",
        "mse",
        "rmse",
        "mape",
        "smape",
        "theil_u1",
        "n",
    ]
    assert frame["series"].tolist() == [
        "output",
        "prices",
        "output",
        "prices",
    ]
    assert frame["rmse"].tolist() == pytest.approx([1.0, 2.0, 1.0, 2.0])


def test_backtest_metrics_by_window_falls_back_to_series_positions():
    """Unnamed multivariate output receives stable public series labels."""
    result = BacktestResult(
        mean=np.ones((1, 2, 2)),
        actual=np.ones((1, 2, 2)),
        lower=None,
        upper=None,
        origins=np.array([10]),
        failures=[],
        model_type="TEST",
        window="expanding",
        target="observed",
    )

    assert result.metrics_by_window["series"].tolist() == [
        "series_0",
        "series_1",
    ]


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
    frame = result.metrics_by_window
    assert frame["n"].tolist() == [0, 0]
    assert (
        frame.drop(columns=["window_start", "window_end", "n"])
        .isna()
        .all()
        .all()
    )


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


def _oos_result(
    mean,
    actual,
    target="observed",
    *,
    lower=None,
    upper=None,
    validation_dates=None,
    series_names=None,
    alpha=None,
):
    """Build a minimal comparable result."""
    mean = np.asarray(mean, dtype=float)
    actual = np.asarray(actual, dtype=float)
    estimation_dates = None
    if validation_dates is not None:
        validation_dates = pd.DatetimeIndex(validation_dates)
        estimation_dates = pd.date_range(
            end=validation_dates[0] - pd.offsets.Day(1),
            periods=10,
            freq="D",
        )

    return OOSResult(
        mean=mean,
        actual=actual,
        lower=lower,
        upper=upper,
        estimation_indices=np.arange(10),
        validation_indices=np.arange(10, 10 + len(mean)),
        estimation_dates=estimation_dates,
        validation_dates=validation_dates,
        model_type="TEST",
        target=target,
        series_names=series_names,
        alpha=alpha,
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


def test_evaluate_models_oos_returns_all_metrics_and_shared_periods():
    """One batch call evaluates and ranks every model on one holdout window."""
    data = np.arange(15.0)
    models = {
        "weak": _MeanModel(data),
        "strong": _MeanModel(data, forecast_bias=6.5),
    }

    report = evaluate_models_oos(
        models,
        estimation_period=(0, 9),
        validation_period=(10, 14),
        rank_by="rmse",
    )

    assert isinstance(report, OOSComparisonResult)
    assert report.ranking == ["strong", "weak"]
    assert report.best_model == "strong"
    assert report.target == "observed"
    assert report.table.index.tolist() == ["strong", "weak"]
    assert report.table.columns.tolist() == [
        "mae",
        "mse",
        "rmse",
        "mape",
        "smape",
        "theil_u1",
        "n",
        "rank",
    ]
    assert report.table.loc["strong", "rmse"] == pytest.approx(np.sqrt(3.0))
    assert report.table.loc["strong", "n"] == 5
    assert report.table.loc["strong", "rank"] == 1
    for evaluation in report.evaluations.values():
        np.testing.assert_array_equal(evaluation.estimation_indices, np.arange(10))
        np.testing.assert_array_equal(
            evaluation.validation_indices,
            np.arange(10, 15),
        )
    assert all(model.result_ is None for model in models.values())


def test_oos_forwards_explicit_fit_method():
    """Direct OOS evaluation forwards an explicit optimizer to the clone."""
    model = _OptimizerMeanModel(np.arange(15.0))

    result = oos(
        model,
        estimation_period=(0, 9),
        validation_period=(10, 14),
        method="lbfgs",
    )

    assert result.mean.shape == (5,)
    assert model.methods == ["lbfgs"]


def test_evaluate_models_oos_forwards_method_to_every_model():
    """Batch OOS evaluation applies one explicit optimizer to every model."""
    data = np.arange(15.0)
    first = _OptimizerMeanModel(data)
    second = _OptimizerMeanModel(data, forecast_bias=6.5)

    report = evaluate_models_oos(
        {"first": first, "second": second},
        estimation_period=(0, 9),
        validation_period=(10, 14),
        method="powell",
    )

    assert report.table.shape[0] == 2
    assert first.methods == ["powell"]
    assert second.methods == ["powell"]


def test_evaluate_models_oos_rejects_unsupported_method_before_fitting():
    """An incompatible named model prevents every fit in the batch."""
    data = np.arange(15.0)
    supported = _OptimizerMeanModel(data)
    unsupported = _MeanModel(data)

    with pytest.raises(TypeError, match=r"unsupported.*method"):
        evaluate_models_oos(
            {"supported": supported, "unsupported": unsupported},
            estimation_period=(0, 9),
            validation_period=(10, 14),
            method="lbfgs",
        )

    assert supported.methods == []
    assert unsupported.fit_windows == []


def test_oos_records_requested_alpha_even_without_intervals():
    """The OOS result retains the requested level independently of bounds."""
    result = oos(
        _MeanModel(np.arange(15.0)),
        estimation_period=(0, 9),
        validation_period=(10, 14),
        alpha=0.10,
    )

    assert result.alpha == pytest.approx(0.10)


def test_forecast_table_uses_all_models_and_excludes_intervals_by_default():
    """The automatic table keeps mapping order and forecast-minus-actual errors."""
    actual = np.array([10.0, 12.0, 14.0])
    lower = np.array([9.0, 10.0, 11.0])
    upper = np.array([13.0, 14.0, 15.0])
    report = OOSComparisonResult(
        {
            "model-a": _oos_result(
                [11.0, 11.0, 15.0],
                actual,
                lower=lower,
                upper=upper,
                alpha=0.05,
            ),
            "model-b": _oos_result([10.0, 13.0, 13.0], actual),
            "model-c": _oos_result([9.0, 12.0, 14.0], actual),
        },
        rank_by="rmse",
    )

    frame = report.forecast_table()

    assert frame.columns.tolist() == [
        "Actual",
        "model-a forecast",
        "model-a error",
        "model-b forecast",
        "model-b error",
        "model-c forecast",
        "model-c error",
    ]
    assert frame.index.tolist() == [10, 11, 12]
    assert frame.index.name == "observation"
    np.testing.assert_allclose(
        frame["model-a error"],
        frame["model-a forecast"] - frame["Actual"],
    )
    assert not any("lower" in column or "upper" in column for column in frame)


def test_forecast_table_can_include_available_intervals_and_date_index():
    """Interval columns are opt-in and absent bounds do not create fake columns."""
    dates = pd.date_range("2025-01-01", periods=2, freq="MS", name="month")
    actual = np.array([4.0, 5.0])
    report = OOSComparisonResult(
        {
            "bounded": _oos_result(
                [4.5, 5.5],
                actual,
                lower=np.array([3.5, 4.5]),
                upper=np.array([5.5, 6.5]),
                validation_dates=dates,
                alpha=0.05,
            ),
            "point-only": _oos_result(
                [4.2, 5.2],
                actual,
                validation_dates=dates,
            ),
        },
        rank_by="rmse",
    )

    frame = report.forecast_table(
        include_errors=False,
        include_intervals=True,
    )

    assert frame.columns.tolist() == [
        "Actual",
        "bounded forecast",
        "bounded lower",
        "bounded upper",
        "point-only forecast",
    ]
    assert frame.index.equals(dates)


def test_forecast_table_selects_multivariate_series_by_name_or_position():
    """Every named vector component can be compared without model-specific code."""
    actual = np.array([[1.0, 10.0], [2.0, 20.0]])
    report = OOSComparisonResult(
        {
            "var-a": _oos_result(
                [[1.5, 11.0], [2.5, 19.0]],
                actual,
                series_names=("output", "prices"),
            ),
            "var-b": _oos_result(
                [[0.5, 9.0], [1.5, 21.0]],
                actual,
                series_names=("output", "prices"),
            ),
        },
        rank_by="rmse",
    )

    by_name = report.forecast_table(series="prices")
    by_position = report.forecast_table(series=1)

    pd.testing.assert_frame_equal(by_name, by_position)
    np.testing.assert_allclose(by_name["Actual"], [10.0, 20.0])
    with pytest.raises(ValueError, match="series is required"):
        report.forecast_table()
    with pytest.raises(ValueError, match="unknown series"):
        report.forecast_table(series="sales")
    with pytest.raises(IndexError, match="out of range"):
        report.forecast_table(series=2)


def test_plot_forecasts_draws_all_models_and_only_available_intervals():
    """The plot composes shared line styling with same-colour interval bands."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    actual = np.array([10.0, 12.0, 14.0])
    report = OOSComparisonResult(
        {
            "model-a": _oos_result(
                [11.0, 11.0, 15.0],
                actual,
                lower=np.array([9.0, 10.0, 12.0]),
                upper=np.array([13.0, 14.0, 16.0]),
                alpha=0.10,
            ),
            "model-b": _oos_result([10.0, 13.0, 13.0], actual),
        },
        rank_by="rmse",
    )

    colors = ["#222222", "#1f77b4", "#ff7f0e"]
    fig, ax = report.plot_forecasts(
        colors=colors,
        title="Holdout comparison",
        grid=True,
    )

    try:
        assert ax.get_title() == "Holdout comparison"
        assert [line.get_label() for line in ax.lines] == [
            "Actual",
            "model-a forecast",
            "model-b forecast",
        ]
        assert [line.get_color() for line in ax.lines] == colors
        assert len(ax.collections) == 1
        assert ax.collections[0].get_facecolor()[0] == pytest.approx(
            to_rgba(colors[1], alpha=0.12)
        )
        labels = ax.get_legend_handles_labels()[1]
        assert "model-a 90% interval" in labels
    finally:
        plt.close(fig)


def test_plot_forecasts_can_hide_intervals_and_reuse_an_axes():
    """Plot options remain composable with a caller-owned Matplotlib axes."""
    import matplotlib.pyplot as plt

    evaluation = _oos_result(
        [1.0, 2.0],
        [1.5, 2.5],
        lower=np.array([0.5, 1.5]),
        upper=np.array([1.5, 2.5]),
        alpha=0.05,
    )
    report = OOSComparisonResult({"model": evaluation}, rank_by="rmse")
    expected_fig, expected_ax = plt.subplots()
    expected_ax.plot([10, 11], [0.0, 0.0], label="Existing")

    fig, ax = report.plot_forecasts(ax=expected_ax, show_intervals=False)

    try:
        assert fig is expected_fig
        assert ax is expected_ax
        assert len(ax.lines) == 3
        assert len(ax.collections) == 0
    finally:
        plt.close(fig)


@pytest.mark.parametrize("interval_alpha", [-0.1, 1.1, np.nan, True])
def test_plot_forecasts_rejects_invalid_interval_opacity(interval_alpha):
    """Invalid fill opacity fails before any plot is created."""
    report = OOSComparisonResult(
        {"model": _oos_result([1.0], [1.0])},
        rank_by="rmse",
    )

    with pytest.raises((TypeError, ValueError), match="interval_alpha"):
        report.plot_forecasts(interval_alpha=interval_alpha)


def test_evaluate_models_oos_rejects_nonfinite_scoring_values():
    """A model cannot win by omitting validation predictions from its score."""
    data = np.arange(15.0)

    with pytest.raises(ValueError, match=r"missing.*non-finite"):
        evaluate_models_oos(
            {
                "complete": _MeanModel(data),
                "missing": _MeanModel(data, missing_forecast=True),
            },
            estimation_period=(0, 9),
            validation_period=(10, 14),
        )


def test_evaluate_models_oos_rejects_different_actual_values():
    """Shared bounds do not conceal models constructed from different data."""
    first = np.arange(15.0)
    second = first.copy()
    second[-1] += 1.0

    with pytest.raises(ValueError, match="same actual values"):
        evaluate_models_oos(
            {
                "first": _MeanModel(first),
                "second": _MeanModel(second),
            },
            estimation_period=(0, 9),
            validation_period=(10, 14),
        )


def test_evaluate_models_oos_validates_rank_metric_before_fitting():
    """Invalid batch configuration fails before any model estimation."""
    model = _MeanModel(np.arange(15.0))

    with pytest.raises(ValueError, match="rank_by"):
        evaluate_models_oos(
            {"mean": model},
            estimation_period=(0, 9),
            validation_period=(10, 14),
            rank_by="accuracy",
        )

    assert model.fit_windows == []


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
    from Ts.TsModels import SARIMAX
    from Ts.TsSims import simulate_sarima

    data = simulate_sarima(
        n=40,
        order=(1, 0, 0),
        ar=[0.5],
        seed=42,
        burn=50,
    ).data
    fitted = SARIMAX(data, order=(1, 0, 0)).fit()

    with pytest.raises(TypeError, match="oos_start"):
        fitted.predict(oos_start=25)


def test_oos_passes_holdout_exog_without_holdout_y():
    from Ts.TsModels import SARIMAX

    dates = pd.date_range("2020-01-01", periods=30, freq="MS")
    exog = pd.DataFrame({"x": np.arange(30.0)}, index=dates)
    rng = np.random.default_rng(2609)
    data = pd.Series(
        2.0 * exog["x"].to_numpy() + rng.normal(scale=0.01, size=len(dates)),
        index=dates,
    )
    model = SARIMAX(
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
    assert result.metrics["rmse"] < 0.05
    assert model.result_ is None


def test_record_mode_reports_missing_future_exog_dates():
    from Ts.TsModels import SARIMAX

    dates = pd.date_range("2022-01-01", periods=23, freq="MS")
    exog = pd.DataFrame(
        {"x": np.arange(23.0)},
        index=dates,
    )
    model = SARIMAX(
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


def _unified_result(
    mean,
    actual,
    *,
    splits=None,
    failures=None,
    dates=None,
    series_names=None,
):
    """Build one unified result for result-contract tests."""
    mean = np.asarray(mean, dtype=float)
    if splits is None:
        splits = RollingOrigin(
            initial_window=10,
            horizon=mean.shape[1],
            step=2,
        ).split(14, dates)
    return ForecastEvaluationResult(
        mean=mean,
        actual=np.asarray(actual, dtype=float),
        lower=mean - 0.5,
        upper=mean + 0.5,
        splits=splits,
        failures=[] if failures is None else failures,
        model_type="TEST",
        target="observed",
        dates=dates,
        series_names=series_names,
        alpha=0.05,
        uses_observed_future_exog=False,
    )


def test_unified_result_keeps_split_and_horizon_axes():
    result = _unified_result(
        [[10.0, 11.0], [12.0, 13.0]],
        [[10.5, 10.5], [12.5, 12.5]],
    )

    assert result.mean.shape == (2, 2)
    assert result.metrics["n"] == 4
    assert [split.split for split in result.splits] == [0, 1]
    result.mean[0, 0] = -99.0
    assert result.predictions.loc[0, "forecast"] == -99.0


def test_unified_result_rejects_method_specific_one_dimensional_shape():
    split = RollingOrigin(initial_window=10, horizon=2).split(12, None)

    with pytest.raises(ValueError, match="split, horizon"):
        ForecastEvaluationResult(
            mean=np.array([1.0, 2.0]),
            actual=np.array([1.0, 2.0]),
            lower=None,
            upper=None,
            splits=split,
            failures=[],
            model_type="TEST",
            target="observed",
        )


def test_unified_result_preserves_dates_and_multivariate_series():
    dates = pd.date_range("2020-01-01", periods=14, freq="MS")
    splits = RollingOrigin(initial_window=10, horizon=2, step=2).split(14, dates)
    mean = np.arange(8.0).reshape(2, 2, 2)
    result = _unified_result(
        mean,
        mean + 1.0,
        splits=splits,
        dates=dates,
        series_names=("output", "prices"),
    )

    assert result.series_names == ("output", "prices")
    assert result.predictions["series"].tolist() == [
        "output",
        "prices",
        "output",
        "prices",
        "output",
        "prices",
        "output",
        "prices",
    ]
    assert result.predictions.loc[0, "target_time"] == dates[10]


def test_unified_result_requires_failed_splits_to_be_atomic_nan_rows():
    splits = RollingOrigin(initial_window=10, horizon=2, step=2).split(14, None)
    mean = np.array([[10.0, 11.0], [np.nan, np.nan]])
    actual = np.array([[10.0, 12.0], [np.nan, np.nan]])
    failure = {"split": 1, "error_type": "RuntimeError", "message": "planned"}

    result = _unified_result(
        mean,
        actual,
        splits=splits,
        failures=[failure],
    )

    assert len(result.failures) == 1
    assert not result.predictions.loc[result.predictions["split"] == 1, "valid"].any()


def test_unified_comparison_scores_only_the_common_finite_sample():
    splits = RollingOrigin(initial_window=10, horizon=2, step=2).split(14, None)
    actual = np.array([[10.0, 12.0], [14.0, 16.0]])
    complete = _unified_result(
        [[9.0, 11.0], [13.0, 15.0]],
        actual,
        splits=splits,
    )
    partial = _unified_result(
        [[10.0, 12.0], [np.nan, np.nan]],
        [[10.0, 12.0], [np.nan, np.nan]],
        splits=splits,
        failures=[
            {"split": 1, "error_type": "RuntimeError", "message": "planned"}
        ],
    )

    report = ForecastComparisonResult(
        {"complete": complete, "partial": partial},
        rank_by="rmse",
    )

    assert report.ranking == ["partial", "complete"]
    assert report.best_model == "partial"
    assert report.table.loc["complete", "n_total"] == 4
    assert report.table.loc["complete", "n_common"] == 2
    assert report.table.loc["partial", "coverage"] == pytest.approx(0.5)
    assert report.table.loc["partial", "failures"] == 1
    assert report.table.loc["partial", "rank"] == 1


def test_unified_comparison_rejects_an_empty_common_sample():
    splits = RollingOrigin(initial_window=10, horizon=1).split(11, None)
    failed = _unified_result(
        [[np.nan]],
        [[np.nan]],
        splits=splits,
        failures=[
            {"split": 0, "error_type": "RuntimeError", "message": "planned"}
        ],
    )

    with pytest.raises(ValueError, match="common finite"):
        ForecastComparisonResult({"a": failed, "b": failed})


def test_unified_metric_tables_group_common_values_by_axis():
    mean = np.array([[10.0, 11.0], [12.0, 13.0]])
    actual = mean + 1.0
    result = _unified_result(mean, actual)
    report = ForecastComparisonResult({"model": result})

    by_horizon = report.metric_table(by="horizon")
    by_origin = report.metric_table(by="origin")

    assert by_horizon["horizon"].tolist() == [1, 2]
    assert by_origin["origin"].tolist() == [10, 12]
    assert by_horizon["rmse"].tolist() == pytest.approx([1.0, 1.0])
    assert by_origin["rmse"].tolist() == pytest.approx([1.0, 1.0])


def test_unified_comparison_predictions_include_every_model_and_error():
    result = _unified_result(
        [[10.0, 11.0], [12.0, 13.0]],
        [[11.0, 10.0], [12.0, 14.0]],
    )
    report = ForecastComparisonResult({"first": result, "second": result})

    frame = report.predictions

    assert frame.columns.tolist() == [
        "model",
        "split",
        "origin",
        "target_time",
        "horizon",
        "series",
        "actual",
        "forecast",
        "error",
        "lower",
        "upper",
        "valid",
    ]
    assert frame["model"].value_counts().to_dict() == {"first": 4, "second": 4}
    assert frame.loc[0, "error"] == pytest.approx(-1.0)


def test_evaluate_forecasts_uses_one_holdout_engine_for_multiple_models():
    data = np.arange(15.0)
    report = evaluate_forecasts(
        {
            "weak": _MeanModel(data),
            "strong": _MeanModel(data, forecast_bias=6.5),
        },
        scheme=Holdout(train=(0, 9), test=(10, 14)),
    )

    assert report.ranking == ["strong", "weak"]
    assert report.results["strong"].mean.shape == (1, 5)
    assert report.splits.loc[0, "train_start"] == 0
    assert report.splits.loc[0, "forecast_end"] == 14


def test_evaluate_forecasts_uses_exact_expanding_and_gapped_windows():
    model = _MeanModel(np.arange(20.0))

    report = evaluate_forecasts(
        {"model": model},
        scheme=RollingOrigin(
            initial_window=10,
            horizon=2,
            step=3,
            gap=1,
        ),
    )

    assert [window.tolist() for window in model.fit_windows] == [
        list(np.arange(10.0)),
        list(np.arange(13.0)),
        list(np.arange(16.0)),
    ]
    assert report.results["model"].mean.shape == (3, 2)
    assert report.splits["forecast_start"].tolist() == [11, 14, 17]


def test_evaluate_forecasts_forwards_all_fit_kwargs_to_every_split():
    model = _FitOptionsMeanModel(np.arange(16.0))

    evaluate_forecasts(
        {"model": model},
        scheme=RollingOrigin(initial_window=10, horizon=1, step=2),
        fit_kwargs={"method": "lbfgs", "maxiter": 500},
    )

    assert model.options == [("lbfgs", 500), ("lbfgs", 500), ("lbfgs", 500)]


def test_evaluate_forecasts_validates_every_fit_keyword_before_fitting():
    supported = _FitOptionsMeanModel(np.arange(15.0))
    unsupported = _MeanModel(np.arange(15.0))

    with pytest.raises(TypeError, match=r"unsupported.*maxiter"):
        evaluate_forecasts(
            {"supported": supported, "unsupported": unsupported},
            scheme=Holdout(train=(0, 9), test=(10, 14)),
            fit_kwargs={"maxiter": 500},
        )

    assert supported.options == []
    assert unsupported.fit_windows == []


def test_evaluate_forecasts_requires_explicit_observed_future_exog():
    model = _MeanModel(np.arange(15.0))
    model.exog = np.arange(15.0)[:, None]

    with pytest.raises(ValueError, match="future_exog='observed'"):
        evaluate_forecasts(
            {"conditional": model},
            scheme=Holdout(train=(0, 9), test=(10, 14)),
        )

    report = evaluate_forecasts(
        {"conditional": model},
        scheme=Holdout(train=(0, 9), test=(10, 14)),
        future_exog="observed",
    )
    assert report.results["conditional"].uses_observed_future_exog


def test_evaluate_forecasts_records_failed_splits_atomically():
    model = _FailingMeanModel(np.arange(16.0), fail_length=12)

    report = evaluate_forecasts(
        {"model": model},
        scheme=RollingOrigin(initial_window=10, horizon=1, step=2),
        on_error="record",
    )
    result = report.results["model"]

    assert len(result.failures) == 1
    assert result.failures[0]["split"] == 1
    assert np.isnan(result.mean[1]).all()
    assert np.isnan(result.actual[1]).all()
    assert report.table.loc["model", "coverage"] == pytest.approx(2 / 3)


def test_evaluate_forecasts_rejects_different_targets_before_fitting():
    first = _MeanModel(np.arange(15.0))
    second = _MeanModel(np.arange(15.0) + 1.0)

    with pytest.raises(ValueError, match="same target values"):
        evaluate_forecasts(
            {"first": first, "second": second},
            scheme=Holdout(train=(0, 9), test=(10, 14)),
        )

    assert first.fit_windows == []
    assert second.fit_windows == []
