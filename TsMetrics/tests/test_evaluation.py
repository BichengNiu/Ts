"""Tests for the unified historical forecast evaluation engine."""

from dataclasses import dataclass
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from Ts.TsMetrics import (
    ForecastComparisonResult,
    ForecastEvaluationResult,
    Holdout,
    RollingOrigin,
    evaluate_forecasts,
)
from Ts.TsModels._base import BaseModel, BaseModelResult, PredictResult


@dataclass
class _MeanResult(BaseModelResult):
    training_mean: float = 0.0
    forecast_bias: float = 0.0

    def predict(self, start=0, end=None, dynamic=False, alpha=0.05):
        del dynamic, alpha
        end = start if end is None else end
        mean = np.full(end - start + 1, self.training_mean + self.forecast_bias)
        return PredictResult(
            mean=mean,
            lower=mean - 0.5,
            upper=mean + 0.5,
            is_oos=np.ones(mean.shape, dtype=bool),
        )


class _MeanModel(BaseModel):
    _evaluation_target_name = "observed"

    def __init__(self, data, dates=None, *, forecast_bias=0.0):
        self.data = np.asarray(data, dtype=float)
        self.dates = dates
        self.forecast_bias = forecast_bias
        self.fit_windows = []
        self.result_ = None

    def fit(self):
        self.fit_windows.append(self.data.copy())
        mean = float(np.mean(self.data))
        return _MeanResult(
            model_type="MEAN",
            params={"mean": mean},
            std_errors={"mean": 0.25},
            p_values={"mean": 0.01},
            aic=0.0,
            bic=0.0,
            log_likelihood=0.0,
            residuals=self.data - mean,
            fitted_values=np.full_like(self.data, mean),
            nobs=len(self.data),
            data=self.data.copy(),
            training_mean=mean,
            forecast_bias=self.forecast_bias,
        )


class _FitOptionsModel(_MeanModel):
    def __init__(self, data):
        super().__init__(data)
        self.options = []

    def fit(self, *, method="bfgs", maxiter=100):
        self.options.append((method, maxiter))
        return super().fit()


class _FailingModel(_MeanModel):
    def __init__(self, data, fail_length):
        super().__init__(data)
        self.fail_length = fail_length

    def fit(self):
        if len(self.data) == self.fail_length:
            raise RuntimeError("planned fit failure")
        return super().fit()


def _result(mean, actual, *, splits=None, failures=None, dates=None, names=None):
    mean = np.asarray(mean, dtype=float)
    if splits is None:
        splits = RollingOrigin(
            initial_window=10, horizon=mean.shape[1], step=2
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
        series_names=names,
        alpha=0.05,
    )


def test_holdout_and_rolling_use_one_engine_and_result_shape():
    data = np.arange(16.0)
    holdout = evaluate_forecasts(
        {"model": _MeanModel(data)},
        scheme=Holdout(train=(0, 9), test=(10, 14)),
    )
    rolling = evaluate_forecasts(
        {"model": _MeanModel(data)},
        scheme=RollingOrigin(initial_window=10, horizon=2, step=2),
    )

    assert holdout.results["model"].mean.shape == (1, 5)
    assert rolling.results["model"].mean.shape == (3, 2)
    assert type(holdout) is type(rolling) is ForecastComparisonResult
    assert holdout.splits.loc[0, "window"] == "holdout"
    assert rolling.splits["window"].eq("expanding").all()


def test_multiple_models_are_ranked_on_the_same_holdout_sample():
    data = np.arange(15.0)
    report = evaluate_forecasts(
        {
            "weak": _MeanModel(data),
            "strong": _MeanModel(data, forecast_bias=6.5),
        },
        scheme=Holdout(train=(0, 9), test=(10, 14)),
    )

    assert report.ranking == ["strong", "weak"]
    assert report.best_model == "strong"
    assert report.table["n_common"].tolist() == [5, 5]


def test_gap_forecasts_through_unscored_period_and_scores_only_targets():
    model = _MeanModel(np.arange(20.0))
    report = evaluate_forecasts(
        {"model": model},
        scheme=RollingOrigin(initial_window=10, horizon=2, step=3, gap=1),
    )

    assert [window.tolist() for window in model.fit_windows] == [
        list(np.arange(10.0)),
        list(np.arange(13.0)),
        list(np.arange(16.0)),
    ]
    assert report.splits["forecast_start"].tolist() == [11, 14, 17]
    assert report.results["model"].actual[0].tolist() == [11.0, 12.0]


def test_fixed_rolling_window_never_expands():
    model = _MeanModel(np.arange(18.0))
    evaluate_forecasts(
        {"model": model},
        scheme=RollingOrigin(
            initial_window=12,
            horizon=1,
            step=2,
            window="rolling",
            window_size=10,
        ),
    )
    assert [len(window) for window in model.fit_windows] == [10, 10, 10]


def test_fit_kwargs_are_validated_for_all_models_before_fitting():
    supported = _FitOptionsModel(np.arange(15.0))
    unsupported = _MeanModel(np.arange(15.0))
    with pytest.raises(TypeError, match=r"unsupported.*maxiter"):
        evaluate_forecasts(
            {"supported": supported, "unsupported": unsupported},
            scheme=Holdout(train=(0, 9), test=(10, 14)),
            fit_kwargs={"maxiter": 500},
        )
    assert supported.options == []
    assert unsupported.fit_windows == []


def test_fit_kwargs_are_forwarded_to_every_split():
    model = _FitOptionsModel(np.arange(16.0))
    evaluate_forecasts(
        {"model": model},
        scheme=RollingOrigin(initial_window=10, horizon=1, step=2),
        fit_kwargs={"method": "lbfgs", "maxiter": 500},
    )
    assert model.options == [("lbfgs", 500)] * 3


def test_observed_future_exog_requires_an_explicit_conditional_policy():
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


def test_record_mode_keeps_a_failed_split_as_one_atomic_nan_row():
    report = evaluate_forecasts(
        {"model": _FailingModel(np.arange(16.0), fail_length=12)},
        scheme=RollingOrigin(initial_window=10, horizon=1, step=2),
        on_error="record",
    )
    result = report.results["model"]
    assert result.failures[0]["split"] == 1
    assert np.isnan(result.mean[1]).all()
    assert np.isnan(result.actual[1]).all()
    assert report.table.loc["model", "coverage"] == pytest.approx(2 / 3)


def test_record_mode_returns_a_failure_report_when_every_split_fails():
    report = evaluate_forecasts(
        {"model": _FailingModel(np.arange(11.0), fail_length=10)},
        scheme=RollingOrigin(initial_window=10),
        on_error="record",
    )
    assert report.best_model is None
    assert len(report.failures) == 1
    assert report.table.loc["model", "n_common"] == 0
    assert np.isnan(report.table.loc["model", "rmse"])


def test_models_must_share_target_values_before_any_fit():
    first = _MeanModel(np.arange(15.0))
    second = _MeanModel(np.arange(15.0) + 1)
    with pytest.raises(ValueError, match="same target values"):
        evaluate_forecasts(
            {"first": first, "second": second},
            scheme=Holdout(train=(0, 9), test=(10, 14)),
        )
    assert first.fit_windows == second.fit_windows == []


def test_dates_are_preserved_in_splits_and_predictions():
    dates = pd.date_range("2020-01-01", periods=15, freq="MS")
    report = evaluate_forecasts(
        {"model": _MeanModel(np.arange(15.0), dates=dates)},
        scheme=Holdout(
            train=(dates[0], dates[9]), test=(dates[10], dates[14])
        ),
    )
    assert report.splits.loc[0, "forecast_start"] == dates[10]
    assert report.predictions.loc[0, "target_time"] == dates[10]


def test_result_contract_keeps_axes_and_long_form_values():
    result = _result(
        [[10.0, 11.0], [12.0, 13.0]],
        [[10.5, 10.5], [12.5, 12.5]],
    )
    assert result.mean.shape == (2, 2)
    assert result.metrics["n"] == 4
    assert result.predictions["horizon"].tolist() == [1, 2, 1, 2]
    assert result.split_table["forecast_start"].tolist() == [10, 12]


def test_result_rejects_method_specific_one_dimensional_shape():
    splits = RollingOrigin(initial_window=10, horizon=2).split(12)
    with pytest.raises(ValueError, match="split, horizon"):
        ForecastEvaluationResult(
            mean=np.array([1.0, 2.0]),
            actual=np.array([1.0, 2.0]),
            lower=None,
            upper=None,
            splits=splits,
            failures=[],
            model_type="TEST",
            target="observed",
        )


def test_multivariate_result_keeps_series_axis_and_names():
    dates = pd.date_range("2020-01-01", periods=14, freq="MS")
    splits = RollingOrigin(initial_window=10, horizon=2, step=2).split(14, dates)
    mean = np.arange(8.0).reshape(2, 2, 2)
    result = _result(
        mean, mean + 1, splits=splits, dates=dates, names=("output", "prices")
    )
    assert result.predictions["series"].tolist() == [
        "output", "prices", "output", "prices",
        "output", "prices", "output", "prices",
    ]


def test_comparison_uses_only_the_common_finite_sample():
    splits = RollingOrigin(initial_window=10, horizon=2, step=2).split(14)
    actual = np.array([[10.0, 12.0], [14.0, 16.0]])
    complete = _result([[9.0, 11.0], [13.0, 15.0]], actual, splits=splits)
    partial = _result(
        [[10.0, 12.0], [np.nan, np.nan]],
        [[10.0, 12.0], [np.nan, np.nan]],
        splits=splits,
        failures=[{"split": 1, "error_type": "RuntimeError", "message": "x"}],
    )
    report = ForecastComparisonResult({"complete": complete, "partial": partial})
    assert report.ranking == ["partial", "complete"]
    assert report.table.loc["complete", "n_common"] == 2
    assert report.table.loc["partial", "coverage"] == pytest.approx(0.5)


def test_grouped_metrics_and_long_predictions_cover_all_models():
    result = _result(
        [[10.0, 11.0], [12.0, 13.0]],
        [[11.0, 10.0], [12.0, 14.0]],
    )
    report = ForecastComparisonResult({"first": result, "second": result})
    assert report.metric_table(by="horizon")["horizon"].tolist() == [1, 2, 1, 2]
    assert report.metric_table(by="origin")["origin"].tolist() == [10, 12, 10, 12]
    assert report.predictions["model"].value_counts().to_dict() == {
        "first": 4, "second": 4
    }


def test_rolling_parameter_table_tracks_each_training_sample_range():
    report = evaluate_forecasts(
        {"model": _MeanModel(np.arange(16.0))},
        scheme=RollingOrigin(initial_window=10, horizon=1, step=2),
    )

    table = report.parameter_table(model="model", parameters="mean")

    assert table["split"].tolist() == [0, 1, 2]
    assert table["train_start"].tolist() == [0, 0, 0]
    assert table["train_end"].tolist() == [9, 11, 13]
    assert table["n_train"].tolist() == [10, 12, 14]
    assert table["estimate"].tolist() == pytest.approx([4.5, 5.5, 6.5])
    assert table["std_error"].tolist() == pytest.approx([0.25] * 3)
    assert table["p_value"].tolist() == pytest.approx([0.01] * 3)


def test_parameter_table_keeps_failed_sample_range_as_nan():
    report = evaluate_forecasts(
        {"model": _FailingModel(np.arange(16.0), fail_length=12)},
        scheme=RollingOrigin(initial_window=10, horizon=1, step=2),
        on_error="record",
    )

    table = report.parameter_table(model="model", parameters="mean")

    assert table["split"].tolist() == [0, 1, 2]
    assert np.isnan(table.loc[table["split"] == 1, "estimate"]).all()


def test_parameter_api_validates_model_and_parameter_names():
    report = evaluate_forecasts(
        {"model": _MeanModel(np.arange(12.0))},
        scheme=RollingOrigin(initial_window=10),
    )

    with pytest.raises(ValueError, match="unknown model"):
        report.parameter_table(model="missing")
    with pytest.raises(ValueError, match="unknown parameter"):
        report.parameter_table(model="model", parameters="missing")


def test_parameter_plot_reuses_shared_series_plotting_contract():
    report = evaluate_forecasts(
        {"model": _MeanModel(np.arange(16.0))},
        scheme=RollingOrigin(initial_window=10, horizon=1, step=2),
    )

    fig, ax = report.plot_parameters(
        model="model",
        parameters="mean",
        title="Rolling mean estimate",
    )

    assert ax.get_title() == "Rolling mean estimate"
    assert ax.get_xlabel() == "Training sample end"
    assert ax.get_ylabel() == "Estimate"
    assert ax.lines[0].get_xdata().tolist() == [9, 11, 13]
    assert ax.lines[0].get_ydata().tolist() == pytest.approx([4.5, 5.5, 6.5])
    plt.close(fig)


def test_plots_reuse_shared_series_plotting_contract():
    result = _result(
        [[10.0, 11.0], [12.0, 13.0]],
        [[10.5, 10.5], [12.5, 12.5]],
    )
    report = ForecastComparisonResult({"model": result})
    with pytest.raises(ValueError, match="horizon is required"):
        report.plot_forecasts()
    fig, ax = report.plot_forecasts(horizon=1, title="Rolling forecasts")
    assert ax.get_title() == "Rolling forecasts"
    assert len(ax.lines) == 2
    plt.close(fig)
    fig, ax = report.plot_metric("rmse", by="origin", ytitle="RMSE")
    assert ax.get_ylabel() == "RMSE"
    plt.close(fig)


def test_structural_protocol_is_supported_without_inheritance():
    class Fitted:
        model_type = "DUCK"

        def __init__(self, nobs):
            self.nobs = nobs

        def predict(self, *, start, end, alpha, **kwargs):
            del start, alpha, kwargs
            return SimpleNamespace(
                mean=np.zeros(end - self.nobs + 1), lower=None, upper=None
            )

    class Duck:
        _evaluation_target_name = "observed"

        def __init__(self, data):
            self.data = np.asarray(data, dtype=float)
            self.dates = None

        def _clone_for_evaluation(self, data, exog=None, *, dates=None):
            del exog, dates
            return type(self)(data)

        def _evaluation_actual(self, observed, train_data):
            del train_data
            return np.asarray(observed)

        def _evaluation_predict_kwargs(self, start, stop):
            del start, stop
            return {}

        def _validate_evaluation(self, context):
            del context

        def fit(self):
            return Fitted(len(self.data))

    report = evaluate_forecasts(
        {"duck": Duck(np.arange(12.0))},
        scheme=Holdout(train=(0, 9), test=(10, 11)),
    )
    assert report.results["duck"].mean.shape == (1, 2)
    assert report.parameter_table(model="duck").empty
