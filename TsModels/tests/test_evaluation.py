"""Cross-cutting model integration tests for unified forecast evaluation."""

from dataclasses import dataclass

import numpy as np
import pytest

from Ts.TsMetrics import Holdout, RollingOrigin, compute_metrics, evaluate_forecasts
from Ts.TsModels import BackcastResult
from Ts.TsModels._base import BaseModel, BaseModelResult, PredictResult


def test_backcast_result_uses_negative_chronological_indices():
    result = BackcastResult(
        mean=np.array([1.0, 2.0, 3.0]),
        lower=None,
        upper=None,
        indices=np.array([-3, -2, -1]),
        model_type="SARIMAX",
        target="observed",
    )
    assert result.indices.tolist() == [-3, -2, -1]


def test_backcast_result_rejects_wrong_index_length():
    with pytest.raises(ValueError, match="indices"):
        BackcastResult(
            mean=np.ones(3),
            lower=None,
            upper=None,
            indices=np.array([-2, -1]),
            model_type="SARIMAX",
            target="observed",
        )


def test_metrics_filter_nonfinite_pairs_and_flatten_axes():
    metrics = compute_metrics(
        np.array([[1.0, np.nan], [3.0, 4.0]]),
        np.array([[1.0, 2.0], [5.0, 5.0]]),
    )
    assert metrics["n"] == 3
    assert metrics["mae"] == pytest.approx(1.0)


def test_clone_replaces_data_exog_and_fitted_state_without_mutating_caller():
    class ConfiguredModel:
        pass

    original = ConfiguredModel()
    original.data = np.arange(8.0)
    original.exog = np.arange(16.0).reshape(8, 2)
    original.order = (2, 0, 1)
    original.result_ = object()
    cloned = BaseModel._clone_for_evaluation(
        original, original.data[:4], exog=original.exog[:4]
    )
    cloned.exog[0, 0] = -1
    assert cloned.order == original.order
    assert cloned.result_ is None
    assert original.exog[0, 0] == 0
    assert original.result_ is not None


@dataclass
class _MeanResult(BaseModelResult):
    fit_mean: float = 0.0

    def predict(self, start=0, end=None, dynamic=False, alpha=0.05):
        del dynamic, alpha
        end = start if end is None else end
        mean = np.full(end - start + 1, self.fit_mean)
        return PredictResult(
            mean=mean,
            lower=mean - 0.5,
            upper=mean + 0.5,
            is_oos=np.ones(mean.shape, dtype=bool),
        )


class _MeanModel(BaseModel):
    def __init__(self, data, fail_length=None):
        self.data = np.asarray(data, dtype=float)
        self.fail_length = fail_length
        self.fit_windows = []
        self.result_ = None

    def fit(self):
        if len(self.data) == self.fail_length:
            raise RuntimeError("planned failure")
        self.fit_windows.append(self.data.copy())
        mean = float(np.mean(self.data))
        return _MeanResult(
            model_type="MEAN",
            params={},
            std_errors={},
            p_values={},
            aic=0.0,
            bic=0.0,
            log_likelihood=0.0,
            residuals=self.data - mean,
            fitted_values=np.full_like(self.data, mean),
            nobs=len(self.data),
            data=self.data.copy(),
            fit_mean=mean,
        )


def test_unified_rolling_refits_only_the_declared_expanding_windows():
    model = _MeanModel(np.arange(20.0))
    report = evaluate_forecasts(
        {"model": model},
        scheme=RollingOrigin(initial_window=10, horizon=3, step=4),
    )
    assert [window.tolist() for window in model.fit_windows] == [
        list(np.arange(10.0)), list(np.arange(14.0))
    ]
    assert report.results["model"].mean.shape == (2, 3)
    assert model.result_ is None


def test_unified_rolling_supports_fixed_window_and_failure_records():
    model = _MeanModel(np.arange(18.0), fail_length=10)
    with pytest.raises(RuntimeError, match="planned"):
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
    report = evaluate_forecasts(
        {"model": _MeanModel(np.arange(18.0), fail_length=14)},
        scheme=RollingOrigin(initial_window=12, horizon=1, step=2),
        on_error="record",
    )
    assert len(report.failures) == 1


def test_unified_holdout_scores_only_the_declared_test_period():
    report = evaluate_forecasts(
        {"model": _MeanModel(np.arange(20.0))},
        scheme=Holdout(train=(2, 11), test=(15, 18)),
    )
    result = report.results["model"]
    assert result.splits[0].train_indices.tolist() == list(range(2, 12))
    assert result.splits[0].target_indices.tolist() == list(range(15, 19))
    assert result.actual.tolist() == [[15.0, 16.0, 17.0, 18.0]]
