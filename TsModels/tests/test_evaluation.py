"""Tests for rolling-origin backtesting and reverse-time backcasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from Ts.TsMetrics import BacktestResult, compute_metrics
from Ts.TsModels import BackcastResult
from Ts.TsModels._base import BaseModel, BaseModelResult, PredictResult


class TestEvaluationResultContracts:
    """Validate result shapes and public data contracts."""

    def test_result_types_are_public(self):
        '''Evaluation result containers are exported by their owning packages.'''
        from Ts.TsMetrics import BacktestResult
        from Ts.TsModels import BackcastResult

        assert BacktestResult.__name__ == 'BacktestResult'
        assert BackcastResult.__name__ == 'BackcastResult'

    def test_backtest_result_keeps_origin_and_horizon_axes(self):
        """BacktestResult preserves forecast-origin and horizon dimensions.

        covers: TsMetrics/_results.py::BacktestResult [class]
        """
        result = BacktestResult(
            mean=np.array([[1.0, 2.0], [3.0, 4.0]]),
            actual=np.array([[1.5, 2.5], [3.5, 4.5]]),
            lower=None,
            upper=None,
            origins=np.array([20, 21]),
            target_indices=np.array([[20, 21], [21, 22]]),
            metrics={},
            metrics_by_horizon=[],
            metrics_by_series=[],
            failures=[],
            model_type="SARIMA",
            window="expanding",
            target="observed",
        )

        assert result.mean.shape == (2, 2)
        assert result.target_indices.tolist() == [[20, 21], [21, 22]]

    def test_backtest_result_rejects_incompatible_shapes(self):
        """BacktestResult rejects actual arrays that cannot match predictions.

        covers: TsMetrics/_results.py::BacktestResult.__post_init__ [function]
        """
        with pytest.raises(ValueError, match="actual"):
            BacktestResult(
                mean=np.ones((2, 2)),
                actual=np.ones((3, 2)),
                lower=None,
                upper=None,
                origins=np.array([20, 21]),
                target_indices=np.array([[20, 21], [21, 22]]),
                metrics={},
                metrics_by_horizon=[],
                metrics_by_series=[],
                failures=[],
                model_type="SARIMA",
                window="expanding",
                target="observed",
            )

    def test_backcast_result_uses_negative_chronological_indices(self):
        """BackcastResult stores chronological negative pre-sample indices.

        covers: TsModels/_backcast.py::BackcastResult [class]
        """
        result = BackcastResult(
            mean=np.array([1.0, 2.0, 3.0]),
            lower=None,
            upper=None,
            indices=np.array([-3, -2, -1]),
            model_type="SARIMA",
            target="observed",
        )

        assert result.indices.tolist() == [-3, -2, -1]

    def test_backcast_result_rejects_wrong_index_length(self):
        """BackcastResult requires one index per backcast period.

        covers: TsModels/_backcast.py::BackcastResult.__post_init__ [function]
        """
        with pytest.raises(ValueError, match="indices"):
            BackcastResult(
                mean=np.ones(3),
                lower=None,
                upper=None,
                indices=np.array([-2, -1]),
                model_type="SARIMA",
                target="observed",
            )


class TestEvaluationMetrics:
    """Check finite-pair filtering and degenerate metric inputs."""

    def test_metrics_ignore_nan_pairs(self):
        """Only positions with finite actual and prediction values are scored.

        covers: TsMetrics/_metrics.py::compute_metrics [function]
        """
        metrics = compute_metrics(
            np.array([1.0, np.nan, 3.0]),
            np.array([1.0, 2.0, 5.0]),
        )

        assert metrics["n"] == 2
        assert metrics["rmse"] == pytest.approx(np.sqrt(2.0))
        assert metrics["mae"] == pytest.approx(1.0)

    def test_metrics_handle_all_missing_pairs(self):
        """An all-missing comparison returns NaN metrics and n=0."""
        metrics = compute_metrics(
            np.array([np.nan, np.nan]),
            np.array([1.0, np.nan]),
        )

        assert metrics["n"] == 0
        assert np.isnan(metrics["rmse"])
        assert np.isnan(metrics["mape"])

    def test_metrics_handle_all_zero_actuals(self):
        """MAPE is NaN when all actual values are zero."""
        metrics = compute_metrics(
            np.zeros(3),
            np.array([0.0, 1.0, 2.0]),
        )

        assert metrics["n"] == 3
        assert np.isnan(metrics["mape"])

    def test_metrics_flatten_multivariate_arrays(self):
        """Metric calculation treats every finite series-period pair equally."""
        actual = np.array([[1.0, 2.0], [3.0, 4.0]])
        predicted = actual + 1.0

        metrics = compute_metrics(actual, predicted)

        assert metrics["n"] == 4
        assert metrics["rmse"] == pytest.approx(1.0)
        assert metrics["mae"] == pytest.approx(1.0)


class TestModelClone:
    """Verify model-window cloning never mutates the caller."""

    def test_clone_replaces_data_and_clears_result(self):
        """_clone_with_data preserves configuration but clears fitted state.

        covers: TsModels/_base.py::_clone_for_evaluation [function]
        """

        class ConfiguredModel:
            pass

        original = ConfiguredModel()
        original.data = np.arange(8.0)
        original.order = (2, 0, 1)
        original.result_ = object()

        cloned = BaseModel._clone_for_evaluation(
            original,
            np.arange(4.0),
        )

        assert cloned is not original
        assert cloned.order == (2, 0, 1)
        assert cloned.result_ is None
        np.testing.assert_array_equal(cloned.data, np.arange(4.0))
        np.testing.assert_array_equal(original.data, np.arange(8.0))
        assert original.result_ is not None

    def test_clone_slices_exog_independently(self):
        """_clone_with_data copies a supplied exogenous-data window."""

        class ExogenousModel:
            pass

        original = ExogenousModel()
        original.data = np.arange(8.0)
        original.exog = np.arange(16.0).reshape(8, 2)
        original.result_ = None
        exog_window = original.exog[:4]

        cloned = BaseModel._clone_for_evaluation(
            original,
            original.data[:4],
            exog=exog_window,
        )
        cloned.exog[0, 0] = -1.0

        assert original.exog[0, 0] == 0.0


@dataclass
class _MeanForecastResult(BaseModelResult):
    """Small deterministic result used to inspect backtest windows."""

    fit_mean: float = 0.0
    horizon_to_fail: int | None = None

    def predict(
        self,
        start=0,
        end=None,
        dynamic=False,
        alpha=0.05,
    ):
        """Forecast the training mean for every requested period."""
        del dynamic, alpha
        if end is None:
            end = start
        horizon = end - start + 1
        if self.horizon_to_fail == horizon:
            raise RuntimeError("planned forecast failure")
        mean = np.full(horizon, self.fit_mean)
        return PredictResult(
            mean=mean,
            lower=mean - 0.5,
            upper=mean + 0.5,
            is_oos=np.ones(horizon, dtype=bool),
        )


class _MeanForecastModel(BaseModel):
    """Deterministic model whose forecasts reveal its training window."""

    def __init__(self, data, fail_train_length=None):
        self.data = np.asarray(data, dtype=float)
        self.fail_train_length = fail_train_length
        self.fit_windows = []
        self.result_ = None

    def fit(self):
        """Fit by storing and averaging the active data window."""
        if self.fail_train_length == len(self.data):
            raise RuntimeError("planned fit failure")
        self.fit_windows.append(self.data.copy())
        result = _MeanForecastResult(
            model_type="MEAN",
            params={},
            std_errors={},
            p_values={},
            aic=0.0,
            bic=0.0,
            log_likelihood=0.0,
            residuals=self.data - np.mean(self.data),
            fitted_values=np.full_like(self.data, np.mean(self.data)),
            nobs=len(self.data),
            data=self.data.copy(),
            fit_mean=float(np.mean(self.data)),
        )
        self.result_ = result
        return result


class TestBacktestBehavior:
    '''Verify leakage-free rolling-origin orchestration.'''

    def test_expanding_windows_use_only_pre_origin_data(self):
        '''Each expanding fit ends immediately before its forecast origin.

        covers: TsMetrics/_backtest.py::backtest [function]
        covers: TsModels/_base.py::BaseModel.backtest [function]
        '''
        model = _MeanForecastModel(np.arange(20.0))

        result = model.backtest(
            initial_window=10,
            horizon=2,
            step=3,
        )

        assert result.origins.tolist() == [10, 13, 16]
        assert [len(window) for window in model.fit_windows] == [10, 13, 16]
        assert [window[-1] for window in model.fit_windows] == [9.0, 12.0, 15.0]
        assert result.target_indices.tolist() == [
            [10, 11],
            [13, 14],
            [16, 17],
        ]
        assert result.mean.shape == (3, 2)
        np.testing.assert_allclose(result.mean[0], np.array([4.5, 4.5]))
        np.testing.assert_allclose(result.actual[0], np.array([10.0, 11.0]))

    def test_rolling_windows_keep_the_requested_size(self):
        '''Rolling fits discard observations older than window_size.'''
        model = _MeanForecastModel(np.arange(20.0))

        result = model.backtest(
            initial_window=10,
            horizon=1,
            step=4,
            window='rolling',
            window_size=10,
        )

        assert result.origins.tolist() == [10, 14, 18]
        assert [len(window) for window in model.fit_windows] == [10, 10, 10]
        np.testing.assert_array_equal(model.fit_windows[1], np.arange(4.0, 14.0))
        np.testing.assert_array_equal(model.fit_windows[2], np.arange(8.0, 18.0))

    def test_confidence_intervals_keep_origin_horizon_shape(self):
        '''Forecast confidence arrays align with point predictions.'''
        model = _MeanForecastModel(np.arange(18.0))

        result = model.backtest(initial_window=10, horizon=3, step=4)

        assert result.lower.shape == result.mean.shape
        assert result.upper.shape == result.mean.shape
        np.testing.assert_allclose(result.upper - result.lower, 1.0)

    def test_record_mode_keeps_failed_origin_as_nan(self):
        '''on_error=record records a fit failure without shifting later rows.'''
        model = _MeanForecastModel(
            np.arange(18.0),
            fail_train_length=12,
        )

        result = model.backtest(
            initial_window=10,
            horizon=1,
            step=2,
            on_error='record',
        )

        assert result.origins.tolist() == [10, 12, 14, 16]
        assert np.isnan(result.mean[1, 0])
        assert result.failures == [
            {
                'origin': 12,
                'error_type': 'RuntimeError',
                'message': 'planned fit failure',
            }
        ]
        assert result.metrics['n'] == 3

    def test_raise_mode_propagates_window_failure(self):
        '''on_error=raise does not hide a failed refit.'''
        model = _MeanForecastModel(
            np.arange(18.0),
            fail_train_length=12,
        )

        with pytest.raises(RuntimeError, match='planned fit failure'):
            model.backtest(initial_window=10, horizon=1, step=2)

    def test_backtest_does_not_replace_existing_result(self):
        '''Backtesting leaves the caller fitted state untouched.'''
        model = _MeanForecastModel(np.arange(16.0))
        sentinel = object()
        model.result_ = sentinel

        model.backtest(initial_window=10, horizon=1, step=3)

        assert model.result_ is sentinel

    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            ({'initial_window': 9}, 'initial_window'),
            ({'initial_window': 10, 'horizon': 0}, 'horizon'),
            ({'initial_window': 10, 'step': 0}, 'step'),
            ({'initial_window': 10, 'window': 'bad'}, 'window'),
            (
                {
                    'initial_window': 10,
                    'window': 'rolling',
                    'window_size': 9,
                },
                'window_size',
            ),
            ({'initial_window': 10, 'alpha': 1.0}, 'alpha'),
            ({'initial_window': 10, 'on_error': 'ignore'}, 'on_error'),
            ({'initial_window': 18, 'horizon': 3}, 'initial_window'),
        ],
    )
    def test_invalid_backtest_arguments_raise(self, kwargs, message):
        '''Backtest validation names the invalid public argument.

        covers: TsMetrics/_backtest.py::_validate_backtest_args [function]
        '''
        model = _MeanForecastModel(np.arange(20.0))

        with pytest.raises(ValueError, match=message):
            model.backtest(**kwargs)


class TestBacktestModelIntegration:
    '''Smoke-test every predictive model family through the shared engine.'''

    @staticmethod
    def _univariate_data():
        '''Return deterministic AR(1) data.'''
        from Ts.TsSims import simulate_sarima

        return simulate_sarima(
            n=80,
            order=(1, 0, 0),
            ar=[0.6],
            seed=42,
            burn=100,
        ).data

    @staticmethod
    def _multivariate_data():
        '''Return deterministic cointegrated bivariate data.'''
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.standard_normal(80))
        y = 2.0 * x + 0.5 * rng.standard_normal(80)
        return np.column_stack([y, x])

    def test_sarima_backtest(self):
        '''SARIMA produces finite multistep rolling-origin forecasts.'''
        from Ts.TsModels import SARIMA

        model = SARIMA(self._univariate_data(), order=(1, 0, 0))
        result = model.backtest(initial_window=60, horizon=2, step=10)

        assert result.mean.shape == (2, 2)
        assert np.isfinite(result.mean).all()
        assert result.target == 'observed'

    def test_var_backtest(self):
        '''VAR preserves the series axis in backtest output.'''
        from Ts.TsModels import VAR

        model = VAR(self._multivariate_data(), lags=1)
        result = model.backtest(initial_window=60, horizon=2, step=10)

        assert result.mean.shape == (2, 2, 2)
        assert len(result.metrics_by_series) == 2
        assert np.isfinite(result.mean).all()

    def test_vecm_backtest(self):
        '''VECM refits and forecasts cointegrated data at each origin.'''
        from Ts.TsModels import VECM

        model = VECM(self._multivariate_data(), lags=2, coint_rank=1)
        result = model.backtest(initial_window=60, horizon=1, step=10)

        assert result.mean.shape == (2, 1, 2)
        assert np.isfinite(result.mean).all()

    def test_svar_backtest(self):
        '''SVAR inherits the shared evaluation path through BaseModel.'''
        from Ts.TsModels import SVAR

        restrictions = np.array([[np.nan, 0.0], [np.nan, np.nan]])
        model = SVAR(
            self._multivariate_data(),
            lags=1,
            C_lr=restrictions,
        )
        result = model.backtest(initial_window=60, horizon=1, step=10)

        assert result.mean.shape == (2, 1, 2)
        assert np.isfinite(result.mean).all()

    def test_garch_backtest_uses_labelled_volatility_proxy(self):
        '''GARCH compares sigma forecasts with a labelled return proxy.'''
        from Ts.TsModels import GARCH
        from Ts.TsSims import simulate_garch

        data = simulate_garch(
            n=90,
            p=1,
            q=1,
            omega=0.1,
            alpha=[0.2],
            beta=[0.7],
            seed=42,
            burn=100,
        ).data
        model = GARCH(data, p=1, q=1, compare_lags=False)
        result = model.backtest(initial_window=70, horizon=2, step=10)

        assert result.mean.shape == (2, 2)
        assert result.target == 'absolute_demeaned_return_proxy'
        assert np.all(result.mean > 0.0)

    def test_auto_sarima_backtest_with_single_candidate(self):
        '''AutoSARIMA can reselect its single candidate in each window.'''
        from Ts.TsModels import AutoSARIMA

        model = AutoSARIMA(
            self._univariate_data(),
            p=(1, 1),
            d=(0, 0),
            q=(0, 0),
            P=(0, 0),
            D=(0, 0),
            Q=(0, 0),
        )
        result = model.backtest(initial_window=70, horizon=1, step=10)

        assert result.mean.shape == (1, 1)
        assert np.isfinite(result.mean).all()

    def test_auto_garch_backtest_with_single_candidate(self):
        '''AutoGARCH can reselect its single candidate in each window.'''
        from Ts.TsModels import AutoGARCH
        from Ts.TsSims import simulate_garch

        data = simulate_garch(
            n=80,
            p=1,
            q=1,
            omega=0.1,
            alpha=[0.2],
            beta=[0.7],
            seed=7,
            burn=100,
        ).data
        model = AutoGARCH(data, p=(1, 1), q=(1, 1), o=(0, 0))
        result = model.backtest(initial_window=70, horizon=1, step=10)

        assert result.mean.shape == (1, 1)
        assert result.target == 'absolute_demeaned_return_proxy'

    def test_arimax_backtest_aligns_training_and_future_context(
        self,
        monkeypatch,
    ):
        from Ts.TsModels import SARIMA
        from Ts.TsModels._intervention import EventSpec

        rng = np.random.default_rng(21)
        dates = pd.date_range("2021-01-01", periods=24, freq="MS")
        exog = pd.DataFrame(
            {"x": rng.normal(size=24)},
            index=dates,
        )
        event_positions = (5, 10, 15)
        event_level = np.zeros(24)
        for position in event_positions:
            event_level[position:] += 1.0
        data = pd.Series(
            1.2 * exog["x"].to_numpy()
            + 0.4 * event_level
            + rng.normal(scale=0.01, size=24),
            index=dates,
        )
        event = EventSpec(
            "policy",
            dates[list(event_positions)],
            "step",
            date_rule="exact",
        )
        model = SARIMA(
            data,
            exog=exog,
            events=[event],
            order=(0, 0, 0),
            trend="n",
        )
        original_data = model.data.copy()
        clone_records = []
        forecast_records = []
        original_clone = model._clone_for_evaluation
        original_context = model._evaluation_predict_kwargs

        def recording_clone(data, exog=None, *, dates=None):
            cloned = original_clone(
                data,
                exog=exog,
                dates=dates,
            )
            clone_records.append(
                {
                    "data": cloned.data.copy(),
                    "exog": cloned.exog.copy(),
                    "dates": cloned.dates.copy(),
                    "events": cloned.events,
                }
            )
            return cloned

        def recording_context(start, stop):
            context = original_context(start, stop)
            forecast_records.append(
                {
                    "exog": context["future_exog"].copy(),
                    "dates": context["future_dates"].copy(),
                }
            )
            return context

        monkeypatch.setattr(
            model,
            "_clone_for_evaluation",
            recording_clone,
        )
        monkeypatch.setattr(
            model,
            "_evaluation_predict_kwargs",
            recording_context,
        )

        result = model.backtest(
            initial_window=12,
            horizon=2,
            step=4,
            window="rolling",
            window_size=12,
        )

        assert result.origins.tolist() == [12, 16, 20]
        for record, start, stop in zip(
            clone_records,
            (0, 4, 8),
            (12, 16, 20),
            strict=True,
        ):
            np.testing.assert_array_equal(
                record["data"],
                model.data[start:stop],
            )
            np.testing.assert_array_equal(
                record["exog"],
                model.exog[start:stop],
            )
            assert record["dates"].equals(model.dates[start:stop])
            assert record["events"] == model.events
        for record, start in zip(
            forecast_records,
            (12, 16, 20),
            strict=True,
        ):
            np.testing.assert_array_equal(
                record["exog"],
                model.exog[start:start + 2],
            )
            assert record["dates"].equals(
                model.dates[start:start + 2]
            )
        np.testing.assert_array_equal(model.data, original_data)
        assert model.result_ is None

    def test_garch_oos_with_exog_remains_explicitly_unsupported(self):
        from Ts.TsModels import GARCH

        model = GARCH(
            np.linspace(-1.0, 1.0, 30),
            p=1,
            q=1,
            exog=np.ones((30, 1)),
        )

        with pytest.raises(NotImplementedError, match="GARCH oos.*exog"):
            model.oos(split=20)


@dataclass
class _SequenceForecastResult(BaseModelResult):
    '''Result whose forecast values expose backcast time ordering.'''

    def predict(
        self,
        start=0,
        end=None,
        dynamic=False,
        alpha=0.05,
    ):
        '''Return the sequence one through the requested horizon.'''
        del dynamic, alpha
        if end is None:
            end = start
        horizon = end - start + 1
        mean = np.arange(1.0, horizon + 1.0)
        return PredictResult(
            mean=mean,
            lower=mean - 0.25,
            upper=mean + 0.25,
            is_oos=np.ones(horizon, dtype=bool),
        )


class _SequenceForecastModel(BaseModel):
    '''Model test double that records reverse-time training data.'''

    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)
        self.fit_windows = []
        self.result_ = None

    def fit(self):
        '''Store the active data and return a sequence forecaster.'''
        self.fit_windows.append(self.data.copy())
        result = _SequenceForecastResult(
            model_type='SEQUENCE',
            params={},
            std_errors={},
            p_values={},
            aic=0.0,
            bic=0.0,
            log_likelihood=0.0,
            residuals=np.zeros_like(self.data),
            fitted_values=self.data.copy(),
            nobs=len(self.data),
            data=self.data.copy(),
        )
        self.result_ = result
        return result


class TestBackcastBehavior:
    '''Verify reverse-fit-forecast-reverse semantics.'''

    def test_backcast_reverses_training_data_and_forecast_axis(self):
        '''Backcast fits reversed data and returns chronological estimates.

        covers: TsModels/_backcast.py::backcast_model [function]
        covers: TsModels/_base.py::BaseModel.backcast [function]
        '''
        model = _SequenceForecastModel(np.arange(12.0))

        result = model.backcast(steps=3)

        np.testing.assert_array_equal(
            model.fit_windows[0],
            np.arange(11.0, -1.0, -1.0),
        )
        np.testing.assert_array_equal(result.mean, np.array([3.0, 2.0, 1.0]))
        np.testing.assert_array_equal(
            result.lower,
            np.array([2.75, 1.75, 0.75]),
        )
        np.testing.assert_array_equal(
            result.upper,
            np.array([3.25, 2.25, 1.25]),
        )
        assert result.indices.tolist() == [-3, -2, -1]
        assert result.target == 'observed'

    def test_backcast_does_not_replace_existing_result(self):
        '''Reverse-time fitting leaves the caller fitted state untouched.'''
        model = _SequenceForecastModel(np.arange(12.0))
        sentinel = object()
        model.result_ = sentinel

        model.backcast(steps=2)

        assert model.result_ is sentinel

    @pytest.mark.parametrize(
        ('kwargs', 'error_type', 'message'),
        [
            ({'steps': 0}, ValueError, 'steps'),
            ({'steps': True}, TypeError, 'steps'),
            ({'steps': 2, 'alpha': 0.0}, ValueError, 'alpha'),
            ({'steps': 2, 'alpha': np.nan}, ValueError, 'alpha'),
        ],
    )
    def test_invalid_backcast_arguments_raise(
        self,
        kwargs,
        error_type,
        message,
    ):
        '''Backcast validation names its invalid public argument.'''
        model = _SequenceForecastModel(np.arange(12.0))

        with pytest.raises(error_type, match=message):
            model.backcast(**kwargs)

    def test_var_backcast_preserves_series_axis(self):
        '''Multivariate backcasting returns steps by series.'''
        from Ts.TsModels import VAR

        data = TestBacktestModelIntegration._multivariate_data()
        result = VAR(data, lags=1).backcast(steps=3)

        assert result.mean.shape == (3, 2)
        assert result.indices.tolist() == [-3, -2, -1]
        assert np.isfinite(result.mean).all()

    def test_garch_backcast_is_positive_conditional_volatility(self):
        '''GARCH backcasts are positive and explicitly labelled volatility.'''
        from Ts.TsModels import GARCH
        from Ts.TsSims import simulate_garch

        data = simulate_garch(
            n=80,
            p=1,
            q=1,
            omega=0.1,
            alpha=[0.2],
            beta=[0.7],
            seed=17,
            burn=100,
        ).data
        result = GARCH(
            data,
            p=1,
            q=1,
            compare_lags=False,
        ).backcast(steps=3)

        assert np.all(result.mean > 0.0)
        assert result.target == 'conditional_volatility'

    @pytest.mark.parametrize('auto', [False, True])
    def test_garch_backcast_with_exog_is_rejected(self, auto):
        '''GARCH variants require explicit pre-sample exogenous values.'''
        from Ts.TsModels import GARCH, AutoGARCH

        data = np.linspace(-1.0, 1.0, 30)
        exog = np.ones((30, 1))
        if auto:
            model = AutoGARCH(
                data,
                p=(1, 1),
                q=(1, 1),
                o=(0, 0),
                exog=exog,
            )
        else:
            model = GARCH(data, p=1, q=1, exog=exog)

        with pytest.raises(NotImplementedError, match='exog'):
            model.backcast(steps=2)
