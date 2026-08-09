"""Regression tests for evaluation protocol and result invariants."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from Ts.TsMetrics import (
    BacktestResult,
    OOSComparisonResult,
    OOSResult,
    backtest,
    compare_forecasts,
    evaluate_models_oos,
    oos,
)


class _Fitted:
    """Minimal fitted forecast result."""

    model_type = "DUCK"

    def __init__(self, nobs):
        self.nobs = nobs

    def predict(self, *, start, end, alpha, **kwargs):
        """Return deterministic forecasts."""
        del start, alpha, kwargs
        horizon = end - self.nobs + 1
        return SimpleNamespace(
            mean=np.zeros(horizon),
            lower=None,
            upper=None,
        )


class _DuckModel:
    """Small structural implementation of the evaluation protocol."""

    _evaluation_target_name = "observed"

    def __init__(self, data, dates=None, fit_counter=None):
        self.data = np.asarray(data, dtype=float)
        self.dates = dates
        self.fit_counter = [] if fit_counter is None else fit_counter

    def _clone_for_evaluation(self, data, exog=None, *, dates=None):
        del exog
        return type(self)(data, dates=dates, fit_counter=self.fit_counter)

    def _evaluation_actual(self, observed, train_data):
        del train_data
        return np.asarray(observed, dtype=float)

    def _evaluation_predict_kwargs(self, start, stop):
        del start, stop
        return {}

    def _validate_evaluation(self, context):
        del context

    def fit(self):
        self.fit_counter.append(self.data.copy())
        return _Fitted(len(self.data))


def _oos_result(mean=(1.0,), actual=(1.0,)):
    """Build one valid public OOS result."""
    return OOSResult(
        mean=np.asarray(mean, dtype=float),
        actual=np.asarray(actual, dtype=float),
        lower=None,
        upper=None,
        estimation_indices=np.arange(10),
        validation_indices=np.arange(10, 10 + len(mean)),
        estimation_dates=None,
        validation_dates=None,
        model_type="TEST",
        target="observed",
    )


def test_multi_model_oos_types_are_public_from_package_root():
    """The batch workflow has identical root and subpackage entry points."""
    from Ts import OOSComparisonResult as RootOOSComparisonResult
    from Ts import evaluate_models_oos as root_evaluate_models_oos

    assert RootOOSComparisonResult is OOSComparisonResult
    assert root_evaluate_models_oos is evaluate_models_oos


def test_protocol_rejects_missing_target_before_fitting():
    """An incomplete model protocol fails before expensive model work."""
    model = _DuckModel(np.arange(12.0))
    model._evaluation_target_name = None

    with pytest.raises(TypeError, match="_evaluation_target_name"):
        oos(model, (0, 9), (10, 11))

    assert model.fit_counter == []


def test_rolling_window_is_fixed_from_the_first_origin():
    """A rolling backtest never begins with an expanding warm-up phase."""
    model = _DuckModel(np.arange(16.0))

    backtest(
        model,
        initial_window=12,
        horizon=1,
        window="rolling",
        window_size=10,
    )

    assert [len(window) for window in model.fit_counter] == [10, 10, 10, 10]


def test_rolling_window_cannot_exceed_first_origin_history():
    """A fixed window must fit inside history available at the first origin."""
    with pytest.raises(ValueError, match="<= initial_window"):
        backtest(
            _DuckModel(np.arange(16.0)),
            initial_window=10,
            horizon=1,
            window="rolling",
            window_size=12,
        )


def test_tuple_dates_are_normalised_before_training_slice():
    """Any accepted datetime-like sequence uses the validated calendar."""
    dates = tuple(np.arange("2020-01", "2021-01", dtype="datetime64[M]"))
    result = oos(
        _DuckModel(np.arange(12.0), dates=dates),
        (dates[0], dates[9]),
        (dates[10], dates[11]),
    )

    assert result.validation_dates.tolist() == list(result.validation_dates)
    assert len(result.validation_dates) == 2

    rolling = backtest(
        _DuckModel(np.arange(12.0), dates=dates),
        initial_window=10,
        horizon=2,
        step=1,
    )
    assert isinstance(rolling.dates, pd.DatetimeIndex)
    assert rolling.metrics_by_window.iloc[-1]["window_end"] == rolling.dates[-1]


def test_record_mode_commits_each_origin_atomically():
    """Late fitted-result failures cannot leave scored partial rows."""

    class InvalidFitted(_Fitted):
        @property
        def model_type(self):
            raise RuntimeError("late metadata failure")

    class InvalidModel(_DuckModel):
        def fit(self):
            return InvalidFitted(len(self.data))

    result = backtest(
        InvalidModel(np.arange(13.0)),
        initial_window=10,
        on_error="record",
    )

    assert len(result.failures) == 3
    assert np.isnan(result.mean).all()
    assert np.isnan(result.actual).all()
    assert result.metrics["n"] == 0


def test_oos_result_rejects_empty_validation_and_half_interval():
    """Public OOS results cannot represent incomplete forecast state."""
    with pytest.raises(ValueError, match="forecast value"):
        _oos_result(mean=(), actual=())

    with pytest.raises(ValueError, match="both be set"):
        OOSResult(
            **{
                **_oos_result().__dict__,
                "lower": np.array([0.5]),
            }
        )


def test_oos_result_preserves_optional_series_and_interval_metadata():
    """Multivariate names and the requested interval level are public metadata."""
    result = OOSResult(
        mean=np.ones((2, 2)),
        actual=np.ones((2, 2)),
        lower=np.zeros((2, 2)),
        upper=np.full((2, 2), 2.0),
        estimation_indices=np.arange(10),
        validation_indices=np.array([10, 11]),
        estimation_dates=None,
        validation_dates=None,
        model_type="TEST",
        target="observed",
        series_names=("output", "prices"),
        alpha=0.10,
    )

    assert result.series_names == ("output", "prices")
    assert result.alpha == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("series_names", "message"),
    [
        (("output",), "one name per series"),
        (("output", "output"), "unique"),
        (("output", ""), "non-empty"),
    ],
)
def test_oos_result_rejects_invalid_multivariate_series_names(
    series_names,
    message,
):
    """Series metadata must identify every multivariate result column."""
    with pytest.raises((TypeError, ValueError), match=message):
        OOSResult(
            mean=np.ones((2, 2)),
            actual=np.ones((2, 2)),
            lower=None,
            upper=None,
            estimation_indices=np.arange(10),
            validation_indices=np.array([10, 11]),
            estimation_dates=None,
            validation_dates=None,
            model_type="TEST",
            target="observed",
            series_names=series_names,
        )


def test_oos_result_metadata_defaults_preserve_positional_construction():
    """Appending optional fields does not break the original public signature."""
    result = OOSResult(
        np.array([1.0]),
        np.array([1.0]),
        None,
        None,
        np.arange(10),
        np.array([10]),
        None,
        None,
        "TEST",
        "observed",
    )

    assert result.series_names is None
    assert result.alpha is None


@pytest.mark.parametrize("alpha", [0.0, 1.0, np.nan, True, "bad"])
def test_oos_result_rejects_invalid_interval_level(alpha):
    """Manually constructed results enforce the evaluator's alpha contract."""
    with pytest.raises((TypeError, ValueError), match="alpha"):
        OOSResult(
            np.array([1.0]),
            np.array([1.0]),
            None,
            None,
            np.arange(10),
            np.array([10]),
            None,
            None,
            "TEST",
            "observed",
            alpha=alpha,
        )


def test_backtest_result_derives_indices_and_rejects_invalid_window():
    """Target positions are derived and window metadata is constrained."""
    values = {
        "mean": np.ones((2, 2)),
        "actual": np.ones((2, 2)),
        "lower": None,
        "upper": None,
        "origins": np.array([10, 12]),
        "failures": [],
        "model_type": "TEST",
        "window": "rolling",
        "target": "observed",
    }
    result = BacktestResult(**values)
    assert result.target_indices.tolist() == [[10, 11], [12, 13]]

    with pytest.raises(ValueError, match="window"):
        BacktestResult(**{**values, "window": "invented"})


def test_backtest_result_metadata_defaults_preserve_positional_construction():
    """Appending metadata does not change the original positional signature."""
    result = BacktestResult(
        np.ones((1, 1)),
        np.ones((1, 1)),
        None,
        None,
        np.array([10]),
        [],
        "TEST",
        "expanding",
        "observed",
    )

    assert result.dates is None
    assert result.series_names is None


def test_backtest_result_preserves_calendar_and_series_names():
    """Valid date and multivariate labels are copied into public metadata."""
    dates = pd.date_range("2020-01-01", periods=14, freq="MS")
    result = BacktestResult(
        mean=np.ones((2, 2, 2)),
        actual=np.ones((2, 2, 2)),
        lower=None,
        upper=None,
        origins=np.array([10, 12]),
        failures=[],
        model_type="TEST",
        window="expanding",
        target="observed",
        dates=dates,
        series_names=("output", "prices"),
    )

    assert result.dates.equals(dates)
    assert result.dates is not dates
    assert result.series_names == ("output", "prices")


@pytest.mark.parametrize(
    ("dates", "message"),
    [
        (pd.DatetimeIndex(["2020-01-01", None]), "missing"),
        (pd.DatetimeIndex(["2020-01-01", "2020-01-01"]), "unique"),
        (pd.DatetimeIndex(["2020-02-01", "2020-01-01"]), "increasing"),
        (pd.date_range("2020-01-01", periods=11, freq="MS"), "cover"),
    ],
)
def test_backtest_result_rejects_invalid_calendar_metadata(dates, message):
    """A result calendar must be strict and cover every forecast target."""
    with pytest.raises(ValueError, match=message):
        BacktestResult(
            mean=np.ones((1, 2)),
            actual=np.ones((1, 2)),
            lower=None,
            upper=None,
            origins=np.array([10]),
            failures=[],
            model_type="TEST",
            window="expanding",
            target="observed",
            dates=dates,
        )


@pytest.mark.parametrize(
    ("mean", "series_names", "message"),
    [
        (np.ones((1, 2)), ("output",), "multivariate"),
        (np.ones((1, 2, 2)), ("output",), "one name per series"),
        (np.ones((1, 2, 2)), ("output", "output"), "unique"),
        (np.ones((1, 2, 2)), ("output", ""), "non-empty"),
    ],
)
def test_backtest_result_rejects_invalid_series_names(
    mean,
    series_names,
    message,
):
    """Series metadata follows the same strict contract as OOS results."""
    with pytest.raises((TypeError, ValueError), match=message):
        BacktestResult(
            mean=mean,
            actual=mean,
            lower=None,
            upper=None,
            origins=np.array([10]),
            failures=[],
            model_type="TEST",
            window="expanding",
            target="observed",
            series_names=series_names,
        )


def test_comparison_recomputes_metrics_from_result_arrays():
    """Mutating a returned metric dictionary cannot corrupt later ranking."""
    result = _oos_result()
    result.metrics["rmse"] = -999.0

    comparison = compare_forecasts({"model": result})

    assert comparison.scores == {"model": 0.0}
