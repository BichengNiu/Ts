"""Calendar and cross-model tests for unified holdout evaluation."""

import numpy as np
import pandas as pd
import pytest

from Ts.TsMetrics import Holdout, evaluate_forecasts
from Ts.TsModels import GARCH, SARIMAX, SVAR, VAR, VECM, AutoGARCH, AutoSARIMAX


def _univariate_factories():
    return [
        lambda data: SARIMAX(data, order=(0, 0, 0)),
        lambda data: AutoSARIMAX(data, p=(0, 0), d=(0, 0), q=(0, 0)),
        lambda data: GARCH(data, p=1, q=1),
        lambda data: AutoGARCH(data, p=(1, 1), q=(1, 1)),
    ]


def _multivariate_factories():
    return [
        lambda data: VAR(data, lags=1),
        lambda data: VECM(data, lags=2, coint_rank=1),
        lambda data: SVAR(
            data, lags=1, A=np.array([[1.0, np.nan], [0.0, 1.0]])
        ),
    ]


@pytest.mark.parametrize("factory", _univariate_factories())
def test_univariate_models_infer_and_evaluate_datetime_index(factory):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    values = pd.Series(rng.normal(size=60), index=dates)
    model = factory(values)
    report = evaluate_forecasts(
        {"model": model},
        scheme=Holdout(
            train=(dates[2], dates[41]), test=(dates[47], dates[54])
        ),
    )
    result = report.results["model"]
    assert model.dates.equals(dates)
    assert result.mean.shape == (1, 8)
    assert result.splits[0].train_indices.tolist() == list(range(2, 42))
    assert result.splits[0].target_indices.tolist() == list(range(47, 55))


@pytest.mark.parametrize("factory", _multivariate_factories())
def test_multivariate_models_evaluate_the_same_date_periods(factory):
    rng = np.random.default_rng(43)
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    trend = np.cumsum(rng.normal(size=60))
    values = pd.DataFrame(
        {"a": trend + rng.normal(scale=0.2, size=60),
         "b": 0.6 * trend + rng.normal(scale=0.2, size=60)},
        index=dates,
    )
    report = evaluate_forecasts(
        {"model": factory(values)},
        scheme=Holdout(
            train=(dates[2], dates[41]), test=(dates[47], dates[54])
        ),
    )
    result = report.results["model"]
    assert result.mean.shape == (1, 8, 2)
    assert result.series_names == ("a", "b")


@pytest.mark.parametrize(
    ("dates", "message"),
    [
        (pd.DatetimeIndex(["2020-01-01"] * 30), "unique"),
        (pd.date_range("2020-01-01", periods=30, freq="MS")[::-1], "increasing"),
        (pd.date_range("2020-01-01", periods=29, freq="MS"), "30 entries"),
    ],
)
def test_invalid_model_dates_are_hard_failures(dates, message):
    with pytest.raises(ValueError, match=message):
        GARCH(np.arange(30.0), p=1, q=1, dates=dates)


def test_sarimax_exog_covers_gap_and_requires_observed_policy():
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    exog = pd.DataFrame({"x": np.linspace(0.0, 2.0, 60)}, index=dates)
    rng = np.random.default_rng(2608)
    data = pd.Series(
        2.0 * exog["x"].to_numpy() + rng.normal(scale=0.01, size=60),
        index=dates,
    )
    model = SARIMAX(data, exog=exog, order=(0, 0, 0), trend="n")
    scheme = Holdout(
        train=(dates[2], dates[41]), test=(dates[47], dates[54])
    )
    with pytest.raises(ValueError, match="future_exog='observed'"):
        evaluate_forecasts({"model": model}, scheme=scheme)
    report = evaluate_forecasts(
        {"model": model}, scheme=scheme, future_exog="observed"
    )
    assert report.table.loc["model", "rmse"] < 0.05


def test_sarimax_reports_exog_missing_inside_forecast_bridge():
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    exog = pd.DataFrame({"x": np.linspace(0.0, 2.0, 60)}, index=dates)
    model = SARIMAX(
        pd.Series(2 * exog["x"].to_numpy(), index=dates),
        exog=exog,
        order=(0, 0, 0),
        trend="n",
    )
    model.exog = model.exog[:45].copy()
    with pytest.raises(ValueError, match="future exog is missing dates"):
        evaluate_forecasts(
            {"model": model},
            scheme=Holdout(
                train=(dates[2], dates[41]), test=(dates[47], dates[54])
            ),
            future_exog="observed",
        )
