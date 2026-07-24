"""Cross-model tests for the shared date-index contract."""

import numpy as np
import pandas as pd
import pytest

from Ts.TsModels import (
    GARCH,
    SARIMA,
    SVAR,
    VAR,
    VECM,
    AutoGARCH,
    AutoSARIMA,
)


def _univariate_factories():
    return [
        lambda data: SARIMA(data, order=(0, 0, 0)),
        lambda data: AutoSARIMA(
            data,
            p=(0, 0),
            d=(0, 0),
            q=(0, 0),
        ),
        lambda data: GARCH(data, p=1, q=1),
        lambda data: AutoGARCH(data, p=(1, 1), q=(1, 1)),
    ]


@pytest.mark.parametrize("factory", _univariate_factories())
def test_univariate_models_infer_datetime_index(factory):
    """All forecast models preserve a Series DatetimeIndex."""
    dates = pd.date_range("2020-01-01", periods=30, freq="MS")
    model = factory(pd.Series(np.arange(30.0), index=dates))

    assert model.dates.equals(dates)


def _multivariate_factories():
    return [
        lambda data: VAR(data, lags=1),
        lambda data: VECM(data, lags=2, coint_rank=1),
        lambda data: SVAR(
            data,
            lags=1,
            A=np.array([[1.0, np.nan], [0.0, 1.0]]),
        ),
    ]


@pytest.mark.parametrize("factory", _multivariate_factories())
def test_multivariate_models_infer_datetime_index(factory):
    """VAR-family models preserve a DataFrame DatetimeIndex."""
    dates = pd.date_range("2020-01-01", periods=30, freq="MS")
    values = np.column_stack([np.arange(30.0), np.arange(30.0) * 0.5])
    model = factory(pd.DataFrame(values, index=dates, columns=["a", "b"]))

    assert model.dates.equals(dates)


@pytest.mark.parametrize(
    ("dates", "message"),
    [
        (pd.DatetimeIndex(["2020-01-01"] * 30), "unique"),
        (pd.date_range("2020-01-01", periods=30, freq="MS")[::-1], "increasing"),
        (pd.date_range("2020-01-01", periods=29, freq="MS"), "30 entries"),
    ],
)
def test_invalid_model_dates_are_hard_failures(dates, message):
    """Invalid calendars fail at model construction, before estimation."""
    with pytest.raises(ValueError, match=message):
        GARCH(np.arange(30.0), p=1, q=1, dates=dates)


def test_missing_drop_keeps_dates_synchronised():
    """Dropped observations remove the same entries from the calendar."""
    dates = pd.date_range("2020-01-01", periods=30, freq="MS")
    values = np.arange(30.0)
    values[[4, 17]] = np.nan

    model = GARCH(
        pd.Series(values, index=dates),
        p=1,
        q=1,
        missing="drop",
    )

    assert model.dates.equals(dates.delete([4, 17]))
    assert model.dropped_positions == (4, 17)


def _evaluation_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    univariate = pd.Series(rng.normal(size=60), index=dates)
    common_trend = np.cumsum(rng.normal(size=60))
    multivariate = pd.DataFrame(
        {
            "a": common_trend + rng.normal(scale=0.2, size=60),
            "b": 0.6 * common_trend + rng.normal(scale=0.2, size=60),
        },
        index=dates,
    )
    return dates, univariate, multivariate


@pytest.mark.parametrize("factory", _univariate_factories())
def test_univariate_models_validate_explicit_date_periods_with_gap(factory):
    """Every univariate forecast model accepts exact non-adjacent periods."""
    dates, univariate, _ = _evaluation_data()

    result = factory(univariate).oos(
        estimation_period=(dates[2], dates[41]),
        validation_period=(dates[47], dates[54]),
    )

    np.testing.assert_array_equal(result.estimation_indices, np.arange(2, 42))
    np.testing.assert_array_equal(result.validation_indices, np.arange(47, 55))
    assert result.estimation_dates.equals(dates[2:42])
    assert result.validation_dates.equals(dates[47:55])
    assert result.mean.shape == (8,)


@pytest.mark.parametrize("factory", _multivariate_factories())
def test_multivariate_models_validate_explicit_date_periods_with_gap(factory):
    """Every VAR-family model accepts exact non-adjacent periods."""
    dates, _, multivariate = _evaluation_data()

    result = factory(multivariate).oos(
        estimation_period=(dates[2], dates[41]),
        validation_period=(dates[47], dates[54]),
    )

    np.testing.assert_array_equal(result.estimation_indices, np.arange(2, 42))
    np.testing.assert_array_equal(result.validation_indices, np.arange(47, 55))
    assert result.estimation_dates.equals(dates[2:42])
    assert result.validation_dates.equals(dates[47:55])
    assert result.mean.shape == (8, 2)


@pytest.mark.parametrize("factory", _univariate_factories())
def test_univariate_models_validate_position_periods(factory):
    """Array-backed univariate models use explicit zero-based positions."""
    _, univariate, _ = _evaluation_data()

    result = factory(univariate.to_numpy()).oos(
        estimation_period=(2, 41),
        validation_period=(47, 54),
    )

    assert result.estimation_dates is None
    assert result.validation_dates is None
    np.testing.assert_array_equal(result.validation_indices, np.arange(47, 55))


@pytest.mark.parametrize("factory", _multivariate_factories())
def test_multivariate_models_validate_position_periods(factory):
    """Array-backed VAR-family models use explicit zero-based positions."""
    _, _, multivariate = _evaluation_data()

    result = factory(multivariate.to_numpy()).oos(
        estimation_period=(2, 41),
        validation_period=(47, 54),
    )

    assert result.estimation_dates is None
    assert result.validation_dates is None
    np.testing.assert_array_equal(result.validation_indices, np.arange(47, 55))


def test_sarima_exog_covers_the_gap_and_validation_period():
    """ARIMAX forecasts through the gap using aligned exogenous values."""
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    exog = pd.DataFrame({"x": np.linspace(0.0, 2.0, 60)}, index=dates)
    data = pd.Series(2.0 * exog["x"].to_numpy(), index=dates)
    model = SARIMA(data, exog=exog, order=(0, 0, 0), trend="n")

    result = model.oos(
        estimation_period=(dates[2], dates[41]),
        validation_period=(dates[47], dates[54]),
    )

    assert result.metrics["rmse"] < 1e-5
    assert result.validation_dates.equals(dates[47:55])


def test_sarima_rejects_exog_missing_inside_forecast_bridge():
    """Missing exog in the gap or validation period is a hard failure."""
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    exog = pd.DataFrame({"x": np.linspace(0.0, 2.0, 60)}, index=dates)
    data = pd.Series(2.0 * exog["x"].to_numpy(), index=dates)
    model = SARIMA(data, exog=exog, order=(0, 0, 0), trend="n")
    model.exog = model.exog[:45].copy()

    with pytest.raises(ValueError, match="future exog is missing dates"):
        model.oos(
            estimation_period=(dates[2], dates[41]),
            validation_period=(dates[47], dates[54]),
        )
