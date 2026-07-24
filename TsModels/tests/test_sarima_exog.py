"""Tests for dated SARIMA/ARIMAX input contracts."""

import numpy as np
import pandas as pd
import pytest

from Ts.TsModels import SARIMA


def test_dataframe_exog_splits_history_and_default_future():
    y_dates = pd.date_range("2025-01-01", periods=12, freq="MS")
    all_dates = pd.date_range("2025-01-01", periods=15, freq="MS")
    y = pd.Series(np.arange(12.0), index=y_dates)
    exog = pd.DataFrame({"price": np.arange(15.0)}, index=all_dates)

    model = SARIMA(y, exog=exog, order=(0, 0, 0), trend="n")

    assert model.exog.shape == (12, 1)
    assert model.exog_names == ("price",)
    assert model.dates.equals(y_dates)
    assert model.future_exog.index.equals(all_dates[12:])
    assert model.future_exog.columns.tolist() == ["price"]


def test_ndarray_exog_requires_names_and_equal_length():
    with pytest.raises(ValueError, match="exog_names"):
        SARIMA(np.arange(12.0), exog=np.ones((12, 1)))
    with pytest.raises(ValueError, match="12 observations"):
        SARIMA(
            np.arange(12.0),
            exog=np.ones((11, 1)),
            exog_names=["x"],
        )


def test_ndarray_exog_must_be_two_dimensional():
    with pytest.raises(ValueError, match="two-dimensional"):
        SARIMA(
            np.arange(12.0),
            exog=np.ones(12),
            exog_names=["x"],
        )


def test_missing_drop_removes_y_exog_and_dates_jointly():
    dates = pd.date_range("2025-01-01", periods=12, freq="D")
    y = pd.Series(np.arange(12.0), index=dates)
    x = pd.DataFrame({"x": np.arange(12.0)}, index=dates)
    y.iloc[2] = np.nan
    x.iloc[4, 0] = np.nan

    model = SARIMA(y, exog=x, missing="drop")

    assert len(model.data) == 10
    assert dates[2] not in model.dates
    assert dates[4] not in model.dates
    assert model.exog.shape == (10, 1)


def test_missing_raise_is_default_and_rejects_infinite_values():
    dates = pd.date_range("2025-01-01", periods=12, freq="D")
    y = pd.Series(np.arange(12.0), index=dates)
    x = pd.DataFrame({"x": np.arange(12.0)}, index=dates)
    y.iloc[1] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        SARIMA(y, exog=x)

    y.iloc[1] = 1.0
    x.iloc[3, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        SARIMA(y, exog=x)


def test_unknown_missing_policy_fails():
    with pytest.raises(ValueError, match="missing"):
        SARIMA(np.arange(12.0), missing="fill")


def test_ndarray_data_accepts_explicit_dates():
    dates = pd.date_range("2025-01-01", periods=12, freq="MS")

    model = SARIMA(np.arange(12.0), dates=dates)

    assert model.dates.equals(dates)


def test_series_rejects_conflicting_dates_argument():
    dates = pd.date_range("2025-01-01", periods=12, freq="MS")
    y = pd.Series(np.arange(12.0), index=dates)

    with pytest.raises(ValueError, match="dates.*Series"):
        SARIMA(y, dates=dates)


@pytest.mark.parametrize(
    "dates",
    [
        pd.DatetimeIndex(
            ["2025-01-01", "2025-01-02", "2025-01-02"] * 4
        ),
        pd.date_range("2025-01-01", periods=12, freq="D")[::-1],
    ],
)
def test_dates_must_be_unique_and_increasing(dates):
    with pytest.raises(ValueError, match="unique|increasing"):
        SARIMA(np.arange(12.0), dates=dates)


def test_dataframe_exog_must_cover_every_historical_date():
    dates = pd.date_range("2025-01-01", periods=12, freq="MS")
    y = pd.Series(np.arange(12.0), index=dates)
    x = pd.DataFrame({"x": np.arange(11.0)}, index=dates.delete(5))

    with pytest.raises(ValueError, match="2025-06-01"):
        SARIMA(y, exog=x)


def test_dataframe_rejects_extra_historical_rows():
    dates = pd.date_range("2025-01-01", periods=12, freq="MS")
    y = pd.Series(np.arange(12.0), index=dates)
    extra_dates = dates.insert(1, pd.Timestamp("2025-01-15"))
    x = pd.DataFrame({"x": np.arange(13.0)}, index=extra_dates)

    with pytest.raises(ValueError, match="extra historical"):
        SARIMA(y, exog=x)


def test_dataframe_columns_are_authoritative():
    dates = pd.date_range("2025-01-01", periods=12, freq="MS")
    y = pd.Series(np.arange(12.0), index=dates)
    x = pd.DataFrame({"x": np.arange(12.0)}, index=dates)

    with pytest.raises(ValueError, match="exog_names.*DataFrame"):
        SARIMA(y, exog=x, exog_names=["renamed"])


def test_exog_names_must_be_unique_nonempty_and_match_columns():
    y = np.arange(12.0)

    with pytest.raises(ValueError, match="one name per"):
        SARIMA(y, exog=np.ones((12, 2)), exog_names=["x"])
    with pytest.raises(ValueError, match="unique"):
        SARIMA(y, exog=np.ones((12, 2)), exog_names=["x", "x"])
    with pytest.raises(ValueError, match="non-empty"):
        SARIMA(y, exog=np.ones((12, 1)), exog_names=[""])


def test_inputs_are_copied_and_plain_sarima_still_works():
    y = np.arange(12.0)
    x = np.ones((12, 1))

    model = SARIMA(y, exog=x, exog_names=["x"])
    plain = SARIMA(y)
    y[:] = -1
    x[:] = -1

    assert np.all(model.data >= 0)
    assert np.all(model.exog == 1)
    assert plain.exog is None
    assert plain.dates is None
    assert plain.future_exog is None
