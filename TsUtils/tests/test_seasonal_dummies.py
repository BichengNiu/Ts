"""Tests for recurring calendar-position dummy generation."""

import numpy as np
import pandas as pd
import pytest

from Ts.TsUtils import seasonal_dummies


def test_public_imports_expose_same_function():
    """The unified and TsUtils namespaces expose the same public function."""
    from Ts import seasonal_dummies as top_level
    from Ts.TsUtils import seasonal_dummies as utils_level

    assert top_level is utils_level
    assert utils_level is seasonal_dummies


@pytest.mark.parametrize("container", ["series", "frame", "index"])
def test_accepts_dated_public_inputs(container):
    """Series, DataFrame, and a direct time index preserve date alignment."""
    index = pd.date_range("2024-01-01", periods=12, freq="MS")
    if container == "series":
        data = pd.Series(np.arange(12.0), index=index)
    elif container == "frame":
        data = pd.DataFrame({"value": np.arange(12.0)}, index=index)
    else:
        data = index

    result = seasonal_dummies(data)

    assert result.index.equals(index)


def test_accepts_period_index():
    """PeriodIndex frequency metadata is accepted without conversion."""
    index = pd.period_range("2024-01", periods=12, freq="M")

    result = seasonal_dummies(index)

    assert result.index.equals(index)


@pytest.mark.parametrize(
    ("data", "error", "match"),
    [
        (
            pd.Series([1.0, 2.0, 3.0]),
            TypeError,
            "DatetimeIndex or PeriodIndex",
        ),
        (pd.DatetimeIndex([]), ValueError, "must not be empty"),
        (
            pd.DatetimeIndex(["2024-01-01", None]),
            ValueError,
            "missing dates",
        ),
        (
            pd.DatetimeIndex(["2024-01-01", "2024-01-01"]),
            ValueError,
            "unique",
        ),
        (
            pd.DatetimeIndex(["2024-01-03", "2024-01-02", "2024-01-01"]),
            ValueError,
            "in increasing order",
        ),
        (
            pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-04"]),
            ValueError,
            "regular frequency",
        ),
        (pd.date_range("2020", periods=3, freq="YS"), ValueError, "does not support"),
        (pd.date_range("2024-01-01", periods=3, freq="h"), ValueError, "does not support"),
        (pd.date_range("2024-01-01", periods=3, freq="2MS"), ValueError, "does not support"),
    ],
)
def test_rejects_invalid_time_indexes(data, error, match):
    """Invalid calendars fail rather than being interpreted positionally."""
    with pytest.raises(error, match=match):
        seasonal_dummies(data)


@pytest.mark.parametrize("drop_first", [1, 0, "yes", None])
def test_drop_first_must_be_boolean(drop_first):
    """The reference-category switch has a strict Boolean contract."""
    index = pd.date_range("2024-01-01", periods=4, freq="QS")

    with pytest.raises(TypeError, match="drop_first must be a boolean"):
        seasonal_dummies(index, drop_first=drop_first)


def test_infers_frequency_from_regular_datetime_index_without_metadata():
    """A regular DatetimeIndex can be inferred when `.freq` is absent."""
    source = pd.date_range("2024-01-01", periods=12, freq="MS")
    index = pd.DatetimeIndex(source.values)

    assert index.freq is None
    result = seasonal_dummies(index)

    assert result.index.equals(index)


def test_calendar_daily_uses_monday_reference_and_full_week_schema():
    """Calendar-day data receives recurring weekday indicators."""
    index = pd.date_range("2024-01-01", periods=7, freq="D")

    result = seasonal_dummies(index)

    assert list(result) == [
        "weekday_Tue",
        "weekday_Wed",
        "weekday_Thu",
        "weekday_Fri",
        "weekday_Sat",
        "weekday_Sun",
    ]
    assert result.loc[index[0]].sum() == 0
    assert result.loc[index[1], "weekday_Tue"] == 1


def test_business_daily_excludes_weekend_categories():
    """Business-day data uses only Monday through Friday categories."""
    index = pd.date_range("2024-01-01", periods=5, freq="B")

    result = seasonal_dummies(index)

    assert list(result) == [
        "weekday_Tue",
        "weekday_Wed",
        "weekday_Thu",
        "weekday_Fri",
    ]
    assert result.loc[index[0]].sum() == 0


def test_weekly_uses_iso_week_one_reference_and_fixed_53_week_schema():
    """Anchored weekly indexes produce ISO week 02 through 53 by default."""
    index = pd.date_range("2024-01-07", periods=4, freq="W-SUN")

    result = seasonal_dummies(index)

    assert list(result) == [f"week_{week:02d}" for week in range(2, 54)]
    assert result.loc[index[0]].sum() == 0
    assert result.loc[index[1], "week_02"] == 1
    assert result["week_53"].eq(0).all()


def test_monthly_defaults_to_january_reference_and_fixed_schema():
    """A short monthly sample still receives every non-reference month."""
    index = pd.date_range("2024-02-01", periods=3, freq="MS")

    result = seasonal_dummies(index)

    assert list(result) == [f"month_{month:02d}" for month in range(2, 13)]
    assert result.loc[index[0], "month_02"] == 1
    assert result["month_12"].eq(0).all()
    assert all(dtype == np.dtype("int8") for dtype in result.dtypes)


@pytest.mark.parametrize("frequency", ["MS", "ME", "BMS", "BME"])
def test_month_begin_end_variants_share_the_month_schema(frequency):
    """Calendar and business month anchors use the same seasonal positions."""
    index = pd.date_range("2024-01-01", periods=3, freq=frequency)

    result = seasonal_dummies(index)

    assert list(result) == [f"month_{month:02d}" for month in range(2, 13)]


def test_drop_first_false_keeps_complete_quarter_schema():
    """The full quarter schema contains one active category per row."""
    index = pd.period_range("2024Q1", periods=4, freq="Q-DEC")

    result = seasonal_dummies(index, drop_first=False)

    assert list(result) == [
        "quarter_Q1",
        "quarter_Q2",
        "quarter_Q3",
        "quarter_Q4",
    ]
    np.testing.assert_array_equal(result.sum(axis=1), np.ones(4, dtype=int))


def test_fiscal_quarter_period_index_uses_its_anchor():
    """A fiscal PeriodIndex retains Q1-Q4 positions from its own frequency."""
    index = pd.period_range("2024Q1", periods=4, freq="Q-MAR")

    result = seasonal_dummies(index, drop_first=False)

    for position, quarter in enumerate(range(1, 5)):
        assert result.iloc[position, quarter - 1] == 1


def test_fiscal_quarter_datetime_index_uses_begin_anchor():
    """QS-APR dates are classified relative to an April fiscal-year start."""
    index = pd.date_range("2024-04-01", periods=4, freq="QS-APR")

    result = seasonal_dummies(index, drop_first=False)

    for position, quarter in enumerate(range(1, 5)):
        assert result.iloc[position, quarter - 1] == 1
