"""Tests for calendar-oriented time-series reshaping."""

import numpy as np
import pandas as pd
import pytest

from Ts import calendar_table as root_calendar_table
from Ts.TsUtils import calendar_table


def test_public_exports_resolve_to_the_same_function():
    """The root and TsUtils namespaces expose one public function."""
    assert root_calendar_table is calendar_table


def test_monthly_series_has_fixed_month_rows_and_year_columns():
    """Monthly observations are arranged by month within natural year."""
    index = pd.date_range("2023-11-01", periods=5, freq="MS")
    source = pd.Series([11.0, 12.0, 1.0, np.nan, 3.0], index=index)

    result = calendar_table(source)

    assert result.index.equals(pd.Index(range(1, 13), name="month"))
    assert result.columns.equals(pd.Index([2023, 2024], name="year"))
    assert result.loc[11, 2023] == 11.0
    assert pd.isna(result.loc[2, 2024])


def test_dataframe_selection_requires_col_only_when_ambiguous():
    """A single column is automatic while multiple columns require col."""
    index = pd.date_range("2024-01-01", periods=3, freq="MS")
    frame = pd.DataFrame({"sales": [1.0, 2.0, 3.0]}, index=index)
    pd.testing.assert_frame_equal(
        calendar_table(frame),
        calendar_table(frame, col="sales"),
    )

    ambiguous = frame.assign(price=[4.0, 5.0, 6.0])
    with pytest.raises(ValueError, match="col is required"):
        calendar_table(ambiguous)


def test_series_rejects_col_and_dataframe_requires_a_unique_existing_col():
    """Column selection never silently ignores or ambiguously selects data."""
    index = pd.date_range("2024-01-01", periods=3, freq="MS")
    series = pd.Series([1.0, 2.0, 3.0], index=index)
    with pytest.raises(ValueError, match="col is only valid"):
        calendar_table(series, col="value")

    frame = pd.DataFrame({"sales": [1.0, 2.0, 3.0]}, index=index)
    with pytest.raises(ValueError, match="not a DataFrame column"):
        calendar_table(frame, col="missing")

    duplicate = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        index=index,
        columns=["sales", "sales"],
    )
    with pytest.raises(ValueError, match="columns must be unique"):
        calendar_table(duplicate, col="sales")


def test_annual_table_uses_one_value_row_and_natural_year_columns():
    """Annual observations are transposed under their natural years."""
    index = pd.date_range("2022-01-01", periods=3, freq="YS")

    result = calendar_table(pd.Series([2.0, 3.0, 4.0], index=index))

    assert result.index.equals(pd.Index(["value"], name="period"))
    assert result.columns.equals(pd.Index([2022, 2023, 2024], name="year"))
    assert result.loc["value", 2023] == 3.0


def test_natural_quarterly_table_uses_q1_to_q4_rows():
    """Calendar quarters are arranged within natural years."""
    index = pd.period_range("2023Q4", periods=3, freq="Q-DEC")

    result = calendar_table(pd.Series([4.0, 1.0, 2.0], index=index))

    assert result.index.equals(pd.Index(range(1, 5), name="quarter"))
    assert result.columns.equals(pd.Index([2023, 2024], name="year"))
    assert result.loc[4, 2023] == 4.0
    assert result.loc[1, 2024] == 1.0
    assert result.loc[2, 2024] == 2.0


def test_daily_table_uses_day_rows_and_year_month_columns():
    """Daily values use days within chronological natural-year months."""
    index = pd.date_range("2024-01-30", periods=4, freq="D")

    result = calendar_table(pd.Series(range(4), index=index, dtype=float))

    assert result.index.equals(pd.Index(range(1, 32), name="day"))
    assert result.columns.names == ["year", "month"]
    assert result.columns.tolist() == [(2024, 1), (2024, 2)]
    assert result.loc[30, (2024, 1)] == 0.0
    assert result.loc[2, (2024, 2)] == 3.0


def test_business_daily_table_keeps_weekend_calendar_cells_missing():
    """Business-day completion does not manufacture weekend observations."""
    index = pd.date_range("2024-01-05", periods=2, freq="B")

    result = calendar_table(pd.Series([5.0, 8.0], index=index))

    assert result.loc[5, (2024, 1)] == 5.0
    assert pd.isna(result.loc[6, (2024, 1)])
    assert pd.isna(result.loc[7, (2024, 1)])
    assert result.loc[8, (2024, 1)] == 8.0


@pytest.mark.parametrize("frequency", ["W-SUN", "W-MON"])
def test_weekly_month_is_determined_by_wednesday_not_timestamp_label(frequency):
    """A cross-month week follows its Wednesday, not its anchored label."""
    first_label = "2024-02-04" if frequency == "W-SUN" else "2024-02-05"
    index = pd.date_range(first_label, periods=2, freq=frequency)

    result = calendar_table(pd.Series([10.0, 20.0], index=index))

    assert result.index.equals(pd.Index(range(1, 6), name="week"))
    assert result.columns.names == ["year", "month"]
    assert result.columns.tolist() == [(2024, 1), (2024, 2)]
    assert result.loc[5, (2024, 1)] == 10.0
    assert result.loc[1, (2024, 2)] == 20.0


def test_weekly_period_index_uses_the_periods_wednesday():
    """Weekly PeriodIndex boundaries carry the same Wednesday semantics."""
    index = pd.period_range("2024-02-04", periods=2, freq="W-SUN")

    result = calendar_table(pd.Series([10.0, 20.0], index=index))

    assert result.loc[5, (2024, 1)] == 10.0
    assert result.loc[1, (2024, 2)] == 20.0
