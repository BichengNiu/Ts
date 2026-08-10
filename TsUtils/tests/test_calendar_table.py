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
