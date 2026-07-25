"""Tests for composable ordinary, log, and year-over-year differences."""

import numpy as np
import pandas as pd
import pytest

from Ts import difference as RootDifference
from Ts.TsUtils import difference


TRANSFORMATIONS = [
    pytest.param(1, False, 1, id="first-difference"),
    pytest.param(1, True, 1, id="first-log-difference"),
    pytest.param(2, False, 1, id="second-difference"),
    pytest.param(2, True, 1, id="second-log-difference"),
    pytest.param(1, False, 12, id="year-over-year-difference"),
    pytest.param(1, True, 12, id="year-over-year-log-difference"),
    pytest.param(2, False, 12, id="second-year-over-year-difference"),
    pytest.param(2, True, 12, id="second-year-over-year-log-difference"),
]


def _expected(values, *, order, log, lag):
    transformed = np.log(values) if log else values.astype(float)
    if order == 1:
        return transformed - transformed.shift(lag)
    return (
        transformed
        - 2.0 * transformed.shift(lag)
        + transformed.shift(2 * lag)
    )


@pytest.mark.parametrize(("order", "log", "lag"), TRANSFORMATIONS)
def test_series_supports_all_requested_transformations(order, log, lag):
    """Every requested operation follows its explicit lag-operator formula."""
    index = pd.date_range("2022-01-01", periods=30, freq="MS")
    series = pd.Series(
        np.linspace(10.0, 80.0, num=30) ** 1.3,
        index=index,
        name="sales",
    )

    result = difference(series, order=order, log=log, lag=lag)
    expected = _expected(series, order=order, log=log, lag=lag)

    pd.testing.assert_series_equal(result, expected)


def test_dataframe_preserves_container_and_metadata_without_mutating_input():
    """Column-wise transforms retain pandas labels and leave caller data intact."""
    index = pd.period_range("2020Q1", periods=12, freq="Q")
    frame = pd.DataFrame(
        {
            "output": np.geomspace(10.0, 40.0, num=12),
            "prices": np.geomspace(50.0, 90.0, num=12),
        },
        index=index,
    )
    original = frame.copy(deep=True)

    result = difference(frame, order=2, log=True, lag=4)
    expected = _expected(frame, order=2, log=True, lag=4)

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(frame, original)
    assert result is not frame
    assert result.index is frame.index
    assert result.columns is frame.columns


def test_series_name_index_and_shape_are_preserved():
    series = pd.Series(
        [2, 4, 8, 16],
        index=pd.Index(["a", "b", "c", "d"], name="period"),
        name="value",
    )

    result = difference(series, order=2)

    assert result.name == "value"
    assert result.index is series.index
    assert result.shape == series.shape
    assert result.iloc[:2].isna().all()


def test_missing_values_are_preserved_and_propagated_by_differencing():
    series = pd.Series([1.0, pd.NA, 4.0, 8.0, 16.0], dtype="Float64")

    result = difference(series)
    expected = pd.Series([1.0, np.nan, 4.0, 8.0, 16.0]).diff()

    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param([1.0, 2.0], id="list"),
        pytest.param(np.array([1.0, 2.0]), id="ndarray"),
        pytest.param(1.0, id="scalar"),
    ],
)
def test_rejects_non_pandas_inputs(data):
    with pytest.raises(TypeError, match="Series or DataFrame"):
        difference(data)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(pd.Series([], dtype=float), id="empty-series"),
        pytest.param(pd.DataFrame({"value": []}), id="empty-dataframe-rows"),
        pytest.param(pd.DataFrame(index=range(2)), id="empty-dataframe-columns"),
    ],
)
def test_rejects_empty_inputs(data):
    with pytest.raises(ValueError, match="at least one"):
        difference(data)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(pd.Series(["1", "2"]), id="string-series"),
        pytest.param(
            pd.DataFrame({"numeric": [1.0], "label": ["a"]}),
            id="mixed-dataframe",
        ),
        pytest.param(pd.Series([True, False]), id="boolean-series"),
        pytest.param(pd.DataFrame({"flag": [True, False]}), id="boolean-dataframe"),
    ],
)
def test_rejects_non_numeric_and_boolean_data(data):
    with pytest.raises(TypeError, match="numeric"):
        difference(data)


def test_rejects_duplicate_dataframe_columns():
    frame = pd.DataFrame([[1.0, 2.0]], columns=["value", "value"])

    with pytest.raises(ValueError, match="unique"):
        difference(frame)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_rejects_infinite_values(value):
    with pytest.raises(ValueError, match="infinite"):
        difference(pd.Series([1.0, value]))


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_log_difference_requires_strictly_positive_values(value):
    series = pd.Series([1.0, value, np.nan, 2.0])

    with pytest.raises(ValueError, match="strictly positive"):
        difference(series, log=True)


@pytest.mark.parametrize("order", [0, 3, 1.0, True, "1"])
def test_order_must_be_one_or_two(order):
    with pytest.raises((TypeError, ValueError), match="order"):
        difference(pd.Series([1.0, 2.0]), order=order)


@pytest.mark.parametrize("lag", [0, -1, 1.0, True, "1"])
def test_lag_must_be_a_positive_integer(lag):
    with pytest.raises((TypeError, ValueError), match="lag"):
        difference(pd.Series([1.0, 2.0]), lag=lag)


@pytest.mark.parametrize("log", [0, 1, None, "yes"])
def test_log_must_be_boolean(log):
    with pytest.raises(TypeError, match="log"):
        difference(pd.Series([1.0, 2.0]), log=log)


def test_difference_is_exported_from_root_package():
    assert RootDifference is difference
