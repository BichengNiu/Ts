"""Tests for auditable Box-Cox transformations."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from Ts import BoxCoxResult as RootBoxCoxResult
from Ts import boxcox as root_boxcox
from Ts.TsUtils import BoxCoxResult, boxcox


def test_series_auto_lambda_matches_scipy_and_preserves_metadata():
    index = pd.date_range("2024-01-01", periods=8, freq="MS")
    series = pd.Series(
        [1.0, 1.5, np.nan, 3.0, 5.0, 8.0, 13.0, 21.0],
        index=index,
        name="sales",
    )
    original = series.copy(deep=True)

    result = boxcox(series)
    observed = series.dropna().to_numpy(dtype=float)
    expected_observed, expected_lmbda = stats.boxcox(observed)
    expected = np.full(series.shape, np.nan)
    expected[series.notna()] = expected_observed

    assert isinstance(result, BoxCoxResult)
    assert isinstance(result.data, pd.Series)
    np.testing.assert_allclose(result.data.to_numpy(), expected, equal_nan=True)
    assert result.lmbda == pytest.approx(expected_lmbda)
    assert result.data.index is series.index
    assert result.data.name == "sales"
    pd.testing.assert_series_equal(series, original)
    assert result.data is not series


@pytest.mark.parametrize("lmbda", [-1.0, 0.0, 0.5, 2.0])
def test_series_explicit_lambda_matches_scipy(lmbda):
    series = pd.Series([1.0, 2.0, 4.0, np.nan, 16.0], name="value")

    result = boxcox(series, lmbda=lmbda)
    expected = stats.boxcox(series.to_numpy(dtype=float), lmbda=lmbda)

    np.testing.assert_allclose(result.data.to_numpy(), expected, equal_nan=True)
    assert result.lmbda == lmbda


def test_dataframe_auto_estimates_each_column_independently():
    index = pd.period_range("2020Q1", periods=6, freq="Q")
    frame = pd.DataFrame(
        {
            "output": [1.0, 2.0, 3.5, 5.0, 8.0, 13.0],
            "prices": [10.0, 12.0, np.nan, 20.0, 31.0, 49.0],
        },
        index=index,
    )
    original = frame.copy(deep=True)

    result = boxcox(frame)

    assert isinstance(result.data, pd.DataFrame)
    assert isinstance(result.lmbda, pd.Series)
    assert result.lmbda.name == "lmbda"
    assert result.lmbda.index.equals(frame.columns)
    for column in frame.columns:
        observed = frame[column].dropna().to_numpy(dtype=float)
        expected_observed, expected_lmbda = stats.boxcox(observed)
        expected = np.full(frame.shape[0], np.nan)
        expected[frame[column].notna()] = expected_observed
        np.testing.assert_allclose(
            result.data[column].to_numpy(),
            expected,
            equal_nan=True,
        )
        assert result.lmbda[column] == pytest.approx(expected_lmbda)

    assert result.data.index is frame.index
    assert result.data.columns is frame.columns
    pd.testing.assert_frame_equal(frame, original)


def test_dataframe_accepts_one_shared_lambda():
    frame = pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 9.0]})

    result = boxcox(frame, lmbda=0.5)

    expected = pd.DataFrame(
        {
            column: stats.boxcox(frame[column].to_numpy(), lmbda=0.5)
            for column in frame.columns
        }
    )
    pd.testing.assert_frame_equal(result.data, expected)
    pd.testing.assert_series_equal(
        result.lmbda,
        pd.Series([0.5, 0.5], index=frame.columns, name="lmbda"),
    )


@pytest.mark.parametrize(
    "lmbda",
    [
        pytest.param({"a": 0.0, "b": 0.5}, id="mapping"),
        pytest.param(
            pd.Series({"a": -0.5, "b": 1.5}),
            id="series",
        ),
    ],
)
def test_dataframe_accepts_per_column_lambdas(lmbda):
    frame = pd.DataFrame({"a": [1.0, 3.0, 9.0], "b": [2.0, 4.0, 8.0]})

    result = boxcox(frame, lmbda=lmbda)

    for column in frame.columns:
        expected = stats.boxcox(
            frame[column].to_numpy(),
            lmbda=float(lmbda[column]),
        )
        np.testing.assert_allclose(result.data[column], expected)
        assert result.lmbda[column] == float(lmbda[column])


def test_fitted_dataframe_lambdas_can_be_reused_on_new_data():
    training = pd.DataFrame({"a": [1.0, 2.0, 4.0, 8.0], "b": [3.0, 5.0, 9.0, 15.0]})
    future = pd.DataFrame({"a": [16.0, 32.0], "b": [25.0, 40.0]})

    fitted = boxcox(training)
    applied = boxcox(future, lmbda=fitted.lmbda)

    pd.testing.assert_series_equal(applied.lmbda, fitted.lmbda)
    for column in future.columns:
        expected = stats.boxcox(
            future[column].to_numpy(),
            lmbda=fitted.lmbda[column],
        )
        np.testing.assert_allclose(applied.data[column], expected)


def test_missing_values_remain_in_their_original_positions():
    series = pd.Series([np.nan, 1.0, 2.0, np.nan, 4.0])

    result = boxcox(series)

    assert result.data.isna().equals(series.isna())


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
        boxcox(data)


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
        boxcox(data)


def test_rejects_duplicate_dataframe_columns():
    frame = pd.DataFrame([[1.0, 2.0]], columns=["value", "value"])

    with pytest.raises(ValueError, match="unique"):
        boxcox(frame, lmbda=0.5)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(pd.Series(["1", "2"]), id="strings"),
        pytest.param(
            pd.DataFrame({"numeric": [1.0], "label": ["a"]}),
            id="mixed-frame",
        ),
        pytest.param(pd.Series([True, False]), id="boolean"),
        pytest.param(pd.Series([1 + 2j, 3 + 4j]), id="complex"),
    ],
)
def test_rejects_non_real_numeric_data(data):
    with pytest.raises(TypeError, match="real numeric"):
        boxcox(data, lmbda=0.5)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_rejects_infinite_values(value):
    with pytest.raises(ValueError, match="infinite"):
        boxcox(pd.Series([1.0, value]), lmbda=0.5)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_requires_strictly_positive_observed_values(value):
    with pytest.raises(ValueError, match="strictly positive"):
        boxcox(pd.Series([1.0, value, np.nan]), lmbda=0.5)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(pd.Series([np.nan, np.nan]), id="all-missing"),
        pytest.param(pd.Series([np.nan, 2.0]), id="one-observation"),
    ],
)
def test_auto_estimation_requires_two_observed_values(data):
    with pytest.raises(ValueError, match="at least two"):
        boxcox(data)


def test_auto_estimation_rejects_constant_values():
    with pytest.raises(ValueError, match="constant"):
        boxcox(pd.Series([2.0, 2.0, np.nan, 2.0]))


@pytest.mark.parametrize(
    "lmbda",
    [True, "0.5", complex(1.0, 1.0), np.nan, np.inf, [0.5]],
)
def test_series_rejects_invalid_lambda(lmbda):
    with pytest.raises((TypeError, ValueError), match="lmbda"):
        boxcox(pd.Series([1.0, 2.0]), lmbda=lmbda)


@pytest.mark.parametrize(
    "lmbda",
    [
        pytest.param({"a": 0.5}, id="missing-column"),
        pytest.param({"a": 0.5, "b": 1.0, "c": 2.0}, id="extra-column"),
        pytest.param({"a": 0.5, "b": np.inf}, id="infinite-value"),
        pytest.param({"a": 0.5, "b": True}, id="boolean-value"),
    ],
)
def test_dataframe_rejects_invalid_lambda_mapping(lmbda):
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]})

    with pytest.raises((TypeError, ValueError), match="lmbda"):
        boxcox(frame, lmbda=lmbda)


def test_dataframe_rejects_duplicate_lambda_labels():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]})
    lmbda = pd.Series([0.5, 1.0], index=["a", "a"])

    with pytest.raises(ValueError, match=r"lmbda.*unique"):
        boxcox(frame, lmbda=lmbda)


def test_boxcox_is_exported_from_both_public_namespaces():
    assert root_boxcox is boxcox
    assert RootBoxCoxResult is BoxCoxResult
