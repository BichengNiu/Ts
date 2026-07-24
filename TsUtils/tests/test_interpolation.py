"""Tests for auditable missing-value interpolation."""

import numpy as np
import pandas as pd
import pytest


def test_linear_interpolation_returns_structured_result():
    """Linear interpolation fills an interior NumPy gap and records masks.

    covers: TsUtils/_interpolation.py [module]
    covers: TsUtils/_interpolation.py::InterpolationResult [class]
    covers: TsUtils/_interpolation.py::interpolate_missing [function]
    """
    from Ts.TsUtils import InterpolationResult, interpolate_missing

    result = interpolate_missing(np.array([0.0, np.nan, 4.0]))

    assert isinstance(result, InterpolationResult)
    np.testing.assert_allclose(result.data, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(result.missing_mask, [False, True, False])
    np.testing.assert_array_equal(result.filled_mask, [False, True, False])
    np.testing.assert_array_equal(result.remaining_mask, [False, False, False])
    assert result.n_missing == 1
    assert result.n_filled == 1
    assert result.n_remaining == 0
    assert result.complete


def test_two_dimensional_array_is_interpolated_by_column():
    """Each column is treated as a separate series."""
    from Ts.TsUtils import interpolate_missing

    data = np.array(
        [
            [0.0, 10.0],
            [np.nan, 20.0],
            [4.0, np.nan],
            [6.0, 40.0],
        ]
    )

    result = interpolate_missing(data)

    np.testing.assert_allclose(
        result.data,
        [[0.0, 10.0], [2.0, 20.0], [4.0, 30.0], [6.0, 40.0]],
    )
    assert result.data.shape == data.shape


def test_series_metadata_and_input_are_preserved():
    """Series output preserves index/name without mutating the caller."""
    from Ts.TsUtils import interpolate_missing

    index = pd.date_range("2024-01-01", periods=3, freq="D")
    data = pd.Series([1.0, np.nan, 3.0], index=index, name="sales")
    original = data.copy()

    result = interpolate_missing(data)

    assert isinstance(result.data, pd.Series)
    assert result.data.index.equals(index)
    assert result.data.name == "sales"
    pd.testing.assert_series_equal(data, original)
    np.testing.assert_allclose(result.data.to_numpy(), [1.0, 2.0, 3.0])


def test_dataframe_metadata_is_preserved():
    """DataFrame output preserves its index and columns."""
    from Ts.TsUtils import interpolate_missing

    index = pd.Index(["a", "b", "c"], name="period")
    data = pd.DataFrame(
        {"sales": [1.0, np.nan, 3.0], "cost": [2.0, 4.0, 6.0]},
        index=index,
    )

    result = interpolate_missing(data)

    assert isinstance(result.data, pd.DataFrame)
    assert result.data.index.equals(index)
    assert result.data.columns.equals(data.columns)
    assert result.data.index.name == "period"
    assert result.data.loc["b", "sales"] == pytest.approx(2.0)


def test_nullable_pandas_missing_value_is_interpolated():
    """pd.NA is treated as missing without losing Series metadata."""
    from Ts.TsUtils import interpolate_missing

    data = pd.Series([1.0, pd.NA, 3.0], dtype="Float64", name="value")

    result = interpolate_missing(data)

    assert result.data.name == "value"
    np.testing.assert_allclose(result.data.to_numpy(dtype=float), [1, 2, 3])


def test_time_interpolation_uses_elapsed_time():
    """Time interpolation uses an irregular datetime coordinate."""
    from Ts.TsUtils import interpolate_missing

    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-11"])
    data = pd.Series([0.0, np.nan, 10.0], index=index)

    result = interpolate_missing(data, method="time")

    assert result.data.iloc[1] == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["nearest", "cubic"])
def test_scipy_interpolation_methods_fill_interior_gap(method):
    """Nearest and cubic methods are exposed through the public API."""
    from Ts.TsUtils import interpolate_missing

    if method == "nearest":
        data = np.array([0.0, np.nan, 10.0, 12.0])
    else:
        data = np.array([0.0, 1.0, np.nan, 27.0, 64.0])

    result = interpolate_missing(data, method=method)

    assert np.isfinite(result.data[2 if method == "cubic" else 1])
    if method == "cubic":
        assert result.data[2] == pytest.approx(8.0)


def test_max_gap_leaves_long_gap_unfilled():
    """A gap longer than max_gap is retained in full, not partially filled."""
    from Ts.TsUtils import interpolate_missing

    data = np.array([0.0, np.nan, np.nan, np.nan, 4.0])

    result = interpolate_missing(data, max_gap=2)

    assert np.isnan(result.data[1:4]).all()
    assert result.n_missing == 3
    assert result.n_filled == 0
    assert result.n_remaining == 3
    assert not result.complete


def test_max_gap_fills_eligible_short_gap():
    """A gap at or below max_gap remains eligible for interpolation."""
    from Ts.TsUtils import interpolate_missing

    result = interpolate_missing(
        np.array([0.0, np.nan, np.nan, 3.0]),
        max_gap=2,
    )

    np.testing.assert_allclose(result.data, [0.0, 1.0, 2.0, 3.0])


def test_edges_are_kept_by_default():
    """True interpolation does not silently extrapolate boundary values."""
    from Ts.TsUtils import interpolate_missing

    result = interpolate_missing(np.array([np.nan, 1.0, 3.0, np.nan]))

    assert np.isnan(result.data[[0, 3]]).all()
    assert result.n_remaining == 2


def test_nearest_edge_policy_fills_eligible_boundaries():
    """Explicit nearest edge handling copies the closest observed value."""
    from Ts.TsUtils import interpolate_missing

    result = interpolate_missing(
        np.array([np.nan, 1.0, 3.0, np.nan]),
        edge="nearest",
    )

    np.testing.assert_allclose(result.data, [1.0, 1.0, 3.0, 3.0])
    assert result.complete


def test_max_gap_also_limits_edge_filling():
    """Long boundary gaps remain missing even when nearest edges are requested."""
    from Ts.TsUtils import interpolate_missing

    result = interpolate_missing(
        np.array([np.nan, np.nan, 2.0, 3.0]),
        max_gap=1,
        edge="nearest",
    )

    assert np.isnan(result.data[:2]).all()
    assert result.n_remaining == 2


def test_all_missing_series_remains_unfilled():
    """Interpolation does not invent a level for an all-missing series."""
    from Ts.TsUtils import interpolate_missing

    result = interpolate_missing(np.array([np.nan, np.nan]))

    assert np.isnan(result.data).all()
    assert result.n_filled == 0
    assert result.n_remaining == 2
    assert not result.complete


def test_infinity_is_rejected_instead_of_treated_as_missing():
    """Infinite values indicate invalid data rather than an imputable gap."""
    from Ts.TsUtils import interpolate_missing

    with pytest.raises(ValueError, match="infinite"):
        interpolate_missing(np.array([1.0, np.inf, 3.0]))


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (np.ones((2, 2, 2)), "one- or two-dimensional"),
        (np.array([]), "at least one observation"),
        (np.empty((2, 0)), "at least one series"),
        (["a", None, "c"], "numeric"),
    ],
)
def test_invalid_input_is_rejected(data, message):
    """Inputs must be non-empty numeric time-series containers."""
    from Ts.TsUtils import interpolate_missing

    with pytest.raises((TypeError, ValueError), match=message):
        interpolate_missing(data)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "polynomial"}, "method must be"),
        ({"max_gap": 0}, "max_gap"),
        ({"max_gap": True}, "max_gap"),
        ({"edge": "extrapolate"}, "edge must be"),
    ],
)
def test_invalid_options_are_rejected(kwargs, message):
    """Interpolation options are validated before backend execution."""
    from Ts.TsUtils import interpolate_missing

    with pytest.raises(ValueError, match=message):
        interpolate_missing(np.array([1.0, np.nan, 3.0]), **kwargs)


def test_time_method_requires_valid_time_index():
    """Time interpolation requires a unique increasing time index."""
    from Ts.TsUtils import interpolate_missing

    with pytest.raises(TypeError, match="DatetimeIndex or TimedeltaIndex"):
        interpolate_missing(np.array([1.0, np.nan, 3.0]), method="time")

    duplicate = pd.Series(
        [1.0, np.nan, 3.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-03"]),
    )
    with pytest.raises(ValueError, match="unique and increasing"):
        interpolate_missing(duplicate, method="time")


def test_summary_reports_configuration_and_counts():
    """The result summary exposes the imputation audit trail."""
    from Ts.TsUtils import interpolate_missing

    result = interpolate_missing(np.array([0.0, np.nan, 2.0]))
    text = result.summary()

    assert "Interpolation Result" in text
    assert "Method          : linear" in text
    assert "Missing values  : 1" in text
    assert "Filled values   : 1" in text
    assert "Remaining       : 0" in text
