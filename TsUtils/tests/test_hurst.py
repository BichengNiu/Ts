"""Tests for the rescaled-range Hurst exponent estimator."""

import numpy as np
import pandas as pd
import pytest

from Ts import hurst_exponent as root_hurst_exponent
from Ts.TsUtils import hurst_exponent


def _reference_hurst(values):
    """Calculate the expected slope independently of the implementation."""
    values = np.asarray(values, dtype=float)
    sizes = [2**power for power in range(2, int(np.log2(len(values) // 2)) + 1)]
    rs_values = []
    for size in sizes:
        blocks = values[: len(values) // size * size].reshape(-1, size)
        block_statistics = []
        for block in blocks:
            centered = block - block.mean()
            standard_deviation = block.std(ddof=1)
            if standard_deviation > np.finfo(float).eps:
                block_statistics.append(
                    np.ptp(np.cumsum(centered)) / standard_deviation
                )
        rs_values.append(np.mean(block_statistics))
    return np.polyfit(np.log(sizes), np.log(rs_values), deg=1)[0]


def test_matches_independent_rescaled_range_reference_and_is_reproducible():
    values = np.random.default_rng(20260902).normal(size=256)

    result = hurst_exponent(values)

    assert result == pytest.approx(_reference_hurst(values))
    assert result == hurst_exponent(values)


def test_persistent_series_has_higher_estimate_than_white_noise():
    rng = np.random.default_rng(20260902)
    white_noise = rng.normal(size=512)
    persistent = np.cumsum(rng.normal(size=512))

    assert hurst_exponent(persistent) > hurst_exponent(white_noise)


def test_missing_values_are_dropped_without_mutating_input():
    values = np.random.default_rng(4).normal(size=64)
    series = pd.Series(values, name="target")
    series.iloc[[3, 17]] = np.nan
    original = series.copy(deep=True)

    result = hurst_exponent(series)

    assert result == pytest.approx(hurst_exponent(series.dropna()))
    pd.testing.assert_series_equal(series, original)


def test_dataframe_selection_and_root_export_are_supported():
    frame = pd.DataFrame(
        {
            "target": np.random.default_rng(3).normal(size=80),
            "other": np.arange(80.0),
        }
    )

    assert root_hurst_exponent is hurst_exponent
    assert hurst_exponent(frame, variable="target") == pytest.approx(
        hurst_exponent(frame["target"])
    )


@pytest.mark.parametrize(
    ("data", "message"),
    [
        pytest.param(np.ones(32), "constant", id="constant"),
        pytest.param(np.arange(19.0), "at least 20", id="short"),
        pytest.param(
            [1.0, np.inf, *list(range(30))],
            "infinite",
            id="infinite",
        ),
        pytest.param([True, False] * 16, "numeric", id="boolean"),
    ],
)
def test_rejects_invalid_series(data, message):
    with pytest.raises((TypeError, ValueError), match=message):
        hurst_exponent(data)


def test_missing_raise_policy_is_explicit():
    values = np.arange(32.0)
    values[4] = np.nan

    with pytest.raises(ValueError, match="must not contain missing"):
        hurst_exponent(values, missing="raise")


def test_dataframe_variable_and_missing_policy_are_validated():
    frame = pd.DataFrame({"a": np.arange(30.0), "b": np.arange(30.0)})

    with pytest.raises(ValueError, match="variable must be specified"):
        hurst_exponent(frame)
    with pytest.raises(ValueError, match="not a DataFrame column"):
        hurst_exponent(frame, variable="missing")
    with pytest.raises(ValueError, match="missing must be one of"):
        hurst_exponent(frame["a"], missing="interpolate")
