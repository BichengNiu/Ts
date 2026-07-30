"""Tests for the extended autocorrelation function diagnostic."""

import numpy as np
import pandas as pd
import pytest

from Ts import EACFResult as RootEACFResult
from Ts import eacf as root_eacf
from Ts.TsUtils import EACFResult, eacf


def _acf_at_lag(values, lag):
    centered = np.asarray(values, dtype=float) - np.mean(values)
    return np.dot(centered[lag:], centered[:-lag]) / np.dot(centered, centered)


def test_returns_numeric_significance_and_symbol_tables():
    rng = np.random.default_rng(20260730)
    result = eacf(rng.normal(size=160), ar_max=3, ma_max=4)

    assert isinstance(result, EACFResult)
    assert result.nobs == 160
    assert result.values.shape == (4, 5)
    assert result.significant.shape == (4, 5)
    assert result.symbols.shape == (4, 5)
    np.testing.assert_array_equal(result.ar_orders, np.arange(4))
    np.testing.assert_array_equal(result.ma_orders, np.arange(5))
    np.testing.assert_array_equal(
        result.symbols,
        np.where(result.significant, "x", "o"),
    )
    assert not result.values.flags.writeable
    assert not result.significant.flags.writeable
    assert not result.symbols.flags.writeable


def test_first_row_equals_the_ordinary_sample_acf():
    values = np.array([2.0, 4.0, 1.0, 5.0, 3.0, 8.0, 7.0, 6.0, 9.0, 4.0])

    result = eacf(values, ar_max=0, ma_max=3)

    expected = np.array([_acf_at_lag(values, lag) for lag in range(1, 5)])
    np.testing.assert_allclose(result.values[0], expected, rtol=1e-12, atol=1e-12)


def test_full_table_matches_iterative_least_squares_reference_values():
    values = np.array(
        [
            2.0,
            4.0,
            1.0,
            5.0,
            3.0,
            8.0,
            7.0,
            6.0,
            9.0,
            4.0,
            10.0,
            12.0,
            8.0,
            11.0,
            15.0,
            13.0,
            14.0,
            16.0,
            12.0,
            18.0,
        ]
    )
    expected = np.array(
        [
            [
                0.601544149630274,
                0.621531100478469,
                0.497759895606786,
                0.303958242714224,
            ],
            [
                -0.583222437365842,
                -0.014648833750531,
                0.328459111368126,
                -0.329387365575431,
            ],
            [
                -0.525598269718784,
                -0.000173663872805,
                0.339874489061918,
                -0.271300379692471,
            ],
        ]
    )

    result = eacf(values, ar_max=2, ma_max=3)

    np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-12)


def test_significance_uses_order_specific_large_sample_threshold():
    rng = np.random.default_rng(17)
    values = rng.normal(size=100)
    result = eacf(values, ar_max=2, ma_max=2)

    thresholds = np.fromfunction(
        lambda p, q: 2.0 / np.sqrt(100 - p - q - 1),
        result.values.shape,
    )
    np.testing.assert_array_equal(
        result.significant,
        np.abs(result.values) > thresholds,
    )


def test_dataframe_variable_is_explicit_and_input_is_not_mutated():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "target": rng.normal(size=40),
            "other": np.arange(40.0),
        }
    )
    original = frame.copy(deep=True)

    with pytest.raises(ValueError, match="variable must be specified"):
        eacf(frame, ar_max=2, ma_max=2)

    result = eacf(frame, ar_max=2, ma_max=2, variable="target")

    assert result.nobs == len(frame)
    pd.testing.assert_frame_equal(frame, original)


def test_dataframe_variable_must_identify_an_existing_unique_column():
    frame = pd.DataFrame({"target": np.arange(20.0), "other": np.arange(20.0)})

    with pytest.raises(ValueError, match="not a DataFrame column"):
        eacf(frame, ar_max=1, ma_max=1, variable="missing")

    duplicated = pd.DataFrame(
        np.random.default_rng(3).normal(size=(20, 2)),
        columns=["target", "target"],
    )
    with pytest.raises(ValueError, match="exactly one column"):
        eacf(duplicated, ar_max=1, ma_max=1, variable="target")


def test_summary_labels_orders_and_symbol_meaning():
    result = eacf(np.random.default_rng(7).normal(size=30), ar_max=2, ma_max=2)

    text = result.summary()

    assert "Extended Autocorrelation Function" in text
    assert r"AR\MA" in text
    assert "x = significant" in text
    assert "o = not significant" in text


@pytest.mark.parametrize("name", ["ar_max", "ma_max"])
@pytest.mark.parametrize("value", [-1, 1.5, True, "2"])
def test_orders_must_be_nonnegative_integers(name, value):
    kwargs = {"ar_max": 2, "ma_max": 2, name: value}

    with pytest.raises((TypeError, ValueError), match=name):
        eacf(np.arange(30.0), **kwargs)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        pytest.param([1.0, np.nan, 2.0, 3.0], "missing", id="missing"),
        pytest.param([1.0, np.inf, 2.0, 3.0], "infinite", id="infinite"),
        pytest.param(np.ones(10), "constant", id="constant"),
        pytest.param([True, False, True, False], "numeric", id="boolean"),
        pytest.param([1.0 + 1.0j, 2.0, 3.0, 4.0], "real numeric", id="complex"),
        pytest.param([[1.0, 2.0], [3.0, 4.0]], "one-dimensional", id="2d"),
        pytest.param(["a", "b", "c", "d"], "numeric", id="text"),
    ],
)
def test_rejects_invalid_series(data, message):
    with pytest.raises((TypeError, ValueError), match=message):
        eacf(data, ar_max=0, ma_max=0)


def test_rejects_samples_too_short_for_requested_table():
    with pytest.raises(ValueError, match="at least 13 observations"):
        eacf(np.arange(12.0) ** 2, ar_max=2, ma_max=3)


def test_rejects_rank_deficient_lagged_regression():
    with pytest.raises(ValueError, match="rank deficient at order 3"):
        eacf(np.arange(20.0), ar_max=1, ma_max=1)


def test_public_exports_are_available_from_subpackage_and_root():
    assert root_eacf is eacf
    assert RootEACFResult is EACFResult
