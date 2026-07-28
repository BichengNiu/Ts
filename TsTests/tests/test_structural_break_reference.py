"""Reference-contract tests for structural-break unit-root methods."""

import numpy as np
import pytest

from Ts.TsTests import PerronTest, ZivotAndrewsTest
from Ts.TsTests._break_utils import (
    _make_perron_break_dummies,
    _make_zivot_break_dummies,
)
from Ts.TsTests._critical_values import _perron_crit


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("intercept", {"DL", "DP"}),
        ("slope", {"DL", "DT"}),
        ("both", {"DL", "DP", "DT"}),
    ],
)
def test_perron_models_use_published_dummy_sets(model, expected):
    assert set(_make_perron_break_dummies(100, 50, model)) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("intercept", {"DL"}),
        ("slope", {"DT"}),
        ("both", {"DL", "DT"}),
    ],
)
def test_zivot_models_keep_distinct_dummy_sets(model, expected):
    assert set(_make_zivot_break_dummies(100, 50, model)) == expected


@pytest.mark.parametrize(
    ("model", "fraction", "significance", "expected"),
    [
        ("intercept", 0.1, 0.01, -4.30),
        ("intercept", 0.9, 0.10, -3.38),
        ("slope", 0.5, 0.05, -3.96),
        ("both", 0.5, 0.01, -4.90),
        ("both", 0.8, 0.025, -4.31),
    ],
)
def test_perron_critical_values_match_published_table(
    model, fraction, significance, expected
):
    assert _perron_crit(model, fraction, significance) == pytest.approx(expected)


def test_perron_critical_values_interpolate_by_break_fraction():
    assert _perron_crit("intercept", 0.15, 0.05) == pytest.approx(-3.725)


def test_perron_result_records_matched_label_and_fraction():
    rng = np.random.default_rng(419)
    y = np.cumsum(rng.normal(size=80))
    labels = 2000 + np.arange(80) * 2
    result = PerronTest(y, break_year=2080, time_index=labels, lags=1).fit()

    assert result.break_index == 40
    assert result.break_year == 2080
    assert result.break_fraction == pytest.approx(0.5)


@pytest.mark.parametrize("test_class", [PerronTest, ZivotAndrewsTest])
def test_time_labels_do_not_change_deterministic_trend(test_class):
    rng = np.random.default_rng(831)
    y = np.cumsum(rng.normal(size=80))
    positional = np.arange(80, dtype=float)
    irregular = np.cumsum(np.linspace(1.0, 3.0, 80))
    if test_class is PerronTest:
        first = test_class(y, break_year=40, time_index=positional, lags=1).fit()
        second = test_class(
            y,
            break_year=irregular[40],
            time_index=irregular,
            lags=1,
        ).fit()
    else:
        first = test_class(y, time_index=positional, lags=1).fit()
        second = test_class(y, time_index=irregular, lags=1).fit()

    assert second.statistic == pytest.approx(first.statistic)
    assert second.break_index == first.break_index


@pytest.mark.parametrize("test_class", [PerronTest, ZivotAndrewsTest])
def test_rejection_text_states_unit_root_null(test_class):
    rng = np.random.default_rng(73)
    y = rng.normal(size=100)
    kwargs = {"break_year": 50} if test_class is PerronTest else {}
    result = test_class(y, lags=0, **kwargs).fit()
    assert "Reject H0 (unit root)" in str(result)
    assert "Reject H0 (stationary with break)" not in str(result)
