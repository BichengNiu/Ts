"""Tests for global Bai-Perron multiple-unknown-break analysis."""

from itertools import combinations, pairwise

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from Ts.TsTests._bai_perron import (
    BaiPerronTest,
    BaiPerronTestResult,
    _global_partitions,
    _global_supf_statistics,
    _information_criteria,
    _sequential_supf_statistics,
)


def _partition_rss(y, x, break_indices):
    bounds = (0, *(index + 1 for index in break_indices), len(y))
    rss = 0.0
    for start, stop in pairwise(bounds):
        beta = np.linalg.lstsq(x[start:stop], y[start:stop], rcond=None)[0]
        residuals = y[start:stop] - x[start:stop] @ beta
        rss += float(residuals @ residuals)
    return rss


def _brute_force_partition(y, x, min_segment_size, n_breaks):
    candidates = range(min_segment_size - 1, len(y) - min_segment_size)
    best_rss = np.inf
    best_breaks = ()
    for breaks in combinations(candidates, n_breaks):
        bounds = (0, *(index + 1 for index in breaks), len(y))
        if min(np.diff(bounds)) < min_segment_size:
            continue
        rss = _partition_rss(y, x, breaks)
        if rss < best_rss:
            best_rss = rss
            best_breaks = breaks
    return best_rss, best_breaks


def test_dynamic_programming_matches_exhaustive_global_search():
    rng = np.random.default_rng(91)
    nobs = 30
    x_raw = rng.normal(size=nobs)
    x = np.column_stack([np.ones(nobs), x_raw])
    y = 1.0 + 0.5 * x_raw + rng.normal(scale=0.3, size=nobs)
    y[10:20] += 2.0 - 1.5 * x_raw[10:20]
    y[20:] -= 1.5

    result = _global_partitions(y, x, min_segment_size=7, max_breaks=2)
    for n_breaks in (1, 2):
        brute_rss, brute_breaks = _brute_force_partition(
            y,
            x,
            min_segment_size=7,
            n_breaks=n_breaks,
        )
        assert result.rss[n_breaks] == pytest.approx(brute_rss)
        assert result.breaks[n_breaks] == brute_breaks


def test_information_criteria_use_documented_bic_and_lwz_penalties():
    rss = {0: 100.0, 1: 70.0, 2: 60.0}
    nobs = 80
    nparams = 2
    bic, lwz = _information_criteria(rss, nobs, nparams)
    deviance = nobs * (np.log(2 * np.pi) + 1 + np.log(rss[1] / nobs))
    parameter_count = 2 * nparams + 1

    assert bic[1] == pytest.approx(deviance + np.log(nobs) * parameter_count)
    assert lwz[1] == pytest.approx(
        deviance + 0.299 * np.log(nobs) ** 2.1 * parameter_count
    )


def test_global_supf_formula_uses_nonpointwise_optimal_rss():
    rss = {0: 120.0, 1: 80.0, 2: 60.0}
    statistics = _global_supf_statistics(rss, nobs=100, nparams=2)
    expected_two = ((120.0 - 60.0) / 4) / (60.0 / 94)
    assert statistics[2] == pytest.approx(expected_two)


def test_sequential_supf_searches_within_fixed_null_segments():
    rng = np.random.default_rng(1)
    nobs = 45
    x = np.column_stack([np.ones(nobs), rng.normal(size=nobs)])
    y = rng.normal(size=nobs)
    y[12:28] += 1.2 * x[12:28, 1] + 1.0
    y[28:] -= 0.8 * x[28:, 1] + 0.5
    partitions = _global_partitions(
        y,
        x,
        min_segment_size=8,
        max_breaks=2,
    )

    null_break = partitions.breaks[1][0]
    fixed_candidates = []
    for segment_start, segment_stop in pairwise((0, null_break + 1, nobs)):
        for split in range(segment_start + 8, segment_stop - 8 + 1):
            candidate_breaks = tuple(sorted((null_break, split - 1)))
            fixed_candidates.append(_partition_rss(y, x, candidate_breaks))
    best_conditional_rss = min(fixed_candidates)
    expected = ((partitions.rss[1] - best_conditional_rss) / x.shape[1]) / (
        best_conditional_rss / (nobs - 3 * x.shape[1])
    )

    statistic = _sequential_supf_statistics(
        partitions,
        nobs,
        x.shape[1],
    )[1]
    globally_reoptimized = ((partitions.rss[1] - partitions.rss[2]) / x.shape[1]) / (
        partitions.rss[2] / (nobs - 3 * x.shape[1])
    )

    assert statistic == pytest.approx(expected)
    assert statistic != pytest.approx(globally_reoptimized)


@pytest.fixture(scope="module")
def strong_two_break_result():
    rng = np.random.default_rng(777)
    nobs = 72
    y = np.r_[
        np.full(24, -2.0),
        np.full(24, 3.0),
        np.full(24, -1.0),
    ]
    y += rng.normal(scale=0.25, size=nobs)
    return BaiPerronTest(
        y,
        min_segment_size=14,
        max_breaks=2,
        n_bootstrap=39,
        random_state=42,
    ).fit()


def test_complete_multiple_break_result_contract(strong_two_break_result):
    result = strong_two_break_result
    assert isinstance(result, BaiPerronTestResult)
    assert result.n_breaks == 2
    assert result.break_indices == (23, 47)
    assert result.break_years == (23.0, 47.0)
    assert result.break_fractions == pytest.approx((24 / 72, 48 / 72))
    assert result.selection_method == "bic"
    assert result.lags is None
    assert set(result.partitions) == {0, 1, 2}
    assert set(result.rss_by_breaks) == {0, 1, 2}
    assert set(result.bic_by_breaks) == {0, 1, 2}
    assert set(result.lwz_by_breaks) == {0, 1, 2}
    assert len(result.segment_coefficients) == 3
    assert result.fitted.shape == (72,)
    assert result.residuals.shape == (72,)


def test_reports_supf_sequential_double_max_and_bootstrap_inference(
    strong_two_break_result,
):
    result = strong_two_break_result
    assert set(result.supf_by_breaks) == {1, 2}
    assert set(result.supf_pvalues) == {1, 2}
    assert set(result.sequential_supf) == {0, 1}
    assert set(result.sequential_pvalues) == {0, 1}
    assert result.udmax == result.statistic
    assert result.udmax_pvalue == result.pvalue
    assert result.udmax_pvalue <= 0.05
    assert result.wdmax_pvalue <= 0.05
    assert set(result.wdmax_weights) == {1, 2}
    assert "UDmax" in result.bootstrap_critical_values
    assert "WDmax" in result.bootstrap_critical_values


def test_bootstrap_break_intervals_and_plot_are_auditable(
    strong_two_break_result,
):
    result = strong_two_break_result
    assert len(result.break_confidence_intervals) == 2
    for estimate, (lower, upper) in zip(
        result.break_indices,
        result.break_confidence_intervals,
        strict=True,
    ):
        assert lower <= estimate <= upper
    assert "not serial-correlation robust" in str(result)

    fig, ax = result.plot_test()
    assert ax.get_title() == "Bai-Perron Global Multiple-Break Partition"
    assert len(ax.lines) == 4
    plt.close(fig)


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(8)
    y = np.r_[np.zeros(20), np.ones(20) * 3]
    y += rng.normal(scale=0.4, size=40)
    kwargs = {
        "min_segment_size": 10,
        "max_breaks": 1,
        "n_bootstrap": 19,
        "random_state": 17,
    }
    first = BaiPerronTest(y, **kwargs).fit()
    second = BaiPerronTest(y, **kwargs).fit()

    assert second.break_indices == first.break_indices
    assert second.udmax_pvalue == first.udmax_pvalue
    assert second.bootstrap_critical_values == first.bootstrap_critical_values
    assert second.break_confidence_intervals == first.break_confidence_intervals


def test_dataframe_input_preserves_named_columns_and_break_labels():
    rng = np.random.default_rng(46)
    nobs = 40
    driver = rng.normal(size=nobs)
    outcome = 1.0 + 0.5 * driver + rng.normal(scale=0.3, size=nobs)
    outcome[20:] += 3.0
    frame = pd.DataFrame(
        {
            "year": np.arange(2000, 2000 + nobs),
            "outcome": outcome,
            "driver": driver,
        }
    )
    result = BaiPerronTest(
        frame,
        y_col="outcome",
        time_col="year",
        exog_cols=["driver"],
        min_segment_size=10,
        max_breaks=1,
        breaks=1,
        n_bootstrap=19,
        random_state=4,
    ).fit()

    assert result.break_years == (2019.0,)
    assert result.break_fractions == pytest.approx((0.5,))
    assert set(result.segment_coefficients[0]) == {"const", "driver"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"trim": 0.0}, "trim"),
        ({"min_segment_size": 1}, "greater than"),
        ({"min_segment_size": 21}, "half"),
        ({"max_breaks": 0}, "at least 1"),
        (
            {"min_segment_size": 10, "max_breaks": 4},
            "admissible maximum",
        ),
        ({"breaks": 2, "max_breaks": 1}, "must not exceed"),
        ({"criterion": "aic"}, "criterion"),
        ({"significance": 0.025}, "significance"),
        ({"confidence_level": 1.0}, "confidence_level"),
        ({"n_bootstrap": 18}, "at least 19"),
    ],
)
def test_configuration_validation(kwargs, message):
    y = np.arange(40, dtype=float) + np.sin(np.arange(40))
    with pytest.raises(ValueError, match=message):
        BaiPerronTest(y, **kwargs)
