"""Tests for the classical known-break Chow test."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.stats import f

from Ts.TsTests._chow import ChowTest, ChowTestResult


def _manual_chow(y, x, break_index):
    split = break_index + 1
    pooled = sm.OLS(y, x).fit()
    before = sm.OLS(y[:split], x[:split]).fit()
    after = sm.OLS(y[split:], x[split:]).fit()
    q = x.shape[1]
    df_denom = len(y) - 2 * q
    rss_split = before.ssr + after.ssr
    statistic = ((pooled.ssr - rss_split) / q) / (rss_split / df_denom)
    return statistic, f.sf(statistic, q, df_denom)


def test_matches_manual_chow_formula_and_result_contract():
    rng = np.random.default_rng(281)
    nobs = 100
    x = rng.normal(size=nobs)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.5, size=nobs)
    result = ChowTest(
        y,
        break_year=50,
        exog=x,
        time_index=np.arange(nobs),
    ).fit()
    design = sm.add_constant(x)
    manual_stat, manual_pvalue = _manual_chow(y, design, 50)

    assert isinstance(result, ChowTestResult)
    assert result.statistic == pytest.approx(manual_stat)
    assert result.pvalue == pytest.approx(manual_pvalue)
    assert result.break_index == 50
    assert result.break_year == 50
    assert result.df_num == 2
    assert result.df_denom == 96
    assert result.lags is None
    assert result.fitted_split.shape == (nobs,)
    assert set(result.coefficients_before) == {"const", "x1"}


def test_detects_large_seeded_coefficient_break():
    rng = np.random.default_rng(52)
    nobs = 120
    x = rng.normal(size=nobs)
    y = np.empty(nobs)
    y[:61] = 1.0 + 0.5 * x[:61] + rng.normal(scale=0.3, size=61)
    y[61:] = -2.0 + 2.5 * x[61:] + rng.normal(scale=0.3, size=59)

    result = ChowTest(y, break_year=60, exog=x).fit()
    assert result.pvalue < 1e-20
    assert result.rss_split < result.rss_pooled


def test_stable_seeded_regression_is_not_rejected():
    rng = np.random.default_rng(9)
    x = rng.normal(size=160)
    y = 1.5 - 0.8 * x + rng.normal(size=160)
    result = ChowTest(y, break_year=80, exog=x).fit()
    assert result.pvalue > 0.05


def test_dataframe_input_uses_explicit_named_columns():
    rng = np.random.default_rng(44)
    frame = pd.DataFrame(
        {
            "year": np.arange(2000, 2040),
            "outcome": rng.normal(size=40),
            "driver": rng.normal(size=40),
        }
    )
    result = ChowTest(
        frame,
        break_year=2019,
        y_col="outcome",
        time_col="year",
        exog_cols=["driver"],
    ).fit()

    assert result.break_year == 2019
    assert set(result.coefficients_pooled) == {"const", "driver"}


def test_break_label_must_exist_and_leave_estimable_regimes():
    y = np.arange(20, dtype=float) ** 2
    x = np.arange(20, dtype=float)
    with pytest.raises(ValueError, match="match exactly one"):
        ChowTest(y, break_year=8.5, exog=x)
    with pytest.raises(ValueError, match="insufficient"):
        ChowTest(y, break_year=1, exog=x).fit()


def test_rank_deficient_regime_is_rejected():
    y = np.arange(20, dtype=float) ** 2
    x = np.r_[np.zeros(10), np.arange(10, dtype=float)]
    with pytest.raises(ValueError, match="rank-deficient"):
        ChowTest(y, break_year=9, exog=x).fit()


def test_zero_split_residual_variance_is_rejected():
    x = np.arange(30, dtype=float)
    y = np.empty(30)
    y[:15] = 1.0 + 2.0 * x[:15]
    y[15:] = -3.0 + 0.5 * x[15:]

    with pytest.raises(ValueError, match="positive residual variance"):
        ChowTest(y, break_year=14, exog=x).fit()


def test_summary_and_plot_are_auditable():
    rng = np.random.default_rng(7)
    x = rng.normal(size=60)
    y = 1.0 + x + rng.normal(size=60)
    result = ChowTest(y, break_year=30, exog=x).fit()

    assert "Lags:           N/A" in str(result)
    assert "known break" in str(result)
    fig, ax = result.plot_test()
    assert ax.get_title() == "Chow Test: Known Regression Break"
    assert len(ax.lines) == 4
    plt.close(fig)
