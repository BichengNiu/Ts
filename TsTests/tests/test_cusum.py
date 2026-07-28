"""Tests for the OLS-residual CUSUM stability test."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.diagnostic import breaks_cusumolsresid

from Ts.TsTests._cusum import CUSUMTest, CUSUMTestResult


def test_matches_statsmodels_statistic_pvalue_critical_values_and_path():
    rng = np.random.default_rng(411)
    nobs = 100
    x = rng.normal(size=nobs)
    y = 1.0 + 0.7 * x + rng.normal(size=nobs)
    result = CUSUMTest(y, exog=x).fit()

    fitted = sm.OLS(y, sm.add_constant(x)).fit()
    statistic, pvalue, critical = breaks_cusumolsresid(fitted.resid, ddof=2)
    scale = np.sqrt(np.sum(fitted.resid**2) * nobs / (nobs - 2))

    assert isinstance(result, CUSUMTestResult)
    assert result.statistic == pytest.approx(statistic)
    assert result.pvalue == pytest.approx(pvalue)
    assert result.critical_values == {f"{level}%": value for level, value in critical}
    assert result.critical_values["5%"] == pytest.approx(1.36)
    np.testing.assert_allclose(result.cusum, np.cumsum(fitted.resid) / scale)
    assert result.lags is None


def test_detects_large_intercept_instability():
    rng = np.random.default_rng(93)
    nobs = 160
    x = rng.normal(size=nobs)
    y = 0.5 + 0.8 * x + rng.normal(scale=0.5, size=nobs)
    y[80:] += 4.0
    result = CUSUMTest(y, exog=x).fit()
    assert result.pvalue < 0.01
    assert result.statistic > result.critical_values["1%"]


def test_stable_seeded_regression_is_not_rejected():
    rng = np.random.default_rng(314)
    x = rng.normal(size=160)
    y = -1.0 + 1.2 * x + rng.normal(size=160)
    result = CUSUMTest(y, exog=x).fit()
    assert result.pvalue > 0.05


def test_dataframe_input_uses_explicit_named_columns():
    rng = np.random.default_rng(45)
    frame = pd.DataFrame(
        {
            "year": np.arange(2000, 2040),
            "outcome": rng.normal(size=40),
            "driver": rng.normal(size=40),
        }
    )
    result = CUSUMTest(
        frame,
        y_col="outcome",
        time_col="year",
        exog_cols=["driver"],
    ).fit()

    assert result.time_index[0] == 2000
    assert set(result.coefficients) == {"const", "driver"}


def test_zero_residual_variance_is_rejected():
    x = np.arange(30, dtype=float)
    y = 1.0 + 2.0 * x
    with pytest.raises(ValueError, match="positive residual variance"):
        CUSUMTest(y, exog=x).fit()


def test_summary_and_plot_contracts():
    rng = np.random.default_rng(18)
    x = rng.normal(size=80)
    y = 1.0 + x + rng.normal(size=80)
    result = CUSUMTest(y, exog=x).fit()

    assert "Regression parameters are stable" in str(result)
    assert "Lags:           N/A" in str(result)
    fig, ax = result.plot_test(alpha=0.05)
    assert ax.get_title() == "OLS Residual CUSUM Stability Test"
    assert len(ax.lines) == 4
    plt.close(fig)

    with pytest.raises(ValueError, match="alpha"):
        result.plot_test(alpha=0.025)
