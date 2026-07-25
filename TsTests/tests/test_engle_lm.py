"""Tests for EngleLMTest (RED phase — tests written before implementation)."""

import numpy as np
import pytest

from Ts.TsTests import EngleLMTest, EngleLMTestResult
from Ts.TsTests._base import BaseTest, BaseTestResult


class TestEngleLMTestResult:
    """Tests for EngleLMTestResult dataclass."""

    def test_result_is_base_test_result(self):
        """EngleLMTestResult must inherit from BaseTestResult."""
        assert issubclass(EngleLMTestResult, BaseTestResult)

    def test_result_stores_common_fields(self):
        """EngleLMTestResult must have: statistic, pvalue, lags, nobs, residuals."""
        resid = np.random.randn(100)
        result = EngleLMTestResult(
            statistic=22.5,
            pvalue=0.013,
            lags=5,
            nobs=95,
            residuals=resid,
        )
        assert result.statistic == 22.5
        assert result.pvalue == 0.013
        assert result.lags == 5
        assert result.nobs == 95
        assert len(result.residuals) == 100

    def test_result_stores_lm_specific_fields(self):
        """EngleLMTestResult must store F statistic and R-squared via base fields."""
        resid = np.random.randn(100)
        result = EngleLMTestResult(
            statistic=22.5,
            pvalue=0.013,
            lags=5,
            nobs=95,
            residuals=resid,
            f_statistic=4.89,
            f_pvalue=0.0004,
            rsquared=0.215,
        )
        assert result.statistic == 22.5
        assert result.pvalue == 0.013
        assert result.f_statistic == 4.89
        assert result.f_pvalue == 0.0004
        assert result.rsquared == 0.215


class TestEngleLMTest:
    """Tests for EngleLMTest class."""

    def test_is_base_test(self):
        """EngleLMTest must inherit from BaseTest."""
        assert issubclass(EngleLMTest, BaseTest)

    def test_fit_returns_result(self):
        """fit() must return an EngleLMTestResult."""
        np.random.seed(42)
        y = np.random.randn(200)
        test = EngleLMTest(y, lags=5)
        result = test.fit()
        assert isinstance(result, EngleLMTestResult)

    def test_fit_sets_result_attribute(self):
        """After fit(), result_ must be populated."""
        np.random.seed(42)
        y = np.random.randn(200)
        test = EngleLMTest(y, lags=5)
        test.fit()
        assert test.result_ is not None
        assert isinstance(test.result_, EngleLMTestResult)

    def test_white_noise_no_arch(self):
        """White noise should not reject H0 (no ARCH effects)."""
        np.random.seed(42)
        y = np.random.randn(500)
        test = EngleLMTest(y, lags=10)
        result = test.fit()
        assert result.pvalue > 0.05

    def test_arch_process_rejects_h0(self):
        """Data with ARCH effects should reject H0."""
        np.random.seed(42)
        n = 500
        eps = np.random.randn(n)
        y = np.zeros(n)
        sigma2 = np.ones(n)
        for t in range(1, n):
            sigma2[t] = 1.0 + 0.5 * y[t - 1] ** 2
            y[t] = np.sqrt(sigma2[t]) * eps[t]
        test = EngleLMTest(y, lags=10)
        result = test.fit()
        assert result.pvalue < 0.05

    def test_with_explicit_residuals(self):
        """User can pass pre-computed residuals."""
        np.random.seed(42)
        y = np.random.randn(200)
        custom_resid = y - np.mean(y)
        test = EngleLMTest(y, lags=5, residuals=custom_resid)
        result = test.fit()
        assert result.pvalue > 0.05  # white noise → no ARCH

    def test_summary_returns_string(self):
        """summary() must return a non-empty string."""
        np.random.seed(42)
        y = np.random.randn(200)
        test = EngleLMTest(y, lags=5)
        s = test.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert "Engle" in s
        assert "LM" in s

    def test_result_str_contains_key_info(self):
        """str(result) must contain LM statistic and R-squared."""
        np.random.seed(42)
        y = np.random.randn(200)
        test = EngleLMTest(y, lags=5)
        result = test.fit()
        s = str(result)
        assert "LM" in s
        assert "R" in s

    def test_invalid_lags_raises(self):
        """lags <= 0 must raise ValueError."""
        y = np.random.randn(100)
        with pytest.raises(ValueError):
            EngleLMTest(y, lags=0)

    def test_insufficient_data_raises(self):
        """Too few observations relative to lags must raise ValueError."""
        y = np.random.randn(5)
        with pytest.raises(ValueError):
            EngleLMTest(y, lags=10)

    def test_insufficient_degrees_of_freedom_raises(self):
        """The auxiliary regression needs enough residual degrees of freedom."""
        y = np.random.randn(12)
        with pytest.raises(ValueError, match="22 observations"):
            EngleLMTest(y, lags=10)

    def test_nan_data_keeps_explicit_residuals_aligned(self):
        """Explicit residuals are filtered with the same mask as the data."""
        rng = np.random.default_rng(42)
        y = rng.normal(size=100)
        residuals = y - y.mean()
        y[20] = np.nan

        result = EngleLMTest(y, lags=5, residuals=residuals).fit()

        assert result.nobs == 94
        assert len(result.residuals) == 99

    def test_constant_data_returns_no_arch_result(self):
        """Rank-deficient auxiliary regressions do not fail on constant data."""
        result = EngleLMTest(np.zeros(100), lags=5).fit()

        assert result.statistic == 0.0
        assert result.pvalue == 1.0
        assert np.isnan(result.f_statistic)

    def test_result_stores_individual_lag_stats(self):
        """EngleLMTestResult must store per-lag LM statistics and p-values."""
        np.random.seed(42)
        y = np.random.randn(200)
        test = EngleLMTest(y, lags=5)
        result = test.fit()
        assert result.individual_lags is not None
        assert result.individual_stats is not None
        assert result.individual_pvalues is not None
        assert len(result.individual_lags) == 5
        assert len(result.individual_stats) == 5
        assert len(result.individual_pvalues) == 5
        assert np.array_equal(result.individual_lags, np.arange(1, 6))
        # str output must include per-lag table
        s = str(result)
        assert "Per-Lag Breakdown" in s
        assert all(f"{i}" in s for i in range(1, 6))
        # Last row must match joint result
        assert abs(result.individual_stats[-1] - result.statistic) < 1e-10
        assert abs(result.individual_pvalues[-1] - result.pvalue) < 1e-10
