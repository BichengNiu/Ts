"""Tests for LjungBoxTest (RED phase — tests written before implementation)."""

import numpy as np
import pytest

# These imports will fail until the modules are created — expected RED behavior.
from Ts.TsTests import LjungBoxTest, LjungBoxTestResult
from Ts.TsTests._base import BaseTest, BaseTestResult


class TestLjungBoxTestResult:
    """Tests for LjungBoxTestResult dataclass."""

    def test_result_is_base_test_result(self):
        """LjungBoxTestResult must inherit from BaseTestResult."""
        assert issubclass(LjungBoxTestResult, BaseTestResult)

    def test_result_stores_common_fields(self):
        """LjungBoxTestResult must have: statistic, pvalue, lags, nobs, residuals."""
        result = LjungBoxTestResult(
            statistic=15.3,
            pvalue=0.12,
            lags=10,
            nobs=100,
        )
        assert result.statistic == 15.3
        assert result.pvalue == 0.12
        assert result.lags == 10
        assert result.nobs == 100
        assert result.residuals is None

    def test_result_stores_individual_lag_stats(self):
        """LjungBoxTestResult must store per-lag statistics after fit()."""
        np.random.seed(42)
        y = np.random.randn(200)
        test = LjungBoxTest(y, lags=5)
        result = test.fit()
        assert result.individual_lags is not None
        assert len(result.individual_lags) == 5
        assert result.individual_stats is not None
        assert len(result.individual_stats) == 5
        assert result.individual_pvalues is not None
        assert len(result.individual_pvalues) == 5
        assert result.apply_squared is True


class TestLjungBoxTest:
    """Tests for LjungBoxTest class."""

    def test_is_base_test(self):
        """LjungBoxTest must inherit from BaseTest."""
        assert issubclass(LjungBoxTest, BaseTest)

    def test_fit_returns_result(self):
        """fit() must return a LjungBoxTestResult."""
        np.random.seed(42)
        y = np.random.randn(200)  # white noise
        test = LjungBoxTest(y, lags=5)
        result = test.fit()
        assert isinstance(result, LjungBoxTestResult)

    def test_fit_sets_result_attribute(self):
        """After fit(), result_ attribute must be populated."""
        np.random.seed(42)
        y = np.random.randn(200)
        test = LjungBoxTest(y, lags=5)
        test.fit()
        assert test.result_ is not None
        assert isinstance(test.result_, LjungBoxTestResult)

    def test_white_noise_no_arch(self):
        """White noise should not reject H0 (no ARCH effects)."""
        np.random.seed(42)
        y = np.random.randn(500)
        test = LjungBoxTest(y, lags=10)
        result = test.fit()
        # For white noise, Q-stat should be small, p-value should be large
        assert result.pvalue > 0.05

    def test_arch_process_rejects_h0(self):
        """Data with ARCH effects should reject H0."""
        np.random.seed(42)
        n = 500
        # Generate ARCH(1) process: y_t = sigma_t * eps_t, sigma_t^2 = 1 + 0.5*y_{t-1}^2
        eps = np.random.randn(n)
        y = np.zeros(n)
        sigma2 = np.ones(n)
        for t in range(1, n):
            sigma2[t] = 1.0 + 0.5 * y[t - 1] ** 2
            y[t] = np.sqrt(sigma2[t]) * eps[t]
        test = LjungBoxTest(y, lags=10)
        result = test.fit()
        # ARCH(1) data should produce significant Q on squared returns
        assert result.pvalue < 0.05

    def test_apply_squared_false_on_raw_series(self):
        """apply_squared=False tests autocorrelation in the raw series."""
        np.random.seed(42)
        y = np.random.randn(200)
        test = LjungBoxTest(y, lags=5, apply_squared=False)
        result = test.fit()
        assert result.apply_squared is False
        # White noise: p-value should be > 0.05 for raw series too
        assert result.pvalue > 0.05

    def test_summary_returns_string(self):
        """summary() must return a non-empty string."""
        np.random.seed(42)
        y = np.random.randn(200)
        test = LjungBoxTest(y, lags=5)
        s = test.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert "Ljung-Box" in s

    def test_result_str_contains_key_info(self):
        """str(result) must contain key statistics."""
        np.random.seed(42)
        y = np.random.randn(200)
        test = LjungBoxTest(y, lags=5)
        result = test.fit()
        s = str(result)
        assert "Ljung-Box" in s
        assert "Q" in s or "statistic" in s.lower()

    def test_invalid_lags_raises(self):
        """lags <= 0 must raise ValueError."""
        y = np.random.randn(100)
        with pytest.raises(ValueError):
            LjungBoxTest(y, lags=0)

    def test_nan_observations_are_ignored(self):
        """Missing observations are removed before the Ljung-Box calculation."""
        rng = np.random.default_rng(42)
        y = rng.normal(size=100)
        y[10] = np.nan

        result = LjungBoxTest(y, lags=5).fit()

        assert result.nobs == 99

    def test_insufficient_data_raises(self):
        """Too few observations relative to lags must raise ValueError."""
        y = np.random.randn(5)
        with pytest.raises(ValueError):
            LjungBoxTest(y, lags=10)
