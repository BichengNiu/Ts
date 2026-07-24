"""Tests for NormalityTest."""

import numpy as np
import pytest

from Ts.TsTests import NormalityTest, NormalityTestResult
from Ts.TsTests._base import BaseTest, BaseTestResult


class TestNormalityTestResult:
    """Tests for NormalityTestResult dataclass."""

    def test_result_is_base_test_result(self):
        """NormalityTestResult must inherit from BaseTestResult."""
        assert issubclass(NormalityTestResult, BaseTestResult)

    def test_result_stores_common_fields(self):
        """NormalityTestResult must have: statistic, pvalue, lags, nobs."""
        result = NormalityTestResult(
            statistic=2.5, pvalue=0.29, lags=0, nobs=200,
        )
        assert result.statistic == 2.5
        assert result.pvalue == 0.29
        assert result.lags == 0
        assert result.nobs == 200

    def test_result_stores_skewness_kurtosis(self):
        """NormalityTestResult must store skewness and kurtosis."""
        np.random.seed(42)
        y = np.random.randn(200)
        test = NormalityTest(y)
        result = test.fit()
        assert hasattr(result, "skewness")
        assert hasattr(result, "kurtosis")
        assert isinstance(result.skewness, float)
        assert isinstance(result.kurtosis, float)


class TestNormalityTest:
    """Tests for NormalityTest class."""

    def test_is_base_test(self):
        """NormalityTest must inherit from BaseTest."""
        assert issubclass(NormalityTest, BaseTest)

    def test_fit_returns_result(self):
        """fit() must return a NormalityTestResult."""
        np.random.seed(42)
        y = np.random.randn(200)
        test = NormalityTest(y)
        result = test.fit()
        assert isinstance(result, NormalityTestResult)

    def test_fit_sets_result_attribute(self):
        """After fit(), result_ attribute must be populated."""
        np.random.seed(42)
        y = np.random.randn(200)
        test = NormalityTest(y)
        test.fit()
        assert test.result_ is not None
        assert isinstance(test.result_, NormalityTestResult)

    def test_normal_data_not_rejected(self):
        """Normal data should not reject H0 (p > 0.05)."""
        np.random.seed(42)
        y = np.random.randn(500)
        test = NormalityTest(y)
        result = test.fit()
        assert result.pvalue > 0.05

    def test_non_normal_data_rejected(self):
        """Non-normal (t with low df) data should reject H0 (p < 0.05)."""
        np.random.seed(42)
        y = np.random.standard_t(df=3, size=500)
        test = NormalityTest(y)
        result = test.fit()
        assert result.pvalue < 0.05

    def test_summary_returns_string(self):
        """summary() must return a non-empty string."""
        np.random.seed(42)
        y = np.random.randn(200)
        test = NormalityTest(y)
        s = test.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert "Jarque-Bera" in s

    def test_result_str_contains_key_info(self):
        """str(result) must contain key statistics."""
        np.random.seed(42)
        y = np.random.randn(200)
        test = NormalityTest(y)
        result = test.fit()
        s = str(result)
        assert "Skewness" in s
        assert "Kurtosis" in s

    def test_too_few_observations_raises(self):
        """Too few observations must raise ValueError."""
        y = np.random.randn(5)
        with pytest.raises(ValueError):
            NormalityTest(y)

    def test_normality_plot_test(self):
        """NormalityTestResult.plot_test() should exist and return fig, ax."""
        import matplotlib
        matplotlib.use('Agg')
        np.random.seed(42)
        data = np.random.randn(100)
        nt = NormalityTest(data)
        nt.fit()
        import matplotlib.pyplot as plt
        fig, ax = nt.result_.plot_test()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
