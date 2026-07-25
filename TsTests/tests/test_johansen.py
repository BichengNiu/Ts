"""Tests for Ts.TsTests._johansen — Johansen cointegration test."""

import numpy as np
import pytest

from Ts.TsTests._johansen import JohansenTest


@pytest.fixture
def bivariate_data():
    """Generate 2-variable data for Johansen testing.

    Two independent random walks, no cointegration (rank=0).
    """
    np.random.seed(42)
    n = 200
    return np.random.randn(n, 2).cumsum(axis=0)


class TestJohansenFit:
    """Test JohansenTest.fit() execution and result structure."""

    def test_fit_returns_result(self, bivariate_data):
        """fit() returns a JohansenTestResult and stores it in result_."""
        from Ts.TsTests._johansen import JohansenTest, JohansenTestResult

        test = JohansenTest(bivariate_data, lags=2, trend="constant")
        result = test.fit()
        assert isinstance(result, JohansenTestResult)
        assert test.result_ is result

    def test_result_has_eigenvalues(self, bivariate_data):
        """Result contains eigenvalues sorted descending."""
        from Ts.TsTests._johansen import JohansenTest

        test = JohansenTest(bivariate_data, lags=2)
        result = test.fit()
        assert result.eigenvalues is not None
        assert len(result.eigenvalues) == bivariate_data.shape[1]
        # eigenvalues should be sorted descending
        for i in range(len(result.eigenvalues) - 1):
            assert result.eigenvalues[i] >= result.eigenvalues[i + 1]

    def test_result_has_trace_statistics(self, bivariate_data):
        """Result contains trace statistics for each rank."""
        from Ts.TsTests._johansen import JohansenTest

        test = JohansenTest(bivariate_data, lags=2)
        result = test.fit()
        k = bivariate_data.shape[1]
        assert len(result.trace_statistics) == k
        assert result.trace_statistics.dtype == np.float64
        # trace stats should be non-negative and generally decreasing
        for i in range(k):
            assert result.trace_statistics[i] >= 0

    def test_result_has_maxeig_statistics(self, bivariate_data):
        """Result contains max-eigenvalue statistics for each rank."""
        from Ts.TsTests._johansen import JohansenTest

        test = JohansenTest(bivariate_data, lags=2)
        result = test.fit()
        k = bivariate_data.shape[1]
        assert len(result.maxeig_statistics) == k
        # max-eig stats should be non-negative
        for i in range(k):
            assert result.maxeig_statistics[i] >= 0

    def test_result_has_critical_values(self, bivariate_data):
        """Result contains critical values for both trace and max-eig tests."""
        from Ts.TsTests._johansen import JohansenTest

        test = JohansenTest(bivariate_data, lags=2)
        result = test.fit()
        k = bivariate_data.shape[1]
        assert result.trace_critical_values is not None
        assert result.maxeig_critical_values is not None
        # critical values shape: (k, 3) for 90%, 95%, 99%
        assert result.trace_critical_values.shape == (k, 3)
        assert result.maxeig_critical_values.shape == (k, 3)

    def test_result_has_rank(self, bivariate_data):
        """Result contains cointegration rank determined by trace test."""
        from Ts.TsTests._johansen import JohansenTest

        test = JohansenTest(bivariate_data, lags=2)
        result = test.fit()
        assert isinstance(result.rank, int)
        assert 0 <= result.rank <= bivariate_data.shape[1]

    def test_result_inherits_base_fields(self, bivariate_data):
        """Result inherits statistic, pvalue, lags, nobs from BaseTestResult."""
        from Ts.TsTests._johansen import JohansenTest

        test = JohansenTest(bivariate_data, lags=2)
        result = test.fit()
        assert isinstance(result.statistic, float)
        assert result.pvalue is None
        assert result.lags == 2
        assert result.nobs > 0


class TestJohansenSummary:
    """Test JohansenTestResult.summary() formatted output."""

    def test_summary_contains_header(self, bivariate_data):
        """summary() contains test name, trend, sample, lags header."""
        from Ts.TsTests._johansen import JohansenTest

        test = JohansenTest(bivariate_data, lags=2, trend="constant")
        result = test.fit()
        output = result.summary()
        assert "Johansen tests for cointegration" in output
        assert "Trend: constant" in output
        assert "Lags = 2" in output

    def test_summary_contains_trace_table(self, bivariate_data):
        """summary() contains trace test table with rank, eigenvalue,
        statistic, and critical value columns.
        """
        from Ts.TsTests._johansen import JohansenTest

        test = JohansenTest(bivariate_data, lags=2)
        result = test.fit()
        output = result.summary()
        assert "Trace test" in output
        assert "rank" in output.lower()
        assert "eigenvalue" in output.lower()
        assert "statistic" in output.lower()
        assert "critical value" in output.lower()

    def test_summary_contains_maxeig_table(self, bivariate_data):
        """summary() contains max-eigenvalue test table."""
        from Ts.TsTests._johansen import JohansenTest

        test = JohansenTest(bivariate_data, lags=2)
        result = test.fit()
        output = result.summary()
        assert "Max-eigenvalue test" in output

    def test_summary_contains_rank_conclusion(self, bivariate_data):
        """summary() reports cointegration rank at the end."""
        from Ts.TsTests._johansen import JohansenTest

        test = JohansenTest(bivariate_data, lags=2)
        result = test.fit()
        output = result.summary()
        assert "Cointegration rank" in output
        assert str(result.rank) in output

    def test_summary_star_marks_first_non_rejected(self, bivariate_data):
        """summary() marks the first non-rejected rank with *."""
        from Ts.TsTests._johansen import JohansenTest

        test = JohansenTest(bivariate_data, lags=2)
        result = test.fit()
        output = result.summary()
        # At least one row (the first non-rejected) should have '*'
        assert "*" in output

    def test_summary_rank_zero_eigenvalue_dot(self, bivariate_data):
        """summary() shows '.' for eigenvalue at rank 0."""
        from Ts.TsTests._johansen import JohansenTest

        test = JohansenTest(bivariate_data, lags=2)
        result = test.fit()
        output = result.summary()
        # Rank 0 row should show '.' for eigenvalue
        assert "." in output


class TestJohansenInit:
    """Test JohansenTest construction and parameter validation."""

    def test_init_stores_data_and_params(self, bivariate_data):
        """JohansenTest stores 2-D data, lags, and trend parameters."""
        from Ts.TsTests._johansen import JohansenTest

        test = JohansenTest(bivariate_data, lags=2, trend="constant")
        assert test.lags == 2
        assert test.trend == "constant"
        assert test.data.shape == (200, 2)
        assert test.result_ is None

    def test_default_trend_is_constant(self, bivariate_data):
        """Default trend is 'constant'."""
        from Ts.TsTests._johansen import JohansenTest

        test = JohansenTest(bivariate_data, lags=2)
        assert test.trend == "constant"

    @pytest.mark.parametrize("trend", ["none", "constant", "trend"])
    def test_supported_trends_return_finite_critical_values(
        self,
        bivariate_data,
        trend,
    ):
        """Every documented trend maps to a supported statsmodels det_order."""
        result = JohansenTest(
            bivariate_data,
            lags=2,
            trend=trend,
        ).fit()

        assert np.all(np.isfinite(result.trace_critical_values))
        assert np.all(np.isfinite(result.maxeig_critical_values))

    def test_1d_data_raises(self, bivariate_data):
        """1-D data raises ValueError."""
        from Ts.TsTests._johansen import JohansenTest

        with pytest.raises(ValueError):
            JohansenTest(np.array([1.0, 2.0, 3.0]), lags=1)

    def test_invalid_trend_raises(self, bivariate_data):
        """Invalid trend specification raises ValueError."""
        from Ts.TsTests._johansen import JohansenTest

        with pytest.raises(ValueError):
            JohansenTest(bivariate_data, lags=2, trend="invalid")

    @pytest.mark.parametrize("trend", ["rconstant", "rtrend"])
    def test_unsupported_five_case_aliases_raise(self, bivariate_data, trend):
        """Do not claim unsupported restricted deterministic specifications."""
        with pytest.raises(ValueError, match="trend must be one of"):
            JohansenTest(bivariate_data, lags=2, trend=trend)

    def test_lags_must_produce_nonnegative_difference_order(self, bivariate_data):
        """lags=0 would pass k_ar_diff=-1 to statsmodels."""
        with pytest.raises(ValueError, match="lags must be >= 1"):
            JohansenTest(bivariate_data, lags=0)

    def test_k1_data_raises(self, bivariate_data):
        """Single variable (k=1) raises ValueError -- no cointegration possible."""
        from Ts.TsTests._johansen import JohansenTest

        with pytest.raises(ValueError):
            JohansenTest(np.random.randn(100, 1), lags=1)
