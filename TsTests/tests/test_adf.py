"""Tests for Ts.TsTests._adf — ADFTest and ADFTestResult."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest


@pytest.fixture
def random_walk():
    """Generate a pure random walk."""
    rng = np.random.default_rng(42)
    return np.cumsum(rng.standard_normal(200))


@pytest.fixture
def ar1_stationary():
    """Generate a stationary AR(1)."""
    rng = np.random.default_rng(42)
    n = 200
    y = np.zeros(n)
    y[0] = rng.standard_normal()
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal()
    return y


class TestADFTestResult:
    """Test ADFTestResult container."""

    def test_str_contains_key_info(self):
        from Ts.TsTests._adf import ADFTestResult

        result = ADFTestResult(
            statistic=-3.5,
            pvalue=0.01,
            lags=2,
            nobs=100,
            trend="c",
            critical_values={"1%": -3.50, "5%": -2.90, "10%": -2.58},
        )
        text = str(result)
        assert "ADF" in text
        assert "Unit root" in text
        assert "stationary" in text.lower()

    def test_plot_test_returns_fig_ax(self):
        from Ts.TsTests._adf import ADFTestResult

        result = ADFTestResult(
            statistic=-3.5,
            pvalue=0.01,
            lags=2,
            nobs=100,
            trend="c",
            critical_values={"1%": -3.50, "5%": -2.90, "10%": -2.58},
        )
        import matplotlib.pyplot as plt

        fig, ax = result.plot_test()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)


class TestADFTest:
    """Test ADFTest class."""

    def test_random_walk_cannot_reject(self, random_walk):
        """ADF on pure random walk: should NOT reject H0 at 5%."""
        from Ts.TsTests._adf import ADFTest

        test = ADFTest(random_walk, trend="c", lags=1)
        result = test.fit()

        # Random walk → likely cannot reject unit root
        # p-value should be > 0.05 (but this is probabilistic)
        assert result.statistic < 0  # ADF stat is negative
        assert "Cannot reject" in str(result) or result.pvalue > 0.05

    def test_ar1_stationary_rejects(self, ar1_stationary):
        """ADF on stationary AR(1): should reject H0."""
        from Ts.TsTests._adf import ADFTest

        test = ADFTest(ar1_stationary, trend="c", lags=1)
        result = test.fit()

        # Stationary AR(1) → should reject unit root
        assert result.pvalue < 0.05
        assert "stationary" in str(result).lower()

    def test_summary_returns_string(self, random_walk):
        """summary() returns a string."""
        from Ts.TsTests._adf import ADFTest

        test = ADFTest(random_walk, trend="c", lags=1)
        text = test.summary()
        assert isinstance(text, str)
        assert "ADF" in text

    def test_auto_lag_selection(self, random_walk):
        """ADF with auto lag selection works."""
        from Ts.TsTests._adf import ADFTest

        test = ADFTest(random_walk, trend="c", max_lags=8)
        result = test.fit()
        assert result.lags >= 0

    def test_fixed_lags_are_used_exactly(self, random_walk):
        """A fixed ``lags`` value is honoured, not treated as an upper bound."""
        from Ts.TsTests._adf import ADFTest

        result = ADFTest(random_walk, trend="c", lags=3).fit()
        assert result.lags == 3

    def test_fixed_lags_zero_is_accepted(self, random_walk):
        """``lags=0`` fits the no-lag regression."""
        from Ts.TsTests._adf import ADFTest

        result = ADFTest(random_walk, trend="c", lags=0).fit()
        assert result.lags == 0

    def test_trend_ct(self, random_walk):
        """ADF with trend='ct' works."""
        from Ts.TsTests._adf import ADFTest

        test = ADFTest(random_walk, trend="ct", lags=1)
        result = test.fit()
        assert result.trend == "ct"

    def test_invalid_trend_raises(self, random_walk):
        """ADF with invalid trend raises ValueError."""
        from Ts.TsTests._adf import ADFTest

        with pytest.raises(ValueError):
            ADFTest(random_walk, trend="invalid")

    def test_deterministic_trend_rejects(self):
        """ADF on deterministic trend should reject unit root."""
        from Ts.TsTests._adf import ADFTest

        rng = np.random.default_rng(42)
        t = np.arange(200)
        y = 5.0 + 0.3 * t + rng.standard_normal(200)  # Trend-stationary

        test = ADFTest(y, trend="ct", lags=1)
        result = test.fit()

        # TS process under correct spec (ct) should reject H0
        assert result.pvalue < 0.05
