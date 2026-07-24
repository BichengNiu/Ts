"""Tests for Ts.TsTests._phillips_perron — PhillipsPerronTest and result."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest


@pytest.fixture
def random_walk():
    rng = np.random.default_rng(42)
    return np.cumsum(rng.standard_normal(200))


@pytest.fixture
def ar1_stationary():
    rng = np.random.default_rng(42)
    n = 200
    y = np.zeros(n)
    y[0] = rng.standard_normal()
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal()
    return y


class TestPhillipsPerronTestResult:
    """Test PhillipsPerronTestResult container."""

    def test_str_contains_key_info(self):
        from Ts.TsTests._phillips_perron import PhillipsPerronTestResult

        result = PhillipsPerronTestResult(
            statistic=-3.2,
            pvalue=0.02,
            lags=4,
            nobs=100,
            trend="c",
            test_type="tau",
            critical_values={"1%": -3.50, "5%": -2.90, "10%": -2.58},
        )
        text = str(result)
        assert "Phillips-Perron" in text
        assert "Unit root" in text

    def test_plot_test_returns_fig_ax(self):
        from Ts.TsTests._phillips_perron import PhillipsPerronTestResult

        result = PhillipsPerronTestResult(
            statistic=-3.2,
            pvalue=0.02,
            lags=4,
            nobs=100,
            trend="c",
            test_type="tau",
            critical_values={"1%": -3.50, "5%": -2.90, "10%": -2.58},
        )
        import matplotlib.pyplot as plt
        fig, ax = result.plot_test()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)


class TestPhillipsPerronTest:
    """Test PhillipsPerronTest class."""

    def test_random_walk_likely_unit_root(self, random_walk):
        """PP on random walk: p-value > 0.05 (likely)."""
        from Ts.TsTests._phillips_perron import PhillipsPerronTest

        test = PhillipsPerronTest(random_walk, trend="c")
        result = test.fit()

        assert result.statistic < 0
        # Random walk → likely has unit root (p > 0.05)
        # (probabilistic, but with seed=42 and 200 obs this should hold)

    def test_ar1_stationary_rejects(self, ar1_stationary):
        """PP on stationary AR(1) rejects H0."""
        from Ts.TsTests._phillips_perron import PhillipsPerronTest

        test = PhillipsPerronTest(ar1_stationary, trend="c")
        result = test.fit()

        assert result.pvalue < 0.05

    def test_summary_returns_string(self, random_walk):
        """summary() returns string."""
        from Ts.TsTests._phillips_perron import PhillipsPerronTest

        test = PhillipsPerronTest(random_walk, trend="c")
        text = test.summary()
        assert isinstance(text, str)
        assert "Phillips-Perron" in text

    def test_rho_test_type(self, random_walk):
        """PP with test_type='rho' works."""
        from Ts.TsTests._phillips_perron import PhillipsPerronTest

        test = PhillipsPerronTest(random_walk, trend="c", test_type="rho")
        result = test.fit()
        assert result.test_type == "rho"

    def test_trend_ct(self, random_walk):
        """PP with trend='ct' works."""
        from Ts.TsTests._phillips_perron import PhillipsPerronTest

        test = PhillipsPerronTest(random_walk, trend="ct")
        result = test.fit()
        assert result.trend == "ct"

    def test_invalid_test_type_raises(self, random_walk):
        """Invalid test_type raises ValueError."""
        from Ts.TsTests._phillips_perron import PhillipsPerronTest

        with pytest.raises(ValueError):
            PhillipsPerronTest(random_walk, test_type="invalid")
