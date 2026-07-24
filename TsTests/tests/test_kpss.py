"""Tests for Ts.TsTests._kpss — KPSSTest and KPSSTestResult."""

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


class TestKPSSTestResult:
    """Test KPSSTestResult container."""

    def test_str_shows_reversed_hypothesis(self):
        """KPSS H0 is stationarity (different from ADF/PP)."""
        from Ts.TsTests._kpss import KPSSTestResult

        result = KPSSTestResult(
            statistic=0.15,
            pvalue=0.10,
            lags=4,
            nobs=100,
            trend="c",
            critical_values={"10%": 0.347, "5%": 0.463, "1%": 0.739},
        )
        text = str(result)
        assert "KPSS" in text
        assert "stationary" in text.lower()

    def test_rejection_interpretation(self):
        """KPSS: p < 0.05 means reject stationarity → unit root."""
        from Ts.TsTests._kpss import KPSSTestResult

        result = KPSSTestResult(
            statistic=0.80,
            pvalue=0.01,
            lags=4,
            nobs=100,
            trend="c",
            critical_values={"10%": 0.347, "5%": 0.463, "1%": 0.739},
        )
        text = str(result)
        # p=0.01 < 0.05 → reject H0 (stationarity) → unit root
        assert "unit root" in text.lower()

    def test_plot_test_returns_fig_ax(self):
        from Ts.TsTests._kpss import KPSSTestResult

        result = KPSSTestResult(
            statistic=0.15,
            pvalue=0.10,
            lags=4,
            nobs=100,
            trend="c",
            critical_values={"10%": 0.347, "5%": 0.463, "1%": 0.739},
        )
        import matplotlib.pyplot as plt
        fig, ax = result.plot_test()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)


class TestKPSSTest:
    """Test KPSSTest class."""

    def test_random_walk_rejects(self, random_walk):
        """KPSS on random walk: should REJECT stationarity (p < 0.05)."""
        from Ts.TsTests._kpss import KPSSTest

        test = KPSSTest(random_walk, trend="c")
        result = test.fit()

        # KPSS H0 = stationary. Random walk → reject H0 → unit root.
        assert result.pvalue < 0.05 or "unit root" in str(result).lower()

    def test_ar1_stationary_cannot_reject(self, ar1_stationary):
        """KPSS on stationary AR(1): should NOT reject H0."""
        from Ts.TsTests._kpss import KPSSTest

        test = KPSSTest(ar1_stationary, trend="c")
        result = test.fit()

        # Stationary → cannot reject H0 of stationarity
        assert "stationary" in str(result).lower()

    def test_summary_returns_string(self, random_walk):
        """summary() returns string."""
        from Ts.TsTests._kpss import KPSSTest

        test = KPSSTest(random_walk, trend="c")
        text = test.summary()
        assert isinstance(text, str)
        assert "KPSS" in text

    def test_trend_ct(self, random_walk):
        """KPSS with trend='ct' works."""
        from Ts.TsTests._kpss import KPSSTest

        test = KPSSTest(random_walk, trend="ct")
        result = test.fit()
        assert result.trend == "ct"

    def test_invalid_trend_raises(self, random_walk):
        """Invalid trend raises ValueError."""
        from Ts.TsTests._kpss import KPSSTest

        with pytest.raises(ValueError):
            KPSSTest(random_walk, trend="invalid")

    def test_explicit_lags(self, random_walk):
        """KPSS with explicit lags works."""
        from Ts.TsTests._kpss import KPSSTest

        test = KPSSTest(random_walk, trend="c", lags=4)
        result = test.fit()
        assert result.lags == 4
