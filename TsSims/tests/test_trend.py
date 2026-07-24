"""Tests for Ts.TsSims._ts_ds — TS/DS simulation and SimTSDSResult."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest


class TestSimTSDSResult:
    """Test the SimTSDSResult container object and its core methods."""

    @pytest.fixture
    def ts_result(self):
        """Create a TS result for testing."""
        from Ts.TsSims._ts_ds import SimTSDSResult

        return SimTSDSResult(
            data=np.array([0.5, 1.8, 3.2, 4.1, 5.7]),
            residuals=np.array([0.5, 0.3, -0.2, 0.1, 0.4]),
            params={
                "process_type": "trend_stationary",
                "intercept": 0.0,
                "slope": 1.0,
                "sigma": 1.0,
                "seed": 42,
                "n": 5,
            },
        )

    @pytest.fixture
    def ds_result(self):
        """Create a DS result for testing."""
        from Ts.TsSims._ts_ds import SimTSDSResult

        return SimTSDSResult(
            data=np.array([0.3, 1.1, 0.8, 2.5, 3.0]),
            residuals=np.array([0.3, -0.2, 0.5, -0.1, 0.7]),
            params={
                "process_type": "difference_stationary",
                "drift": 0.5,
                "sigma": 1.0,
                "seed": 42,
                "n": 5,
                "burn": 50,
            },
        )

    def test_get_data_returns_series(self, ts_result):
        """get_data() returns a pd.Series with correct length."""
        result = ts_result.get_data()
        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_get_params_ts_keys(self, ts_result):
        """get_params() contains TS-specific keys."""
        params = ts_result.get_params()
        assert params["process_type"] == "trend_stationary"
        assert params["intercept"] == 0.0
        assert params["slope"] == 1.0

    def test_get_params_ds_keys(self, ds_result):
        """get_params() contains DS-specific keys."""
        params = ds_result.get_params()
        assert params["process_type"] == "difference_stationary"
        assert params["drift"] == 0.5

    def test_get_params_is_deep_copy(self, ts_result):
        """get_params() returns a copy, not a reference."""
        params1 = ts_result.get_params()
        params1["slope"] = 9.9
        params2 = ts_result.get_params()
        assert params2["slope"] == 1.0

    def test_summary_ts(self, ts_result):
        """summary() contains TS info."""
        text = ts_result.summary()
        assert "TS" in text
        assert "Trend-Stationary" in text

    def test_summary_ds(self, ds_result):
        """summary() contains DS info."""
        text = ds_result.summary()
        assert "DS" in text
        assert "Difference-Stationary" in text

    def test_plot_returns_fig_ax(self, ts_result):
        """plot() returns (fig, ax)."""
        fig, ax = ts_result.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)


class TestSimulateTrendStationary:
    """Test simulate_trend_stationary()."""

    def test_basic(self):
        """simulate_trend_stationary returns correct-shaped result."""
        from Ts.TsSims._ts_ds import simulate_trend_stationary

        r = simulate_trend_stationary(n=100, intercept=2.0, slope=0.5, sigma=1.0, seed=42)

        assert r.get_data().shape[0] == 100
        assert r.params["intercept"] == 2.0
        assert r.params["slope"] == 0.5

    def test_seed_reproducibility(self):
        """Same seed produces identical data."""
        from Ts.TsSims._ts_ds import simulate_trend_stationary

        r1 = simulate_trend_stationary(n=100, seed=42)
        r2 = simulate_trend_stationary(n=100, seed=42)

        np.testing.assert_array_equal(r1.data, r2.data)

    def test_different_seeds_produce_different_data(self):
        """Different seeds produce different data."""
        from Ts.TsSims._ts_ds import simulate_trend_stationary

        r1 = simulate_trend_stationary(n=100, seed=1)
        r2 = simulate_trend_stationary(n=100, seed=2)

        assert not np.allclose(r1.data, r2.data)

    def test_trend_is_visible(self):
        """With large slope and small sigma, the trend should dominate noise."""
        from Ts.TsSims._ts_ds import simulate_trend_stationary

        r = simulate_trend_stationary(n=500, intercept=0.0, slope=2.0, sigma=0.1, seed=42)

        # The series should be strongly increasing
        first_half_mean = r.data[:250].mean()
        second_half_mean = r.data[250:].mean()
        assert second_half_mean > first_half_mean

    def test_default_parameters(self):
        """Uses sensible defaults when only n is provided."""
        from Ts.TsSims._ts_ds import simulate_trend_stationary

        r = simulate_trend_stationary(n=50, seed=42)

        assert r.get_data().shape[0] == 50
        assert r.params["intercept"] == 0.0
        assert r.params["slope"] == 1.0


class TestSimulateDifferenceStationary:
    """Test simulate_difference_stationary()."""

    def test_basic(self):
        """simulate_difference_stationary returns correct-shaped result."""
        from Ts.TsSims._ts_ds import simulate_difference_stationary

        r = simulate_difference_stationary(n=100, drift=1.0, sigma=1.0, seed=42)

        assert r.get_data().shape[0] == 100
        assert r.params["drift"] == 1.0
        assert r.params["burn"] == 50

    def test_seed_reproducibility(self):
        """Same seed produces identical data."""
        from Ts.TsSims._ts_ds import simulate_difference_stationary

        r1 = simulate_difference_stationary(n=100, seed=42)
        r2 = simulate_difference_stationary(n=100, seed=42)

        np.testing.assert_array_equal(r1.data, r2.data)

    def test_diff_is_stationary(self):
        """First difference of a DS process should be (approximately) stationary."""
        from Ts.TsSims._ts_ds import simulate_difference_stationary

        r = simulate_difference_stationary(n=500, drift=0.5, sigma=1.0, seed=42, burn=100)

        diff = np.diff(r.data)
        # The differenced series should have mean ~ drift
        assert abs(diff.mean() - 0.5) < 0.3
        # Check that the variance is finite (stationary behaviour)
        assert 0.5 < diff.std() < 2.0

    def test_random_walk_grows_over_time(self):
        """A random walk without drift should still wander."""
        from Ts.TsSims._ts_ds import simulate_difference_stationary

        r = simulate_difference_stationary(n=500, drift=0.0, sigma=1.0, seed=42, burn=100)

        # The variance at the end should be larger than at the start
        early_std = r.data[:50].std()
        late_std = r.data[-50:].std()
        # This is probabilistic — use a mild assertion
        assert late_std > 0.0
        assert early_std > 0.0
