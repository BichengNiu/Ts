"""Tests for Ts.TsSims._sarima — SARIMA simulation and SimSARIMAResult."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest


class TestSimSARIMAResult:
    """Test the SimSARIMAResult container object and its three core methods."""

    @pytest.fixture
    def simple_result(self):
        """Create a minimal SimSARIMAResult for testing."""
        from Ts.TsSims._sarima import SimSARIMAResult

        return SimSARIMAResult(
            data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            residuals=np.array([0.1, 0.2, 0.1, -0.1, 0.0]),
            params={
                "order": (1, 0, 0),
                "seasonal_order": (0, 0, 0, 0),
                "ar": [0.5],
                "ma": None,
                "const": 0.0,
                "sigma2": 1.0,
                "seed": 42,
                "n": 5,
                "burn": 100,
            },
        )

    def test_get_data_returns_series(self, simple_result):
        """get_data() returns a pd.Series with correct length."""
        result = simple_result.get_data()
        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_get_params_returns_dict_with_correct_keys(self, simple_result):
        """get_params() returns a dict containing all parameter keys."""
        params = simple_result.get_params()
        assert isinstance(params, dict)
        assert params["order"] == (1, 0, 0)
        assert params["ar"] == [0.5]
        assert params["seed"] == 42

    def test_get_params_is_deep_copy(self, simple_result):
        """get_params() returns a copy, not a reference to internal params."""
        params1 = simple_result.get_params()
        params1["ar"] = [0.9]
        params2 = simple_result.get_params()
        assert params2["ar"] == [0.5]

    def test_summary_returns_string_with_key_info(self, simple_result):
        """summary() returns a non-empty string with parameter info."""
        text = simple_result.summary()
        assert isinstance(text, str)
        assert "SARIMA" in text
        assert "0.5" in text

    def test_plot_returns_fig_ax(self, simple_result):
        """plot() returns (fig, ax) and uses TsPlots style."""
        fig, ax = simple_result.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)


class TestSimulateSARIMA:
    """Test the simulate_sarima() convenience function."""

    def test_ar1_basic(self):
        """simulate_sarima with AR(1) returns correct-shaped result."""
        from Ts.TsSims._sarima import simulate_sarima

        result = simulate_sarima(n=100, order=(1, 0, 0), ar=[0.5], seed=42, burn=50)

        assert result.get_data().shape[0] == 100
        assert result.params["order"] == (1, 0, 0)

    def test_seed_reproducibility(self):
        """Same seed produces identical data."""
        from Ts.TsSims._sarima import simulate_sarima

        r1 = simulate_sarima(n=100, order=(1, 0, 0), ar=[0.5], seed=42)
        r2 = simulate_sarima(n=100, order=(1, 0, 0), ar=[0.5], seed=42)

        np.testing.assert_array_equal(r1.data, r2.data)

    def test_different_seed_produces_different_data(self):
        """Different seeds produce different data."""
        from Ts.TsSims._sarima import simulate_sarima

        r1 = simulate_sarima(n=100, order=(1, 0, 0), ar=[0.5], seed=1)
        r2 = simulate_sarima(n=100, order=(1, 0, 0), ar=[0.5], seed=2)

        assert not np.allclose(r1.data, r2.data)

    def test_ma1_basic(self):
        """simulate_sarima with MA(1) returns result."""
        from Ts.TsSims._sarima import simulate_sarima

        result = simulate_sarima(n=100, order=(0, 0, 1), ma=[0.5], seed=42)

        assert result.get_data().shape[0] == 100

    def test_arima_with_differencing(self):
        """simulate_sarima with d=1 integrates correctly (length preserved)."""
        from Ts.TsSims._sarima import simulate_sarima

        result = simulate_sarima(n=100, order=(1, 1, 0), ar=[0.3], seed=42)

        assert result.get_data().shape[0] == 100

    def test_seasonal_sarima(self):
        """simulate_sarima with seasonal order generates data."""
        from Ts.TsSims._sarima import simulate_sarima

        result = simulate_sarima(
            n=200,
            order=(1, 0, 0),
            ar=[0.5],
            seasonal_order=(1, 0, 0, 4),
            seasonal_ar=[0.3],
            seed=42,
        )

        assert result.get_data().shape[0] == 200

    def test_default_parameters(self):
        """simulate_sarima with only n=50 uses sensible defaults."""
        from Ts.TsSims._sarima import simulate_sarima

        result = simulate_sarima(n=50, seed=42)

        assert result.get_data().shape[0] == 50
        assert "order" in result.get_params()


class TestSimulateSARIMAX:
    """Test SARIMAX response simulation built on the SARIMA simulator."""

    def test_constant_is_propagated_through_the_ar_recursion(self):
        """The ARMA intercept produces its long-run mean before integration."""
        from Ts.TsSims._sarima import simulate_sarima

        baseline = simulate_sarima(
            n=40,
            order=(1, 0, 0),
            ar=[0.5],
            const=0.0,
            seed=42,
            burn=0,
        )
        shifted = simulate_sarima(
            n=40,
            order=(1, 0, 0),
            ar=[0.5],
            const=2.0,
            seed=42,
            burn=0,
        )

        np.testing.assert_allclose(shifted.data - baseline.data, 4.0)

    def test_initial_value_conditions_a_differenced_sarimax_path(self):
        """A supplied initial value anchors every simulated integrated path."""
        from Ts.TsSims._sarima import simulate_sarimax

        deterministic = np.linspace(1.0, 2.0, 40)
        baseline = simulate_sarimax(
            n=40,
            order=(1, 1, 0),
            ar=[0.5],
            const=2.0,
            deterministic=deterministic,
            seed=42,
            burn=30,
        )
        anchored = simulate_sarimax(
            n=40,
            order=(1, 1, 0),
            ar=[0.5],
            const=2.0,
            deterministic=deterministic,
            initial_value=100.0,
            seed=42,
            burn=30,
        )

        assert anchored.data[0] == pytest.approx(101.0)
        np.testing.assert_allclose(
            np.diff(anchored.data - deterministic),
            np.diff(baseline.data - deterministic),
        )

    def test_deterministic_path_is_added_after_sarima_simulation(self):
        """A deterministic response path shifts the same seeded error path."""
        from Ts.TsSims._sarima import simulate_sarima, simulate_sarimax

        deterministic = np.linspace(1.0, 2.0, 40)
        base = simulate_sarima(
            n=40,
            order=(1, 0, 0),
            ar=[0.4],
            seed=42,
            burn=30,
        )
        result = simulate_sarimax(
            n=40,
            order=(1, 0, 0),
            ar=[0.4],
            deterministic=deterministic,
            seed=42,
            burn=30,
        )

        np.testing.assert_allclose(result.data, base.data + deterministic)
        assert result.params["model"] == "SARIMAX"
        assert result.params["deterministic"] == pytest.approx(deterministic.tolist())

    @pytest.mark.parametrize(
        "deterministic",
        [np.ones(9), np.full(10, np.inf), np.ones((10, 1))],
    )
    def test_deterministic_path_requires_finite_one_dimensional_length(self, deterministic):
        """Invalid deterministic paths fail before random simulation starts."""
        from Ts.TsSims._sarima import simulate_sarimax

        with pytest.raises((TypeError, ValueError), match="deterministic"):
            simulate_sarimax(n=10, deterministic=deterministic)
