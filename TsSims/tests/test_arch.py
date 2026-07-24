"""Tests for Ts.TsSims._garch — GARCH simulation and SimGARCHResult."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


class TestSimGARCHResult:
    """Test the SimGARCHResult container object and its core methods."""

    @pytest.fixture
    def simple_result(self):
        """Create a minimal SimGARCHResult for testing."""
        from Ts.TsSims._garch_result import SimGARCHResult

        return SimGARCHResult(
            data=np.array([0.5, -0.3, 0.8, -0.2, 0.1]),
            residuals=np.array([0.5, -0.3, 0.8, -0.2, 0.1]),
            conditional_volatility=np.array([0.63, 0.71, 0.83, 0.68, 0.65]),
            params={
                "p": 1,
                "q": 0,
                "omega": 0.4,
                "alpha": [0.5],
                "beta": [],
                "mean_model": "constant",
                "mean_const": 0.0,
                "mean_ar": [],
                "dist": "normal",
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

    def test_get_params_returns_dict(self, simple_result):
        """get_params() returns a dict with correct values."""
        params = simple_result.get_params()
        assert isinstance(params, dict)
        assert params["p"] == 1
        assert params["omega"] == 0.4
        assert params["alpha"] == [0.5]

    def test_get_params_is_deep_copy(self, simple_result):
        """get_params() returns a copy, not a reference."""
        params1 = simple_result.get_params()
        params1["omega"] = 0.9
        params2 = simple_result.get_params()
        assert params2["omega"] == 0.4

    def test_summary_returns_string(self, simple_result):
        """summary() returns a non-empty string with key info."""
        text = simple_result.summary()
        assert isinstance(text, str)
        assert "ARCH" in text
        assert "0.5" in text

    def test_plot_returns_fig_ax(self, simple_result):
        """plot() returns (fig, ax) with two-panel axes."""
        fig, ax = simple_result.plot()
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        assert len(ax) == 2
        assert isinstance(ax[0], plt.Axes)
        assert isinstance(ax[1], plt.Axes)

    def test_to_dataframe_has_columns(self, simple_result):
        """to_dataframe() returns DataFrame with data, errors, vol columns."""
        df = simple_result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "data" in df.columns
        assert "residuals" in df.columns
        assert "volatility" in df.columns


class TestSimulateGARCH:
    """Test simulate_garch()."""

    def test_arch_q0_basic(self):
        """simulate_garch with q=0 (pure ARCH) returns correct-shaped result."""
        from Ts.TsSims._garch import simulate_garch

        result = simulate_garch(n=100, p=1, q=0, omega=0.4, alpha=[0.5], seed=42, burn=50)

        assert result.get_data().shape[0] == 100
        assert result.conditional_volatility.shape[0] == 100

    def test_seed_reproducibility(self):
        """Same seed produces identical data."""
        from Ts.TsSims._garch import simulate_garch

        r1 = simulate_garch(n=100, p=1, q=0, omega=0.4, alpha=[0.5], seed=42)
        r2 = simulate_garch(n=100, p=1, q=0, omega=0.4, alpha=[0.5], seed=42)

        np.testing.assert_array_equal(r1.data, r2.data)

    def test_arch_kurtosis_gt_normal(self):
        """ARCH(1) data should have excess kurtosis > 0 (fatter tails than normal)."""
        from Ts.TsSims._garch import simulate_garch

        result = simulate_garch(
            n=2000, p=1, q=0, omega=0.4, alpha=[0.5], seed=42, burn=500,
        )

        kurt = pd.Series(result.data).kurtosis()
        assert kurt > 0.0

    def test_garch_basic(self):
        """simulate_garch with p=1, q=1 generates data."""
        from Ts.TsSims._garch import simulate_garch

        result = simulate_garch(
            n=100, p=1, q=1,
            omega=0.2, alpha=[0.3], beta=[0.5],
            seed=42,
        )

        assert result.get_data().shape[0] == 100

    def test_positive_volatility(self):
        """Conditional volatility should be strictly positive."""
        from Ts.TsSims._garch import simulate_garch

        result = simulate_garch(n=100, p=1, q=0, omega=0.4, alpha=[0.3], seed=42)

        assert np.all(result.conditional_volatility > 0)

    def test_default_parameters(self):
        """simulate_garch with minimal args uses sensible defaults."""
        from Ts.TsSims._garch import simulate_garch

        result = simulate_garch(n=50, seed=42)

        assert result.get_data().shape[0] == 50

    def test_nonstationary_garch_warns(self):
        """GARCH with alpha+beta >= 1 warns but still simulates."""
        import warnings
        from Ts.TsSims._garch import simulate_garch

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = simulate_garch(
                n=100, p=1, q=1,
                omega=0.2, alpha=[0.6], beta=[0.5],
                seed=42,
            )
        assert len(w) == 1
        assert "non-stationary" in str(w[0].message)
        assert result.get_data().shape[0] == 100

    def test_student_t_df_too_low_raises(self):
        """Student's t with df <= 2 raises ValueError."""
        from Ts.TsSims._garch import simulate_garch

        with pytest.raises(ValueError, match="df > 2"):
            simulate_garch(
                n=100, p=1, q=0,
                dist="t", dist_params={"df": 1.5},
                seed=42,
            )


class TestSimulateGJRGARCH:
    """Test simulate_gjr_garch() — asymmetric GJR-GARCH simulation."""

    def test_basic_generation(self):
        """simulate_gjr_garch generates data with correct shape."""
        from Ts.TsSims._garch_ext import simulate_gjr_garch

        result = simulate_gjr_garch(
            n=200, p=1, q=1, o=1,
            omega=0.05, alpha=[0.10], gamma=[0.15], beta=[0.70],
            seed=42, burn=100,
        )

        assert result.get_data().shape[0] == 200
        assert result.conditional_volatility.shape[0] == 200
        assert result.residuals.shape[0] == 200

    def test_seed_reproducibility(self):
        """Same seed produces identical GJR-GARCH data."""
        from Ts.TsSims._garch_ext import simulate_gjr_garch

        r1 = simulate_gjr_garch(
            n=100, p=1, q=1, o=1,
            omega=0.05, alpha=[0.10], gamma=[0.15], beta=[0.70],
            seed=42,
        )
        r2 = simulate_gjr_garch(
            n=100, p=1, q=1, o=1,
            omega=0.05, alpha=[0.10], gamma=[0.15], beta=[0.70],
            seed=42,
        )

        np.testing.assert_array_equal(r1.data, r2.data)

    def test_positive_volatility(self):
        """GJR-GARCH conditional volatility is strictly positive."""
        from Ts.TsSims._garch_ext import simulate_gjr_garch

        result = simulate_gjr_garch(
            n=200, p=1, q=1, o=1,
            omega=0.05, alpha=[0.10], gamma=[0.15], beta=[0.70],
            seed=42, burn=100,
        )

        assert np.all(result.conditional_volatility > 0)

    def test_nonstationary_warns(self):
        """GJR-GARCH with alpha+0.5*gamma+beta >= 1 warns."""
        import warnings
        from Ts.TsSims._garch_ext import simulate_gjr_garch

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = simulate_gjr_garch(
                n=100, p=1, q=1, o=1,
                omega=0.05, alpha=[0.40], gamma=[0.30], beta=[0.60],
                seed=42,
            )
        assert len(w) == 1
        assert "non-stationary" in str(w[0].message)
        assert result.get_data().shape[0] == 100

    def test_default_parameters(self):
        """simulate_gjr_garch with minimal args uses sensible defaults."""
        from Ts.TsSims._garch_ext import simulate_gjr_garch

        result = simulate_gjr_garch(n=50, seed=42)

        assert result.get_data().shape[0] == 50
        params = result.get_params()
        assert params["o"] == 1

    def test_o_zero_is_standard_garch(self):
        """GJR-GARCH with o=0 reduces to standard GARCH (no gamma terms)."""
        from Ts.TsSims._garch_ext import simulate_gjr_garch

        result = simulate_gjr_garch(
            n=100, p=1, q=1, o=0,
            omega=0.1, alpha=[0.3], beta=[0.5],
            seed=42,
        )

        params = result.get_params()
        assert params["gamma"] == []

    def test_summary_shows_gjr(self):
        """GJR-GARCH summary displays model type."""
        from Ts.TsSims._garch_ext import simulate_gjr_garch

        result = simulate_gjr_garch(n=100, p=1, q=1, o=1, seed=42)
        text = result.summary()
        assert "GJR-GARCH" in text


class TestSimulateEGARCH:
    """Test simulate_egarch() — Exponential GARCH simulation."""

    def test_basic_generation(self):
        """simulate_egarch generates data with correct shape."""
        from Ts.TsSims._garch_ext import simulate_egarch

        result = simulate_egarch(
            n=200, p=1, q=1, o=1,
            omega=0.0, alpha=[0.20], gamma=[0.10], beta=[0.30],
            seed=42, burn=100,
        )

        assert result.get_data().shape[0] == 200
        assert result.conditional_volatility.shape[0] == 200

    def test_seed_reproducibility(self):
        """Same seed produces identical EGARCH data."""
        from Ts.TsSims._garch_ext import simulate_egarch

        r1 = simulate_egarch(
            n=100, p=1, q=1, o=1,
            omega=0.0, alpha=[0.20], gamma=[0.10], beta=[0.30],
            seed=42,
        )
        r2 = simulate_egarch(
            n=100, p=1, q=1, o=1,
            omega=0.0, alpha=[0.20], gamma=[0.10], beta=[0.30],
            seed=42,
        )

        np.testing.assert_array_equal(r1.data, r2.data)

    def test_positive_volatility(self):
        """EGARCH conditional volatility is strictly positive (by construction)."""
        from Ts.TsSims._garch_ext import simulate_egarch

        result = simulate_egarch(
            n=200, p=1, q=1, o=1,
            omega=0.0, alpha=[0.20], gamma=[0.10], beta=[0.30],
            seed=42, burn=100,
        )

        assert np.all(result.conditional_volatility > 0)

    def test_default_parameters(self):
        """simulate_egarch with minimal args uses sensible defaults."""
        from Ts.TsSims._garch_ext import simulate_egarch

        result = simulate_egarch(n=50, seed=42)

        assert result.get_data().shape[0] == 50

    def test_symmetric_egarch(self):
        """EGARCH with o=0 has no gamma (symmetric EGARCH)."""
        from Ts.TsSims._garch_ext import simulate_egarch

        result = simulate_egarch(
            n=100, p=1, q=1, o=0,
            omega=0.0, alpha=[0.20], beta=[0.30],
            seed=42,
        )

        params = result.get_params()
        assert params["gamma"] == []

    def test_summary_shows_egarch(self):
        """EGARCH summary displays model type."""
        from Ts.TsSims._garch_ext import simulate_egarch

        result = simulate_egarch(n=100, p=1, q=1, o=1, seed=42)
        text = result.summary()
        assert "EGARCH" in text


class TestSimulateGARCHM:
    """Test simulate_garch_m() — GARCH-in-Mean simulation."""

    def test_basic_generation(self):
        """simulate_garch_m generates data with correct shape."""
        from Ts.TsSims._garch_ext import simulate_garch_m

        result = simulate_garch_m(
            n=200, p=1, q=1,
            omega=0.1, alpha=[0.2], beta=[0.6],
            garch_m_kappa=0.2, garch_m_form="vol",
            seed=42, burn=100,
        )

        assert result.get_data().shape[0] == 200
        assert result.conditional_volatility.shape[0] == 200

    def test_seed_reproducibility(self):
        """Same seed produces identical GARCH-M data."""
        from Ts.TsSims._garch_ext import simulate_garch_m

        r1 = simulate_garch_m(
            n=100, p=1, q=1,
            omega=0.1, alpha=[0.2], beta=[0.6],
            garch_m_kappa=0.2, garch_m_form="vol",
            seed=42,
        )
        r2 = simulate_garch_m(
            n=100, p=1, q=1,
            omega=0.1, alpha=[0.2], beta=[0.6],
            garch_m_kappa=0.2, garch_m_form="vol",
            seed=42,
        )

        np.testing.assert_array_equal(r1.data, r2.data)

    def test_positive_volatility(self):
        """GARCH-M conditional volatility is strictly positive."""
        from Ts.TsSims._garch_ext import simulate_garch_m

        result = simulate_garch_m(
            n=200, p=1, q=1,
            omega=0.1, alpha=[0.2], beta=[0.6],
            garch_m_kappa=0.2, seed=42, burn=100,
        )

        assert np.all(result.conditional_volatility > 0)

    def test_form_var(self):
        """GARCH-M with form='var' uses variance in mean."""
        from Ts.TsSims._garch_ext import simulate_garch_m

        result = simulate_garch_m(
            n=100, p=1, q=1,
            omega=0.1, alpha=[0.2], beta=[0.6],
            garch_m_kappa=0.2, garch_m_form="var",
            seed=42,
        )

        assert result.get_data().shape[0] == 100

    def test_form_log(self):
        """GARCH-M with form='log' uses log-variance in mean."""
        from Ts.TsSims._garch_ext import simulate_garch_m

        result = simulate_garch_m(
            n=100, p=1, q=1,
            omega=0.1, alpha=[0.2], beta=[0.6],
            garch_m_kappa=0.2, garch_m_form="log",
            seed=42,
        )

        assert result.get_data().shape[0] == 100

    def test_default_parameters(self):
        """simulate_garch_m with minimal args uses sensible defaults."""
        from Ts.TsSims._garch_ext import simulate_garch_m

        result = simulate_garch_m(n=50, seed=42)

        assert result.get_data().shape[0] == 50

    def test_kappa_zero_equals_standard_garch(self):
        """GARCH-M with kappa=0 produces same data as standard GARCH."""
        from Ts.TsSims._garch_ext import simulate_garch_m
        from Ts.TsSims._garch import simulate_garch

        r_m = simulate_garch_m(
            n=150, p=1, q=1,
            omega=0.1, alpha=[0.2], beta=[0.6],
            garch_m_kappa=0.0, seed=42, burn=100,
        )
        r_g = simulate_garch(
            n=150, p=1, q=1,
            omega=0.1, alpha=[0.2], beta=[0.6],
            seed=42, burn=100,
        )

        np.testing.assert_array_equal(r_m.data, r_g.data)

    def test_summary_shows_garch_m(self):
        """GARCH-M summary displays model type."""
        from Ts.TsSims._garch_ext import simulate_garch_m

        result = simulate_garch_m(n=100, p=1, q=1, garch_m_kappa=0.2, seed=42)
        text = result.summary()
        assert "GARCH-M" in text


class TestIGARCH:
    """Test simulate_igarch() — IGARCH simulation."""

    def test_igarch_does_not_mutate_caller_beta(self):
        """simulate_igarch must not modify the caller's beta list."""
        from Ts.TsSims import simulate_igarch
        beta_original = [0.5]
        beta_input = [0.5]
        simulate_igarch(n=100, p=1, q=1, alpha=[0.2], beta=beta_input, seed=42)
        assert beta_input == beta_original, (
            f"Caller's beta was mutated: {beta_input} != {beta_original}"
        )

    def test_igarch_satisfies_unit_sum(self):
        """IGARCH must enforce sum(alpha) + sum(beta) == 1."""
        from Ts.TsSims import simulate_igarch
        r = simulate_igarch(n=100, p=1, q=1, alpha=[0.3], beta=[0.5], seed=42)
        params = r.get_params()
        total = sum(params['alpha']) + sum(params['beta'])
        assert abs(total - 1.0) < 1e-10, f"Sum should be 1.0, got {total}"

    def test_igarch_basic_simulation(self):
        """IGARCH simulation should produce valid data."""
        from Ts.TsSims import simulate_igarch
        r = simulate_igarch(n=200, p=1, q=1, seed=42)
        data = r.get_data()
        assert len(data) == 200
        assert isinstance(r.summary(), str)
