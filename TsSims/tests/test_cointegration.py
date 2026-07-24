"""Tests for Ts.TsSims._cointegration — cointegrated data simulation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def rng():
    """Seeded numpy random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_result():
    """Create a minimal SimCointegratedResult (k=2, r=1, n=5)."""
    from Ts.TsSims._cointegration import SimCointegratedResult

    return SimCointegratedResult(
        data=np.array([
            [0.5, -0.3],
            [1.2, 0.8],
            [2.1, 1.5],
            [3.0, 2.2],
            [4.2, 3.1],
        ]),
        residuals=np.array([
            [0.5, -0.3],
            [0.7, 0.1],
            [0.2, 0.4],
            [0.1, 0.5],
            [0.5, 0.3],
        ]),
        params={
            "k": 2,
            "coint_rank": 1,
            "alpha": np.array([[-0.5], [0.0]]),
            "beta": np.array([[1.0], [0.0]]),
            "sigma": 1.0,
            "seed": 42,
            "n": 5,
            "burn": 100,
        },
    )


@pytest.fixture
def result_k3_r2():
    """Create a SimCointegratedResult (k=3, r=2, n=5)."""
    from Ts.TsSims._cointegration import SimCointegratedResult

    return SimCointegratedResult(
        data=np.array([
            [0.5, -0.3, 0.1],
            [1.2, 0.8, 0.9],
            [2.1, 1.5, 1.8],
            [3.0, 2.2, 2.5],
            [4.2, 3.1, 3.4],
        ]),
        residuals=np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5],
        ]),
        params={
            "k": 3,
            "coint_rank": 2,
            "alpha": np.array([[-0.5, 0.0], [0.0, -0.5], [0.0, 0.0]]),
            "beta": np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
            "sigma": 1.0,
            "seed": 42,
            "n": 5,
            "burn": 100,
        },
    )


# ===========================================================================
# TestSimCointegratedResult — container object
# ===========================================================================

class TestSimCointegratedResult:

    def test_get_data_returns_dataframe(self, simple_result):
        """covers: code/python/Ts/TsSims/__init__.py [module]
        covers: code/python/Ts/TsSims/_base.py [module]
        covers: code/python/Ts/TsSims/_base.py::BaseSimResult [class]
        covers: code/python/Ts/TsSims/_base.py::BaseSimResult.get_data [function]
        covers: code/python/Ts/TsSims/_cointegration.py [module]
        covers: code/python/Ts/TsSims/_cointegration.py::SimCointegratedResult [class]
        covers: code/python/Ts/TsSims/_garch.py [module]
        covers: code/python/Ts/TsSims/_garch.py::simulate_garch [function]
        covers: code/python/Ts/TsSims/_garch.py::simulate_igarch [function]
        covers: code/python/Ts/TsSims/_garch_core.py [module]
        covers: code/python/Ts/TsSims/_garch_core.py::_to_list [function]
        covers: code/python/Ts/TsSims/_garch_core.py::_normalize_coef [function]
        covers: code/python/Ts/TsSims/_garch_core.py::_make_standard_variance_fn [function]
        covers: code/python/Ts/TsSims/_garch_core.py::_make_standard_variance_fn._variance_fn [function]
        covers: code/python/Ts/TsSims/_garch_core.py::_generate_innovations [function]
        covers: code/python/Ts/TsSims/_garch_core.py::_compute_mean [function]
        covers: code/python/Ts/TsSims/_garch_core.py::_run_garch_simulation [function]
        covers: code/python/Ts/TsSims/_garch_ext.py [module]
        covers: code/python/Ts/TsSims/_garch_ext.py::_simulate_gjr_garch [function]
        covers: code/python/Ts/TsSims/_garch_ext.py::_simulate_gjr_garch._init_sigma2_fn [function]
        covers: code/python/Ts/TsSims/_garch_ext.py::_simulate_gjr_garch._variance_fn [function]
        covers: code/python/Ts/TsSims/_garch_ext.py::_simulate_egarch [function]
        covers: code/python/Ts/TsSims/_garch_ext.py::_simulate_egarch._variance_fn [function]
        covers: code/python/Ts/TsSims/_garch_ext.py::simulate_gjr_garch [function]
        covers: code/python/Ts/TsSims/_garch_ext.py::simulate_egarch [function]
        covers: code/python/Ts/TsSims/_garch_ext.py::simulate_garch_m [function]
        covers: code/python/Ts/TsSims/_garch_result.py [module]
        covers: code/python/Ts/TsSims/_garch_result.py::SimGARCHResult [class]
        covers: code/python/Ts/TsSims/_garch_result.py::SimGARCHResult.conditional_variance [function]
        covers: code/python/Ts/TsSims/_garch_result.py::SimGARCHResult._detect_model_type [function]
        covers: code/python/Ts/TsSims/_garch_result.py::SimGARCHResult.summary [function]
        covers: code/python/Ts/TsSims/_garch_result.py::SimGARCHResult.plot [function]
        covers: code/python/Ts/TsSims/_garch_result.py::SimGARCHResult.to_dataframe [function]
        covers: code/python/Ts/TsSims/_sarima.py [module]
        covers: code/python/Ts/TsSims/_sarima.py::SimSARIMAResult [class]
        covers: code/python/Ts/TsSims/_sarima.py::SimSARIMAResult.summary [function]
        covers: code/python/Ts/TsSims/_sarima.py::SimSARIMAResult.plot [function]
        covers: code/python/Ts/TsSims/_sarima.py::_expand_seasonal_poly [function]
        covers: code/python/Ts/TsSims/_sarima.py::_build_ar_ma_polynomials [function]
        covers: code/python/Ts/TsSims/_sarima.py::_apply_inverse_differencing [function]
        covers: code/python/Ts/TsSims/_sarima.py::simulate_sarima [function]
        covers: code/python/Ts/TsSims/_ts_ds.py [module]
        covers: code/python/Ts/TsSims/_ts_ds.py::SimTSDSResult [class]
        covers: code/python/Ts/TsSims/_ts_ds.py::SimTSDSResult.summary [function]
        covers: code/python/Ts/TsSims/_ts_ds.py::SimTSDSResult.plot [function]
        covers: code/python/Ts/TsSims/_ts_ds.py::simulate_trend_stationary [function]
        covers: code/python/Ts/TsSims/_ts_ds.py::simulate_difference_stationary [function]
        get_data() returns a pd.DataFrame with k columns."""
        result = simple_result.get_data()
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 2)

    def test_get_data_column_names(self, simple_result):
        """get_data() columns are y0, y1, ..."""
        result = simple_result.get_data()
        assert list(result.columns) == ["y0", "y1"]

    def test_get_params_is_deep_copy(self, simple_result):
        """covers: code/python/Ts/TsSims/_base.py::BaseSimResult.get_params [function]
        get_params() returns a copy, not a reference."""
        params1 = simple_result.get_params()
        params1["k"] = 99
        params2 = simple_result.get_params()
        assert params2["k"] == 2

    def test_get_params_contains_keys(self, simple_result):
        """get_params() contains all simulation keys."""
        params = simple_result.get_params()
        assert params["k"] == 2
        assert params["coint_rank"] == 1
        assert params["sigma"] == 1.0
        assert params["seed"] == 42

    def test_summary_returns_string(self, simple_result):
        """covers: code/python/Ts/TsSims/_base.py::BaseSimResult.summary [function]
        covers: code/python/Ts/TsSims/_cointegration.py::SimCointegratedResult.summary [function]
        summary() returns a non-empty string with key info."""
        text = simple_result.summary()
        assert isinstance(text, str)
        assert len(text) > 0
        assert "Cointegrated" in text

    def test_summary_shows_dimensions(self, simple_result):
        """summary() reports k, coint_rank, n."""
        text = simple_result.summary()
        assert "2" in text
        assert "1" in text
        assert "5" in text

    def test_summary_k3_r2(self, result_k3_r2):
        """summary() works for k=3, r=2 case."""
        text = result_k3_r2.summary()
        assert "3" in text
        assert "2" in text

    def test_plot_returns_fig_axes(self, simple_result):
        """covers: code/python/Ts/TsSims/_base.py::BaseSimResult.plot [function]
        covers: code/python/Ts/TsSims/_cointegration.py::SimCointegratedResult.plot [function]
        plot() returns (fig, axes) with k subplots."""
        fig, axes = simple_result.plot()
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 2
        plt.close(fig)

    def test_plot_k3_returns_3_axes(self, result_k3_r2):
        """plot() returns k axes for k=3."""
        fig, axes = result_k3_r2.plot()
        assert len(axes) == 3
        plt.close(fig)


# ===========================================================================
# TestSimulateCointegratedBasic — function basic behavior
# ===========================================================================

class TestSimulateCointegratedBasic:

    def test_returns_result_with_correct_shape(self, rng):
        """covers: code/python/Ts/TsSims/_cointegration.py::simulate_cointegrated [function]
        Returns SimCointegratedResult with data (n, k)."""
        from Ts.TsSims._cointegration import simulate_cointegrated, SimCointegratedResult

        r = simulate_cointegrated(n=100, k=2, coint_rank=1, seed=42)

        assert isinstance(r, SimCointegratedResult)
        assert r.data.shape == (100, 2)
        assert r.residuals.shape == (100, 2)

    def test_seed_reproducibility(self):
        """Same seed produces identical data."""
        from Ts.TsSims._cointegration import simulate_cointegrated

        r1 = simulate_cointegrated(n=100, k=2, coint_rank=1, seed=42)
        r2 = simulate_cointegrated(n=100, k=2, coint_rank=1, seed=42)

        np.testing.assert_array_equal(r1.data, r2.data)

    def test_different_seeds_different_data(self):
        """Different seeds produce different data."""
        from Ts.TsSims._cointegration import simulate_cointegrated

        r1 = simulate_cointegrated(n=100, k=2, coint_rank=1, seed=1)
        r2 = simulate_cointegrated(n=100, k=2, coint_rank=1, seed=2)

        assert not np.allclose(r1.data, r2.data)

    def test_default_parameters(self, rng):
        """covers: code/python/Ts/TsSims/_cointegration.py::_make_default_beta [function]
        covers: code/python/Ts/TsSims/_cointegration.py::_make_default_alpha [function]
        Works with only n and k specified."""
        from Ts.TsSims._cointegration import simulate_cointegrated

        r = simulate_cointegrated(n=50, k=3, coint_rank=2, seed=42)

        assert r.data.shape == (50, 3)
        assert r.params["k"] == 3
        assert r.params["coint_rank"] == 2

    def test_burn_increases_total_length(self):
        """burn parameter extends internal generation beyond n."""
        from Ts.TsSims._cointegration import simulate_cointegrated

        r1 = simulate_cointegrated(n=50, k=2, coint_rank=1, burn=0, seed=42)
        r2 = simulate_cointegrated(n=50, k=2, coint_rank=1, burn=500, seed=42)

        assert r1.data.shape[0] == 50
        assert r2.data.shape[0] == 50
        assert not np.allclose(r1.data, r2.data)


# ===========================================================================
# TestSimulateCointegratedValidation — parameter validation
# ===========================================================================

class TestSimulateCointegratedValidation:

    def test_k_less_than_2_raises(self, rng):
        """covers: code/python/Ts/TsSims/_cointegration.py::_validate_params [function]
        k < 2 raises ValueError."""
        from Ts.TsSims._cointegration import simulate_cointegrated

        with pytest.raises(ValueError, match="k"):
            simulate_cointegrated(n=100, k=1, coint_rank=1, seed=42)

    def test_coint_rank_ge_k_raises(self):
        """coint_rank >= k raises ValueError."""
        from Ts.TsSims._cointegration import simulate_cointegrated

        with pytest.raises(ValueError, match="coint_rank"):
            simulate_cointegrated(n=100, k=2, coint_rank=2, seed=42)

    def test_coint_rank_zero_raises(self):
        """coint_rank < 1 raises ValueError."""
        from Ts.TsSims._cointegration import simulate_cointegrated

        with pytest.raises(ValueError, match="coint_rank"):
            simulate_cointegrated(n=100, k=2, coint_rank=0, seed=42)

    def test_alpha_wrong_shape_raises(self):
        """alpha with wrong dimensions raises ValueError."""
        from Ts.TsSims._cointegration import simulate_cointegrated

        bad_alpha = np.array([[-0.5], [-0.3], [-0.2]])

        with pytest.raises(ValueError, match="alpha"):
            simulate_cointegrated(n=100, k=2, coint_rank=1, alpha=bad_alpha, seed=42)

    def test_beta_wrong_shape_raises(self):
        """beta with wrong dimensions raises ValueError."""
        from Ts.TsSims._cointegration import simulate_cointegrated

        bad_beta = np.array([[1.0], [0.0], [0.0]])

        with pytest.raises(ValueError, match="beta"):
            simulate_cointegrated(n=100, k=2, coint_rank=1, beta=bad_beta, seed=42)

    def test_unstable_eigenvalues_raises(self, rng):
        """covers: code/python/Ts/TsSims/_cointegration.py::_check_stability [function]
        alpha/beta producing unstable eigenvalues raises ValueError."""
        from Ts.TsSims._cointegration import simulate_cointegrated

        unstable_alpha = np.array([[2.0], [0.0]])
        beta = np.array([[1.0], [0.0]])

        with pytest.raises(ValueError, match="stable"):
            simulate_cointegrated(n=100, k=2, coint_rank=1,
                                  alpha=unstable_alpha, beta=beta, seed=42)


# ===========================================================================
# TestSimulateCointegratedCustom — custom parameters
# ===========================================================================

class TestSimulateCointegratedCustom:

    def test_custom_alpha_beta_passed_to_params(self):
        """Custom alpha/beta are stored in params."""
        from Ts.TsSims._cointegration import simulate_cointegrated

        alpha = np.array([[-0.3], [0.0]])
        beta = np.array([[1.0], [-0.5]])

        r = simulate_cointegrated(n=50, k=2, coint_rank=1,
                                  alpha=alpha, beta=beta, seed=42)

        params = r.get_params()
        np.testing.assert_array_equal(params["alpha"], alpha)
        np.testing.assert_array_equal(params["beta"], beta)

    def test_default_alpha_is_diagonal(self):
        """Default alpha has -0.5 * I_r structure."""
        from Ts.TsSims._cointegration import simulate_cointegrated

        r = simulate_cointegrated(n=50, k=4, coint_rank=3, seed=42)

        alpha = r.params["alpha"]
        assert alpha.shape == (4, 3)
        np.testing.assert_array_equal(alpha[:3, :3], -0.5 * np.eye(3))
        np.testing.assert_array_equal(alpha[3:, :], np.zeros((1, 3)))

    def test_default_beta_is_identity_stack(self):
        """Default beta is [I_r; 0] stack."""
        from Ts.TsSims._cointegration import simulate_cointegrated

        r = simulate_cointegrated(n=50, k=4, coint_rank=2, seed=42)

        beta = r.params["beta"]
        assert beta.shape == (4, 2)
        np.testing.assert_array_equal(beta[:2, :2], np.eye(2))
        np.testing.assert_array_equal(beta[2:, :], np.zeros((2, 2)))


# ===========================================================================
# TestCointegrationRoundTrip — closed-loop verification
# ===========================================================================

class TestCointegrationRoundTrip:

    def test_johansen_recovers_coint_rank_k2_r1(self):
        """Johansen trace test recovers true coint_rank for k=2, r=1."""
        from Ts.TsSims._cointegration import simulate_cointegrated
        from Ts.TsTests._johansen import JohansenTest

        data = simulate_cointegrated(
            n=500, k=2, coint_rank=1,
            alpha=np.array([[-0.3], [0.0]]),
            beta=np.array([[1.0], [-1.0]]),
            sigma=0.5, seed=42, burn=200,
        )

        jt = JohansenTest(data.data, lags=1, trend="constant")
        jt.fit()

        trace_stats = jt.result_.trace_statistics
        trace_crit = jt.result_.trace_critical_values[:, 1]

        assert trace_stats[0] > trace_crit[0], (
            f"Expected H0: r<=0 rejected, got trace={trace_stats[0]:.2f} <= crit={trace_crit[0]:.2f}"
        )
        assert trace_stats[1] <= trace_crit[1], (
            f"Expected H0: r<=1 NOT rejected, got trace={trace_stats[1]:.2f} > crit={trace_crit[1]:.2f}"
        )

    def test_vecm_recovers_pi_matrix(self):
        """VECM estimate of Pi = alpha @ beta.T is close to true Pi."""
        from Ts.TsSims._cointegration import simulate_cointegrated
        from Ts.TsModels._vecm import VECM

        alpha_true = np.array([[-0.3], [0.0]])
        beta_true = np.array([[1.0], [-1.0]])
        pi_true = alpha_true @ beta_true.T

        data = simulate_cointegrated(
            n=500, k=2, coint_rank=1,
            alpha=alpha_true, beta=beta_true,
            sigma=0.3, seed=42, burn=200,
        )

        vecm = VECM(data.data, lags=2, coint_rank=1, trend="n")
        result = vecm.fit()

        pi_est = result.alpha @ result.beta.T

        rel_error = np.linalg.norm(pi_est - pi_true) / np.linalg.norm(pi_true)
        assert rel_error < 0.5, (
            f"Pi recovery error too large: {rel_error:.3f}"
        )

    def test_k3_r2_johansen_rank_recovery(self):
        """Johansen sequential trace test recovers r=2 in k=3 system."""
        from Ts.TsSims._cointegration import simulate_cointegrated
        from Ts.TsTests._johansen import JohansenTest

        alpha = np.array([[-0.3, 0.0], [0.0, -0.3], [0.0, 0.0]])
        beta = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])

        data = simulate_cointegrated(
            n=500, k=3, coint_rank=2, alpha=alpha, beta=beta,
            sigma=0.3, seed=42, burn=200,
        )

        jt = JohansenTest(data.data, lags=1, trend="constant")
        jt.fit()

        selected_rank = jt.result_.rank
        assert selected_rank >= 2, (
            f"Expected rank >= 2, got {selected_rank}"
        )
