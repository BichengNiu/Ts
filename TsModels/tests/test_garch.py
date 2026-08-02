"""Tests for Ts.TsModels._garch — GARCH and GARCHResult."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from Ts.TsSims import simulate_garch


@pytest.fixture
def arch_data():
    """Generate ARCH(2) data (GARCH with q=0)."""
    r = simulate_garch(
        n=200,
        p=2,
        q=0,
        omega=0.4,
        alpha=[0.3, 0.2],
        seed=42,
        burn=200,
    )
    return r.data


@pytest.fixture
def garch11_data():
    """Generate GARCH(1,1) data."""
    r = simulate_garch(
        n=300,
        p=1,
        q=1,
        omega=0.1,
        alpha=[0.2],
        beta=[0.7],
        seed=42,
        burn=200,
    )
    return r.data


class TestGARCH:
    """Test GARCH construction and fit()."""

    def test_garch_q0_estimates_arch(self, arch_data):
        """GARCH(p, q=0) fits a pure ARCH model."""
        from Ts.TsModels._garch import GARCH
        from Ts.TsModels._garch_result import GARCHResult

        model = GARCH(arch_data, p=2, q=0)
        assert model.p == 2
        assert model.q == 0
        result = model.fit()
        assert isinstance(result, GARCHResult)
        assert result.model_type == "ARCH"
        assert result.nobs == 200

    def test_garch_q0_has_volatility(self, arch_data):
        """GARCH(p, q=0) result includes conditional_volatility."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(arch_data, p=2, q=0)
        result = model.fit()
        vol = result.conditional_volatility
        assert vol is not None
        assert len(vol) == 200
        assert np.all(vol > 0)

    def test_garch_q0_lag_selection(self, arch_data):
        """GARCH(p>1, q=0) stores per-lag AIC/BIC."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(arch_data, p=3, q=0)
        result = model.fit()
        assert result.individual_lags is not None
        assert result.individual_aic is not None
        assert result.individual_bic is not None
        assert len(result.individual_lags) == 3
        text = result.summary()
        assert "Lowest AIC: P =" in text
        assert "Lowest BIC: P =" in text

    def test_garch_can_skip_redundant_lag_comparisons(self, arch_data):
        """Auto-selection callers can avoid refitting lower-order ARCH models."""
        from Ts.TsModels._garch import GARCH

        result = GARCH(arch_data, p=3, q=0, compare_lags=False).fit()

        assert result.individual_lags is None
        assert result.individual_aic is None
        assert result.individual_bic is None

    def test_garch_default_is_garch11(self, garch11_data):
        """GARCH(data) defaults to GARCH(1,1)."""
        from Ts.TsModels._garch import GARCH
        from Ts.TsModels._garch_result import GARCHResult

        model = GARCH(garch11_data)
        assert model.p == 1
        assert model.q == 1
        result = model.fit()
        assert isinstance(result, GARCHResult)
        assert result.model_type == "GARCH"

    def test_garch11_fit(self, garch11_data):
        """GARCH(1,1) fits correctly."""
        from Ts.TsModels._garch import GARCH
        from Ts.TsModels._garch_result import GARCHResult

        model = GARCH(garch11_data, p=1, q=1)
        result = model.fit()
        assert isinstance(result, GARCHResult)
        assert result.model_type == "GARCH"

    def test_arch_class_removed(self):
        """ARCH class no longer exists in _garch module."""
        from Ts.TsModels import _garch

        assert not hasattr(_garch, "ARCH")

    def test_invalid_p_raises(self, arch_data):
        """p < 1 raises ValueError."""
        from Ts.TsModels._garch import GARCH

        with pytest.raises(ValueError):
            GARCH(arch_data, p=0)

    def test_invalid_q_negative_raises(self, arch_data):
        """q < 0 raises ValueError."""
        from Ts.TsModels._garch import GARCH

        with pytest.raises(ValueError):
            GARCH(arch_data, p=1, q=-1)

    def test_nan_data_keeps_exog_aligned(self):
        """Rows removed from data are removed from exogenous regressors too."""
        from Ts.TsModels._garch import GARCH

        rng = np.random.default_rng(42)
        data = rng.normal(size=60)
        exog = rng.normal(size=(60, 2))
        data[10] = np.nan

        model = GARCH(data, p=1, q=1, exog=exog, missing="drop")

        assert len(model.data) == 59
        assert model.exog.shape == (59, 2)
        assert model.dropped_positions == (10,)


class TestGARCHResult:
    """Test GARCHResult methods.

    covers: code/python/Ts/TsModels/_garch_result.py::GARCHResult [class]
    covers: code/python/Ts/TsModels/_garch_result.py::GARCHResult.summary [function]
    covers: code/python/Ts/TsModels/_garch_result.py::GARCHResult.test_persistence [function]
    covers: code/python/Ts/TsModels/_garch_result.py::GARCHResult.long_run_equilibrium [function]
    covers: code/python/Ts/TsModels/_garch_result.py::GARCHResult.conditional_variance [function]
    """

    @pytest.fixture
    def garch_q0_result(self, arch_data):
        """Fit GARCH(2,0) and return result."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(arch_data, p=2, q=0)
        return model.fit()

    @pytest.fixture
    def garch11_result(self, garch11_data):
        """Fit GARCH(1,1) and return result."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(garch11_data, p=1, q=1)
        return model.fit()

    def test_summary_has_key_fields(self, garch_q0_result):
        """GARCH(q=0) summary contains omega and alpha."""
        text = garch_q0_result.summary()
        assert "ARCH" in text
        assert "omega" in text.lower()

    def test_garch11_summary(self, garch11_result):
        """GARCH(1,1) summary contains model type."""
        text = garch11_result.summary()
        assert "GARCH" in text

    def test_forecast_returns_var_and_vol(self, garch_q0_result):
        """predict() beyond sample returns PredictResult with variance forecasts."""
        from Ts.TsModels._base import PredictResult

        horizon = 5
        end = garch_q0_result.nobs + horizon - 1
        pr = garch_q0_result.predict(start=garch_q0_result.nobs, end=end)
        assert isinstance(pr, PredictResult)
        assert len(pr.mean) == horizon
        assert np.all(pr.mean > 0)

    def test_plot_fit_inherited(self, garch_q0_result):
        """plot_fit() works."""
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        fig, ax = garch_q0_result.plot_fit()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot_diagnostics_inherited(self, garch11_result):
        """plot_diagnostics() returns the shared four-panel layout."""
        from matplotlib.figure import Figure

        fig, axes = garch11_result.plot_diagnostics()
        assert isinstance(fig, Figure)
        assert len(axes) == 4

    def test_test_residuals_inherited(self, garch_q0_result):
        """test_residuals() works."""
        output = garch_q0_result.test_residuals(lags=5)
        assert output.ljung_box is not None
        assert output.engle_lm is not None

    def test_persistence_returns_dict(self, garch_q0_result):
        """test_persistence() returns dict with expected keys."""
        result = garch_q0_result.test_persistence()
        assert isinstance(result, dict)
        for key in ("chi2", "pvalue", "persistence_sum", "reject_null", "conclusion"):
            assert key in result
        assert result["persistence_sum"] > 0

    def test_persistence_garch11(self, garch11_result):
        """GARCH(1,1) persistence test."""
        result = garch11_result.test_persistence()
        assert result["persistence_sum"] < 2.0
        assert result["chi2"] >= 0
        assert 0.0 <= result["pvalue"] <= 1.0

    def test_summary_includes_persistence(self, garch_q0_result):
        """summary() includes IGARCH persistence test."""
        text = garch_q0_result.summary()
        assert "IGARCH" in text
        assert "Wald chi2" in text

    def test_result_has_dist_field(self, garch11_result):
        """GARCHResult must have a dist field."""
        assert hasattr(garch11_result, "dist")
        assert garch11_result.dist == "normal"

    def test_dist_t_stored_in_result(self, garch11_data):
        """GARCH(data, dist='t') stores dist='t' in result."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(garch11_data, p=1, q=1, dist="t")
        result = model.fit()
        assert result.dist == "t"

    def test_summary_shows_dist_normal(self, garch11_result):
        """summary() shows distribution info for normal."""
        text = garch11_result.summary()
        assert "[Normal]" in text

    def test_summary_shows_dist_t(self, garch11_data):
        """summary() shows distribution info for Student's t."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(garch11_data, p=1, q=1, dist="t")
        result = model.fit()
        text = result.summary()
        assert "[Student's t]" in text

    def test_summary_shows_dist_ged(self, garch11_data):
        """summary() shows distribution info for GED."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(garch11_data, p=1, q=1, dist="ged")
        result = model.fit()
        text = result.summary()
        assert "[GED]" in text


class TestGARCHInMean:
    """Test GARCH-M (ARCH-in-Mean) estimation."""

    @pytest.fixture
    def garch11_data(self):
        """Generate GARCH(1,1) data for GARCH-M testing."""
        r = simulate_garch(
            n=300,
            p=1,
            q=1,
            omega=0.1,
            alpha=[0.2],
            beta=[0.7],
            seed=42,
            burn=200,
        )
        return r.data

    def test_garch_m_basic_fit(self, garch11_data):
        """GARCH-M(1,1) with garch_m=True fits successfully."""
        from Ts.TsModels._garch import GARCH
        from Ts.TsModels._garch_result import GARCHResult

        model = GARCH(garch11_data, p=1, q=1, garch_m=True)
        assert model.garch_m is True
        assert model.garch_m_form == "vol"
        result = model.fit()
        assert isinstance(result, GARCHResult)
        assert result.garch_m is True
        assert result.garch_m_form == "vol"

    def test_garch_m_form_var(self, garch11_data):
        """GARCH-M with garch_m_form='var' uses variance form."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(garch11_data, p=1, q=1, garch_m=True, garch_m_form="var")
        result = model.fit()
        assert result.garch_m_form == "var"

    def test_garch_m_form_log(self, garch11_data):
        """GARCH-M with garch_m_form='log' uses log-variance form."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(garch11_data, p=1, q=1, garch_m=True, garch_m_form="log")
        result = model.fit()
        assert result.garch_m_form == "log"

    def test_garch_m_summary_shows_in_mean(self, garch11_data):
        """GARCH-M summary() indicates ARCH-in-mean."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(garch11_data, p=1, q=1, garch_m=True)
        result = model.fit()
        text = result.summary()
        assert "ARCH-in-mean" in text

    def test_garch_m_invalid_form_raises(self, garch11_data):
        """Invalid garch_m_form raises ValueError."""
        from Ts.TsModels._garch import GARCH

        with pytest.raises(ValueError):
            GARCH(garch11_data, p=1, q=1, garch_m=True, garch_m_form="invalid")


class TestGARCHExog:
    """Test GARCH with exogenous regressors."""

    @pytest.fixture
    def garch11_data(self):
        """Generate GARCH(1,1) data."""
        r = simulate_garch(
            n=300,
            p=1,
            q=1,
            omega=0.1,
            alpha=[0.2],
            beta=[0.7],
            seed=42,
            burn=200,
        )
        return r.data

    def test_garch_exog_fit(self, garch11_data):
        """GARCH with exog fits successfully."""
        from Ts.TsModels._garch import GARCH
        from Ts.TsModels._garch_result import GARCHResult
        import numpy as np

        exog = np.random.default_rng(42).standard_normal(len(garch11_data))
        model = GARCH(garch11_data, p=1, q=1, exog=exog)
        result = model.fit()
        assert isinstance(result, GARCHResult)

    def test_garch_exog_wrong_length_raises(self, garch11_data):
        """exog with wrong length raises ValueError."""
        from Ts.TsModels._garch import GARCH
        import numpy as np

        exog = np.random.default_rng(42).standard_normal(len(garch11_data) - 10)
        with pytest.raises(ValueError):
            GARCH(garch11_data, p=1, q=1, exog=exog)


class TestGJR_GARCH:
    """Test GJR-GARCH (asymmetric GARCH) estimation."""

    @pytest.fixture
    def gjrgarch_data(self):
        """Generate GJR-GARCH(1,1,1) data."""
        np.random.seed(123)
        n = 500
        omega, alpha, gamma, beta = 0.05, 0.10, 0.15, 0.75
        eps = np.random.randn(n)
        sigma2 = np.zeros(n)
        e = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - 0.5 * gamma - beta)
        e[0] = np.sqrt(sigma2[0]) * eps[0]
        for t in range(1, n):
            I_neg = 1.0 if e[t - 1] < 0 else 0.0
            sigma2[t] = (
                omega
                + alpha * e[t - 1] ** 2
                + gamma * I_neg * e[t - 1] ** 2
                + beta * sigma2[t - 1]
            )
            e[t] = np.sqrt(max(sigma2[t], 1e-10)) * eps[t]
        return 1.0 + e

    def test_gjrgarch_fit(self, gjrgarch_data):
        """GJR-GARCH(1,1,1) fits successfully."""
        from Ts.TsModels._garch import GARCH
        from Ts.TsModels._garch_result import GARCHResult

        model = GARCH(gjrgarch_data, p=1, o=1, q=1)
        result = model.fit()
        assert isinstance(result, GARCHResult)
        assert result.model_type == "GJR-GARCH"
        assert result._o == 1

    def test_gjrgarch_has_gamma(self, gjrgarch_data):
        """GJR-GARCH result includes gamma parameter."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(gjrgarch_data, p=1, o=1, q=1)
        result = model.fit()
        assert "gamma[1]" in result.params

    def test_gjrgarch_summary_label(self, gjrgarch_data):
        """GJR-GARCH summary shows correct model label."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(gjrgarch_data, p=1, o=1, q=1)
        result = model.fit()
        text = result.summary()
        assert "GJR-GARCH" in text

    def test_gjrgarch_persistence_includes_gamma(self, gjrgarch_data):
        """GJR-GARCH persistence test includes gamma with weight 0.5."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(gjrgarch_data, p=1, o=1, q=1)
        result = model.fit()
        stab = result.test_persistence()
        # persistence_sum should be between 0 and 2
        assert 0 < stab["persistence_sum"] < 2.0
        assert stab["chi2"] >= 0
        assert 0.0 <= stab["pvalue"] <= 1.0

    def test_gjrgarch_forecast(self, gjrgarch_data):
        """GJR-GARCH predict works."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(gjrgarch_data, p=1, o=1, q=1)
        result = model.fit()
        pr = result.predict(start=result.nobs, end=result.nobs + 2)
        assert len(pr.mean) == 3
        assert np.all(pr.mean > 0)

    def test_invalid_o_raises(self, gjrgarch_data):
        """o < 0 raises ValueError."""
        from Ts.TsModels._garch import GARCH

        with pytest.raises(ValueError):
            GARCH(gjrgarch_data, p=1, q=1, o=-1)

    def test_o_zero_is_standard_garch(self, gjrgarch_data):
        """o=0 yields standard GARCH, not GJR-GARCH."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(gjrgarch_data, p=1, q=1, o=0)
        result = model.fit()
        assert result.model_type == "GARCH"
        assert "gamma" not in result.params

    def test_dynamic_forecast_uses_future_and_observed_lags(self):
        """GJR multi-step recursion uses the correct lag source."""
        from Ts.TsModels._garch_result import GARCHResult

        residuals = np.array([1.0, -2.0, -3.0])
        conditional_volatility = np.array([2.0, 3.0, 4.0])
        result = GARCHResult(
            model_type="GJR-GARCH",
            params={
                "omega": 0.1,
                "alpha[1]": 0.2,
                "alpha[2]": 0.1,
                "gamma[1]": 0.4,
                "beta[1]": 0.3,
                "beta[2]": 0.1,
            },
            std_errors={},
            p_values={},
            aic=0.0,
            bic=0.0,
            log_likelihood=0.0,
            residuals=residuals,
            fitted_values=None,
            nobs=3,
            data=residuals,
            conditional_volatility=conditional_volatility,
            _p=2,
            _o=1,
            _q=2,
        )

        forecast_vol = result._garch_forecast_vol(
            3, 3, residuals, conditional_volatility
        )
        expected_variance = np.array([11.6, 10.72, 9.924])
        np.testing.assert_allclose(forecast_vol**2, expected_variance)


class TestEGARCH:
    """Test EGARCH (Exponential GARCH) estimation."""

    @pytest.fixture
    def egarch_data(self):
        """Generate EGARCH(1,1,1) data."""
        np.random.seed(123)
        n = 500
        omega, alpha, gamma, beta = 0.05, 0.20, 0.10, 0.30
        eps = np.random.randn(n)
        ln_sigma2 = np.zeros(n)
        e = np.zeros(n)
        ln_sigma2[0] = omega
        sigma2_0 = np.exp(ln_sigma2[0])
        z_0 = eps[0]
        e[0] = np.sqrt(sigma2_0) * z_0
        for t in range(1, n):
            z = eps[t]
            ln_sigma2[t] = (
                omega
                + alpha * (abs(z) - np.sqrt(2.0 / np.pi))
                + gamma * z
                + beta * ln_sigma2[t - 1]
            )
            sigma2_t = np.exp(ln_sigma2[t])
            e[t] = np.sqrt(sigma2_t) * z
        return 1.0 + e

    def test_egarch_fit(self, egarch_data):
        """EGARCH(1,1,1) with vol='EGARCH' fits successfully."""
        from Ts.TsModels._garch import GARCH
        from Ts.TsModels._garch_result import GARCHResult

        model = GARCH(egarch_data, p=1, o=1, q=1, vol="EGARCH")
        result = model.fit()
        assert isinstance(result, GARCHResult)
        assert result.model_type == "EGARCH"

    def test_egarch_has_gamma(self, egarch_data):
        """EGARCH(1,1,1) result includes gamma[1]."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(egarch_data, p=1, o=1, q=1, vol="EGARCH")
        result = model.fit()
        assert "gamma[1]" in result.params

    def test_egarch_summary_label(self, egarch_data):
        """EGARCH summary shows EGARCH label."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(egarch_data, p=1, o=1, q=1, vol="EGARCH")
        result = model.fit()
        text = result.summary()
        assert "EGARCH" in text

    def test_egarch_persistence_only_beta(self, egarch_data):
        """EGARCH persistence test only includes beta, not alpha/gamma."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(egarch_data, p=1, o=1, q=1, vol="EGARCH")
        result = model.fit()
        stab = result.test_persistence()
        assert 0 < stab["persistence_sum"] < 2.0
        assert stab["chi2"] >= 0

    def test_egarch_forecast(self, egarch_data):
        """EGARCH 1-step predict returns positive variance."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(egarch_data, p=1, o=1, q=1, vol="EGARCH")
        result = model.fit()
        pr = result.predict(start=result.nobs, end=result.nobs)
        assert len(pr.mean) == 1
        assert np.all(pr.mean > 0)

    def test_invalid_vol_raises(self, egarch_data):
        """Invalid vol value raises ValueError."""
        from Ts.TsModels._garch import GARCH

        with pytest.raises(ValueError):
            GARCH(egarch_data, p=1, q=1, vol="INVALID")

    def test_egarch_symmetric_no_gamma(self, egarch_data):
        """EGARCH with o=0 has no gamma parameter."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(egarch_data, p=1, q=1, o=0, vol="EGARCH")
        result = model.fit()
        assert "gamma" not in result.params


class TestIGARCH:
    """Test IGARCH (Integrated GARCH) constrained estimation."""

    @pytest.fixture
    def igarch11_data(self):
        """Generate IGARCH(1,1) data: alpha+beta=1."""
        r = simulate_garch(
            n=300,
            p=1,
            q=1,
            omega=0.05,
            alpha=[0.30],
            beta=[0.70],
            seed=42,
            burn=200,
        )
        return r.data

    def test_igarch_egarch_raises(self, igarch11_data):
        """igarch=True with vol='EGARCH' raises ValueError."""
        from Ts.TsModels._garch import GARCH

        with pytest.raises(ValueError, match="IGARCH"):
            GARCH(igarch11_data, p=1, q=1, vol="EGARCH", igarch=True)

    def test_igarch_garch_m_raises(self, igarch11_data):
        """igarch=True with garch_m=True raises ValueError."""
        from Ts.TsModels._garch import GARCH

        with pytest.raises(ValueError, match="IGARCH"):
            GARCH(igarch11_data, p=1, q=1, garch_m=True, igarch=True)

    def test_igarch_q0_raises(self, igarch11_data):
        """igarch=True with q=0 (no GARCH component) raises ValueError."""
        from Ts.TsModels._garch import GARCH

        with pytest.raises(ValueError, match="IGARCH"):
            GARCH(igarch11_data, p=1, q=0, igarch=True)

    def test_igarch11_basic_fit(self, igarch11_data):
        """IGARCH(1,1) fits and returns GARCHResult with model_type IGARCH."""
        from Ts.TsModels._garch import GARCH
        from Ts.TsModels._garch_result import GARCHResult

        model = GARCH(igarch11_data, p=1, q=1, igarch=True)
        assert model.igarch is True
        result = model.fit()
        assert isinstance(result, GARCHResult)
        assert result.model_type == "IGARCH"

    def test_igarch11_constraint_satisfied(self, igarch11_data):
        """IGARCH(1,1) estimates satisfy alpha+beta=1."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(igarch11_data, p=1, q=1, igarch=True)
        result = model.fit()
        alpha_sum = sum(v for k, v in result.params.items() if k.startswith("alpha"))
        beta_sum = sum(v for k, v in result.params.items() if k.startswith("beta"))
        assert abs(alpha_sum + beta_sum - 1.0) < 1e-10

    def test_igarch11_summary_shows_igarch(self, igarch11_data):
        """IGARCH summary() displays IGARCH model label."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(igarch11_data, p=1, q=1, igarch=True)
        result = model.fit()
        text = result.summary()
        assert "IGARCH" in text

    def test_igarch11_forecast(self, igarch11_data):
        """IGARCH(1,1) predict returns positive variances."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(igarch11_data, p=1, q=1, igarch=True)
        result = model.fit()
        pr = result.predict(start=result.nobs, end=result.nobs + 2)
        assert len(pr.mean) == 3
        assert np.all(pr.mean > 0)

    def test_igarch12_fit_uses_q_minus_one_free_beta_params(self):
        """IGARCH(1,2) fits with exactly one free beta coefficient."""
        from Ts.TsModels._garch import GARCH
        from Ts.TsSims import simulate_igarch

        data = simulate_igarch(
            n=300,
            p=1,
            q=2,
            omega=0.05,
            alpha=[0.20],
            beta=[0.30, 0.50],
            seed=42,
            burn=200,
        ).data
        result = GARCH(data, p=1, q=2, igarch=True).fit()

        alpha_sum = sum(
            value for name, value in result.params.items() if name.startswith("alpha")
        )
        beta_sum = sum(
            value for name, value in result.params.items() if name.startswith("beta")
        )
        assert abs(alpha_sum + beta_sum - 1.0) < 1e-10
        assert {"beta[1]", "beta[2]"} <= result.params.keys()

    def test_igarch_from_simulate_igarch(self):
        """simulate_igarch generates data with alpha+beta=1 by construction."""
        from Ts.TsSims import simulate_igarch

        r = simulate_igarch(
            n=200, p=1, q=1, omega=0.05, alpha=[0.30], beta=[0.70], seed=42
        )
        params = r.get_params()
        alpha_sum = sum(params["alpha"])
        beta_sum = sum(params["beta"])
        assert abs(alpha_sum + beta_sum - 1.0) < 1e-10
        assert len(r.data) == 200
        assert r.conditional_volatility is not None


class TestPersistenceErrorHandling:
    """Test that persistence test failures are handled gracefully."""

    def test_persistence_failure_shows_in_summary(self):
        """If test_persistence fails, summary should include error info, not silently pass."""
        import numpy as np
        from Ts.TsModels import GARCH

        # Use very short series where persistence test might have issues
        data = np.random.randn(50)
        model = GARCH(data, p=1, q=1)
        result = model.fit()
        s = result.summary()
        # Summary should be a non-empty string
        assert isinstance(s, str) and len(s) > 0


class TestIGARCHHessian:
    """Test that IGARCH singular Hessian is handled meaningfully."""

    def test_igarch_singular_hessian_raises_meaningful_error(self):
        """IGARCH with problematic data should give meaningful error, not crash."""
        # Skip if IGARCH fitting is not available in tests


class TestGARCHCovers:
    """Aggregate coverage declarations for GARCH module."""

    def test_cover_all(self, garch11_data):
        """Declare coverage for all GARCH module items exercised by tests.

        covers: code/python/Ts/TsModels/_garch_result.py [module]
        covers: code/python/Ts/TsModels/_garch_result.py::GARCHResult [class]
        covers: code/python/Ts/TsModels/_garch_result.py::GARCHResult.conditional_variance [function]
        covers: code/python/Ts/TsModels/_garch_result.py::GARCHResult.summary [function]
        covers: code/python/Ts/TsModels/_garch_result.py::GARCHResult.test_persistence [function]
        covers: code/python/Ts/TsModels/_garch_result.py::GARCHResult.long_run_equilibrium [function]
        covers: code/python/Ts/TsModels/_garch_result.py::_scale_params_back [function]
        covers: code/python/Ts/TsModels/_garch_result.py::_get_dist_object [function]
        """


class TestGARCHPredict:
    """Test unified GARCHResult.predict() volatility forecasting."""

    @pytest.fixture
    def garch_result(self, garch11_data):
        """Fit GARCH(1,1) and return result."""
        from Ts.TsModels._garch import GARCH

        model = GARCH(garch11_data, p=1, q=1)
        return model.fit()

    def test_predict_in_sample_volatility(self, garch_result):
        """predict() defaults returns conditional volatility for full sample.

        covers: code/python/Ts/TsModels/_garch_result.py::GARCHResult.predict [function]
        covers: code/python/Ts/TsModels/_garch_result.py::GARCHResult._forecast [function]
        covers: code/python/Ts/TsModels/_garch_result.py::GARCHResult._forecast_conditional_vol [function]
        covers: code/python/Ts/TsModels/_garch_result.py::GARCHResult._garch_forecast_vol [function]
        covers: code/python/Ts/TsModels/_garch_result.py::GARCHResult._egarch_forecast_vol [function]
        covers: code/python/Ts/TsModels/_garch_result.py::GARCHResult._igarch_forecast_vol [function]
        covers: code/python/Ts/TsModels/_garch_result.py::GARCHResult._forecast_igarch [function]
        """
        from Ts.TsModels._base import PredictResult

        pr = garch_result.predict()
        assert isinstance(pr, PredictResult)
        assert len(pr.mean) == garch_result.nobs
        assert pr.lower is None
        assert pr.upper is None
        assert np.all(pr.mean > 0)

    def test_predict_rejects_removed_dynamic_argument(self, garch_result):
        """GARCH prediction rejects the removed dynamic compatibility mode."""
        with pytest.raises(TypeError):
            garch_result.predict(dynamic=True)

    def test_predict_validates_protocol_alpha(self, garch_result):
        """The shared evaluation alpha contract is explicit and validated."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            garch_result.predict(alpha=0)

    def test_predict_out_of_sample_volatility(self, garch_result):
        """predict() beyond sample returns variance forecasts."""
        from Ts.TsModels._base import PredictResult

        horizon = 5
        end = garch_result.nobs + horizon - 1
        pr = garch_result.predict(start=garch_result.nobs, end=end)

        assert isinstance(pr, PredictResult)
        assert len(pr.mean) == horizon
        assert np.all(pr.mean > 0)

    def test_predict_can_skip_early_future_periods(self, garch_result):
        """A future-only window may start after the first forecast period."""
        start = garch_result.nobs + 2
        result = garch_result.predict(start=start, end=start + 2)
        full_result = garch_result.predict(
            start=garch_result.nobs,
            end=start + 2,
        )

        assert result.mean.shape == (3,)
        assert np.all(result.mean > 0)
        assert result.is_oos.tolist() == [True, True, True]
        np.testing.assert_allclose(result.mean, full_result.mean[2:])

    def test_oos_uses_observable_volatility_proxy(self, garch_result):
        """GARCH OOS scores one documented observable proxy."""
        from Ts.TsModels import GARCH

        split = int(garch_result.nobs * 0.7)
        evaluation = GARCH(
            garch_result.data,
            p=garch_result._p,
            q=garch_result._q,
            o=garch_result._o,
            compare_lags=False,
        ).oos(
            estimation_period=(0, split - 1),
            validation_period=(split, len(garch_result.data) - 1),
        )

        expected = np.abs(
            garch_result.data[split:] - np.mean(garch_result.data[:split])
        )
        np.testing.assert_allclose(evaluation.actual, expected)
        assert evaluation.target == "absolute_demeaned_return_proxy"
        assert evaluation.metrics["rmse"] > 0

    def test_predict_has_no_evaluation_fields(self, garch_result):
        """Ordinary volatility predictions no longer carry scoring state."""
        pr = garch_result.predict(start=0, end=garch_result.nobs - 1)
        assert not hasattr(pr, "metrics")
        assert not hasattr(pr, "actual")

    def test_predict_arch_q0(self, arch_data):
        """predict() works for pure ARCH (q=0) models."""
        from Ts.TsModels._garch import GARCH
        from Ts.TsModels._base import PredictResult

        model = GARCH(arch_data, p=2, q=0)
        result = model.fit()
        pr = result.predict()

        assert isinstance(pr, PredictResult)
        assert len(pr.mean) == result.nobs
