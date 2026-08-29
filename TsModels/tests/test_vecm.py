"""Tests for Ts.TsModels._vecm -- VECM and VECMResult."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vecm_data_2d():
    """Generate 2-variable cointegrated data.

    DGP: x_t = x_{t-1} + e_{xt}           (I(1) random walk)
         y_t = 2*x_t + u_{yt}              (cointegrated with x)

    Cointegration vector: (1, -2) -> rank=1
    """
    np.random.seed(42)
    n = 200
    e_x = np.random.randn(n)
    u_y = np.random.randn(n)
    x = np.cumsum(e_x)
    y = 2.0 * x + u_y
    return np.column_stack([y, x])


@pytest.fixture
def vecm_data_3d():
    """Generate 3-variable cointegrated data with rank=2.

    DGP: x_t = x_{t-1} + e_{xt}           (common stochastic trend)
         y_t = 2*x_t + u_{yt}              (cointegrated)
         z_t = 3*x_t + u_{zt}              (cointegrated)

    Cointegration rank = 2.
    """
    np.random.seed(123)
    n = 300
    e_x = np.random.randn(n)
    u_y = 0.5 * np.random.randn(n)
    u_z = 0.5 * np.random.randn(n)
    x = np.cumsum(e_x)
    y = 2.0 * x + u_y
    z = 3.0 * x + u_z
    return np.column_stack([z, y, x])


# ---------------------------------------------------------------------------
# TestVECMInit
# ---------------------------------------------------------------------------


class TestVECMInit:
    """Test VECM construction and parameter validation.

    covers: code/python/Ts/TsModels/_vecm.py [module]
    covers: code/python/Ts/TsModels/_vecm.py::VECM [class]
    """

    def test_init_stores_data_and_params(self, vecm_data_2d):
        """VECM stores data, lags, coint_rank, and trend.

        covers: code/python/Ts/TsModels/_vecm.py [module]
        covers: code/python/Ts/TsModels/_vecm.py::VECM [class]
        covers: code/python/Ts/TsModels/_vecm.py::VECM.__init__ [function]
        """
        from Ts.TsModels._vecm import VECM

        model = VECM(vecm_data_2d, lags=2, coint_rank=1, trend="c")
        assert model.lags == 2
        assert model.coint_rank == 1
        assert model.trend == "c"
        assert model.data.shape == (200, 2)
        assert model.result_ is None

    def test_default_trend_is_c(self, vecm_data_2d):
        """Default trend is 'c'.

        covers: code/python/Ts/TsModels/_vecm.py::VECM.__init__ [function]
        """
        from Ts.TsModels._vecm import VECM

        model = VECM(vecm_data_2d, lags=2, coint_rank=1)
        assert model.trend == "c"

    def test_invalid_lags_less_than_1_raises(self, vecm_data_2d):
        """lags < 1 raises ValueError.

        covers: code/python/Ts/TsModels/_vecm.py::VECM.__init__ [function]
        """
        from Ts.TsModels._vecm import VECM

        with pytest.raises(ValueError):
            VECM(vecm_data_2d, lags=0, coint_rank=1)

    def test_invalid_coint_rank_raises(self, vecm_data_2d):
        """coint_rank must be between 1 and k-1.

        covers: code/python/Ts/TsModels/_vecm.py::VECM.__init__ [function]
        """
        from Ts.TsModels._vecm import VECM

        with pytest.raises(ValueError):
            VECM(vecm_data_2d, lags=2, coint_rank=0)

        with pytest.raises(ValueError):
            VECM(vecm_data_2d, lags=2, coint_rank=2)

    def test_1d_data_raises(self):
        """1-D data raises ValueError.

        covers: code/python/Ts/TsModels/_vecm.py::VECM.__init__ [function]
        """
        from Ts.TsModels._vecm import VECM

        with pytest.raises(ValueError):
            VECM(np.array([1.0, 2.0, 3.0]), lags=2, coint_rank=1)

    def test_invalid_trend_raises(self, vecm_data_2d):
        """Invalid trend string raises ValueError.

        covers: code/python/Ts/TsModels/_vecm.py::VECM.__init__ [function]
        """
        from Ts.TsModels._vecm import VECM

        with pytest.raises(ValueError):
            VECM(vecm_data_2d, lags=2, coint_rank=1, trend="invalid")

    def test_cols_auto_generated(self, vecm_data_2d):
        """Default cols auto-generate y0, y1, ...

        covers: code/python/Ts/TsModels/_vecm.py::VECM.__init__ [function]
        """
        from Ts.TsModels._vecm import VECM

        model = VECM(vecm_data_2d, lags=2, coint_rank=1)
        assert model.data_names == ["y0", "y1"]

    def test_custom_cols(self, vecm_data_2d):
        """Custom cols are stored correctly.

        covers: code/python/Ts/TsModels/_vecm.py::VECM.__init__ [function]
        """
        from Ts.TsModels._vecm import VECM

        model = VECM(vecm_data_2d, lags=2, coint_rank=1, cols=["Z", "Y"])
        assert model.data_names == ["Z", "Y"]


# ---------------------------------------------------------------------------
# TestVECMFit
# ---------------------------------------------------------------------------


class TestVECMFit:
    """Test VECM.fit() estimation.

    covers: code/python/Ts/TsModels/_vecm.py::VECM.fit [function]
    """

    def test_fit_returns_vecm_result(self, vecm_data_3d):
        """fit() returns VECMResult and stores in result_.

        covers: code/python/Ts/TsModels/_vecm.py::VECM.fit [function]
        covers: code/python/Ts/TsModels/_vecm.py::VECMResult [class]
        """
        from Ts.TsModels._vecm import VECM, VECMResult

        model = VECM(vecm_data_3d, lags=2, coint_rank=2, trend="c")
        result = model.fit()

        assert isinstance(result, VECMResult)
        assert model.result_ is result

    def test_fit_alpha_beta_gamma_shapes(self, vecm_data_3d):
        """alpha, beta, gamma have correct dimensions.

        covers: code/python/Ts/TsModels/_vecm.py::VECM.fit [function]
        """
        from Ts.TsModels._vecm import VECM

        model = VECM(vecm_data_3d, lags=2, coint_rank=2, trend="c")
        result = model.fit()

        k = 3
        r = 2
        assert result.alpha.shape == (k, r)
        assert result.beta.shape == (k, r)
        assert result.gamma.shape == (k, k)  # k_ar_diff=1 -> (k, k)

    def test_fit_stores_params(self, vecm_data_3d):
        """Result stores params, std_errors, p_values dicts.

        covers: code/python/Ts/TsModels/_vecm.py::VECM.fit [function]
        """
        from Ts.TsModels._vecm import VECM

        model = VECM(vecm_data_3d, lags=2, coint_rank=2, trend="c")
        result = model.fit()

        assert isinstance(result.params, dict)
        assert len(result.params) > 0
        assert isinstance(result.std_errors, dict)
        assert isinstance(result.p_values, dict)

    def test_fit_stores_info_criteria(self, vecm_data_3d):
        """Result stores aic, bic, log_likelihood, sigma_u.

        covers: code/python/Ts/TsModels/_vecm.py::VECM.fit [function]
        """
        from Ts.TsModels._vecm import VECM

        model = VECM(vecm_data_3d, lags=2, coint_rank=2, trend="c")
        result = model.fit()

        assert isinstance(result.aic, float)
        assert isinstance(result.bic, float)
        assert isinstance(result.log_likelihood, float)
        assert result.sigma_u.shape == (3, 3)

    def test_info_criteria_count_only_free_parameters(self, vecm_data_3d):
        """AIC/BIC count normalized beta and deterministic terms correctly."""
        from Ts.TsModels._vecm import VECM

        result = VECM(vecm_data_3d, lags=2, coint_rank=2, trend="c").fit()
        k, rank, k_ar_diff = 3, 2, 1
        n_params = k * rank + (k - rank) * rank
        n_params += k * k * k_ar_diff
        n_params += result._vecm_result.det_coef.size
        expected_aic = -2.0 * result.log_likelihood + 2.0 * n_params
        assert result.aic == pytest.approx(expected_aic)

    def test_fit_with_trend_n(self, vecm_data_3d):
        """fit() works with trend='n'.

        covers: code/python/Ts/TsModels/_vecm.py::VECM.fit [function]
        """
        from Ts.TsModels._vecm import VECM

        model = VECM(vecm_data_3d, lags=2, coint_rank=2, trend="n")
        result = model.fit()
        assert result is not None

    @pytest.mark.parametrize(
        ("trend", "deterministic", "expected_keys"),
        [
            ("n", "n", set()),
            ("rconstant", "ci", {"_cons.ce1"}),
            ("c", "co", {"_cons.D_y0"}),
            ("rtrend", "coli", {"_cons.D_y0", "_trend.ce1"}),
            ("ct", "colo", {"_cons.D_y0", "_trend.D_y0"}),
        ],
    )
    def test_all_supported_trends_fit_and_summarize(
        self, vecm_data_3d, trend, deterministic, expected_keys
    ):
        """Every public trend maps correctly and produces a valid summary."""
        from Ts.TsModels._vecm import VECM

        result = VECM(vecm_data_3d, lags=2, coint_rank=2, trend=trend).fit()

        assert result._vecm_result.model.deterministic == deterministic
        assert expected_keys <= result.params.keys()
        assert "Vector error-correction model" in result.summary()

    def test_fit_with_custom_cols(self, vecm_data_3d):
        """Custom column names appear in result.

        covers: code/python/Ts/TsModels/_vecm.py::VECM.fit [function]
        """
        from Ts.TsModels._vecm import VECM

        model = VECM(
            vecm_data_3d, lags=2, coint_rank=2, trend="c", cols=["Z", "Y", "X"]
        )
        result = model.fit()
        assert result._data_names == ["Z", "Y", "X"]


# ---------------------------------------------------------------------------
# TestVECMResultSummary
# ---------------------------------------------------------------------------


class TestVECMResultSummary:
    """Test VECMResult.summary() output format.

    covers: code/python/Ts/TsModels/_vecm.py::VECMResult.summary [function]
    """

    @pytest.fixture
    def fitted(self, vecm_data_3d):
        """Return a fitted VECMResult for summary tests."""
        from Ts.TsModels._vecm import VECM

        model = VECM(
            vecm_data_3d, lags=2, coint_rank=2, trend="c", cols=["Z", "Y", "X"]
        )
        return model.fit()

    def test_summary_contains_header(self, fitted):
        """summary() contains 'Vector error-correction model' header.

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult.summary [function]
        """
        s = fitted.summary()
        assert "Vector error-correction model" in s

    def test_summary_contains_info_criteria(self, fitted):
        """summary() contains AIC, Log-Likelihood.

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult.summary [function]
        """
        s = fitted.summary()
        assert "AIC" in s
        assert "Log-Likelihood" in s

    def test_summary_contains_alpha_section(self, fitted):
        """summary() contains alpha adjustment coefficients section.

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult.summary [function]
        """
        s = fitted.summary()
        assert "Adjustment coefficients" in s
        assert "D_" in s

    def test_summary_contains_ce_labels(self, fitted):
        """summary() contains _ce1, _ce2 error-correction labels.

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult.summary [function]
        """
        s = fitted.summary()
        assert "_ce1" in s
        assert "_ce2" in s

    def test_summary_contains_beta_section(self, fitted):
        """summary() contains cointegrating vectors section.

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult.summary [function]
        """
        s = fitted.summary()
        assert "Cointegrating vectors" in s


# ---------------------------------------------------------------------------
# TestVECMResultIRF
# ---------------------------------------------------------------------------


class TestVECMResultIRF:
    """Test VECM impulse response functions.

    covers: code/python/Ts/TsModels/_vecm.py::VECMResult.irf [function]
    """

    @pytest.fixture
    def fitted(self, vecm_data_3d):
        """Return a fitted VECMResult for IRF tests."""
        from Ts.TsModels._vecm import VECM

        model = VECM(
            vecm_data_3d, lags=2, coint_rank=2, trend="c", cols=["Z", "Y", "X"]
        )
        return model.fit()

    def test_irf_returns_irf_result(self, fitted):
        """irf() returns IRFResult.

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult.irf [function]
        """
        from Ts.TsModels._var import IRFResult

        irf = fitted.irf(periods=10, orth=False)
        assert isinstance(irf, IRFResult)

    def test_irf_correct_shape(self, fitted):
        """IRF values have shape (periods+1, k, k).

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult.irf [function]
        """
        irf = fitted.irf(periods=10, orth=False)
        assert irf.values.shape == (11, 3, 3)

    def test_irf_orthogonalized(self, fitted):
        """orth=True returns orthogonalized IRF.

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult.irf [function]
        """
        irf = fitted.irf(periods=5, orth=True)
        assert irf.orth is True
        assert irf.values.shape == (6, 3, 3)


# ---------------------------------------------------------------------------
# TestVECMResultPredict
# ---------------------------------------------------------------------------


class TestVECMResultPredict:
    """Test VECM predict method.

    covers: code/python/Ts/TsModels/_vecm.py::VECMResult.predict [function]
    """

    @pytest.fixture
    def fitted(self, vecm_data_3d):
        """Return a fitted VECMResult for predict tests."""
        from Ts.TsModels._vecm import VECM

        model = VECM(
            vecm_data_3d, lags=2, coint_rank=2, trend="c", cols=["Z", "Y", "X"]
        )
        return model.fit()

    def test_predict_in_sample_shape(self, fitted):
        """In-sample predict returns correct shape.

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult.predict [function]
        """
        pr = fitted.predict(start=0, end=9)
        assert pr.mean.shape == (10, 3)

    def test_predict_oos_shape(self, fitted):
        """Out-of-sample predict extends beyond nobs.

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult.predict [function]
        """
        nobs = fitted.nobs
        pr = fitted.predict(start=nobs - 5, end=nobs + 3)
        assert pr.mean.shape[0] == 9  # (nobs+3) - (nobs-5) + 1

    def test_predict_can_skip_early_future_periods(self, fitted):
        """A future-only window returns only the requested later periods."""
        start = fitted.nobs + 2
        result = fitted.predict(start=start, end=start + 2)
        full_result = fitted.predict(start=fitted.nobs, end=start + 2)

        assert result.mean.shape == (3, 3)
        assert result.is_oos.tolist() == [True, True, True]
        np.testing.assert_allclose(result.mean, full_result.mean[2:])

    def test_predict_rejects_unsupported_dynamic_mode(self, fitted):
        """VECM does not silently ignore a requested dynamic mode."""
        with pytest.raises(TypeError, match="dynamic"):
            fitted.predict(dynamic=True)


# ---------------------------------------------------------------------------
# TestVECMResultDiagnostics
# ---------------------------------------------------------------------------


class TestVECMResultDiagnostics:
    """Test VECM diagnostic plots and residual tests.

    covers: code/python/Ts/TsModels/_base.py::BaseModelResult.plot_diagnostics [function]
    covers: code/python/Ts/TsModels/_base.py::BaseModelResult.test_residuals [function]
    """

    @pytest.fixture
    def fitted(self, vecm_data_3d):
        """Return a fitted VECMResult for diagnostics tests."""
        from Ts.TsModels._vecm import VECM

        model = VECM(
            vecm_data_3d, lags=2, coint_rank=2, trend="c", cols=["Z", "Y", "X"]
        )
        return model.fit()

    def test_plot_diagnostics_returns_fig_axes(self, fitted):
        """plot_diagnostics() returns fig and axes.

        covers: code/python/Ts/TsModels/_base.py::BaseModelResult.plot_diagnostics [function]
        """
        fig, axes = fitted.plot_diagnostics()
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        assert axes is not None
        assert axes.shape == (3, 3)

        standardized = fitted.standardized_residuals
        np.testing.assert_allclose(
            np.std(standardized, axis=0, ddof=0),
            np.ones(standardized.shape[1]),
        )
        for position in range(standardized.shape[1]):
            displayed = np.asarray(
                axes[position, 0].lines[0].get_ydata(), dtype=float
            )
            np.testing.assert_allclose(displayed, standardized[:, position])
            assert axes[position, 0].get_title().endswith(
                "Standardized Residuals"
            )
            assert axes[position, 0].get_ylabel() == ""
            assert axes[position, 0].get_xlabel() == ""

    def test_test_residuals_returns_dict(self, fitted):
        """test_residuals() returns dict of ResidualTestResults.

        covers: code/python/Ts/TsModels/_base.py::BaseModelResult.test_residuals [function]
        """
        results = fitted.test_residuals(lags=5)
        assert isinstance(results, dict)
        assert len(results) == 3
        assert "Z" in results


# ---------------------------------------------------------------------------
# TestVECMSelectOrder
# ---------------------------------------------------------------------------


class TestVECMSelectOrder:
    """Test VECM.select_order static method.

    covers: code/python/Ts/TsModels/_vecm.py::VECM.select_order [function]
    covers: code/python/Ts/TsModels/_vecm.py::VECMOrderResult [class]
    """

    def test_select_order_returns_result(self, vecm_data_3d):
        """select_order returns VECMOrderResult with selected lag.

        covers: code/python/Ts/TsModels/_vecm.py::VECMOrderResult [class]
        covers: code/python/Ts/TsModels/_vecm.py::VECM.select_order [function]
        """
        from Ts.TsModels._vecm import VECM

        result = VECM.select_order(
            vecm_data_3d, max_lags=4, coint_rank=2, criterion="aic"
        )
        assert result.selected_lag >= 1

    def test_vecm_order_result_summary(self, vecm_data_3d):
        """VECMOrderResult.summary() returns string.

        covers: code/python/Ts/TsModels/_vecm.py::VECMOrderResult.summary [function]
        """
        from Ts.TsModels._vecm import VECM

        result = VECM.select_order(
            vecm_data_3d, max_lags=3, coint_rank=2, criterion="aic"
        )
        s = result.summary()
        assert "VECM Lag Order" in s
        assert "AIC" in s

    def test_vecm_order_result_repr(self, vecm_data_3d):
        """VECMOrderResult.__repr__ returns string.

        covers: code/python/Ts/TsModels/_vecm.py::VECMOrderResult.__repr__ [function]
        """
        from Ts.TsModels._vecm import VECM

        result = VECM.select_order(
            vecm_data_3d, max_lags=3, coint_rank=2, criterion="aic"
        )
        r = repr(result)
        assert isinstance(r, str)
        assert "VECM Lag Order" in r

    def test_select_order_bic_uses_bic_values(self, vecm_data_3d):
        """BIC selection reports BIC values rather than relabelled AIC."""
        from Ts.TsModels._vecm import VECM

        aic = VECM.select_order(vecm_data_3d, max_lags=3, coint_rank=2, criterion="aic")
        bic = VECM.select_order(vecm_data_3d, max_lags=3, coint_rank=2, criterion="bic")

        assert bic.criterion == "bic"
        assert any(
            bic.values[lag] != pytest.approx(aic.values[lag]) for lag in bic.values
        )

    def test_select_order_rejects_invalid_criterion(self, vecm_data_3d):
        """Unknown selection criteria fail before fitting candidates."""
        from Ts.TsModels._vecm import VECM

        with pytest.raises(ValueError, match="criterion"):
            VECM.select_order(vecm_data_3d, max_lags=3, coint_rank=2, criterion="hqic")

    def test_select_order_raises_when_all_candidates_fail(self):
        """A failed search cannot silently report lag one as successful."""
        from Ts.TsModels._vecm import VECM

        with pytest.raises(RuntimeError, match="No VECM candidate converged"):
            VECM.select_order(np.ones((30, 2)), max_lags=3, coint_rank=1)


# ---------------------------------------------------------------------------
# TestVECMResultFEVD
# ---------------------------------------------------------------------------


class TestVECMResultFEVD:
    """Test VECM FEVD method.

    covers: code/python/Ts/TsModels/_vecm.py::VECMResult.fevd [function]
    """

    @pytest.fixture
    def fitted(self, vecm_data_3d):
        """Return a fitted VECMResult for FEVD tests."""
        from Ts.TsModels._vecm import VECM

        model = VECM(
            vecm_data_3d, lags=2, coint_rank=2, trend="c", cols=["Z", "Y", "X"]
        )
        return model.fit()

    def test_fevd_returns_fevd_result(self, fitted):
        """fevd() returns FEVDResult.

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult.fevd [function]
        """
        from Ts.TsModels._var import FEVDResult

        f = fitted.fevd(periods=5)
        assert isinstance(f, FEVDResult)

    def test_fevd_correct_shape(self, fitted):
        """FEVD values have shape (periods, k, k).

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult.fevd [function]
        """
        f = fitted.fevd(periods=5)
        assert f.values.shape == (5, 3, 3)

    def test_fevd_rejects_removed_alpha_argument(self, fitted):
        """Point-only VECM FEVD does not accept an unused alpha argument."""
        with pytest.raises(TypeError):
            fitted.fevd(periods=5, alpha=0.1)


# ---------------------------------------------------------------------------
# TestVECMResultGranger
# ---------------------------------------------------------------------------


class TestVECMResultGranger:
    """Test VECM Granger causality.

    covers: code/python/Ts/TsModels/_vecm.py::VECMResult.granger_causality [function]
    """

    @pytest.fixture
    def fitted(self, vecm_data_3d):
        """Return a fitted VECMResult for Granger tests."""
        from Ts.TsModels._vecm import VECM

        model = VECM(
            vecm_data_3d, lags=2, coint_rank=2, trend="c", cols=["Z", "Y", "X"]
        )
        return model.fit()

    def test_granger_single_pair(self, fitted):
        """Granger test for single pair returns result.

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult.granger_causality [function]
        """
        gc = fitted.granger_causality(caused="Z", causing="Y", kind="chi2")
        assert len(gc) == 1
        assert gc.tests[0].caused == "Z"

    def test_granger_all_pairs(self, fitted):
        """Granger all-pairs mode returns multiple tests.

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult._granger_causality_all [function]
        """
        gc = fitted.granger_causality(kind="chi2")
        assert len(gc) >= 6  # at least k*(k-1) pairwise tests

    def test_granger_df_is_int_tuple(self, fitted):
        """Granger test entries have tuple df with int elements.

        VECM granger always uses F-test (statsmodels API), so df is
        (numerator, denominator).

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult.granger_causality [function]
        """
        gc = fitted.granger_causality(caused="Z", causing="Y")
        df = gc.tests[0].df
        assert isinstance(df, tuple)
        assert len(df) == 2
        assert isinstance(df[0], int)
        assert isinstance(df[1], int)

    def test_granger_all_df_types(self, fitted):
        """All-pairs Granger test entries all have tuple-of-int df.

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult._granger_causality_all [function]
        """
        gc = fitted.granger_causality()
        for entry in gc:
            assert isinstance(entry.df, tuple), (
                f"Expected tuple df, got {type(entry.df)} for "
                f"{entry.caused} <- {entry.causing}"
            )
            assert isinstance(entry.df[0], int)
            assert isinstance(entry.df[1], int)


# ---------------------------------------------------------------------------
# TestVECMResultIsStable
# ---------------------------------------------------------------------------


class TestVECMResultIsStable:
    """Test VECM is_stable property.

    covers: code/python/Ts/TsModels/_vecm.py::VECMResult.is_stable [function]
    """

    @pytest.fixture
    def fitted(self, vecm_data_3d):
        """Return a fitted VECMResult for stability tests."""
        from Ts.TsModels._vecm import VECM

        model = VECM(
            vecm_data_3d, lags=2, coint_rank=2, trend="c", cols=["Z", "Y", "X"]
        )
        return model.fit()

    def test_is_stable_returns_bool(self, fitted):
        """is_stable returns a bool.

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult.is_stable [function]
        """
        s = fitted.is_stable
        assert isinstance(s, bool)


# ---------------------------------------------------------------------------
# TestVECMResultPlotRoots
# ---------------------------------------------------------------------------


class TestVECMResultPlotRoots:
    """Test VECM plot_roots method.

    covers: code/python/Ts/TsModels/_vecm.py::VECMResult.plot_roots [function]
    """

    @pytest.fixture
    def fitted(self, vecm_data_3d):
        """Return a fitted VECMResult for plot_roots tests."""
        from Ts.TsModels._vecm import VECM

        model = VECM(
            vecm_data_3d, lags=2, coint_rank=2, trend="c", cols=["Z", "Y", "X"]
        )
        return model.fit()

    def test_plot_roots_returns_fig_ax(self, fitted):
        """plot_roots() returns fig and ax.

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult.plot_roots [function]
        """
        import matplotlib.pyplot as plt

        fig, _ax = fitted.plot_roots()
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# TestVECMResultRepr
# ---------------------------------------------------------------------------


class TestVECMResultRepr:
    """Test VECMResult __repr__.

    covers: code/python/Ts/TsModels/_vecm.py::VECMResult.__repr__ [function]
    """

    @pytest.fixture
    def fitted(self, vecm_data_3d):
        """Return a fitted VECMResult for repr tests."""
        from Ts.TsModels._vecm import VECM

        model = VECM(
            vecm_data_3d, lags=2, coint_rank=2, trend="c", cols=["Z", "Y", "X"]
        )
        return model.fit()

    def test_repr_returns_string(self, fitted):
        """repr() returns string containing model info.

        covers: code/python/Ts/TsModels/_vecm.py::VECMResult.__repr__ [function]
        """
        r = repr(fitted)
        assert isinstance(r, str)
        assert "Vector error-correction" in r
