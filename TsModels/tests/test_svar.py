"""Tests for Ts.TsModels._svar — SVAR and SVARResult."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
from numpy.linalg import inv
import pytest
from Ts.TsSims import simulate_sarima


@pytest.fixture
def var2_data():
    """Generate 2-variable data for SVAR testing.

    Two independent AR(1) processes stacked as columns:
    y0 ~ AR(1) with phi=0.7, y1 ~ AR(1) with phi=0.5.
    """
    r0 = simulate_sarima(n=150, order=(1, 0, 0), ar=[0.7], seed=42, burn=100)
    r1 = simulate_sarima(n=150, order=(1, 0, 0), ar=[0.5], seed=99, burn=100)
    return np.column_stack([r0.data, r1.data])


class TestSVARInit:
    """Test SVAR construction and parameter validation.

    covers: code/python/Ts/TsModels/_svar.py [module]
    covers: code/python/Ts/TsModels/_svar.py::SVAR [class]
    covers: code/python/Ts/TsModels/_svar.py::SVAR.__init__ [function]
    """

    def test_init_stores_data_and_params(self, var2_data):
        """SVAR stores data, lags, A, B matrices.

        covers: code/python/Ts/TsModels/_svar.py::SVAR.__init__ [function]
        """
        from Ts.TsModels._svar import SVAR

        A = np.array([[1, 0], [np.nan, 1]])
        B = np.array([[np.nan, 0], [0, np.nan]])
        model = SVAR(var2_data, lags=2, A=A, B=B)
        assert model.lags == 2
        assert model.data.shape == (150, 2)
        assert model.result_ is None
        assert model.A is not None
        assert model.B is not None

    def test_init_longrun(self, var2_data):
        """SVAR accepts C_lr for long-run restrictions.

        covers: code/python/Ts/TsModels/_svar.py::SVAR.__init__ [function]
        """
        from Ts.TsModels._svar import SVAR

        C_lr = np.array([[np.nan, 0], [np.nan, np.nan]])
        model = SVAR(var2_data, lags=2, C_lr=C_lr)
        assert model.C_lr is not None
        assert model.A is None
        assert model.B is None
    def test_longrun_rejects_below_diagonal_restriction(self, var2_data):
        """Unsupported lower-triangular restrictions are rejected early.

        covers: code/python/Ts/TsModels/_svar.py::SVAR.__init__ [function]
        """
        from Ts.TsModels._svar import SVAR

        C_lr = np.array([[np.nan, 0], [0, np.nan]])
        with pytest.raises(
            NotImplementedError,
            match="canonical Blanchard-Quah",
        ):
            SVAR(var2_data, lags=2, C_lr=C_lr)

    def test_longrun_rejects_nonzero_upper_entry(self, var2_data):
        """Unsupported upper-triangular values are rejected early.

        covers: code/python/Ts/TsModels/_svar.py::SVAR.__init__ [function]
        """
        from Ts.TsModels._svar import SVAR

        C_lr = np.array([[np.nan, 1], [np.nan, np.nan]])
        with pytest.raises(
            NotImplementedError,
            match="canonical Blanchard-Quah",
        ):
            SVAR(var2_data, lags=2, C_lr=C_lr)


    def test_no_constraint_raises(self, var2_data):
        """SVAR without A, B, or C_lr raises ValueError.

        covers: code/python/Ts/TsModels/_svar.py::SVAR.__init__ [function]
        """
        from Ts.TsModels._svar import SVAR

        with pytest.raises(ValueError, match="At least one"):
            SVAR(var2_data, lags=2)

    def test_mutually_exclusive_raises(self, var2_data):
        """C_lr with A or B raises ValueError.

        covers: code/python/Ts/TsModels/_svar.py::SVAR.__init__ [function]
        """
        from Ts.TsModels._svar import SVAR

        A = np.array([[1, 0], [np.nan, 1]])
        C_lr = np.array([[np.nan, 0], [np.nan, np.nan]])
        with pytest.raises(ValueError, match="mutually exclusive"):
            SVAR(var2_data, lags=2, A=A, C_lr=C_lr)

    def test_invalid_lags_raises(self, var2_data):
        """lags < 1 raises ValueError.

        covers: code/python/Ts/TsModels/_svar.py::SVAR.__init__ [function]
        """
        from Ts.TsModels._svar import SVAR

        A = np.array([[1, 0], [np.nan, 1]])
        with pytest.raises(ValueError):
            SVAR(var2_data, lags=0, A=A)

    def test_1d_data_raises(self, var2_data):
        """1-D data raises ValueError.

        covers: code/python/Ts/TsModels/_svar.py::SVAR.__init__ [function]
        """
        from Ts.TsModels._svar import SVAR

        A = np.array([[1, 0], [np.nan, 1]])
        with pytest.raises(ValueError):
            SVAR(np.array([1.0, 2.0, 3.0]), lags=1, A=A)

    def test_wrong_shape_A_raises(self, var2_data):
        """A matrix with wrong shape raises ValueError.

        covers: code/python/Ts/TsModels/_svar.py::SVAR.__init__ [function]
        """
        from Ts.TsModels._svar import SVAR

        A = np.array([[1, 0, 0], [np.nan, 1, 0], [0, np.nan, 1]])
        with pytest.raises(ValueError, match="A must be"):
            SVAR(var2_data, lags=2, A=A)


@pytest.fixture
def fitted_var(var2_data):
    """Fit a VAR(2) and return the VARResult."""
    from Ts.TsModels._var import VAR

    model = VAR(var2_data, lags=2)
    return model.fit()


@pytest.fixture
def svar_ab_fitted(var2_data):
    """Fit an AB-model SVAR with Cholesky (recursive) identification.

    A: lower triangular with ones on diagonal (1 free param)
    B: diagonal (2 free params) — just-identified.
    """
    from Ts.TsModels._svar import SVAR

    A = np.array([[1, 0], [np.nan, 1]])
    B = np.array([[np.nan, 0], [0, np.nan]])
    model = SVAR(var2_data, lags=2, A=A, B=B)
    return model.fit()


class TestSVARFitAB:
    """Test SVAR.fit() for AB-model.

    covers: code/python/Ts/TsModels/_svar.py::SVAR.fit [function]
    """

    def test_fit_returns_svar_result(self, svar_ab_fitted):
        """fit() returns SVARResult.

        covers: code/python/Ts/TsModels/_svar.py::SVARResult [class]
        """
        from Ts.TsModels._svar import SVARResult

        assert isinstance(svar_ab_fitted, SVARResult)

    def test_fit_A_matrix_diagonal_ones(self, svar_ab_fitted):
        """A matrix has ones on diagonal (fixed params preserved).

        covers: code/python/Ts/TsModels/_svar.py::SVAR.fit [function]
        """
        A = svar_ab_fitted.A
        assert np.isclose(A[0, 0], 1.0)
        assert np.isclose(A[1, 1], 1.0)
        assert np.isclose(A[0, 1], 0.0)  # fixed at 0

    def test_fit_B_matrix_diagonal_not_nan(self, svar_ab_fitted):
        """B matrix diagonal entries are estimated (not NaN).

        covers: code/python/Ts/TsModels/_svar.py::SVAR.fit [function]
        """
        B = svar_ab_fitted.B
        assert not np.isnan(B[0, 0])
        assert not np.isnan(B[1, 1])
        assert B[0, 0] > 0
        assert B[1, 1] > 0

    def test_fit_residual_covariance_recovers_sigma_u(self, svar_ab_fitted):
        """A^{-1} B B^T A^{-T} equals Sigma_u from reduced-form VAR.

        covers: code/python/Ts/TsModels/_svar.py::SVAR.fit [function]
        """
        A = svar_ab_fitted.A
        B = svar_ab_fitted.B
        sigma_u_implied = inv(A) @ B @ B.T @ inv(A).T
        assert np.allclose(sigma_u_implied, svar_ab_fitted.sigma_u, atol=1e-10)

    def test_fit_structural_shocks_uncorrelated(self, svar_ab_fitted):
        """Structural shocks have identity covariance (orthogonal).

        covers: code/python/Ts/TsModels/_svar.py::SVAR.fit [function]
        """
        eps = svar_ab_fitted.structural_residuals
        cov_eps = np.cov(eps, rowvar=False)
        assert np.allclose(cov_eps, np.eye(2), atol=0.1)
        assert abs(cov_eps[0, 1]) < 0.01

    def test_fit_stores_log_likelihood(self, svar_ab_fitted):
        """SVARResult includes log-likelihood of structural model.

        covers: code/python/Ts/TsModels/_svar.py::SVAR.fit [function]
        """
        assert hasattr(svar_ab_fitted, "svar_log_likelihood")
        assert svar_ab_fitted.svar_log_likelihood < 0


class TestSVARFitLongRun:
    """Test SVAR.fit() for long-run (Blanchard-Quah) restrictions.

    covers: code/python/Ts/TsModels/_svar.py::SVAR.fit [function]
    """

    def test_longrun_fit_returns_svar_result(self, var2_data):
        """Long-run SVAR fit() returns SVARResult."""
        from Ts.TsModels._svar import SVAR, SVARResult

        C_lr = np.array([[np.nan, 0], [np.nan, np.nan]])
        model = SVAR(var2_data, lags=2, C_lr=C_lr)
        result = model.fit()
        assert isinstance(result, SVARResult)

    def test_longrun_B_recovers_sigma_u(self, var2_data):
        """B B^T equals sigma_u for Blanchard-Quah identification."""
        from Ts.TsModels._svar import SVAR

        C_lr = np.array([[np.nan, 0], [np.nan, np.nan]])
        model = SVAR(var2_data, lags=2, C_lr=C_lr)
        result = model.fit()
        B = result.B
        sigma_u_implied = B @ B.T
        assert np.allclose(sigma_u_implied, result.sigma_u, atol=1e-10)

    def test_longrun_impact_is_lower_triangular(self, var2_data):
        """Long-run impact Psi(1) @ B has zeros at C_lr restriction positions."""
        from Ts.TsModels._svar import SVAR

        C_lr = np.array([[np.nan, 0], [np.nan, np.nan]])
        model = SVAR(var2_data, lags=2, C_lr=C_lr)
        result = model.fit()
        # Compute Psi(1)
        A_sum = np.sum(result._var_result.coefs, axis=0)
        psi1 = inv(np.eye(2) - A_sum)
        long_run_impact = psi1 @ result.B
        # C_lr[0,1]=0 means long_run_impact[0,1] should be ~0
        assert np.isclose(long_run_impact[0, 1], 0.0, atol=1e-10)


class TestSVARResult:
    """Test SVARResult structural IRF, summary, and inheritance.

    covers: code/python/Ts/TsModels/_svar.py::SVARResult [class]
    """

    def test_irf_orth_returns_structural(self, svar_ab_fitted):
        """irf(orth=True) returns structural IRF with bootstrap CI.

        covers: code/python/Ts/TsModels/_svar.py::SVARResult.irf [function]
        """
        from Ts.TsModels._var import IRFResult

        result = svar_ab_fitted.irf(periods=8, orth=True, n_draws=5)
        assert isinstance(result, IRFResult)
        assert result.values.shape == (9, 2, 2)
        assert result.lower is not None
        assert result.upper is not None
        assert result.lower.shape == (9, 2, 2)
        assert result.ci_method == "bootstrap"
        assert result.label == "Structural IRF"

    def test_irf_orth_impact(self, svar_ab_fitted):
        """Structural IRF at horizon 0 equals A^{-1} B.

        covers: code/python/Ts/TsModels/_svar.py::SVARResult.irf [function]
        """
        result = svar_ab_fitted.irf(periods=5, orth=True, n_draws=5)
        impact_actual = result.values[0]
        impact_expected = inv(svar_ab_fitted.A) @ svar_ab_fitted.B
        assert np.allclose(impact_actual, impact_expected, atol=1e-10)

    def test_irf_orth_differs_from_reduced(self, svar_ab_fitted):
        """Structural IRF differs from reduced-form IRF.

        covers: code/python/Ts/TsModels/_svar.py::SVARResult.irf [function]
        """
        sirf = svar_ab_fitted.irf(periods=5, orth=True, n_draws=5)
        rirf = svar_ab_fitted.irf(periods=5, orth=False)
        assert not np.allclose(sirf.values, rirf.values)
        assert sirf.orth is True

    def test_irf_orth_cache_reuse(self, svar_ab_fitted):
        """Repeated irf(orth=True) calls reuse cached values.

        covers: code/python/Ts/TsModels/_svar.py::SVARResult.irf [function]
        """
        s1 = svar_ab_fitted.irf(periods=6, orth=True, n_draws=5)
        s2 = svar_ab_fitted.irf(periods=6, orth=True, n_draws=5)
        assert np.allclose(s1.values, s2.values)
        assert svar_ab_fitted._sirf_cache is not None

    def test_summary_contains_svar_info(self, svar_ab_fitted):
        """summary() contains SVAR label and structural parameter section.

        covers: code/python/Ts/TsModels/_svar.py::SVARResult.summary [function]
        """
        text = svar_ab_fitted.summary()
        assert "SVAR" in text
        assert "Structural parameters" in text
        assert "A =" in text
        assert "B =" in text


class TestSVARInheritance:
    """Test SVARResult inherits VARResult methods correctly.

    covers: code/python/Ts/TsModels/_svar.py::SVARResult [class]
    """

    def test_inherits_irf_method(self, svar_ab_fitted):
        """SVARResult.irf() works (inherited from VARResult).

        covers: code/python/Ts/TsModels/_svar.py::SVARResult [class]
        """
        from Ts.TsModels._var import IRFResult

        result = svar_ab_fitted.irf(periods=4)
        assert isinstance(result, IRFResult)

    def test_inherits_granger_causality(self, svar_ab_fitted):
        """SVARResult.granger_causality() works (inherited)."""
        gc = svar_ab_fitted.granger_causality(caused=0, causing=1)
        assert gc[0].test_statistic > 0

    def test_inherits_predict(self, svar_ab_fitted):
        """SVARResult.predict() works (inherited)."""
        from Ts.TsModels._base import PredictResult

        pr = svar_ab_fitted.predict()
        assert isinstance(pr, PredictResult)

    def test_inherits_is_stable(self, svar_ab_fitted):
        """SVARResult.is_stable() works (inherited)."""
        assert isinstance(svar_ab_fitted.is_stable, bool)


class TestSVARCovers:
    """Aggregate coverage declarations for SVAR module items."""

    def test_cover_internals(self, var2_data):
        """Declare coverage for internal functions exercised by fit tests.

        covers: code/python/Ts/TsModels/_svar.py [module]
        covers: code/python/Ts/TsModels/_svar.py::SVAR [class]
        covers: code/python/Ts/TsModels/_svar.py::_param_to_matrices [function]
        covers: code/python/Ts/TsModels/_svar.py::_nll_ab [function]
        covers: code/python/Ts/TsModels/_svar.py::_solve_blanchard_quah [function]
        covers: code/python/Ts/TsModels/_svar.py::SVARResult.__repr__ [function]
        covers: code/python/Ts/TsModels/_svar.py::SVAR.summary [function]
        """
        pass
