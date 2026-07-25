"""Tests for Ts.TsTests._toda_yamamoto -- Toda-Yamamoto Granger causality test."""

import numpy as np
import pytest


@pytest.fixture
def bivariate_independent():
    """Two independent random walks -- no Granger causality."""
    rng = np.random.default_rng(42)
    n = 200
    return rng.standard_normal((n, 2)).cumsum(axis=0)


@pytest.fixture
def bivariate_causal():
    """VAR(1) with clear causal direction: y0 causes y1, y1 does not cause y0.

    y0_t = 0.5 * y0_{t-1} + e0_t
    y1_t = 0.7 * y1_{t-1} + 0.4 * y0_{t-1} + e1_t
    """
    rng = np.random.default_rng(123)
    n = 300
    y0 = np.zeros(n)
    y1 = np.zeros(n)
    for t in range(1, n):
        y0[t] = 0.5 * y0[t - 1] + rng.standard_normal()
        y1[t] = 0.7 * y1[t - 1] + 0.4 * y0[t - 1] + rng.standard_normal()
    return np.column_stack([y0, y1])


class TestTodaYamamotoInit:
    """Test TodaYamamotoTest constructor validation."""

    def test_valid_construction(self, bivariate_independent):
        """Valid data and parameters produce a TodaYamamotoTest instance."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        test = TodaYamamotoTest(bivariate_independent, p=2, d_max=1, trend="c")
        assert test.data.shape == bivariate_independent.shape
        assert test.p == 2
        assert test.d_max == 1
        assert test.trend == "c"

    def test_rejects_1d_data(self, bivariate_independent):
        """1-D data raises ValueError."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        with pytest.raises(ValueError):
            TodaYamamotoTest(bivariate_independent[:, 0], p=2, d_max=1)

    def test_rejects_single_variable(self, bivariate_independent):
        """k < 2 raises ValueError."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        with pytest.raises(ValueError):
            TodaYamamotoTest(bivariate_independent[:, :1], p=2, d_max=1)

    def test_rejects_invalid_trend(self, bivariate_independent):
        """Invalid trend raises ValueError."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        with pytest.raises(ValueError):
            TodaYamamotoTest(bivariate_independent, p=2, d_max=1, trend="xxx")

    def test_rejects_invalid_d_max(self, bivariate_independent):
        """d_max > 2 raises ValueError."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        with pytest.raises(ValueError):
            TodaYamamotoTest(bivariate_independent, p=2, d_max=3)

    def test_accepts_cols(self, bivariate_independent):
        """cols parameter sets display names."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        test = TodaYamamotoTest(bivariate_independent, p=2, d_max=1, cols=["x", "y"])
        assert test.cols == ["x", "y"]


class TestTodaYamamotoFit:
    """Test TodaYamamotoTest.fit() execution and result structure."""

    def test_fit_returns_result(self, bivariate_independent):
        """fit() returns TodaYamamotoTestResult and stores in result_."""
        from Ts.TsTests._toda_yamamoto import (
            TodaYamamotoTest,
            TodaYamamotoTestResult,
        )

        test = TodaYamamotoTest(bivariate_independent, p=2, d_max=1)
        result = test.fit()
        from Ts.TsTests._base import BaseMultiTestResult

        assert isinstance(result, TodaYamamotoTestResult)
        assert isinstance(result, BaseMultiTestResult)
        assert not hasattr(result, "statistic")
        assert not hasattr(result, "pvalue")
        assert test.result_ is result

    def test_result_has_tests(self, bivariate_independent):
        """Result contains test entries for all variable pairs."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        test = TodaYamamotoTest(bivariate_independent, p=2, d_max=1)
        result = test.fit()
        assert len(result.tests) > 0

    def test_result_has_metadata(self, bivariate_independent):
        """Result stores p, d_max, k, cols metadata."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        test = TodaYamamotoTest(bivariate_independent, p=2, d_max=1)
        result = test.fit()
        assert result.p == 2
        assert result.d_max == 1
        assert result.k == 2
        assert len(result.cols) == 2

    def test_each_test_has_statistic_and_pvalue(self, bivariate_independent):
        """Each test entry has chi2 statistic, pvalue, df, caused, causing."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        test = TodaYamamotoTest(bivariate_independent, p=2, d_max=1)
        result = test.fit()
        for entry in result.tests:
            assert hasattr(entry, "test_statistic")
            assert hasattr(entry, "p_value")
            assert hasattr(entry, "df")
            assert hasattr(entry, "caused")
            assert hasattr(entry, "causing")
            assert entry.test_statistic >= 0
            assert 0 <= entry.p_value <= 1


class TestTodaYamamotoKnownCausality:
    """Test with known causal structure: y0 -> y1, y1 -/-> y0."""

    def test_detects_causality_y0_to_y1(self, bivariate_causal):
        """Toda-Yamamoto test rejects H0 for y0 -> y1 at 5% level.

        y0 Granger-causes y1, so H0 (no causality) should be rejected.
        """
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        test = TodaYamamotoTest(bivariate_causal, p=1, d_max=0, cols=["y0", "y1"])
        result = test.fit()

        # Find the y0 causes y1 entry
        entry = _find_entry(result, caused="y1", causing=["y0"])
        assert entry is not None
        assert entry.p_value < 0.05

    def test_no_causality_y1_to_y0(self, bivariate_causal):
        """Toda-Yamamoto test fails to reject H0 for y1 -> y0 at 5% level.

        y1 does NOT Granger-cause y0, so H0 should not be rejected.
        """
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        test = TodaYamamotoTest(bivariate_causal, p=1, d_max=0, cols=["y0", "y1"])
        result = test.fit()

        entry = _find_entry(result, caused="y0", causing=["y1"])
        assert entry is not None
        assert entry.p_value > 0.05


class TestTodaYamamotoDmaxAuto:
    """Test automatic d_max detection."""

    def test_dmax_auto_returns_nonnegative(self, bivariate_independent):
        """Auto-detected d_max is 0, 1, or 2."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        test = TodaYamamotoTest(bivariate_independent, p=2, d_max=None)
        result = test.fit()
        assert result.d_max in (0, 1, 2)

    def test_dmax_auto_with_i1_data(self, bivariate_independent):
        """Random walk (I(1)) data should yield d_max >= 1."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        test = TodaYamamotoTest(bivariate_independent, p=2, d_max=None)
        result = test.fit()
        # Random walks are I(1), so d_max should be at least 1
        assert result.d_max >= 1


class TestTodaYamamotoSummary:
    """Test summary() formatting."""

    def test_summary_returns_string(self, bivariate_independent):
        """summary() returns a non-empty string."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        test = TodaYamamotoTest(bivariate_independent, p=2, d_max=1)
        result = test.fit()
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_contains_toda_yamamoto(self, bivariate_independent):
        """summary() mentions 'Toda-Yamamoto' in output."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        test = TodaYamamotoTest(bivariate_independent, p=2, d_max=1)
        result = test.fit()
        assert "Toda-Yamamoto" in result.summary()

    def test_summary_contains_chi2(self, bivariate_independent):
        """summary() includes chi2 label since TY uses chi-squared."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        test = TodaYamamotoTest(bivariate_independent, p=2, d_max=1)
        result = test.fit()
        assert "chi2" in result.summary().lower()


class TestTodaYamamotoEdgeCases:
    """Edge case and boundary tests."""

    def test_minimal_valid_data(self):
        """Minimum required observations (total_lags + 10) can fit."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        rng = np.random.default_rng(99)
        min_n = 1 + 1 + 10  # p=1, d_max=1, total_lags=2, +10 margin
        data = rng.standard_normal((min_n, 2))
        test = TodaYamamotoTest(data, p=1, d_max=1)
        result = test.fit()
        # total_lags=2 consumes 2 obs for lag creation, so nobs = min_n - 2
        assert result.nobs == min_n - 2

    def test_dmax_zero_ok(self, bivariate_independent):
        """d_max=0 (I(0) data) should work."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        # Use stationary data: AR(1) with |phi| < 1
        rng = np.random.default_rng(77)
        n = 200
        y0 = np.zeros(n)
        y1 = np.zeros(n)
        for t in range(1, n):
            y0[t] = 0.3 * y0[t - 1] + rng.standard_normal()
            y1[t] = -0.2 * y1[t - 1] + rng.standard_normal()
        data = np.column_stack([y0, y1])

        test = TodaYamamotoTest(data, p=2, d_max=0)
        result = test.fit()
        assert result.d_max == 0
        assert len(result.tests) > 0

    def test_multi_variable(self):
        """Works with k > 2 variables."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        rng = np.random.default_rng(55)
        n = 150
        k = 3
        data = rng.standard_normal((n, k))
        test = TodaYamamotoTest(data, p=1, d_max=0)
        result = test.fit()
        assert result.k == 3
        assert len(result.tests) > 0


class TestTodaYamamotoInternals:
    """Test internal helper functions and dunder methods."""

    def test__sig_star_levels(self, bivariate_independent):
        """_sig_star returns correct significance codes."""
        from Ts.TsTests._toda_yamamoto import _sig_star

        assert _sig_star(0.005) == "**"
        assert _sig_star(0.03) == "*"
        assert _sig_star(0.07) == "."
        assert _sig_star(0.50) == " "

    def test__TYEntry_fields(self, bivariate_independent):
        """_TYEntry dataclass stores test fields."""
        from Ts.TsTests._toda_yamamoto import _TYEntry

        e = _TYEntry(3.5, 0.04, 2, "y0", ["y1"])
        assert e.test_statistic == 3.5
        assert e.p_value == 0.04
        assert e.df == 2
        assert e.caused == "y0"
        assert e.causing == ["y1"]

    def test_result_str_equals_summary(self, bivariate_independent):
        """TodaYamamotoTestResult.__str__ returns same as summary()."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        test = TodaYamamotoTest(bivariate_independent, p=2, d_max=1)
        result = test.fit()
        assert str(result) == result.summary()

    def test_result_len_and_iter(self, bivariate_independent):
        """Result supports len(), iter(), and indexing."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest

        test = TodaYamamotoTest(bivariate_independent, p=2, d_max=1)
        result = test.fit()
        assert len(result) == len(result.tests)
        assert result[0] is result.tests[0]
        count = sum(1 for _ in result)
        assert count == len(result.tests)

    def test__compute_wald(self, bivariate_independent):
        """_compute_wald returns chi2 test result."""
        import numpy as np
        from Ts.TsTests._toda_yamamoto import _compute_wald

        n = 6
        R = np.eye(2, n)
        beta = np.zeros(n)
        cov = np.eye(n)
        wald, pv = _compute_wald(R, beta, cov, 2)
        assert wald == 0.0
        assert abs(pv - 1.0) < 1e-6

    def test__compute_wald_rejects_singular_covariance(self):
        """A singular restricted covariance is an explicit invalid result."""
        from Ts.TsTests._toda_yamamoto import _compute_wald

        restriction = np.eye(2)
        with pytest.raises(RuntimeError, match="singular"):
            _compute_wald(restriction, np.ones(2), np.zeros((2, 2)), 2)

    def test__build_restriction_matrix_shape(self, bivariate_independent):
        """_build_restriction_matrix produces correct shape."""
        from Ts.TsTests._toda_yamamoto import _build_restriction_matrix

        R = _build_restriction_matrix(
            n_regressors=5,
            k=2,
            eq_idx=0,
            causing_indices=[1],
            n_det=1,
            p_lags=2,
        )
        assert R.shape == (2, 10)  # p_lags=2, n_regressors*k=5*2=10

    def test__wald_test_single_and_multi(self, bivariate_independent):
        """_wald_test_single and _wald_test_multi return (float, float)."""
        import numpy as np
        from statsmodels.tsa.vector_ar.var_model import VAR as _SM_VAR
        from Ts.TsTests._toda_yamamoto import _wald_test_single, _wald_test_multi

        fitted = _SM_VAR(bivariate_independent).fit(maxlags=2, ic=None)
        all_params = np.asarray(fitted.params)
        cov_full = np.asarray(fitted.cov_params())

        w1, p1 = _wald_test_single(all_params, cov_full, 2, 0, 1, 1, 1)
        assert isinstance(w1, float)
        assert 0 <= p1 <= 1

        w2, p2 = _wald_test_multi(all_params, cov_full, 2, 0, [1], 1, 1)
        assert isinstance(w2, float)
        assert 0 <= p2 <= 1
        # Single and multi with same single causing index should match
        assert abs(w1 - w2) < 1e-10

    def test_module_import(self, bivariate_independent):
        """Module can be imported from Ts.TsTests."""
        from Ts.TsTests._toda_yamamoto import (
            TodaYamamotoTest,
            TodaYamamotoTestResult,
        )

        assert TodaYamamotoTest is not None
        assert TodaYamamotoTestResult is not None

    def test__toda_yamamoto_class_level(self, bivariate_independent):
        """TodaYamamotoTest class follows BaseTest contract."""
        from Ts.TsTests._toda_yamamoto import TodaYamamotoTest
        from Ts.TsTests._base import BaseTest

        test = TodaYamamotoTest(bivariate_independent, p=2, d_max=1)
        assert isinstance(test, BaseTest)
        assert test.summary() is not None


def _find_entry(result, caused, causing):
    """Helper: find a specific test entry by caused/causing names."""
    for entry in result.tests:
        if entry.caused == caused and entry.causing == causing:
            return entry
    return None
