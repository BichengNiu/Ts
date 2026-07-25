"""Characterization tests for PerronTest (Mode B — baseline before refactoring).

These tests capture the current behavior of PerronTest. They must PASS now
(baseline) and continue to PASS after refactoring to use BaseTest/BaseTestResult.
"""

import numpy as np
import pytest
from Ts.TsTests import PerronTest, PerronTestResult


class TestPerronResultBaseline:
    """Characterize current PerronTestResult behavior."""

    def test_result_fields_exist(self):
        """Verify all expected fields exist on PerronTestResult."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.1 * t  # trend + noise

        test = PerronTest(y, break_year=50, time_index=t, model="intercept")
        result = test.fit()

        # Core fields
        assert hasattr(result, "statistic")
        assert hasattr(result, "rho_hat")
        assert hasattr(result, "rho_se")
        assert hasattr(result, "lags")
        assert hasattr(result, "break_year")
        assert hasattr(result, "break_index")
        assert hasattr(result, "model")
        assert hasattr(result, "nobs")
        assert hasattr(result, "cv_01")
        assert hasattr(result, "cv_05")
        assert hasattr(result, "cv_10")
        assert hasattr(result, "coefficients")
        assert hasattr(result, "pvalues")
        assert hasattr(result, "residuals")
        assert hasattr(result, "fitted")
        assert hasattr(result, "rsquared")
        assert hasattr(result, "rmse")

    def test_result_types(self):
        """Verify field types."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.1 * t
        test = PerronTest(y, break_year=50, time_index=t, model="intercept")
        result = test.fit()

        assert isinstance(result.statistic, float)
        assert isinstance(result.rho_hat, float)
        assert isinstance(result.rho_se, float)
        assert isinstance(result.lags, int)
        assert isinstance(result.break_year, (int, float))
        assert isinstance(result.break_index, (int, np.integer))
        assert isinstance(result.model, str)
        assert isinstance(result.nobs, int)
        assert isinstance(result.cv_01, float)
        assert isinstance(result.cv_05, float)
        assert isinstance(result.cv_10, float)
        assert isinstance(result.coefficients, dict)
        assert isinstance(result.pvalues, dict)
        assert isinstance(result.residuals, np.ndarray)
        assert isinstance(result.fitted, np.ndarray)
        assert isinstance(result.rsquared, float)
        assert isinstance(result.rmse, float)

    def test_str_contains_keywords(self):
        """str(result) must contain expected keywords."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.1 * t
        test = PerronTest(y, break_year=50, time_index=t, model="intercept")
        result = test.fit()
        s = str(result)

        assert "Perron (1989)" in s
        assert "Unit Root Test" in s
        assert "Break point" in s
        assert "lags" in s.lower() or "Number of lags" in s
        assert "rho" in s.lower() or "ρ" in s

    def test_three_models_work(self):
        """All three models use their intended, distinct break regressors."""
        from Ts.TsTests._break_utils import _make_break_dummies

        expected_columns = {
            "intercept": {"DL", "DP"},
            "slope": {"DT"},
            "both": {"DL", "DP", "DT"},
        }
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.1 * t

        statistics = {}
        for model in ["intercept", "slope", "both"]:
            dummies = _make_break_dummies(
                n,
                50,
                model,
                include_pulse=True,
            )
            assert set(dummies) == expected_columns[model]

            test = PerronTest(
                y,
                break_year=50,
                time_index=t,
                model=model,
                lags=1,
            )
            result = test.fit()
            assert result.model == model
            assert isinstance(result.statistic, float)
            statistics[model] = result.statistic

        assert statistics["slope"] != statistics["both"]


class TestPerronTestBaseline:
    """Characterize PerronTest class behavior."""

    def test_fit_returns_perron_result(self):
        """fit() returns PerronTestResult."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.1 * t
        test = PerronTest(y, break_year=50, time_index=t, model="intercept")
        result = test.fit()
        assert isinstance(result, PerronTestResult)

    def test_fit_stores_result(self):
        """fit() stores result_."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.1 * t
        test = PerronTest(y, break_year=50, time_index=t, model="intercept")
        test.fit()
        assert test.result_ is not None
        assert isinstance(test.result_, PerronTestResult)

    def test_summary_returns_string(self):
        """summary() returns a string."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.1 * t
        test = PerronTest(y, break_year=50, time_index=t, model="intercept")
        s = test.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert "Perron" in s

    def test_fixed_lags(self):
        """Using fixed lags works."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.1 * t
        test = PerronTest(y, break_year=50, time_index=t, model="intercept", lags=3)
        result = test.fit()
        assert result.lags == 3

    def test_aic_lag_selection(self):
        """AIC lag selection works."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.1 * t
        test = PerronTest(
            y,
            break_year=50,
            time_index=t,
            model="intercept",
            lag_method="aic",
            max_lags=4,
        )
        result = test.fit()
        assert isinstance(result.lags, int)
        assert 0 <= result.lags <= 4

    def test_bic_lag_selection(self):
        """BIC lag selection works."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.1 * t
        test = PerronTest(
            y,
            break_year=50,
            time_index=t,
            model="intercept",
            lag_method="bic",
            max_lags=4,
        )
        result = test.fit()
        assert isinstance(result.lags, int)
        assert 0 <= result.lags <= 4

    def test_invalid_model_raises(self):
        """Invalid model raises ValueError."""
        y = np.random.randn(100)
        with pytest.raises(ValueError):
            PerronTest(y, break_year=50, model="invalid")

    def test_invalid_lag_method_raises(self):
        """Invalid lag_method raises ValueError."""
        y = np.random.randn(100)
        with pytest.raises(ValueError):
            PerronTest(y, break_year=50, lag_method="invalid")

    def test_perron_constant_data_raises(self):
        """PerronTest with constant data should raise a meaningful error, not crash."""
        import numpy as np
        from Ts.TsTests import PerronTest

        data = np.ones(100)  # constant - singular design matrix
        pt = PerronTest(data, break_year=50, model="intercept")
        with pytest.raises(ValueError, match="requires non-constant data"):
            pt.fit()

    def test_perron_nonfinite_data_raises(self):
        """PerronTest rejects non-finite observations explicitly."""
        data = np.arange(100, dtype=float)
        data[25] = np.nan
        with pytest.raises(ValueError, match="only finite values"):
            PerronTest(data, break_year=50, model="intercept")

    def test_perron_does_not_swallow_unexpected_ols_errors(self, monkeypatch):
        """Programming errors from OLS escape instead of being reclassified."""

        def raise_unexpected(*args, **kwargs):
            raise RuntimeError("unexpected implementation error")

        monkeypatch.setattr("Ts.TsTests._perron.sm.OLS", raise_unexpected)
        data = np.arange(100, dtype=float) + np.sin(np.arange(100))
        test = PerronTest(data, break_year=50, lags=1)
        with pytest.raises(RuntimeError, match="unexpected implementation error"):
            test.fit()

    def test_lag_selection_does_not_swallow_unexpected_errors(self, monkeypatch):
        """Lag selection suppresses only expected numerical fit failures."""
        from Ts.TsTests._break_utils import _select_lags_by_tstat

        def raise_unexpected(*args, **kwargs):
            raise RuntimeError("unexpected selection error")

        monkeypatch.setattr("Ts.TsTests._break_utils.sm.OLS", raise_unexpected)
        data = np.arange(30, dtype=float) + np.sin(np.arange(30))
        with pytest.raises(RuntimeError, match="unexpected selection error"):
            _select_lags_by_tstat(data, {}, 1, np.arange(30), 1.6)
