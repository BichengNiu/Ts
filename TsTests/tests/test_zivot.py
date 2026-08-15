"""Characterization tests for ZivotAndrewsTest (Mode B — baseline before refactoring).

These tests capture the current behavior of ZivotAndrewsTest. They must PASS now
(baseline) and continue to PASS after refactoring to use BaseTest/BaseTestResult.
"""

import numpy as np
import pytest
from Ts.TsTests import ZivotAndrewsTest, ZivotAndrewsTestResult


class TestZivotResultBaseline:
    """Characterize current ZivotAndrewsTestResult behavior."""

    def test_result_fields_exist(self):
        """Verify all expected fields exist on ZivotAndrewsTestResult."""
        np.random.seed(42)
        n = 150
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.05 * t

        test = ZivotAndrewsTest(y, time_index=t, model="intercept", trim=0.15)
        result = test.fit()

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
        assert hasattr(result, "all_t_stats")
        assert hasattr(result, "all_break_years")
        assert hasattr(result, "lag_method")
        assert hasattr(result, "coefficients")
        assert hasattr(result, "pvalues")
        assert hasattr(result, "residuals")
        assert hasattr(result, "fitted")

    def test_result_types(self):
        """Verify field types."""
        np.random.seed(42)
        n = 150
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.05 * t
        test = ZivotAndrewsTest(y, time_index=t, model="intercept", trim=0.15)
        result = test.fit()

        assert isinstance(result.statistic, float)
        assert isinstance(result.rho_hat, float)
        assert isinstance(result.rho_se, float)
        assert isinstance(result.lags, int)
        assert isinstance(result.break_year, (int, float, np.floating))
        assert isinstance(result.break_index, (int, np.integer))
        assert isinstance(result.model, str)
        assert isinstance(result.nobs, int)
        assert isinstance(result.cv_01, float)
        assert isinstance(result.cv_05, float)
        assert isinstance(result.cv_10, float)
        assert isinstance(result.all_t_stats, np.ndarray)
        assert isinstance(result.all_break_years, np.ndarray)

    def test_str_contains_keywords(self):
        """str(result) must contain expected keywords."""
        np.random.seed(42)
        n = 150
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.05 * t
        test = ZivotAndrewsTest(y, time_index=t, model="intercept", trim=0.15)
        result = test.fit()
        s = str(result)

        assert "Zivot & Andrews" in s
        assert "Unit Root Test" in s
        assert "Optimal break point" in s
        assert "lags" in s.lower() or "Number of lags" in s

    def test_three_models_work(self):
        """All three models use their intended, distinct break regressors."""
        from Ts.TsTests._break_utils import _make_zivot_break_dummies

        expected_columns = {
            "intercept": {"DL"},
            "slope": {"DT"},
            "both": {"DL", "DT"},
        }
        np.random.seed(42)
        n = 150
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.05 * t

        statistics = {}
        for model in ["intercept", "slope", "both"]:
            assert (
                set(_make_zivot_break_dummies(n, 50, model)) == expected_columns[model]
            )

            test = ZivotAndrewsTest(
                y,
                time_index=t,
                model=model,
                lags=1,
                trim=0.15,
            )
            result = test.fit()
            assert result.model == model
            assert isinstance(result.statistic, float)
            statistics[model] = result.statistic

        assert statistics["slope"] != statistics["both"]


class TestZivotAndrewsTestBaseline:
    """Characterize ZivotAndrewsTest class behavior."""

    def test_fit_returns_zivot_result(self):
        """fit() returns ZivotAndrewsTestResult."""
        np.random.seed(42)
        n = 150
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.05 * t
        test = ZivotAndrewsTest(y, time_index=t, model="intercept", trim=0.15)
        result = test.fit()
        assert isinstance(result, ZivotAndrewsTestResult)

    def test_fit_stores_result(self):
        """fit() stores result_."""
        np.random.seed(42)
        n = 150
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.05 * t
        test = ZivotAndrewsTest(y, time_index=t, model="intercept", trim=0.15)
        test.fit()
        assert test.result_ is not None
        assert isinstance(test.result_, ZivotAndrewsTestResult)

    def test_summary_returns_string(self):
        """summary() returns a string."""
        np.random.seed(42)
        n = 150
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.05 * t
        test = ZivotAndrewsTest(y, time_index=t, model="intercept", trim=0.15)
        s = test.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert "Zivot" in s

    def test_fixed_lags(self):
        """Using fixed lags works."""
        np.random.seed(42)
        n = 150
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.05 * t
        test = ZivotAndrewsTest(y, time_index=t, model="intercept", lags=3, trim=0.15)
        result = test.fit()
        assert result.lags == 3

    def test_aic_lag_selection(self):
        """AIC lag selection works."""
        np.random.seed(42)
        n = 150
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.05 * t
        test = ZivotAndrewsTest(
            y, time_index=t, model="intercept", lag_method="aic", max_lags=4, trim=0.15
        )
        result = test.fit()
        assert isinstance(result.lags, int)
        assert 0 <= result.lags <= 4

    def test_bic_lag_selection(self):
        """BIC lag selection works."""
        np.random.seed(42)
        n = 150
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.05 * t
        test = ZivotAndrewsTest(
            y, time_index=t, model="intercept", lag_method="bic", max_lags=4, trim=0.15
        )
        result = test.fit()
        assert isinstance(result.lags, int)
        assert 0 <= result.lags <= 4

    def test_invalid_model_raises(self):
        """Invalid model raises ValueError."""
        y = np.random.randn(100)
        with pytest.raises(ValueError):
            ZivotAndrewsTest(y, model="invalid")

    def test_invalid_lag_method_raises(self):
        """Invalid lag_method raises ValueError."""
        y = np.random.randn(100)
        with pytest.raises(ValueError):
            ZivotAndrewsTest(y, lag_method="invalid")

    def test_plot_test_returns_fig_ax(self):
        """plot_test() returns (fig, ax) tuple."""
        np.random.seed(42)
        n = 150
        t = np.arange(n)
        y = np.cumsum(np.random.randn(n)) + 0.05 * t
        test = ZivotAndrewsTest(y, time_index=t, model="intercept", trim=0.15)
        result = test.fit()
        fig, ax = result.plot_test()
        # Should return a matplotlib figure and axes
        assert fig is not None
        assert ax is not None

    def test_does_not_swallow_unexpected_ols_errors(self, monkeypatch):
        """Programming errors from candidate regressions escape immediately."""

        def raise_unexpected(*args, **kwargs):
            raise RuntimeError("unexpected implementation error")

        monkeypatch.setattr("Ts.TsTests._zivot.sm.OLS", raise_unexpected)
        data = np.arange(60, dtype=float) + np.sin(np.arange(60))
        test = ZivotAndrewsTest(data, lags=1)
        with pytest.raises(RuntimeError, match="unexpected implementation error"):
            test.fit()
