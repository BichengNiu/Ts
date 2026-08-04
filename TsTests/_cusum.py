"""OLS-residual CUSUM test for unknown regression instability."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import breaks_cusumolsresid

from ._base import BaseTest, BaseTestResult
from ._regression_break_utils import (
    RegressionBreakDesign,
    _prepare_regression_break_design,
)


@dataclass
class CUSUMTestResult(BaseTestResult):
    """Result of the full-sample OLS-residual CUSUM stability test.

    Parameters
    ----------
    statistic, pvalue, lags, nobs, residuals : see BaseTestResult
        Supremum statistic, p-value, unused lag field, sample size, and OLS
        residuals.
    critical_values : dict
        Reference critical values keyed by significance level.
    cusum : numpy.ndarray or None
        Standardized cumulative residual path.
    time_index : numpy.ndarray or None
        Labels aligned with ``cusum``.
    coefficients : dict
        Full-sample OLS coefficients.
    fitted : numpy.ndarray or None
        Full-sample fitted values.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import CUSUMTest
    >>> result = CUSUMTest(np.random.default_rng(42).normal(size=80)).fit()
    >>> result.cusum.shape
    (80,)
    """

    critical_values: dict[str, float] = field(default_factory=dict)
    cusum: np.ndarray | None = None
    time_index: np.ndarray | None = None
    coefficients: dict[str, float] = field(default_factory=dict)
    fitted: np.ndarray | None = None

    def __str__(self) -> str:
        header = self._format_conclusion(
            "OLS Residual CUSUM Test",
            "Regression parameters are stable",
        )
        conclusion = (
            "Reject H0: regression instability detected."
            if self.pvalue is not None and self.pvalue < 0.05
            else "Cannot reject H0: no regression instability detected."
        )
        return f"{header}\n  Conclusion (5%): {conclusion}"

    def plot_test(self, alpha: float = 0.05, ax=None):
        """Plot the normalized cumulative residual process and critical limit.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for the displayed critical limit.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created when omitted.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import CUSUMTest
        >>> result = CUSUMTest(np.random.default_rng(42).normal(size=80)).fit()
        >>> fig, ax = result.plot_test()
        """
        level = f"{int(alpha * 100)}%"
        if level not in self.critical_values:
            raise ValueError("alpha must be one of 0.01, 0.05, or 0.10")
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure
        limit = self.critical_values[level]
        ax.plot(self.time_index, self.cusum, label="Scaled cumulative residuals")
        ax.axhline(limit, color="red", linestyle="--", label=f"{level} limits")
        ax.axhline(-limit, color="red", linestyle="--")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("OLS Residual CUSUM Stability Test")
        ax.set_xlabel("Time")
        ax.set_ylabel("CUSUM")
        ax.legend()
        return fig, ax


class CUSUMTest(BaseTest):
    """Test regression stability using cumulative full-sample OLS residuals.

    Parameters
    ----------
    data : array-like or pandas.DataFrame
        Response observations, or a table containing selected columns.
    exog : array-like, optional
        External regressors supplied separately from ``data``.
    time_index : array-like, optional
        Ordered labels aligned with observations.
    trend : {"n", "c", "ct"}, default "c"
        Deterministic regressors in the full-sample OLS model.
    y_col : str or int, optional
        Response column in a DataFrame.
    time_col : str or int, optional
        Time-label column in a DataFrame.
    exog_cols : sequence of str or int, optional
        Regressor columns selected from a DataFrame.

    Attributes
    ----------
    result_ : CUSUMTestResult or None
        Fitted stability-test result.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import CUSUMTest
    >>> rng = np.random.default_rng(42)
    >>> result = CUSUMTest(rng.normal(size=100), trend="c").fit()
    >>> 0.0 <= result.pvalue <= 1.0
    True
    """

    def __init__(
        self,
        data,
        *,
        exog=None,
        time_index=None,
        trend: str = "c",
        y_col: str | int | None = None,
        time_col: str | int | None = None,
        exog_cols: list[str | int] | tuple[str | int, ...] | None = None,
    ):
        self.design: RegressionBreakDesign = _prepare_regression_break_design(
            data,
            exog=exog,
            time_index=time_index,
            trend=trend,
            y_col=y_col,
            time_col=time_col,
            exog_cols=exog_cols,
        )
        self.result_: CUSUMTestResult | None = None

    def fit(self) -> CUSUMTestResult:
        """Fit OLS and run statsmodels' published CUSUM diagnostic.

        Returns
        -------
        CUSUMTestResult

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import CUSUMTest
        >>> result = CUSUMTest(np.random.default_rng(42).normal(size=80)).fit()
        >>> 0.0 <= result.pvalue <= 1.0
        True
        """
        y = self.design.endog
        x = self.design.exog
        nobs, nparams = x.shape
        result = sm.OLS(y, x).fit()
        residuals = np.asarray(result.resid, dtype=float)
        ssr = float(np.sum(residuals**2))
        variance_floor = np.finfo(float).eps * max(1.0, float(np.sum(y**2)))
        scale = np.sqrt(ssr * nobs / (nobs - nparams))
        if not np.isfinite(scale) or ssr <= variance_floor:
            raise ValueError("CUSUM test requires a positive residual variance")
        statistic, pvalue, critical = breaks_cusumolsresid(
            residuals,
            ddof=nparams,
        )
        cusum = np.cumsum(residuals) / scale
        critical_values = {f"{level}%": float(value) for level, value in critical}
        coefficients = {
            name: float(result.params[index])
            for index, name in enumerate(self.design.column_names)
        }
        self.result_ = CUSUMTestResult(
            statistic=float(statistic),
            pvalue=float(pvalue),
            lags=None,
            nobs=nobs,
            residuals=residuals,
            critical_values=critical_values,
            cusum=np.asarray(cusum),
            time_index=self.design.time_index.copy(),
            coefficients=coefficients,
            fitted=np.asarray(result.fittedvalues),
        )
        return self.result_
