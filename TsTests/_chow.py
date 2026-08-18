"""Classical Chow test for a pre-specified regression breakpoint."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import statsmodels.api as sm
from scipy.stats import f

from ._base import BaseTest, BaseTestResult
from ._break_utils import _locate_known_break
from ._regression_break_utils import (
    RegressionBreakDesign,
    _coefficient_dict,
    _prepare_regression_break_design,
)


@dataclass
class ChowTestResult(BaseTestResult):
    """Result of a classical known-break Chow F test.

    Parameters
    ----------
    statistic, pvalue, lags, nobs, residuals : see BaseTestResult
        Chow F statistic, p-value, unused lag field, sample size, and pooled
        residuals.
    break_year : float
        Matched known-break label.
    break_index : int
        Zero-based break position, included in the first regime.
    df_num, df_denom : int
        Numerator and denominator degrees of freedom.
    rss_pooled, rss_split : float
        Residual sums of squares for pooled and split regressions.
    coefficients_pooled, coefficients_before, coefficients_after : dict
        Coefficients from the pooled and regime-specific regressions.
    fitted_pooled, fitted_split : numpy.ndarray or None
        Pooled and split fitted values.
    time_index, observed : numpy.ndarray or None
        Original time labels and response observations.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import ChowTest
    >>> rng = np.random.default_rng(42)
    >>> data = np.r_[rng.normal(size=40), 3 + rng.normal(size=40)]
    >>> result = ChowTest(data, break_year=39).fit()
    >>> result.break_index
    39
    """

    break_year: float = 0.0
    break_index: int = 0
    df_num: int = 0
    df_denom: int = 0
    rss_pooled: float = 0.0
    rss_split: float = 0.0
    coefficients_pooled: dict[str, float] = field(default_factory=dict)
    coefficients_before: dict[str, float] = field(default_factory=dict)
    coefficients_after: dict[str, float] = field(default_factory=dict)
    fitted_pooled: np.ndarray | None = None
    fitted_split: np.ndarray | None = None
    time_index: np.ndarray | None = None
    observed: np.ndarray | None = None

    def __str__(self) -> str:
        header = self._format_conclusion(
            "Chow Test",
            "All selected regression coefficients are stable at the known break",
        )
        conclusion = (
            "Reject H0: coefficients differ across regimes."
            if self.pvalue is not None and self.pvalue < 0.05
            else "Cannot reject H0: no coefficient break detected."
        )
        return (
            f"{header}\n"
            f"  Break point:     {self.break_year}\n"
            f"  Degrees freedom: ({self.df_num}, {self.df_denom})\n"
            f"  Conclusion (5%): {conclusion}"
        )

    def plot_test(self, ax=None):
        """Plot observed values and pooled/split fitted regressions.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created when omitted.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import ChowTest
        >>> years = np.arange(2000, 2080, dtype=float)
        >>> data = np.r_[np.zeros(40), np.ones(40)]
        >>> result = ChowTest(data, 2040, time_index=years).fit()
        >>> fig, ax = result.plot_test()
        """
        from Ts.TsPlots.style import (
            BLACK,
            DARK_BLUE,
            DARK_RED,
            GRAY,
            _FigureContext,
        )

        context = _FigureContext(ax=ax)
        ax = context.ax
        ax.plot(
            self.time_index,
            self.observed,
            label="Observed",
            color=BLACK,
        )
        ax.plot(
            self.time_index,
            self.fitted_pooled,
            label="Pooled fit",
            linestyle="--",
            color=GRAY,
        )
        ax.plot(
            self.time_index,
            self.fitted_split,
            label="Split fit",
            color=DARK_BLUE,
        )
        ax.axvline(
            self.break_year,
            color=DARK_RED,
            linestyle=":",
            label="Known break",
        )
        context.finalize(
            title="Chow Test: Known Regression Break",
            xtitle="Time",
            ytitle="Value",
        )
        return context.fig, context.ax


class ChowTest(BaseTest):
    """Test a regression coefficient break at a date specified a priori.

    The classical F reference distribution assumes independent,
    homoskedastic errors. The first regime includes the matched break
    observation; the second begins at the following observation.

    Parameters
    ----------
    data : array-like or pandas.DataFrame
        Response observations, or a table containing selected columns.
    break_year : float
        Known break label; the matched observation ends the first regime.
    exog : array-like, optional
        External regressors supplied separately from ``data``.
    time_index : array-like, optional
        Ordered labels used to locate ``break_year``.
    trend : {"n", "c", "ct"}, default "c"
        Deterministic regressors included in every regime.
    y_col : str or int, optional
        Response column in a DataFrame.
    time_col : str or int, optional
        Time-label column in a DataFrame.
    exog_cols : sequence of str or int, optional
        Regressor columns selected from a DataFrame.

    Attributes
    ----------
    result_ : ChowTestResult or None
        Fitted known-break test.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import ChowTest
    >>> rng = np.random.default_rng(42)
    >>> data = np.r_[rng.normal(size=40), 3 + rng.normal(size=40)]
    >>> result = ChowTest(data, break_year=39, trend="c").fit()
    >>> result.pvalue < 0.05
    True
    """

    def __init__(
        self,
        data,
        break_year: float,
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
        self.break_index = _locate_known_break(self.design.time_index, break_year)
        self.break_year = float(self.design.time_index[self.break_index])
        self.result_: ChowTestResult | None = None

    def fit(self) -> ChowTestResult:
        """Estimate pooled and split regressions and compute the Chow F test.

        Returns
        -------
        ChowTestResult

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import ChowTest
        >>> years = np.arange(2000, 2080, dtype=float)
        >>> data = np.r_[np.zeros(40), np.ones(40)]
        >>> result = ChowTest(data, 2040, time_index=years).fit()
        >>> result.break_year
        2040.0
        """
        y = self.design.endog
        x = self.design.exog
        nobs, nparams = x.shape
        split = self.break_index + 1
        if split <= nparams or nobs - split <= nparams:
            raise ValueError(
                "break leaves insufficient residual degrees of freedom in "
                "at least one regime"
            )
        if (
            np.linalg.matrix_rank(x[:split]) < nparams
            or np.linalg.matrix_rank(x[split:]) < nparams
        ):
            raise ValueError("break produces a rank-deficient regime design")

        pooled = sm.OLS(y, x).fit()
        before = sm.OLS(y[:split], x[:split]).fit()
        after = sm.OLS(y[split:], x[split:]).fit()
        rss_pooled = float(pooled.ssr)
        rss_split = float(before.ssr + after.ssr)
        variance_floor = np.finfo(float).eps * max(1.0, float(np.sum(y**2))) * 100
        if not np.isfinite(rss_split) or rss_split <= variance_floor:
            raise ValueError(
                "Chow test requires positive residual variance in the split regression"
            )
        df_num = nparams
        df_denom = nobs - 2 * nparams
        numerator = max(rss_pooled - rss_split, 0.0) / df_num
        denominator = rss_split / df_denom
        statistic = float(numerator / denominator)
        pvalue = float(f.sf(statistic, df_num, df_denom))
        if not np.isfinite(statistic) or not np.isfinite(pvalue):
            raise FloatingPointError("invalid Chow test statistic")

        def coefficients(result) -> dict[str, float]:
            return _coefficient_dict(result.params, self.design.column_names)

        fitted_split = np.concatenate([before.fittedvalues, after.fittedvalues])
        residuals_split = np.concatenate([before.resid, after.resid])
        self.result_ = ChowTestResult(
            statistic=statistic,
            pvalue=pvalue,
            lags=None,
            nobs=nobs,
            residuals=residuals_split,
            break_year=self.break_year,
            break_index=self.break_index,
            df_num=df_num,
            df_denom=df_denom,
            rss_pooled=rss_pooled,
            rss_split=rss_split,
            coefficients_pooled=coefficients(pooled),
            coefficients_before=coefficients(before),
            coefficients_after=coefficients(after),
            fitted_pooled=np.asarray(pooled.fittedvalues),
            fitted_split=np.asarray(fitted_split),
            time_index=self.design.time_index.copy(),
            observed=y.copy(),
        )
        return self.result_
