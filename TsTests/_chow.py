"""Classical Chow test for a pre-specified regression breakpoint."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.stats import f

from ._base import BaseTest, BaseTestResult
from ._break_utils import _locate_known_break
from ._regression_break_utils import (
    RegressionBreakDesign,
    _prepare_regression_break_design,
)


@dataclass
class ChowTestResult(BaseTestResult):
    """Result of a classical known-break Chow F test."""

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
        """Plot observed values and pooled/split fitted regressions."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure
        ax.plot(self.time_index, self.observed, label="Observed", color="black")
        ax.plot(
            self.time_index,
            self.fitted_pooled,
            label="Pooled fit",
            linestyle="--",
        )
        ax.plot(self.time_index, self.fitted_split, label="Split fit")
        ax.axvline(self.break_year, color="red", linestyle=":", label="Known break")
        ax.set_title("Chow Test: Known Regression Break")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        return fig, ax


class ChowTest(BaseTest):
    """Test a regression coefficient break at a date specified a priori.

    The classical F reference distribution assumes independent,
    homoskedastic errors. The first regime includes the matched break
    observation; the second begins at the following observation.
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
        """Estimate pooled and split regressions and compute the Chow F test."""
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
            return {
                name: float(result.params[index])
                for index, name in enumerate(self.design.column_names)
            }

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
