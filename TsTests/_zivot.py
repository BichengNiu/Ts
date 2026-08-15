"""Zivot & Andrews (1992) unit root test with an unknown break date.

Reference
---------
Zivot, E. & Andrews, D. W. K. (1992). "Further Evidence on the Great Crash,
the Oil Price Shock, and the Unit-Root Hypothesis." *Journal of Business &
Economic Statistics*, 10(3), 251–270.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._base import BaseTest, BaseTestResult
from ._critical_values import _za_crit
from ._utils import _parse_input, _validate_model
from ._break_utils import (
    _extract_rho_stats,
    _extract_coefficients,
    _fit_unitroot_ols,
    _format_break_test_summary,
    _make_zivot_break_dummies,
    _select_lags_by_ic,
    _select_lags_by_tstat,
    _validate_lag_parameters,
    _validate_time_axis,
    _validate_trim,
)
from ._unitroot_plot import _render_tstat_plot, _render_ic_plot


@dataclass
class ZivotAndrewsTestResult(BaseTestResult):
    """Container for Zivot-Andrews (1992) test results.

    Parameters
    ----------
    statistic, pvalue, lags, nobs, residuals : see BaseTestResult
        Minimum unit-root statistic, optional p-value, selected lag count,
        effective sample size, and regression residuals.
    rho_hat, rho_se : float
        Autoregressive coefficient and standard error at the selected break.
    break_year : float
        Selected break label.
    break_index : int
        Zero-based selected break position.
    model : str
        Deterministic break specification.
    cv_01, cv_05, cv_10 : float
        One-, five-, and ten-percent critical values.
    all_t_stats, all_break_years : numpy.ndarray or None
        Search-path statistics and candidate break labels.
    lag_method : str
        Lag-selection method used.
    ic_by_lag : numpy.ndarray or None
        Information-criterion values when applicable.
    coefficients, pvalues : dict
        Selected regression estimates and p-values.
    fitted : numpy.ndarray or None
        Fitted values at the selected break.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import ZivotAndrewsTest
    >>> rng = np.random.default_rng(42)
    >>> data = np.r_[rng.normal(size=40), 2 + rng.normal(size=40)]
    >>> result = ZivotAndrewsTest(data, lags=0).fit()
    >>> bool(0 < result.break_index < len(data) - 1)
    True
    """

    rho_hat: float = 0.0  # estimated ρ at optimal break
    rho_se: float = 0.0  # std. error of ρ̂
    break_year: float = 0.0  # optimal break point (original units)
    break_index: int = 0  # 0-based index of optimal break
    model: str = ""  # "intercept", "slope", or "both"
    cv_01: float = 0.0  # critical value at 1%
    cv_05: float = 0.0  # critical value at 5%
    cv_10: float = 0.0  # critical value at 10%
    all_t_stats: np.ndarray | None = None  # t-statistics for all break points
    all_break_years: np.ndarray | None = None  # corresponding break years
    lag_method: str = "tstat"  # lag selection method used
    ic_by_lag: np.ndarray | None = None  # IC values for k = 0..max_lags
    coefficients: dict[str, float] = field(default_factory=dict)
    pvalues: dict[str, float] = field(default_factory=dict)
    fitted: np.ndarray | None = None

    def __str__(self) -> str:
        return _format_break_test_summary(
            "Zivot & Andrews (1992) Unit Root Test",
            "Optimal break point",
            "min t(ρ̂) = ρ̂ / s.e.",
            self.model,
            self.break_year,
            self.lags,
            self.nobs,
            self.rho_hat,
            self.rho_se,
            self.statistic,
            self.cv_01,
            self.cv_05,
            self.cv_10,
        )

    def plot_test(self, ax=None):
        """Plot diagnostic information based on the lag selection method.

        - ``"tstat"`` — sequence of t(ρ̂) across candidate break points,
          with optimal break point and critical value lines.
        - ``"aic"`` or ``"bic"`` — information criterion values across
          lag orders at the optimal break point.

        When the information-criterion sequence is unavailable (e.g. an
        explicit ``lags`` value was fitted), the t-statistic search path
        is shown instead.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure is created.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import ZivotAndrewsTest
        >>> data = np.cumsum(np.random.default_rng(42).normal(size=100))
        >>> result = ZivotAndrewsTest(data, max_lags=4).fit()
        >>> fig, ax = result.plot_test()
        """
        if self.lag_method in ("aic", "bic") and self.ic_by_lag is not None:
            return _render_ic_plot(self, ax)
        return _render_tstat_plot(self, ax)


class ZivotAndrewsTest(BaseTest):
    """Zivot & Andrews (1992) unit root test with an *unknown* structural
    break date.

    The break point is chosen endogenously by searching over all possible
    break dates (excluding a trimming region at the boundaries) and selecting
    the one that minimises the t-statistic on the autoregressive coefficient.

    Parameters
    ----------
    data : array-like
        The time series to test.
    time_index : array-like, optional
        Time index (e.g., years). If ``None``, a 0-based index is used.
    model : str, optional
        Model specification: ``"intercept"`` (intercept break), ``"slope"`` (trend break),
        or ``"both"`` (both). Default is ``"intercept"``.
    lags : int, optional
        Number of lagged differences (k). If ``None``, automatic selection
        is performed using the method specified by *lag_method*.
    max_lags : int, optional
        Maximum number of lags for automatic selection. Ignored
        if *lags* is provided. Default is 8.
    lag_crit : float, optional
        t-statistic threshold for lag selection (only used when
        *lag_method* is ``"tstat"``). Default is 1.60.
    lag_method : str, optional
        Lag selection method when *lags* is ``None``:
        ``"tstat"`` (default) — general-to-specific t-statistic method;
        ``"aic"`` — minimise AIC over k = 0..max_lags;
        ``"bic"`` — minimise BIC over k = 0..max_lags.
    trim : float, optional
        Trimming proportion at each end of the sample. Break points within
        the first or last ``trim * T`` observations are excluded. Default is
        0.15 (15%).

    y_col : str or int, optional
        Response column when ``data`` is a DataFrame.
    time_col : str or int, optional
        Time-label column when it is stored inside ``data``.

    Attributes
    ----------
    result_ : ZivotAndrewsTestResult
        Full test results after calling :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import ZivotAndrewsTest
    >>> rng = np.random.default_rng(42)
    >>> data = np.r_[rng.normal(size=40), 2 + rng.normal(size=40)]
    >>> result = ZivotAndrewsTest(
    ...     data, model="intercept", lags=0, trim=0.15
    ... ).fit()
    >>> bool(np.isfinite(result.statistic))
    True
    """

    def __init__(
        self,
        data,
        time_index=None,
        model: str = "intercept",
        lags: int | None = None,
        max_lags: int = 8,
        lag_crit: float = 1.60,
        lag_method: str = "tstat",
        trim: float = 0.15,
        y_col=None,
        time_col=None,
    ):
        self.data, self.time_index = _parse_input(data, time_index, y_col, time_col)
        _validate_time_axis(self.time_index)
        _validate_model(model)
        self.model = model
        self.lags, self.max_lags, self.lag_crit, self.lag_method = (
            _validate_lag_parameters(lags, max_lags, lag_crit, lag_method)
        )
        self.trim = _validate_trim(trim)
        self.result_: ZivotAndrewsTestResult | None = None

    def fit(self) -> ZivotAndrewsTestResult:
        """Run the Zivot-Andrews (1992) test.

        Returns
        -------
        ZivotAndrewsTestResult

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import ZivotAndrewsTest
        >>> data = np.cumsum(np.random.default_rng(42).normal(size=100))
        >>> result = ZivotAndrewsTest(data, max_lags=4).fit()
        >>> bool(0 < result.break_index < len(data) - 1)
        True
        """
        y = self.data
        T = len(y)
        if np.ptp(y) == 0.0:
            raise ValueError("Zivot-Andrews test requires non-constant data.")
        time_idx = self.time_index

        # Determine search range (exclude trimming region)
        start_idx = int(np.floor(self.trim * T))
        end_idx = int(np.ceil((1.0 - self.trim) * T)) - 1
        if end_idx <= start_idx:
            raise ValueError(
                f"Trimming {self.trim * 100:.0f}% leaves no candidate break "
                f"points (T={T}). Reduce trim."
            )

        candidate_indices = np.arange(start_idx, end_idx + 1)
        candidate_years = time_idx[candidate_indices]
        n_candidates = len(candidate_indices)

        t_stats = np.full(n_candidates, np.nan)
        best_t = np.inf
        best_idx = -1
        best_k = self.lags if self.lags is not None else 0
        best_ic_values = None
        best_res = None
        best_reg_cols = None

        for i, bp_idx in enumerate(candidate_indices):
            dummies = _make_zivot_break_dummies(T, bp_idx, self.model)

            # Lag selection for this break point
            if self.lags is None:
                if self.lag_method == "tstat":
                    k = _select_lags_by_tstat(
                        y, dummies, self.max_lags, self.lag_crit
                    )
                else:
                    k, ic_values = _select_lags_by_ic(
                        y, dummies, self.max_lags, self.lag_method
                    )
            else:
                k = self.lags

            # Build regression
            fitted = _fit_unitroot_ols(y, dummies, k)
            if fitted is None:
                continue
            res, reg_cols = fitted

            rho_hat, rho_se, t_stat = _extract_rho_stats(res, reg_cols)
            if (
                res.df_resid <= 0
                or not np.isfinite(rho_se)
                or rho_se <= 0
                or not np.isfinite(t_stat)
                or not np.all(np.isfinite(res.resid))
            ):
                continue

            t_stats[i] = t_stat

            if t_stat < best_t:
                best_t = t_stat
                best_idx = bp_idx
                best_k = k
                best_res = res
                best_reg_cols = reg_cols
                if self.lags is None and self.lag_method in ("aic", "bic"):
                    best_ic_values = ic_values

        if best_res is None:
            raise RuntimeError(
                "No valid regression could be estimated for any break point."
            )

        # Extract results at optimal break
        rho_hat, rho_se, _ = _extract_rho_stats(best_res, best_reg_cols)

        # Critical values (asymptotic)
        cv_01 = _za_crit(self.model, 0.01)
        cv_05 = _za_crit(self.model, 0.05)
        cv_10 = _za_crit(self.model, 0.10)

        # Collect coefficients
        coefs, pvals = _extract_coefficients(best_res, best_reg_cols)

        self.result_ = ZivotAndrewsTestResult(
            statistic=best_t,
            pvalue=None,
            lags=best_k,
            nobs=len(best_res.resid),
            residuals=best_res.resid,
            rho_hat=rho_hat,
            rho_se=rho_se,
            break_year=float(time_idx[best_idx]),
            break_index=best_idx,
            model=self.model,
            cv_01=cv_01,
            cv_05=cv_05,
            cv_10=cv_10,
            all_t_stats=t_stats,
            all_break_years=candidate_years,
            lag_method=self.lag_method,
            ic_by_lag=best_ic_values,
            coefficients=coefs,
            pvalues=pvals,
            fitted=best_res.fittedvalues,
        )
        return self.result_
