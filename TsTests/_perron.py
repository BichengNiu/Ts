"""Perron (1989) unit root test with a known structural break date.

Reference
---------
Perron, P. (1989). "The Great Crash, the Oil Price Shock, and the Unit Root
Hypothesis." *Econometrica*, 57(6), 1361–1401.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import statsmodels.api as sm

from ._base import BaseTest, BaseTestResult
from ._critical_values import _perron_crit
from ._utils import _parse_input, _validate_model
from ._break_utils import (
    _build_regression_data,
    _extract_rho_stats,
    _extract_coefficients,
    _get_regression_columns,
    _locate_known_break,
    _make_perron_break_dummies,
    _select_lags_by_ic,
    _select_lags_by_tstat,
    _validate_nonnegative_int,
    _validate_time_axis,
)


@dataclass
class PerronTestResult(BaseTestResult):
    """Container for Perron (1989) test results.

    Parameters
    ----------
    statistic, pvalue, lags, nobs, residuals : see BaseTestResult
        Unit-root statistic, optional p-value, selected lag count, effective
        sample size, and regression residuals.
    rho_hat, rho_se : float
        Estimated autoregressive coefficient and standard error.
    break_year : float
        Matched known-break label.
    break_index : int
        Zero-based position of the matched break observation.
    break_fraction : float
        Break position divided by sample length.
    model : str
        Deterministic break specification.
    cv_01, cv_05, cv_10 : float
        One-, five-, and ten-percent critical values.
    coefficients, pvalues : dict
        Regression estimates and p-values by term.
    fitted : numpy.ndarray or None
        Fitted test-regression values.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import PerronTest
    >>> rng = np.random.default_rng(42)
    >>> data = np.r_[rng.normal(size=40), 2 + rng.normal(size=40)]
    >>> result = PerronTest(data, break_year=39, lags=0).fit()
    >>> result.break_index
    39
    """

    rho_hat: float = 0.0  # estimated ρ
    rho_se: float = 0.0  # std. error of ρ̂
    break_year: float = 0.0  # break point (in original units)
    break_index: int = 0  # 0-based index of break
    break_fraction: float = 0.0  # break_index / total observations
    model: str = ""  # "intercept", "slope", or "both"
    cv_01: float = 0.0  # critical value at 1%
    cv_05: float = 0.0  # critical value at 5%
    cv_10: float = 0.0  # critical value at 10%
    coefficients: dict[str, float] = field(default_factory=dict)
    pvalues: dict[str, float] = field(default_factory=dict)
    fitted: np.ndarray | None = None

    def __str__(self) -> str:
        return (
            f"Perron (1989) Unit Root Test — Model {self.model}\n"
            f"{'=' * 50}\n"
            f"Break point          : {self.break_year}\n"
            f"Number of lags (k)   : {self.lags}\n"
            f"Effective obs. (T)   : {self.nobs}\n"
            f"\n"
            f"ρ̂ (coeff on y_t-1)   : {self.rho_hat:.4f}\n"
            f"s.e.(ρ̂)              : {self.rho_se:.4f}\n"
            f"t(ρ̂) = ρ̂ / s.e.    : {self.statistic:.3f}\n"
            f"\n"
            f"Critical values:\n"
            f"  1%                 : {self.cv_01:.3f}\n"
            f"  5%                 : {self.cv_05:.3f}\n"
            f"  10%                : {self.cv_10:.3f}\n"
            f"\n"
            f"Conclusion (5%): "
            f"{'Reject H0 (unit root); evidence favors stationarity around a breaking trend' if self.statistic < self.cv_05 else 'Cannot reject H0 (unit root)'}\n"
        )

    @property
    def critical_values(self) -> dict[str, float]:
        """Critical values as a dict, for unified plotting."""
        return {
            "1%": self.cv_01,
            "5%": self.cv_05,
            "10%": self.cv_10,
        }

    def plot_test(self, ax=None):
        """Plot the test statistic against critical values.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Creates a new figure if ``None``.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import PerronTest
        >>> years = np.arange(2000, 2080, dtype=float)
        >>> data = np.cumsum(np.random.default_rng(42).normal(size=80))
        >>> result = PerronTest(data, break_year=2040, time_index=years).fit()
        >>> fig, ax = result.plot_test()
        """
        from ._unitroot_plot import _render_critical_value_plot

        return _render_critical_value_plot(self, "Perron (1989)", ax)


class PerronTest(BaseTest):
    """Perron (1989) unit root test with a *known* structural break date.

    Parameters
    ----------
    data : array-like
        The time series to test.
    break_year : float
        The known break date (in the units of *time_index*). The break is
        assumed to occur *after* this observation.
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

    y_col : str or int, optional
        Response column when ``data`` is a DataFrame.
    time_col : str or int, optional
        Time-label column when it is stored inside ``data``.

    Attributes
    ----------
    result_ : PerronTestResult
        Full test results after calling :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import PerronTest
    >>> rng = np.random.default_rng(42)
    >>> data = np.r_[rng.normal(size=40), 2 + rng.normal(size=40)]
    >>> test = PerronTest(data, break_year=39, model="intercept", lags=0)
    >>> result = test.fit()
    >>> result.break_year
    39.0
    """

    def __init__(
        self,
        data,
        break_year: float,
        time_index=None,
        model: str = "intercept",
        lags: int | None = None,
        max_lags: int = 8,
        lag_crit: float = 1.60,
        lag_method: str = "tstat",
        y_col=None,
        time_col=None,
    ):
        self.data, self.time_index = _parse_input(data, time_index, y_col, time_col)
        _validate_time_axis(self.time_index)
        self.break_index = _locate_known_break(self.time_index, break_year)
        self.break_year = float(self.time_index[self.break_index])
        break_fraction = self.break_index / len(self.data)
        if not 0.1 <= break_fraction <= 0.9:
            raise ValueError(
                "break_year must locate a break fraction between 0.1 and 0.9"
            )
        self.break_fraction = float(break_fraction)
        _validate_model(model)
        self.model = model
        self.lags = (
            None if lags is None else _validate_nonnegative_int(lags, name="lags")
        )
        self.max_lags = _validate_nonnegative_int(max_lags, name="max_lags")
        if isinstance(lag_crit, bool) or not np.isscalar(lag_crit):
            raise TypeError("lag_crit must be a positive finite scalar")
        self.lag_crit = float(lag_crit)
        if not np.isfinite(self.lag_crit) or self.lag_crit <= 0:
            raise ValueError("lag_crit must be a positive finite scalar")
        if lag_method not in ("tstat", "aic", "bic"):
            raise ValueError(
                f"lag_method must be 'tstat', 'aic', or 'bic', got {lag_method!r}"
            )
        self.lag_method = lag_method
        self.result_: PerronTestResult | None = None

    def fit(self) -> PerronTestResult:
        """Run the Perron (1989) test.

        Returns
        -------
        PerronTestResult

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import PerronTest
        >>> years = np.arange(2000, 2080, dtype=float)
        >>> data = np.cumsum(np.random.default_rng(42).normal(size=80))
        >>> result = PerronTest(data, break_year=2040, time_index=years).fit()
        >>> result.break_year
        2040.0
        """
        y = self.data
        T = len(y)
        if not np.all(np.isfinite(y)):
            raise ValueError("Perron test data must contain only finite values.")
        if np.ptp(y) == 0.0:
            raise ValueError("Perron test requires non-constant data.")

        break_idx = self.break_index

        # Create break dummies
        dummies = _make_perron_break_dummies(T, break_idx, self.model)

        # Lag selection
        if self.lags is None:
            if self.lag_method == "tstat":
                k = _select_lags_by_tstat(y, dummies, self.max_lags, self.lag_crit)
            else:
                k, _ = _select_lags_by_ic(y, dummies, self.max_lags, self.lag_method)
        else:
            k = self.lags

        # Build regression data
        df = _build_regression_data(y, dummies, k)

        # Define regressors
        reg_cols = _get_regression_columns(dummies, k)

        X = df[reg_cols].values
        y_dep = df["dy"].values
        if X.shape[0] <= X.shape[1]:
            raise ValueError(
                "Perron test has insufficient residual degrees of freedom "
                f"({X.shape[0]} observations, {X.shape[1]} regressors)."
            )
        if np.linalg.matrix_rank(X) < X.shape[1]:
            raise ValueError("Perron test regression design matrix is rank deficient.")

        # OLS estimation
        try:
            res = sm.OLS(y_dep, X).fit()
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as e:
            raise RuntimeError(
                f"Perron test OLS estimation failed. "
                f"This may be caused by singular design matrix "
                f"(e.g., constant or near-constant data). "
                f"Original error: {e}"
            ) from e

        # Extract key statistics
        rho_hat, rho_se, t_stat = _extract_rho_stats(res, reg_cols)
        if (
            res.df_resid <= 0
            or not np.isfinite(rho_se)
            or rho_se <= 0
            or not np.isfinite(t_stat)
            or not np.all(np.isfinite(res.resid))
        ):
            raise RuntimeError("Perron test produced an invalid numerical fit.")

        # Critical values
        cv_01 = _perron_crit(self.model, self.break_fraction, 0.01)
        cv_05 = _perron_crit(self.model, self.break_fraction, 0.05)
        cv_10 = _perron_crit(self.model, self.break_fraction, 0.10)

        # Collect coefficients
        coefs, pvals = _extract_coefficients(res, reg_cols)

        self.result_ = PerronTestResult(
            statistic=t_stat,
            pvalue=None,
            lags=k,
            nobs=len(y_dep),
            residuals=res.resid,
            rho_hat=rho_hat,
            rho_se=rho_se,
            break_year=self.break_year,
            break_index=break_idx,
            break_fraction=self.break_fraction,
            model=self.model,
            cv_01=cv_01,
            cv_05=cv_05,
            cv_10=cv_10,
            coefficients=coefs,
            pvalues=pvals,
            fitted=res.fittedvalues,
        )
        return self.result_
