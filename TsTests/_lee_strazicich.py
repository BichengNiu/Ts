"""Lee-Strazicich minimum LM unit-root test with two unknown breaks."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from ._base import BaseTest, BaseTestResult
from ._break_utils import (
    _validate_nonnegative_int,
    _validate_time_axis,
)
from ._utils import _parse_input


_LS_MODEL_A_CRITICAL = {
    "1%": -4.545,
    "5%": -3.842,
    "10%": -3.504,
}

_LS_MODEL_C_CRITICAL = {
    (0.2, 0.4): {"1%": -6.16, "5%": -5.59, "10%": -5.27},
    (0.2, 0.6): {"1%": -6.41, "5%": -5.74, "10%": -5.32},
    (0.2, 0.8): {"1%": -6.33, "5%": -5.71, "10%": -5.33},
    (0.4, 0.6): {"1%": -6.45, "5%": -5.67, "10%": -5.31},
    (0.4, 0.8): {"1%": -6.42, "5%": -5.65, "10%": -5.32},
    (0.6, 0.8): {"1%": -6.32, "5%": -5.73, "10%": -5.32},
}


@dataclass(frozen=True)
class _LMRegression:
    statistic: float
    lags: int
    coefficients: dict[str, float]
    tvalues: dict[str, float]
    pvalues: dict[str, float]
    fitted: np.ndarray
    residuals: np.ndarray
    effective_indices: np.ndarray
    ic_by_lag: np.ndarray | None


def _ls_deterministic_terms(
    nobs: int,
    first_break_index: int,
    second_break_index: int,
    model: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Construct Lee-Strazicich deterministic terms excluding the constant."""
    positions = np.arange(nobs, dtype=float)
    du1 = (positions > first_break_index).astype(float)
    du2 = (positions > second_break_index).astype(float)
    if model == "A":
        return (
            np.column_stack([positions + 1.0, du1, du2]),
            ("trend", "DU1", "DU2"),
        )
    dt1 = np.maximum(positions - first_break_index, 0.0)
    dt2 = np.maximum(positions - second_break_index, 0.0)
    return (
        np.column_stack([positions + 1.0, du1, dt1, du2, dt2]),
        ("trend", "DU1", "DT1", "DU2", "DT2"),
    )


def _ls_critical_values(
    model: str,
    first_fraction: float,
    second_fraction: float,
) -> tuple[dict[str, float], tuple[float, float] | None]:
    """Select published Lee-Strazicich two-break critical values."""
    if model == "A":
        return _LS_MODEL_A_CRITICAL.copy(), None
    cell = min(
        _LS_MODEL_C_CRITICAL,
        key=lambda pair: (
            (pair[0] - first_fraction) ** 2 + (pair[1] - second_fraction) ** 2
        ),
    )
    return _LS_MODEL_C_CRITICAL[cell].copy(), cell


def _fit_lm_regression(
    endog: np.ndarray,
    deterministic: np.ndarray,
    deterministic_names: tuple[str, ...],
    lags: int,
) -> _LMRegression:
    """Fit the LM test regression for one break pair and lag order."""
    nobs = len(endog)
    dy = np.diff(endog)
    dz = np.diff(deterministic, axis=0)
    initial_coefficients, _, initial_rank, _ = np.linalg.lstsq(
        dz,
        dy,
        rcond=None,
    )
    if initial_rank < dz.shape[1]:
        raise np.linalg.LinAlgError("detrending design is rank deficient")
    initial_level = float(endog[0] - deterministic[0] @ initial_coefficients)
    detrended = endog - initial_level - deterministic @ initial_coefficients
    detrended_difference = np.diff(detrended)

    dependent = dy[lags:]
    regressors = [detrended[:-1][lags:], dz[lags:]]
    names = ["S_lag1", *(f"d_{name}" for name in deterministic_names)]
    for lag in range(1, lags + 1):
        regressors.append(detrended_difference[lags - lag : nobs - 1 - lag])
        names.append(f"dS_lag{lag}")
    design = np.column_stack(regressors)
    if design.shape[0] <= design.shape[1]:
        raise ValueError("insufficient residual degrees of freedom")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise np.linalg.LinAlgError("LM regression design is rank deficient")

    result = sm.OLS(dependent, design).fit()
    standard_error = float(result.bse[0])
    statistic = float(result.params[0] / standard_error)
    if (
        result.df_resid <= 0
        or not np.isfinite(standard_error)
        or standard_error <= 0
        or not np.isfinite(statistic)
        or not np.all(np.isfinite(result.resid))
    ):
        raise FloatingPointError("invalid LM regression")
    return _LMRegression(
        statistic=statistic,
        lags=lags,
        coefficients={
            name: float(result.params[position]) for position, name in enumerate(names)
        },
        tvalues={
            name: float(result.tvalues[position]) for position, name in enumerate(names)
        },
        pvalues={
            name: float(result.pvalues[position]) for position, name in enumerate(names)
        },
        fitted=np.asarray(result.fittedvalues),
        residuals=np.asarray(result.resid),
        effective_indices=np.arange(lags + 1, nobs),
        ic_by_lag=None,
    )


def _select_lm_regression(
    endog: np.ndarray,
    deterministic: np.ndarray,
    deterministic_names: tuple[str, ...],
    *,
    lags: int | None,
    max_lags: int,
    lag_method: str,
    lag_crit: float,
) -> _LMRegression:
    """Select the augmented lag order for one candidate break pair."""
    if lags is not None:
        return _fit_lm_regression(
            endog,
            deterministic,
            deterministic_names,
            lags,
        )

    fits: list[_LMRegression | None] = []
    criteria = np.full(max_lags + 1, np.nan, dtype=float)
    for lag in range(max_lags + 1):
        try:
            fit = _fit_lm_regression(
                endog,
                deterministic,
                deterministic_names,
                lag,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            fit = None
        fits.append(fit)
        if fit is not None:
            n_effective = len(fit.residuals)
            nparams = len(fit.coefficients)
            ssr = float(fit.residuals @ fit.residuals)
            if lag_method == "aic":
                criteria[lag] = np.log(ssr / n_effective) + (2 * nparams / n_effective)
            elif lag_method == "bic":
                criteria[lag] = np.log(ssr / n_effective) + (
                    nparams * np.log(n_effective) / n_effective
                )

    if lag_method in ("aic", "bic"):
        if np.all(np.isnan(criteria)):
            raise np.linalg.LinAlgError("no estimable lag specification")
        selected_lag = int(np.nanargmin(criteria))
    else:
        selected_lag = 0
        for lag in range(max_lags, 0, -1):
            fit = fits[lag]
            if fit is None:
                continue
            tstat = fit.tvalues[f"dS_lag{lag}"]
            if abs(tstat) >= lag_crit:
                selected_lag = lag
                break
        if fits[selected_lag] is None:
            valid_lags = [lag for lag, fit in enumerate(fits) if fit is not None]
            if not valid_lags:
                raise np.linalg.LinAlgError("no estimable lag specification")
            selected_lag = min(valid_lags)

    selected = fits[selected_lag]
    if selected is None:
        raise np.linalg.LinAlgError("selected lag specification is not estimable")
    return _LMRegression(
        statistic=selected.statistic,
        lags=selected.lags,
        coefficients=selected.coefficients,
        tvalues=selected.tvalues,
        pvalues=selected.pvalues,
        fitted=selected.fitted,
        residuals=selected.residuals,
        effective_indices=selected.effective_indices,
        ic_by_lag=criteria if lag_method in ("aic", "bic") else None,
    )


@dataclass
class LeeStrazicichTwoBreakTestResult(BaseTestResult):
    """Result of the Lee-Strazicich two-unknown-break minimum LM test.

    Parameters
    ----------
    statistic, pvalue, lags, nobs, residuals : see BaseTestResult
        Minimum LM statistic, optional p-value, selected lag count, effective
        sample size, and regression residuals.
    model : {"A", "C"}
        Level-shift or level-and-trend-shift specification.
    break_indices : tuple of int
        Two selected zero-based break positions.
    break_years : tuple of float
        Corresponding time labels.
    break_fractions : tuple of float
        Break locations as shares of the sample.
    cv_01, cv_05, cv_10 : float
        One-, five-, and ten-percent critical values.
    critical_value_cell : tuple of float or None
        Lookup-table cell used for interpolation.
    lag_method : str
        Lag-selection method used.
    ic_by_lag : numpy.ndarray or None
        Information-criterion path when applicable.
    coefficients, pvalues : dict
        Selected regression estimates and p-values.
    fitted : numpy.ndarray or None
        Fitted values from the selected regression.
    regression_time_index : numpy.ndarray or None
        Time labels aligned with the regression sample.
    all_candidate_break_indices, all_candidate_statistics : numpy.ndarray or None
        Candidate break pairs and their LM statistics.
    all_candidate_lags : numpy.ndarray or None
        Selected lag count for each candidate pair.
    time_index, observed : numpy.ndarray or None
        Original time labels and observations.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import LeeStrazicichTwoBreakTest
    >>> rng = np.random.default_rng(42)
    >>> data = np.r_[rng.normal(size=30), 2 + rng.normal(size=30), rng.normal(size=30)]
    >>> result = LeeStrazicichTwoBreakTest(data, lags=0).fit()
    >>> len(result.break_indices)
    2
    """

    model: str = "A"
    break_indices: tuple[int, int] = (0, 0)
    break_years: tuple[float, float] = (0.0, 0.0)
    break_fractions: tuple[float, float] = (0.0, 0.0)
    cv_01: float = 0.0
    cv_05: float = 0.0
    cv_10: float = 0.0
    critical_value_cell: tuple[float, float] | None = None
    lag_method: str = "tstat"
    ic_by_lag: np.ndarray | None = None
    coefficients: dict[str, float] = field(default_factory=dict)
    pvalues: dict[str, float] = field(default_factory=dict)
    fitted: np.ndarray | None = None
    regression_time_index: np.ndarray | None = None
    all_candidate_break_indices: np.ndarray | None = None
    all_candidate_statistics: np.ndarray | None = None
    all_candidate_lags: np.ndarray | None = None
    time_index: np.ndarray | None = None
    observed: np.ndarray | None = None

    @property
    def critical_values(self) -> dict[str, float]:
        """Critical values keyed by significance label."""
        return {"1%": self.cv_01, "5%": self.cv_05, "10%": self.cv_10}

    def __str__(self) -> str:
        model_description = (
            "Model A (two level breaks)"
            if self.model == "A"
            else "Model C (two level and trend breaks)"
        )
        header = self._format_conclusion(
            "Lee-Strazicich Two-Break Minimum LM Unit Root Test",
            "Unit root with two structural breaks",
        )
        conclusion = (
            "Reject H0: evidence favors break-stationarity."
            if self.statistic < self.cv_05
            else "Cannot reject H0: unit root remains plausible."
        )
        return (
            f"{header}\n"
            f"  Specification:   {model_description}\n"
            f"  Break points:    {self.break_years}\n"
            f"  Break fractions: {self.break_fractions}\n"
            f"  Conclusion (5%): {conclusion}"
        )

    def plot_test(self, ax=None):
        """Plot the series and the two minimizing break locations.

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
        >>> from Ts.TsTests import LeeStrazicichTwoBreakTest
        >>> rng = np.random.default_rng(7)
        >>> data = np.r_[rng.normal(size=25), 2 + rng.normal(size=25), rng.normal(size=25)]
        >>> result = LeeStrazicichTwoBreakTest(data, model="A", lags=0).fit()
        >>> fig, ax = result.plot_test()
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure
        ax.plot(self.time_index, self.observed, color="black", label="Observed")
        for position, break_year in enumerate(self.break_years):
            label = "Estimated breaks" if position == 0 else None
            ax.axvline(break_year, color="red", linestyle="--", label=label)
        ax.set_title("Lee-Strazicich Two-Unknown-Break LM Test")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        return fig, ax


class LeeStrazicichTwoBreakTest(BaseTest):
    """Minimum LM unit-root test allowing two endogenous breaks.

    ``model="A"`` permits two level shifts. ``model="C"`` permits two level
    and trend shifts. Breaks enter under both the null and alternative.

    Parameters
    ----------
    data : array-like or pandas.DataFrame
        Time series to test.
    time_index : array-like, optional
        Explicit ordered time labels; defaults to zero-based positions.
    model : {"A", "C"}, default "A"
        Two level shifts, or two level-and-trend shifts.
    lags : int, optional
        Fixed lagged-difference count. ``None`` enables selection.
    max_lags : int, default 8
        Largest candidate lag when selection is enabled.
    lag_method : {"tstat", "aic", "bic"}, default "tstat"
        Lag-selection rule.
    lag_crit : float, default 1.645
        Absolute t-statistic cutoff for general-to-specific selection.
    trim : float, default 0.1
        Fraction excluded at each sample boundary.
    min_break_distance : int, optional
        Minimum number of observations between the two breaks.
    y_col : str or int, optional
        Response column when ``data`` is a DataFrame.
    time_col : str or int, optional
        Time column when it is stored inside ``data``.

    Attributes
    ----------
    result_ : LeeStrazicichTwoBreakTestResult or None
        Fitted minimum-LM result.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import LeeStrazicichTwoBreakTest
    >>> rng = np.random.default_rng(7)
    >>> data = np.r_[rng.normal(size=30), 2 + rng.normal(size=30), rng.normal(size=30)]
    >>> test = LeeStrazicichTwoBreakTest(data, model="A", lags=0)
    >>> result = test.fit()
    >>> result.model
    'A'
    """

    def __init__(
        self,
        data,
        *,
        time_index=None,
        model: str = "A",
        lags: int | None = None,
        max_lags: int = 8,
        lag_method: str = "tstat",
        lag_crit: float = 1.645,
        trim: float = 0.10,
        min_break_distance: int | None = None,
        y_col=None,
        time_col=None,
    ):
        self.data, self.time_index = _parse_input(
            data,
            time_index,
            y_col,
            time_col,
        )
        _validate_time_axis(self.time_index)
        if np.ptp(self.data) == 0.0:
            raise ValueError("Lee-Strazicich test requires non-constant data")
        if not isinstance(model, str) or model.upper() not in ("A", "C"):
            raise ValueError("model must be 'A' or 'C'")
        self.model = model.upper()
        self.lags = (
            None if lags is None else _validate_nonnegative_int(lags, name="lags")
        )
        self.max_lags = _validate_nonnegative_int(max_lags, name="max_lags")
        if lag_method not in ("tstat", "aic", "bic"):
            raise ValueError("lag_method must be 'tstat', 'aic', or 'bic'")
        self.lag_method = lag_method
        if isinstance(lag_crit, bool) or not np.isscalar(lag_crit):
            raise TypeError("lag_crit must be a positive finite scalar")
        self.lag_crit = float(lag_crit)
        if not np.isfinite(self.lag_crit) or self.lag_crit <= 0:
            raise ValueError("lag_crit must be a positive finite scalar")
        if isinstance(trim, bool) or not np.isscalar(trim):
            raise TypeError("trim must be a finite scalar between 0 and 0.5")
        self.trim = float(trim)
        if not np.isfinite(self.trim) or not 0 < self.trim < 0.5:
            raise ValueError("trim must be between 0 and 0.5")
        default_distance = 2 if self.model == "A" else 3
        self.min_break_distance = (
            default_distance
            if min_break_distance is None
            else _validate_nonnegative_int(
                min_break_distance,
                name="min_break_distance",
            )
        )
        if self.min_break_distance < default_distance:
            raise ValueError(
                f"min_break_distance must be at least {default_distance} "
                f"for Model {self.model}"
            )
        self.result_: LeeStrazicichTwoBreakTestResult | None = None

    def fit(self) -> LeeStrazicichTwoBreakTestResult:
        """Search admissible break pairs and return the minimum LM statistic.

        Returns
        -------
        LeeStrazicichTwoBreakTestResult

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import LeeStrazicichTwoBreakTest
        >>> rng = np.random.default_rng(7)
        >>> data = np.r_[rng.normal(size=25), 2 + rng.normal(size=25), rng.normal(size=25)]
        >>> result = LeeStrazicichTwoBreakTest(data, model="A", lags=0).fit()
        >>> len(result.break_indices)
        2
        """
        nobs = len(self.data)
        lag_bound = self.max_lags if self.lags is None else self.lags
        lower = max(int(np.round(self.trim * nobs)), lag_bound + 3)
        upper = int(np.round((1.0 - self.trim) * nobs))
        candidate_pairs: list[tuple[int, int]] = []
        candidate_statistics: list[float] = []
        candidate_lags: list[int] = []
        best_fit: _LMRegression | None = None
        best_pair: tuple[int, int] | None = None

        for first_shift in range(lower, upper + 1):
            for second_shift in range(
                first_shift + self.min_break_distance,
                upper + 1,
            ):
                first_index = first_shift - 1
                second_index = second_shift - 1
                deterministic, names = _ls_deterministic_terms(
                    nobs,
                    first_index,
                    second_index,
                    self.model,
                )
                try:
                    fit = _select_lm_regression(
                        self.data,
                        deterministic,
                        names,
                        lags=self.lags,
                        max_lags=self.max_lags,
                        lag_method=self.lag_method,
                        lag_crit=self.lag_crit,
                    )
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    continue
                candidate_pairs.append((first_index, second_index))
                candidate_statistics.append(fit.statistic)
                candidate_lags.append(fit.lags)
                if best_fit is None or fit.statistic < best_fit.statistic:
                    best_fit = fit
                    best_pair = (first_index, second_index)

        if best_fit is None or best_pair is None:
            raise ValueError(
                "no estimable Lee-Strazicich break pair remains after trimming"
            )
        fractions = tuple((index + 1) / nobs for index in best_pair)
        critical, cell = _ls_critical_values(
            self.model,
            fractions[0],
            fractions[1],
        )
        break_years = tuple(float(self.time_index[index]) for index in best_pair)
        self.result_ = LeeStrazicichTwoBreakTestResult(
            statistic=best_fit.statistic,
            pvalue=None,
            lags=best_fit.lags,
            nobs=len(best_fit.residuals),
            residuals=best_fit.residuals,
            model=self.model,
            break_indices=best_pair,
            break_years=break_years,
            break_fractions=fractions,
            cv_01=critical["1%"],
            cv_05=critical["5%"],
            cv_10=critical["10%"],
            critical_value_cell=cell,
            lag_method=self.lag_method,
            ic_by_lag=best_fit.ic_by_lag,
            coefficients=best_fit.coefficients,
            pvalues=best_fit.pvalues,
            fitted=best_fit.fitted,
            regression_time_index=self.time_index[best_fit.effective_indices].copy(),
            all_candidate_break_indices=np.asarray(candidate_pairs, dtype=int),
            all_candidate_statistics=np.asarray(
                candidate_statistics,
                dtype=float,
            ),
            all_candidate_lags=np.asarray(candidate_lags, dtype=int),
            time_index=self.time_index.copy(),
            observed=self.data.copy(),
        )
        return self.result_
