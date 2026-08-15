"""Bai-Perron global multiple-unknown-break regression analysis.

The implementation estimates pure structural-change models in which every
column of the supplied regression design may change at each breakpoint.
Break locations are global minimum-SSR partitions obtained by dynamic
programming. Non-standard test distributions and break-date intervals are
estimated with a reproducible Rademacher wild bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from ._base import BaseTest, BaseTestResult
from ._break_utils import _validate_trim
from ._regression_break_utils import (
    RegressionBreakDesign,
    _coefficient_dict,
    _prepare_regression_break_design,
)
from Ts.TsUtils._validation import validate_int


@dataclass(frozen=True)
class _PartitionSet:
    """Optimal partitions for zero through ``max_breaks`` breaks."""

    costs: np.ndarray
    rss: dict[int, float]
    breaks: dict[int, tuple[int, ...]]


@dataclass(frozen=True)
class _BootstrapInference:
    """Bootstrap global/sequential test results and break-date intervals."""

    supf: dict[int, float]
    supf_pvalues: dict[int, float]
    sequential: dict[int, float]
    sequential_pvalues: dict[int, float]
    udmax: float
    udmax_pvalue: float
    wdmax: float
    wdmax_pvalue: float
    weights: dict[int, float]
    critical_values: dict[str, float]
    confidence_intervals: tuple[tuple[int, int], ...]


def _segment_cost_matrix(
    endog: np.ndarray,
    exog: np.ndarray,
    min_segment_size: int,
) -> np.ndarray:
    """Compute OLS SSR for every admissible half-open segment ``[i, j)``."""
    nobs, nparams = exog.shape
    cumulative_xx = np.zeros((nobs + 1, nparams, nparams), dtype=float)
    cumulative_xy = np.zeros((nobs + 1, nparams), dtype=float)
    cumulative_yy = np.zeros(nobs + 1, dtype=float)
    for index in range(nobs):
        row = exog[index]
        cumulative_xx[index + 1] = cumulative_xx[index] + np.outer(row, row)
        cumulative_xy[index + 1] = cumulative_xy[index] + row * endog[index]
        cumulative_yy[index + 1] = cumulative_yy[index] + endog[index] ** 2

    costs = np.full((nobs + 1, nobs + 1), np.inf, dtype=float)
    for start in range(nobs - min_segment_size + 1):
        for stop in range(start + min_segment_size, nobs + 1):
            xx = cumulative_xx[stop] - cumulative_xx[start]
            xy = cumulative_xy[stop] - cumulative_xy[start]
            yy = cumulative_yy[stop] - cumulative_yy[start]
            try:
                coefficients = np.linalg.solve(xx, xy)
            except np.linalg.LinAlgError:
                continue
            ssr = float(yy - coefficients @ xy)
            tolerance = np.finfo(float).eps * max(1.0, abs(yy)) * 100
            if ssr < -tolerance or not np.isfinite(ssr):
                continue
            costs[start, stop] = max(ssr, 0.0)
    return costs


def _global_partitions(
    endog: np.ndarray,
    exog: np.ndarray,
    min_segment_size: int,
    max_breaks: int,
) -> _PartitionSet:
    """Solve all global minimum-SSR partitions by dynamic programming."""
    nobs = len(endog)
    costs = _segment_cost_matrix(endog, exog, min_segment_size)
    max_segments = max_breaks + 1
    dynamic = np.full((max_segments + 1, nobs + 1), np.inf, dtype=float)
    previous = np.full((max_segments + 1, nobs + 1), -1, dtype=int)

    for stop in range(min_segment_size, nobs + 1):
        dynamic[1, stop] = costs[0, stop]

    for segments in range(2, max_segments + 1):
        first_stop = segments * min_segment_size
        for stop in range(first_stop, nobs + 1):
            candidates = np.arange(
                (segments - 1) * min_segment_size,
                stop - min_segment_size + 1,
            )
            candidate_costs = (
                dynamic[segments - 1, candidates] + costs[candidates, stop]
            )
            best_position = int(np.argmin(candidate_costs))
            best_cost = float(candidate_costs[best_position])
            if np.isfinite(best_cost):
                dynamic[segments, stop] = best_cost
                previous[segments, stop] = int(candidates[best_position])

    rss: dict[int, float] = {}
    partitions: dict[int, tuple[int, ...]] = {}
    for breaks in range(max_breaks + 1):
        segments = breaks + 1
        rss_value = float(dynamic[segments, nobs])
        if not np.isfinite(rss_value):
            raise ValueError(
                f"no estimable global partition exists for {breaks} breaks"
            )
        endpoints: list[int] = []
        stop = nobs
        for current_segments in range(segments, 1, -1):
            split = int(previous[current_segments, stop])
            if split < 0:
                raise RuntimeError("dynamic-programming traceback failed")
            endpoints.append(split - 1)
            stop = split
        rss[breaks] = rss_value
        partitions[breaks] = tuple(reversed(endpoints))
    return _PartitionSet(costs=costs, rss=rss, breaks=partitions)


def _partition_boundaries(
    nobs: int,
    break_indices: tuple[int, ...],
) -> tuple[int, ...]:
    """Convert last-observation break indices to half-open segment bounds."""
    return (0, *(index + 1 for index in break_indices), nobs)


def _fit_partition(
    design: RegressionBreakDesign,
    break_indices: tuple[int, ...],
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    """Fit every regime and return coefficients, fitted values, and residuals."""
    coefficients: list[dict[str, float]] = []
    fitted = np.empty(len(design.endog), dtype=float)
    residuals = np.empty(len(design.endog), dtype=float)
    bounds = _partition_boundaries(len(design.endog), break_indices)
    for start, stop in pairwise(bounds):
        x_segment = design.exog[start:stop]
        y_segment = design.endog[start:stop]
        beta, _, rank, _ = np.linalg.lstsq(x_segment, y_segment, rcond=None)
        if rank < design.exog.shape[1]:
            raise ValueError("selected partition contains a rank-deficient regime")
        fitted[start:stop] = x_segment @ beta
        residuals[start:stop] = y_segment - fitted[start:stop]
        coefficients.append(_coefficient_dict(beta, design.column_names))
    return coefficients, fitted, residuals


def _information_criteria(
    rss_by_breaks: dict[int, float],
    nobs: int,
    nparams: int,
) -> tuple[dict[int, float], dict[int, float]]:
    """Return Gaussian BIC and Liu-Wu-Zidek strengthened BIC."""
    bic: dict[int, float] = {}
    lwz: dict[int, float] = {}
    bic_penalty = np.log(nobs)
    lwz_penalty = 0.299 * np.log(nobs) ** 2.1
    for breaks, rss in rss_by_breaks.items():
        if rss <= 0:
            raise ValueError("information criteria require positive residual variance")
        deviance = nobs * (np.log(2 * np.pi) + 1.0 + np.log(rss / nobs))
        parameter_count = (breaks + 1) * nparams + breaks
        bic[breaks] = float(deviance + bic_penalty * parameter_count)
        lwz[breaks] = float(deviance + lwz_penalty * parameter_count)
    return bic, lwz


def _global_supf_statistics(
    rss_by_breaks: dict[int, float],
    nobs: int,
    nparams: int,
) -> dict[int, float]:
    """Compute global supF(m|0) statistics from optimal partitions."""
    rss_zero = rss_by_breaks[0]
    statistics: dict[int, float] = {}
    for breaks in range(1, len(rss_by_breaks)):
        rss_alternative = rss_by_breaks[breaks]
        df_denom = nobs - (breaks + 1) * nparams
        numerator = (rss_zero - rss_alternative) / (breaks * nparams)
        denominator = rss_alternative / df_denom
        statistics[breaks] = float(max(numerator, 0.0) / denominator)
    return statistics


def _sequential_supf_statistics(
    partitions: _PartitionSet,
    nobs: int,
    nparams: int,
) -> dict[int, float]:
    """Compute conditional Bai-Perron ``supF(l+1|l)`` statistics.

    For each globally estimated ``l``-break partition, hold its break dates
    fixed and search every admissible segment for one additional break. This
    is distinct from comparing separately re-optimized ``l``- and
    ``l+1``-break partitions.
    """
    statistics: dict[int, float] = {}
    for current_breaks in range(len(partitions.rss) - 1):
        rss_null = partitions.rss[current_breaks]
        bounds = _partition_boundaries(
            nobs,
            partitions.breaks[current_breaks],
        )
        rss_alternative = np.inf
        for start, stop in pairwise(bounds):
            segment_rss = partitions.costs[start, stop]
            for split in range(start + 1, stop):
                left_rss = partitions.costs[start, split]
                right_rss = partitions.costs[split, stop]
                if not np.isfinite(left_rss) or not np.isfinite(right_rss):
                    continue
                candidate_rss = rss_null - segment_rss + left_rss + right_rss
                rss_alternative = min(rss_alternative, candidate_rss)
        if not np.isfinite(rss_alternative):
            raise ValueError(
                "no admissible additional break exists within the "
                f"{current_breaks}-break partition"
            )
        df_denom = nobs - (current_breaks + 2) * nparams
        numerator = (rss_null - rss_alternative) / nparams
        denominator = rss_alternative / df_denom
        statistics[current_breaks] = float(max(numerator, 0.0) / denominator)
    return statistics


def _bootstrap_pvalue(observed: float, simulated: np.ndarray) -> float:
    """Return a finite-bootstrap upper-tail p-value."""
    return float((1 + np.count_nonzero(simulated >= observed)) / (len(simulated) + 1))


def _critical_values(simulated: np.ndarray) -> dict[str, float]:
    """Return upper-tail bootstrap critical values."""
    return {
        "10%": float(np.quantile(simulated, 0.90)),
        "5%": float(np.quantile(simulated, 0.95)),
        "1%": float(np.quantile(simulated, 0.99)),
    }


def _wild_bootstrap_sample(
    fitted: np.ndarray,
    residuals: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a Rademacher wild-bootstrap regression sample."""
    weights = rng.choice(np.array([-1.0, 1.0]), size=len(residuals))
    return fitted + residuals * weights


@dataclass
class BaiPerronTestResult(BaseTestResult):
    """Full output for global multiple-unknown-break regression analysis.

    Parameters
    ----------
    statistic, pvalue, lags, nobs, residuals : see BaseTestResult
        Selected test statistic, p-value, unused lag field, sample size, and
        residuals from the selected partition.
    n_breaks : int
        Selected number of breaks.
    break_indices, break_years, break_fractions : tuple
        Selected positions, labels, and fractional locations.
    min_segment_size : int
        Minimum admissible regime length.
    selection_method : str
        Fixed-break or information-criterion selection rule.
    partitions : dict
        Globally optimal break positions by candidate break count.
    rss_by_breaks, bic_by_breaks, lwz_by_breaks : dict
        Fit and information criteria by candidate break count.
    supf_by_breaks, supf_pvalues : dict
        Global SupF statistics and bootstrap p-values.
    sequential_supf, sequential_pvalues : dict
        Sequential ``l`` versus ``l+1`` break tests.
    udmax, udmax_pvalue, wdmax, wdmax_pvalue : float
        Double-maximum statistics and bootstrap p-values.
    wdmax_weights : dict
        Weights used by the weighted double-maximum statistic.
    bootstrap_critical_values : dict
        Bootstrap critical values for reported statistics.
    break_confidence_intervals, break_confidence_years : tuple
        Position- and label-scale break confidence intervals.
    confidence_level : float
        Confidence level used for break intervals.
    segment_coefficients : list of dict
        Regression coefficients for each selected regime.
    fitted : numpy.ndarray or None
        Fitted values under the selected partition.
    time_index, observed : numpy.ndarray or None
        Original time labels and response observations.
    n_bootstrap : int
        Number of bootstrap replications.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import BaiPerronTest
    >>> rng = np.random.default_rng(42)
    >>> data = np.r_[rng.normal(size=30), 3 + rng.normal(size=30)]
    >>> result = BaiPerronTest(
    ...     data, breaks=1, max_breaks=1, n_bootstrap=19, random_state=42
    ... ).fit()
    >>> result.n_breaks
    1
    """

    n_breaks: int = 0
    break_indices: tuple[int, ...] = ()
    break_years: tuple[float, ...] = ()
    break_fractions: tuple[float, ...] = ()
    min_segment_size: int = 0
    selection_method: str = "bic"
    partitions: dict[int, tuple[int, ...]] = field(default_factory=dict)
    rss_by_breaks: dict[int, float] = field(default_factory=dict)
    bic_by_breaks: dict[int, float] = field(default_factory=dict)
    lwz_by_breaks: dict[int, float] = field(default_factory=dict)
    supf_by_breaks: dict[int, float] = field(default_factory=dict)
    supf_pvalues: dict[int, float] = field(default_factory=dict)
    sequential_supf: dict[int, float] = field(default_factory=dict)
    sequential_pvalues: dict[int, float] = field(default_factory=dict)
    udmax: float = 0.0
    udmax_pvalue: float = 1.0
    wdmax: float = 0.0
    wdmax_pvalue: float = 1.0
    wdmax_weights: dict[int, float] = field(default_factory=dict)
    bootstrap_critical_values: dict[str, dict[str, float]] = field(default_factory=dict)
    break_confidence_intervals: tuple[tuple[int, int], ...] = ()
    break_confidence_years: tuple[tuple[float, float], ...] = ()
    confidence_level: float = 0.95
    segment_coefficients: list[dict[str, float]] = field(default_factory=list)
    fitted: np.ndarray | None = None
    time_index: np.ndarray | None = None
    observed: np.ndarray | None = None
    n_bootstrap: int = 0

    def __str__(self) -> str:
        header = self._format_conclusion(
            "Bai-Perron Multiple Break Test",
            "No structural break against up to the configured maximum",
        )
        return (
            f"{header}\n"
            f"  Selected breaks: {self.n_breaks} ({self.selection_method})\n"
            f"  Break points:    {self.break_years or 'None'}\n"
            f"  UDmax p-value:   {self.udmax_pvalue:.6f}\n"
            f"  WDmax p-value:   {self.wdmax_pvalue:.6f}\n"
            "  Inference:       Rademacher wild bootstrap "
            "(heteroskedasticity robust; not serial-correlation robust)"
        )

    def plot_test(self, ax=None):
        """Plot observed data, selected piecewise fit, breaks, and intervals.

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
        >>> from Ts.TsTests import BaiPerronTest
        >>> rng = np.random.default_rng(42)
        >>> data = np.r_[rng.normal(size=30), 3 + rng.normal(size=30)]
        >>> result = BaiPerronTest(data, breaks=1, max_breaks=1, n_bootstrap=19).fit()
        >>> fig, ax = result.plot_test()
        """
        from Ts.TsPlots.style import DEFAULT_PALETTE, _FigureContext

        context = _FigureContext(ax=ax)
        ax = context.ax
        ax.plot(
            self.time_index,
            self.observed,
            color=DEFAULT_PALETTE[0],
            label="Observed",
        )
        ax.plot(
            self.time_index,
            self.fitted,
            label="Piecewise regression fit",
            color=DEFAULT_PALETTE[1],
        )
        for position, break_year in enumerate(self.break_years):
            label = "Estimated breaks" if position == 0 else None
            ax.axvline(
                break_year,
                color=DEFAULT_PALETTE[4],
                linestyle="--",
                label=label,
            )
            if position < len(self.break_confidence_years):
                lower, upper = self.break_confidence_years[position]
                ax.axvspan(lower, upper, color=DEFAULT_PALETTE[4], alpha=0.10)
        context.finalize(
            title="Bai-Perron Global Multiple-Break Partition",
            xtitle="Time",
            ytitle="Value",
        )
        return context.fig, context.ax


class BaiPerronTest(BaseTest):
    """Estimate and test multiple unknown breaks in a linear regression.

    All regression coefficients are allowed to change across regimes. Break
    indices identify the final observation of the preceding regime, matching
    :class:`PerronTest` semantics.

    Parameters
    ----------
    data : array-like or pandas.DataFrame
        Response observations, or a table containing selected columns.
    exog : array-like, optional
        External regressors supplied separately from ``data``.
    time_index : array-like, optional
        Ordered labels used when reporting breaks.
    trend : {"n", "c", "ct"}, default "c"
        Deterministic regressors allowed to change across regimes.
    y_col : str or int, optional
        Response column in a DataFrame.
    time_col : str or int, optional
        Time-label column in a DataFrame.
    exog_cols : sequence of str or int, optional
        Regressor columns selected from a DataFrame.
    trim : float, default 0.15
        Minimum regime length as a fraction of the sample.
    min_segment_size : int, optional
        Explicit minimum regime length; overrides the trim-derived value.
    max_breaks : int, default 5
        Largest candidate break count.
    breaks : int, optional
        Fix the number of breaks instead of selecting it.
    criterion : {"bic", "lwz"}, default "bic"
        Information criterion used when ``breaks`` is omitted.
    significance : float, default 0.05
        Sequential-test significance level.
    confidence_level : float, default 0.95
        Break-interval confidence level.
    n_bootstrap : int, default 99
        Wild-bootstrap replication count.
    random_state : int, optional
        Seed for reproducible bootstrap inference.

    Attributes
    ----------
    result_ : BaiPerronTestResult or None
        Fitted global partition and inference results.

    Examples
    --------
    Select one known candidate count and use a fixed seed for reproducibility.

    >>> import numpy as np
    >>> from Ts.TsTests import BaiPerronTest
    >>> rng = np.random.default_rng(42)
    >>> data = np.r_[rng.normal(size=30), 3 + rng.normal(size=30)]
    >>> test = BaiPerronTest(
    ...     data, breaks=1, max_breaks=1, n_bootstrap=19, random_state=42
    ... )
    >>> result = test.fit()
    >>> len(result.break_indices)
    1
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
        trim: float = 0.15,
        min_segment_size: int | None = None,
        max_breaks: int = 5,
        breaks: int | None = None,
        criterion: str = "bic",
        significance: float = 0.05,
        confidence_level: float = 0.95,
        n_bootstrap: int = 99,
        random_state: int | None = None,
    ):
        self.design = _prepare_regression_break_design(
            data,
            exog=exog,
            time_index=time_index,
            trend=trend,
            y_col=y_col,
            time_col=time_col,
            exog_cols=exog_cols,
        )
        nobs, nparams = self.design.exog.shape
        self.trim = _validate_trim(trim)
        if min_segment_size is None:
            segment_size = int(np.floor(self.trim * nobs))
        else:
            segment_size = validate_int(
                "min_segment_size",
                min_segment_size,
                minimum=0,
            )
        if segment_size <= nparams:
            raise ValueError(
                "min_segment_size must be greater than the number of regressors"
            )
        if segment_size > nobs // 2:
            raise ValueError("min_segment_size must not exceed half the observations")
        self.min_segment_size = segment_size

        self.max_breaks = validate_int("max_breaks", max_breaks, minimum=0)
        if self.max_breaks < 1:
            raise ValueError("max_breaks must be at least 1")
        max_admissible = nobs // segment_size - 1
        if self.max_breaks > max_admissible:
            raise ValueError(
                f"max_breaks={self.max_breaks} exceeds the admissible maximum "
                f"{max_admissible} for min_segment_size={segment_size}"
            )
        if breaks is not None:
            self.breaks = validate_int("breaks", breaks, minimum=0)
            if self.breaks > self.max_breaks:
                raise ValueError("breaks must not exceed max_breaks")
        else:
            self.breaks = None
        if criterion not in ("bic", "lwz"):
            raise ValueError("criterion must be 'bic' or 'lwz'")
        self.criterion = criterion
        if significance not in (0.10, 0.05, 0.01):
            raise ValueError("significance must be 0.10, 0.05, or 0.01")
        self.significance = significance
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        self.confidence_level = float(confidence_level)
        self.n_bootstrap = validate_int(
            "n_bootstrap",
            n_bootstrap,
            minimum=0,
        )
        if self.n_bootstrap < 19:
            raise ValueError("n_bootstrap must be at least 19")
        if random_state is not None and (
            isinstance(random_state, bool)
            or not isinstance(random_state, (int, np.integer))
        ):
            raise TypeError("random_state must be an integer or None")
        self.random_state = None if random_state is None else int(random_state)
        self.result_: BaiPerronTestResult | None = None

    def _bootstrap_inference(
        self,
        observed_partitions: _PartitionSet,
        selected_breaks: int,
        fitted_by_breaks: dict[int, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> _BootstrapInference:
        """Bootstrap global/sequential tests and selected break-date intervals."""
        nobs, nparams = self.design.exog.shape
        global_draws = np.empty(
            (self.n_bootstrap, self.max_breaks),
            dtype=float,
        )
        fitted_zero, residuals_zero = fitted_by_breaks[0]
        for draw in range(self.n_bootstrap):
            sample = _wild_bootstrap_sample(fitted_zero, residuals_zero, rng)
            partitions = _global_partitions(
                sample,
                self.design.exog,
                self.min_segment_size,
                self.max_breaks,
            )
            statistics = _global_supf_statistics(
                partitions.rss,
                nobs,
                nparams,
            )
            global_draws[draw] = [
                statistics[breaks] for breaks in range(1, self.max_breaks + 1)
            ]

        observed_global = _global_supf_statistics(
            observed_partitions.rss,
            nobs,
            nparams,
        )
        supf_pvalues = {
            breaks: _bootstrap_pvalue(
                observed_global[breaks],
                global_draws[:, breaks - 1],
            )
            for breaks in range(1, self.max_breaks + 1)
        }
        critical_values = {
            f"supF({breaks}|0)": _critical_values(global_draws[:, breaks - 1])
            for breaks in range(1, self.max_breaks + 1)
        }

        quantile = 1.0 - self.significance
        marginal_critical = {
            breaks: float(np.quantile(global_draws[:, breaks - 1], quantile))
            for breaks in range(1, self.max_breaks + 1)
        }
        reference = marginal_critical[1]
        weights = {
            breaks: reference / marginal_critical[breaks]
            for breaks in range(1, self.max_breaks + 1)
        }
        observed_udmax = max(observed_global.values())
        observed_wdmax = max(
            weights[breaks] * observed_global[breaks] for breaks in observed_global
        )
        udmax_draws = np.max(global_draws, axis=1)
        wdmax_draws = np.max(
            global_draws
            * np.asarray([weights[breaks] for breaks in range(1, self.max_breaks + 1)]),
            axis=1,
        )
        critical_values["UDmax"] = _critical_values(udmax_draws)
        critical_values["WDmax"] = _critical_values(wdmax_draws)

        observed_sequential = _sequential_supf_statistics(
            observed_partitions,
            nobs,
            nparams,
        )
        sequential_pvalues: dict[int, float] = {}
        sequential_draws_by_null: dict[int, np.ndarray] = {}
        sequential_draws_by_null[0] = global_draws[:, 0]
        sequential_pvalues[0] = _bootstrap_pvalue(
            observed_sequential[0],
            sequential_draws_by_null[0],
        )
        critical_values["supF(1|0) sequential"] = _critical_values(
            sequential_draws_by_null[0]
        )
        for null_breaks in range(1, self.max_breaks):
            fitted_null, residuals_null = fitted_by_breaks[null_breaks]
            draws = np.empty(self.n_bootstrap, dtype=float)
            for draw in range(self.n_bootstrap):
                sample = _wild_bootstrap_sample(fitted_null, residuals_null, rng)
                partitions = _global_partitions(
                    sample,
                    self.design.exog,
                    self.min_segment_size,
                    null_breaks + 1,
                )
                statistic = _sequential_supf_statistics(
                    partitions,
                    nobs,
                    nparams,
                )
                draws[draw] = statistic[null_breaks]
            sequential_draws_by_null[null_breaks] = draws
            sequential_pvalues[null_breaks] = _bootstrap_pvalue(
                observed_sequential[null_breaks],
                draws,
            )
            critical_values[f"supF({null_breaks + 1}|{null_breaks}) sequential"] = (
                _critical_values(draws)
            )

        confidence_intervals: tuple[tuple[int, int], ...] = ()
        if selected_breaks > 0:
            fitted_selected, residuals_selected = fitted_by_breaks[selected_breaks]
            bootstrap_breaks = np.empty(
                (self.n_bootstrap, selected_breaks),
                dtype=int,
            )
            for draw in range(self.n_bootstrap):
                sample = _wild_bootstrap_sample(
                    fitted_selected,
                    residuals_selected,
                    rng,
                )
                partitions = _global_partitions(
                    sample,
                    self.design.exog,
                    self.min_segment_size,
                    selected_breaks,
                )
                bootstrap_breaks[draw] = partitions.breaks[selected_breaks]
            tail = (1.0 - self.confidence_level) / 2.0
            intervals: list[tuple[int, int]] = []
            estimates = observed_partitions.breaks[selected_breaks]
            for position, estimate in enumerate(estimates):
                lower = int(np.floor(np.quantile(bootstrap_breaks[:, position], tail)))
                upper = int(
                    np.ceil(np.quantile(bootstrap_breaks[:, position], 1.0 - tail))
                )
                intervals.append((min(lower, estimate), max(upper, estimate)))
            confidence_intervals = tuple(intervals)

        return _BootstrapInference(
            supf=observed_global,
            supf_pvalues=supf_pvalues,
            sequential=observed_sequential,
            sequential_pvalues=sequential_pvalues,
            udmax=observed_udmax,
            udmax_pvalue=_bootstrap_pvalue(observed_udmax, udmax_draws),
            wdmax=observed_wdmax,
            wdmax_pvalue=_bootstrap_pvalue(observed_wdmax, wdmax_draws),
            weights=weights,
            critical_values=critical_values,
            confidence_intervals=confidence_intervals,
        )

    def fit(self) -> BaiPerronTestResult:
        """Estimate global partitions, select break count, and bootstrap inference.

        Returns
        -------
        BaiPerronTestResult

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import BaiPerronTest
        >>> rng = np.random.default_rng(42)
        >>> data = np.r_[rng.normal(size=30), 3 + rng.normal(size=30)]
        >>> result = BaiPerronTest(data, breaks=1, max_breaks=1, n_bootstrap=19).fit()
        >>> result.n_breaks
        1
        """
        nobs, nparams = self.design.exog.shape
        partitions = _global_partitions(
            self.design.endog,
            self.design.exog,
            self.min_segment_size,
            self.max_breaks,
        )
        bic, lwz = _information_criteria(partitions.rss, nobs, nparams)
        if self.breaks is None:
            criterion_values = bic if self.criterion == "bic" else lwz
            selected_breaks = min(criterion_values, key=criterion_values.get)
            selection_method = self.criterion
        else:
            selected_breaks = self.breaks
            selection_method = "fixed"

        coefficients_by_breaks: dict[int, list[dict[str, float]]] = {}
        fitted_by_breaks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for breaks, break_indices in partitions.breaks.items():
            coefficients, fitted, residuals = _fit_partition(
                self.design,
                break_indices,
            )
            coefficients_by_breaks[breaks] = coefficients
            fitted_by_breaks[breaks] = (fitted, residuals)

        rng = np.random.default_rng(self.random_state)
        inference = self._bootstrap_inference(
            partitions,
            selected_breaks,
            fitted_by_breaks,
            rng,
        )

        selected_indices = partitions.breaks[selected_breaks]
        selected_years = tuple(
            float(self.design.time_index[index]) for index in selected_indices
        )
        confidence_years = tuple(
            (
                float(self.design.time_index[lower]),
                float(self.design.time_index[upper]),
            )
            for lower, upper in inference.confidence_intervals
        )
        fitted, residuals = fitted_by_breaks[selected_breaks]
        self.result_ = BaiPerronTestResult(
            statistic=inference.udmax,
            pvalue=inference.udmax_pvalue,
            lags=None,
            nobs=nobs,
            residuals=residuals,
            n_breaks=selected_breaks,
            break_indices=selected_indices,
            break_years=selected_years,
            break_fractions=tuple((index + 1) / nobs for index in selected_indices),
            min_segment_size=self.min_segment_size,
            selection_method=selection_method,
            partitions=partitions.breaks,
            rss_by_breaks=partitions.rss,
            bic_by_breaks=bic,
            lwz_by_breaks=lwz,
            supf_by_breaks=inference.supf,
            supf_pvalues=inference.supf_pvalues,
            sequential_supf=inference.sequential,
            sequential_pvalues=inference.sequential_pvalues,
            udmax=inference.udmax,
            udmax_pvalue=inference.udmax_pvalue,
            wdmax=inference.wdmax,
            wdmax_pvalue=inference.wdmax_pvalue,
            wdmax_weights=inference.weights,
            bootstrap_critical_values=inference.critical_values,
            break_confidence_intervals=inference.confidence_intervals,
            break_confidence_years=confidence_years,
            confidence_level=self.confidence_level,
            segment_coefficients=coefficients_by_breaks[selected_breaks],
            fitted=fitted,
            time_index=self.design.time_index.copy(),
            observed=self.design.endog.copy(),
            n_bootstrap=self.n_bootstrap,
        )
        return self.result_
