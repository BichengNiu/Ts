"""Box-Jenkins residual cross-correlation diagnostics for transfer models.

The lag-wise coefficients use a fixed-n biased cross-covariance convention.
For lags zero through K, the joint statistic is
``S* = n**2 * sum(r[k]**2 / (n-k))`` with chi-square degrees of freedom
``K + 1 - m``, where m counts fitted transfer-polynomial parameters only.

References
----------
Box, G. E. P. & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting
and Control*, revised edition, pp. 395-396.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from statsmodels.tsa.stattools import ccf

from ._base import BaseMultiTestResult, BaseTest


def _validate_lags(lags):
    if isinstance(lags, (bool, np.bool_)) or not isinstance(lags, (int, np.integer)):
        raise TypeError("lags must be a positive integer")
    lags = int(lags)
    if lags < 1:
        raise ValueError("lags must be a positive integer")
    return lags


def _validate_alpha(alpha):
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.integer, np.floating))
        or not 0.0 < float(alpha) < 1.0
    ):
        raise ValueError("alpha must be between 0 and 1")
    return float(alpha)


def _clean_residuals(values, name):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not len(array):
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_names(names, count):
    names = tuple(names)
    if len(names) != count:
        raise ValueError("input_names must contain one name per input")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("input names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("input names must be unique")
    return names


def _normalise_inputs(input_residuals, input_names):
    if isinstance(input_residuals, Mapping):
        if input_names is not None:
            raise ValueError("input_names must not be provided with a mapping")
        if not input_residuals:
            raise ValueError("input_residuals must contain at least one input")
        names = _validate_names(input_residuals.keys(), len(input_residuals))
        return {
            name: _clean_residuals(input_residuals[name], f"input residuals {name!r}")
            for name in names
        }

    if isinstance(input_residuals, pd.DataFrame):
        if input_names is not None:
            raise ValueError("input_names must not be provided with a DataFrame")
        if input_residuals.shape[1] == 0:
            raise ValueError("input_residuals must contain at least one input")
        names = _validate_names(input_residuals.columns, input_residuals.shape[1])
        return {
            name: _clean_residuals(input_residuals[name], f"input residuals {name!r}")
            for name in names
        }

    if isinstance(input_residuals, pd.Series):
        if input_names is None:
            name = input_residuals.name or "x"
            names = _validate_names((name,), 1)
        else:
            names = _validate_names(input_names, 1)
        return {
            names[0]: _clean_residuals(
                input_residuals,
                f"input residuals {names[0]!r}",
            )
        }

    try:
        values = np.asarray(input_residuals, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("input residuals must contain numeric values") from error
    if values.ndim == 1:
        names = ("x",) if input_names is None else _validate_names(input_names, 1)
        return {
            names[0]: _clean_residuals(values, f"input residuals {names[0]!r}")
        }
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("input residuals must be one- or two-dimensional")
    if input_names is None:
        raise ValueError("input_names is required for two-dimensional array input")
    names = _validate_names(input_names, values.shape[1])
    return {
        name: _clean_residuals(values[:, position], f"input residuals {name!r}")
        for position, name in enumerate(names)
    }


def _validate_parameter_count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"transfer_params for {name!r} must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"transfer_params for {name!r} must be non-negative")
    return value


def _normalise_transfer_params(transfer_params, names, lags):
    if isinstance(transfer_params, Mapping):
        unknown = [name for name in transfer_params if name not in names]
        if unknown:
            raise ValueError(
                f"transfer_params contains unknown input {unknown[0]!r}"
            )
        missing = [name for name in names if name not in transfer_params]
        if missing:
            raise ValueError(
                f"transfer_params is missing input {missing[0]!r}"
            )
        counts = {
            name: _validate_parameter_count(transfer_params[name], name)
            for name in names
        }
    else:
        count = _validate_parameter_count(transfer_params, "all inputs")
        counts = dict.fromkeys(names, count)

    for name, count in counts.items():
        if lags + 1 - count <= 0:
            raise ValueError(
                "joint S* test requires positive degrees of freedom: "
                f"lags + 1 - transfer_params for {name!r} must be positive"
            )
    return counts


@dataclass
class ResidualCCFInputResult:
    """Residual cross-correlation diagnostics for one transfer input.

    Parameters
    ----------
    input_name : str
        Name of the prewhitened transfer input.
    correlations : pandas.Series
        Residual cross-correlations for lags zero through ``lags``.
    standard_errors : pandas.Series
        Approximate standard errors, each equal to ``1 / sqrt(nobs)``.
    confidence_limits : pandas.DataFrame
        Symmetric null confidence limits with ``lower`` and ``upper`` columns.
    statistic, pvalue : float
        Joint Box-Jenkins S* statistic and chi-square p-value.
    df : int
        Chi-square degrees of freedom, ``lags + 1 - transfer_params``.
    lags, nobs, transfer_params : int
        Maximum lag, aligned sample size, and fitted transfer parameters.
    alpha : float
        Significance level used for the confidence band and decisions.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import ResidualCCFTest, ResidualCCFInputResult
    >>> rng = np.random.default_rng(42)
    >>> result = ResidualCCFTest(rng.normal(size=80), rng.normal(size=80), lags=4).fit()
    >>> isinstance(result.get("x"), ResidualCCFInputResult)
    True
    """

    input_name: str
    correlations: pd.Series
    standard_errors: pd.Series
    confidence_limits: pd.DataFrame
    statistic: float
    pvalue: float
    df: int
    lags: int
    nobs: int
    transfer_params: int
    alpha: float

    @property
    def reject(self):
        """Whether the joint no-cross-correlation null is rejected."""
        return bool(self.pvalue < self.alpha)

    @property
    def significant_lags(self):
        """Lags whose correlations exceed the pointwise confidence band."""
        lower = self.confidence_limits["lower"].to_numpy()
        upper = self.confidence_limits["upper"].to_numpy()
        values = self.correlations.to_numpy()
        mask = (values < lower) | (values > upper)
        return tuple(int(lag) for lag in self.correlations.index[mask])

    def summary(self):
        """Return the per-input residual-CCF and joint S* report.

        Returns
        -------
        str
            Self-contained pointwise and joint diagnostic report.

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import ResidualCCFTest
        >>> rng = np.random.default_rng(42)
        >>> item = ResidualCCFTest(rng.normal(size=60), rng.normal(size=60), lags=3).fit().get("x")
        >>> "S* statistic" in item.summary()
        True
        """
        conclusion = (
            "Reject H0 (remaining input-output cross-correlation)"
            if self.reject
            else "Cannot reject H0 (no remaining cross-correlation detected)"
        )
        significant = (
            ", ".join(str(lag) for lag in self.significant_lags)
            if self.significant_lags
            else "None"
        )
        return "\n".join(
            [
                "=" * 62,
                f"  Residual cross-correlation: {self.input_name}",
                "=" * 62,
                f"  S* statistic       : {self.statistic:.6f}",
                f"  Chi-square df      : {self.df}",
                f"  P-value            : {self.pvalue:.6f}",
                f"  Maximum lag        : {self.lags}",
                f"  Effective obs.     : {self.nobs}",
                f"  Transfer parameters: {self.transfer_params}",
                f"  Pointwise peaks    : {significant}",
                "  H0: residual cross-correlations at lags 0..K are jointly zero",
                f"  Conclusion ({100 * self.alpha:g}%): {conclusion}",
            ]
        )

    def plot_test(self, ax=None, **kwargs):
        """Plot lag-wise residual correlations and pointwise null bands.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axis to reuse.
        **kwargs
            Additional options forwarded to
            :func:`Ts.TsPlots.plot_correlogram`.

        Returns
        -------
        tuple
            Matplotlib ``(fig, ax)`` pair.

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import ResidualCCFTest
        >>> rng = np.random.default_rng(42)
        >>> item = ResidualCCFTest(rng.normal(size=60), rng.normal(size=60), lags=3).fit().get("x")
        >>> fig, ax = item.plot_test()
        >>> len(ax.patches)
        4
        """
        from Ts.TsPlots import plot_correlogram

        kwargs.setdefault("ytitle", "Residual CCF")
        return plot_correlogram(
            self.correlations,
            self.confidence_limits["upper"],
            ax=ax,
            **kwargs,
        )

    def __str__(self):
        return self.summary()


@dataclass
class ResidualCCFTestResult(BaseMultiTestResult):
    """Structured residual-CCF results for one or more transfer inputs.

    Parameters
    ----------
    lags : int
        Common maximum lag.
    nobs : int
        Smallest effective sample size among tested inputs.
    residuals : None
        Reserved base-result field; aligned innovations stay internal.
    inputs : tuple of ResidualCCFInputResult
        Per-input lag-wise and joint results.
    alpha : float
        Significance level used for inference.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import ResidualCCFTest, ResidualCCFTestResult
    >>> rng = np.random.default_rng(42)
    >>> result = ResidualCCFTest(rng.normal(size=80), rng.normal(size=80), lags=4).fit()
    >>> isinstance(result, ResidualCCFTestResult)
    True
    """

    inputs: tuple[ResidualCCFInputResult, ...] = ()
    alpha: float = 0.05

    @property
    def input_names(self):
        """Tested input names in result order."""
        return tuple(item.input_name for item in self.inputs)

    @property
    def tests(self):
        """Return one compact joint S* test row per input."""
        rows = [
            {
                "input": item.input_name,
                "s_statistic": item.statistic,
                "df": item.df,
                "p_value": item.pvalue,
                "reject": item.reject,
                "nobs": item.nobs,
                "transfer_params": item.transfer_params,
            }
            for item in self.inputs
        ]
        return pd.DataFrame(
            rows,
            columns=[
                "input",
                "s_statistic",
                "df",
                "p_value",
                "reject",
                "nobs",
                "transfer_params",
            ],
        )

    def get(self, input_name):
        """Return residual-CCF diagnostics for one named input.

        Parameters
        ----------
        input_name : str
            Tested input name.

        Returns
        -------
        ResidualCCFInputResult
            Matching per-input result.

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import ResidualCCFTest
        >>> rng = np.random.default_rng(42)
        >>> result = ResidualCCFTest(rng.normal(size=60), rng.normal(size=60), lags=3).fit()
        >>> result.get("x").input_name
        'x'
        """
        for item in self.inputs:
            if item.input_name == input_name:
                return item
        raise KeyError(f"unknown input {input_name!r}")

    def summary(self):
        """Return all per-input residual-CCF and joint S* reports.

        Returns
        -------
        str
            Reports in tested-input order.

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import ResidualCCFTest
        >>> rng = np.random.default_rng(42)
        >>> result = ResidualCCFTest(rng.normal(size=60), {"x": rng.normal(size=60)}, lags=3).fit()
        >>> "Residual cross-correlation: x" in result.summary()
        True
        """
        return "\n\n".join(item.summary() for item in self.inputs)

    def plot_test(self, inputs=None, **kwargs):
        """Plot residual CCFs and pointwise bands for selected inputs.

        Parameters
        ----------
        inputs : str or sequence of str, optional
            Inputs to plot. The default plots every tested input.
        **kwargs
            Additional options forwarded to
            :func:`Ts.TsPlots.plot_correlogram`.

        Returns
        -------
        tuple
            ``(fig, ax)`` for one input or ``(fig, axes)`` for facets.

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import ResidualCCFTest
        >>> rng = np.random.default_rng(42)
        >>> result = ResidualCCFTest(rng.normal(size=60), {"x": rng.normal(size=60)}, lags=3).fit()
        >>> fig, ax = result.plot_test()
        >>> len(ax.patches)
        4
        """
        if inputs is None:
            selected = self.input_names
        elif isinstance(inputs, str):
            selected = (inputs,)
        else:
            try:
                selected = tuple(inputs)
            except TypeError as error:
                raise TypeError(
                    "inputs must be a name or an iterable of names"
                ) from error
        if not selected:
            raise ValueError("inputs must contain at least one input")
        if len(set(selected)) != len(selected):
            raise ValueError("inputs must be unique")
        unknown = [name for name in selected if name not in self.input_names]
        if unknown:
            raise ValueError(f"inputs contains unknown input {unknown[0]!r}")

        items = [self.get(name) for name in selected]
        correlations = pd.concat([item.correlations for item in items], axis=1)
        confidence_bands = pd.concat(
            [
                item.confidence_limits["upper"].rename(item.input_name)
                for item in items
            ],
            axis=1,
        )
        kwargs.setdefault("ytitle", "Residual CCF")

        from Ts.TsPlots import plot_correlogram

        return plot_correlogram(
            correlations,
            confidence_bands,
            **kwargs,
        )

    def __str__(self):
        return self.summary()


class ResidualCCFTest(BaseTest):
    """Test final-model residuals against prewhitened input innovations.

    At lag ``k``, the reported coefficient correlates the current output-model
    residual with the input innovation from ``k`` periods earlier. Unequal
    residual arrays are aligned at their common sample end, which matches
    post-burn innovations from models fitted through the same final date.

    Parameters
    ----------
    output_residuals : array-like
        Final transfer-model residuals.
    input_residuals : mapping, Series, DataFrame, or array-like
        One or more prewhitened input innovation sequences.
    lags : int
        Positive maximum lag K; coefficients cover lags 0 through K.
    input_names : sequence of str, optional
        Required for a two-dimensional array and forbidden for mappings or
        DataFrames.
    transfer_params : int or mapping of str to int, default 0
        Fitted transfer-function parameter count m for the S* degrees of
        freedom ``K + 1 - m``. Noise-model parameters are excluded.
    alpha : float, default 0.05
        Significance level for pointwise bands and joint decisions.

    Attributes
    ----------
    result_ : ResidualCCFTestResult or None
        Structured result populated by :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import ResidualCCFTest
    >>> rng = np.random.default_rng(42)
    >>> test = ResidualCCFTest(
    ...     rng.normal(size=100),
    ...     {"price": rng.normal(size=100)},
    ...     lags=6,
    ...     transfer_params={"price": 2},
    ... )
    >>> test.fit().tests.shape
    (1, 7)
    """

    def __init__(
        self,
        output_residuals,
        input_residuals,
        lags,
        *,
        input_names=None,
        transfer_params=0,
        alpha=0.05,
    ):
        self.lags = _validate_lags(lags)
        self.alpha = _validate_alpha(alpha)
        self.output_residuals = _clean_residuals(
            output_residuals,
            "output residuals",
        )
        self.input_residuals = _normalise_inputs(input_residuals, input_names)
        self.transfer_params = _normalise_transfer_params(
            transfer_params,
            tuple(self.input_residuals),
            self.lags,
        )

        for name, input_values in self.input_residuals.items():
            nobs = min(len(self.output_residuals), len(input_values))
            if nobs <= self.lags:
                raise ValueError(
                    f"Need at least {self.lags + 1} aligned observations for "
                    f"{self.lags} lags, got {nobs} for {name!r}"
                )
            output = self.output_residuals[-nobs:]
            input_ = input_values[-nobs:]
            if np.std(output, ddof=0) == 0:
                raise ValueError("output residuals must vary over the aligned sample")
            if np.std(input_, ddof=0) == 0:
                raise ValueError(
                    f"input residuals must vary over the aligned sample for {name!r}"
                )
        self.result_: ResidualCCFTestResult | None = None

    def fit(self):
        """Compute lag-wise residual CCFs and joint Box-Jenkins S* tests.

        Returns
        -------
        ResidualCCFTestResult
            Per-input correlations, confidence bands, and joint tests.

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import ResidualCCFTest
        >>> rng = np.random.default_rng(42)
        >>> result = ResidualCCFTest(rng.normal(size=60), rng.normal(size=60), lags=3).fit()
        >>> result.get("x").correlations.shape[0]
        4
        """
        results = []
        lag_index = pd.RangeIndex(self.lags + 1, name="lag")
        critical = float(norm.ppf(1.0 - self.alpha / 2.0))

        for name, input_values in self.input_residuals.items():
            nobs = min(len(self.output_residuals), len(input_values))
            output = self.output_residuals[-nobs:]
            input_ = input_values[-nobs:]
            values = np.asarray(
                ccf(
                    output,
                    input_,
                    adjusted=False,
                    fft=False,
                    nlags=self.lags + 1,
                ),
                dtype=float,
            )
            standard_error = 1.0 / np.sqrt(nobs)
            band = critical * standard_error
            correlations = pd.Series(values, index=lag_index, name=name)
            standard_errors = pd.Series(
                np.full(self.lags + 1, standard_error),
                index=lag_index,
                name="standard_error",
            )
            confidence_limits = pd.DataFrame(
                {
                    "lower": np.full(self.lags + 1, -band),
                    "upper": np.full(self.lags + 1, band),
                },
                index=lag_index,
            )
            lags = np.arange(self.lags + 1, dtype=float)
            statistic = float(nobs**2 * np.sum(values**2 / (nobs - lags)))
            parameter_count = self.transfer_params[name]
            df = self.lags + 1 - parameter_count
            pvalue = float(chi2.sf(statistic, df))
            results.append(
                ResidualCCFInputResult(
                    input_name=name,
                    correlations=correlations,
                    standard_errors=standard_errors,
                    confidence_limits=confidence_limits,
                    statistic=statistic,
                    pvalue=pvalue,
                    df=df,
                    lags=self.lags,
                    nobs=nobs,
                    transfer_params=parameter_count,
                    alpha=self.alpha,
                )
            )

        self.result_ = ResidualCCFTestResult(
            lags=self.lags,
            nobs=min(item.nobs for item in results),
            residuals=None,
            inputs=tuple(results),
            alpha=self.alpha,
        )
        return self.result_
