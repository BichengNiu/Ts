"""Conditional feedback checking for stochastic distributed-lag inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ._base import BaseMultiTestResult, BaseTest
from ._utils import _validate_alpha, _validate_lags


_TRENDS = frozenset({"n", "c", "t", "ct"})


def _normalise_exog(exog, exog_names):
    if isinstance(exog, pd.DataFrame):
        if exog_names is not None:
            raise ValueError("exog_names must not be provided with a DataFrame")
        frame = exog.copy()
    elif isinstance(exog, pd.Series):
        if exog.name is None:
            names = () if exog_names is None else tuple(exog_names)
            if len(names) != 1:
                raise ValueError(
                    "an unnamed exog Series requires exactly one exog_names value"
                )
            frame = exog.rename(names[0]).to_frame()
        else:
            if exog_names is not None:
                raise ValueError("exog_names must not be provided with a named Series")
            frame = exog.to_frame()
    else:
        values = np.asarray(exog, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("exog must be a non-empty one- or two-dimensional input")
        if exog_names is None:
            raise ValueError("exog_names is required for array exog")
        names = tuple(exog_names)
        if len(names) != values.shape[1]:
            raise ValueError("exog_names must contain one name per exog column")
        frame = pd.DataFrame(values, columns=names)

    names = tuple(frame.columns)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("exog names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("exog names must be unique")
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("exog must contain numeric values") from error
    return frame


def _normalise_y(y):
    if isinstance(y, pd.Series):
        series = y.copy()
        name = y.name if isinstance(y.name, str) and y.name else "y"
    else:
        values = np.asarray(y, dtype=float)
        if values.ndim != 1:
            raise ValueError("y must be one-dimensional")
        series = pd.Series(values)
        name = "y"
    try:
        series = series.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("y must contain numeric values") from error
    return series.rename(name)


def _normalise_tested_inputs(tested_inputs, names):
    if tested_inputs is None:
        return tuple(names)
    if isinstance(tested_inputs, str):
        tested_inputs = (tested_inputs,)
    else:
        try:
            tested_inputs = tuple(tested_inputs)
        except TypeError as error:
            raise TypeError("tested_inputs must be a name or an iterable of names") from error
    if not tested_inputs:
        raise ValueError("tested_inputs must contain at least one input")
    if len(set(tested_inputs)) != len(tested_inputs):
        raise ValueError("tested_inputs must be unique")
    unknown = [name for name in tested_inputs if name not in names]
    if unknown:
        raise ValueError(f"tested_inputs contains unknown input {unknown[0]!r}")
    return tested_inputs


@dataclass
class FeedbackEquationResult:
    """Result for one conditional input feedback equation.

    Parameters
    ----------
    input_name : str
        Current input used as the regression response.
    output_name : str
        Original model-output name whose lags are jointly tested.
    lags : int
        Common lag order.
    regression : statsmodels regression result
        Full fitted OLS result.
    f_statistic, pvalue : float
        Joint feedback F statistic and p-value.
    df_num, df_denom : float
        Numerator and denominator degrees of freedom.
    alpha : float
        Significance level used by :attr:`reject`.
    observation_index : pandas.Index
        Original observations retained by the lag regression.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import FeedbackTest, FeedbackEquationResult
    >>> rng = np.random.default_rng(42)
    >>> result = FeedbackTest(rng.normal(size=80), rng.normal(size=(80, 1)), lags=1, exog_names=["x"]).fit()
    >>> isinstance(result.get("x"), FeedbackEquationResult)
    True
    """

    input_name: str
    output_name: str
    lags: int
    regression: object
    f_statistic: float
    pvalue: float
    df_num: float
    df_denom: float
    alpha: float
    observation_index: pd.Index

    @property
    def reject(self):
        """Whether the joint no-feedback null is rejected."""
        return bool(self.pvalue < self.alpha)

    def summary(self) -> str:
        """Return the full OLS report followed by the joint feedback test.

        Returns
        -------
        str
            Self-contained equation and F-test report.

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import FeedbackTest
        >>> rng = np.random.default_rng(42)
        >>> equation = FeedbackTest(rng.normal(size=60), rng.normal(size=(60, 1)), lags=1, exog_names=["x"]).fit().get("x")
        >>> "Joint Feedback F Test" in equation.summary()
        True
        """
        restrictions = " = ".join(
            [f"{self.output_name}.L{lag}" for lag in range(1, self.lags + 1)]
            + ["0"]
        )
        conclusion = "Reject H0 (feedback detected)" if self.reject else (
            "Cannot reject H0 (no feedback detected)"
        )
        regression = self.regression
        confidence = np.asarray(regression.conf_int(), dtype=float)
        regression_lines = [
            "=" * 78,
            f"OLS Regression Results: {self.input_name}",
            "=" * 78,
            f"Observations: {int(regression.nobs)}",
            f"Df model: {regression.df_model:g}",
            f"Df residuals: {regression.df_resid:g}",
            f"R-squared: {regression.rsquared:.6f}",
            f"Adjusted R-squared: {regression.rsquared_adj:.6f}",
            f"F-statistic: {float(regression.fvalue):.6f}",
            f"Prob (F-statistic): {float(regression.f_pvalue):.6f}",
            f"Log likelihood: {regression.llf:.6f}",
            f"AIC: {regression.aic:.6f}",
            f"BIC: {regression.bic:.6f}",
            f"Covariance type: {regression.cov_type}",
            "-" * 78,
            (
                f"{'Variable':<24}{'Coef.':>11}{'Std.Err.':>11}"
                f"{'t':>11}{'P>|t|':>11}{'[0.025':>11}{'0.975]':>11}"
            ),
            "-" * 78,
        ]
        for position, name in enumerate(regression.params.index):
            regression_lines.append(
                f"{name:<24}"
                f"{regression.params.iloc[position]:>11.5f}"
                f"{regression.bse.iloc[position]:>11.5f}"
                f"{regression.tvalues.iloc[position]:>11.5f}"
                f"{regression.pvalues.iloc[position]:>11.5f}"
                f"{confidence[position, 0]:>11.5f}"
                f"{confidence[position, 1]:>11.5f}"
            )
        regression_lines.append("=" * 78)

        lines = [
            f"Feedback equation: {self.input_name}",
            "\n".join(regression_lines),
            "",
            "Joint Feedback F Test",
            "-" * 50,
            f"H0: {restrictions}",
            f"F({self.df_num:g}, {self.df_denom:g}) = {self.f_statistic:.6f}",
            f"P-value = {self.pvalue:.6f}",
            f"Conclusion ({self.alpha:.1%}): {conclusion}",
        ]
        return "\n".join(lines)

    def __str__(self):
        return self.summary()


@dataclass
class FeedbackTestResult(BaseMultiTestResult):
    """Structured results for one or more conditional feedback equations.

    Parameters
    ----------
    lags : int
        Common lag order.
    nobs : int
        Effective common regression sample size.
    residuals : numpy.ndarray or None
        Equation residuals in tested-input order.
    equations : tuple of FeedbackEquationResult
        Fitted equation results.
    alpha : float
        Significance level used for decisions.
    output_name : str
        Original model-output name.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import FeedbackTest, FeedbackTestResult
    >>> rng = np.random.default_rng(42)
    >>> result = FeedbackTest(rng.normal(size=80), rng.normal(size=(80, 1)), lags=1, exog_names=["x"]).fit()
    >>> isinstance(result, FeedbackTestResult)
    True
    """

    equations: tuple[FeedbackEquationResult, ...] = ()
    alpha: float = 0.05
    output_name: str = "y"

    @property
    def input_names(self):
        """Tested input names in equation order."""
        return tuple(equation.input_name for equation in self.equations)

    @property
    def regressions(self):
        """Return full fitted OLS results keyed by tested input."""
        return {
            equation.input_name: equation.regression for equation in self.equations
        }

    @property
    def tests(self):
        """Return one row per joint feedback F test."""
        rows = [
            {
                "input": equation.input_name,
                "f_statistic": equation.f_statistic,
                "df_num": equation.df_num,
                "df_denom": equation.df_denom,
                "p_value": equation.pvalue,
                "reject": equation.reject,
                "nobs": int(equation.regression.nobs),
            }
            for equation in self.equations
        ]
        return pd.DataFrame(
            rows,
            columns=[
                "input",
                "f_statistic",
                "df_num",
                "df_denom",
                "p_value",
                "reject",
                "nobs",
            ],
        )

    def get(self, input_name) -> FeedbackEquationResult:
        """Return the result for one tested input.

        Parameters
        ----------
        input_name : str
            Tested input name.

        Returns
        -------
        FeedbackEquationResult
            Matching equation result.

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import FeedbackTest
        >>> rng = np.random.default_rng(42)
        >>> result = FeedbackTest(rng.normal(size=60), rng.normal(size=(60, 1)), lags=1, exog_names=["x"]).fit()
        >>> result.get("x").input_name
        'x'
        """
        for equation in self.equations:
            if equation.input_name == input_name:
                return equation
        raise KeyError(f"unknown tested input {input_name!r}")

    def summary(self) -> str:
        """Return all regression reports and joint F tests.

        Returns
        -------
        str
            Reports in tested-input order.

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import FeedbackTest
        >>> rng = np.random.default_rng(42)
        >>> result = FeedbackTest(rng.normal(size=60), rng.normal(size=(60, 1)), lags=1, exog_names=["x"]).fit()
        >>> "Feedback equation: x" in result.summary()
        True
        """
        return "\n\n".join(equation.summary() for equation in self.equations)

    def __str__(self):
        return self.summary()


class FeedbackTest(BaseTest):
    """Test whether past model output predicts current stochastic inputs.

    For each tested input, OLS regresses its current value on lags 1 through
    ``lags`` of every input and the original model output. The joint null is
    that all output-lag coefficients equal zero. Rejection is evidence of
    conditional predictive feedback, not proof of structural causality.

    Parameters
    ----------
    y : array-like
        Original model output whose lag coefficients are tested.
    exog : Series, DataFrame, or array-like
        One or more stochastic model inputs.
    lags : int
        Positive common lag order.
    exog_names : sequence of str, optional
        Required for array ``exog`` and forbidden for named pandas inputs.
    tested_inputs : str or sequence of str, optional
        Inputs whose current values receive separate equations. All exogenous
        inputs remain as lagged controls. The default tests every input.
    trend : {"n", "c", "t", "ct"}, default "c"
        Deterministic regression terms: none, constant, trend, or both.
    missing : {"raise", "drop"}, default "raise"
        Missing-data policy. Drop is applied after the lag matrix is built so
        gaps never become falsely adjacent observations.
    alpha : float, default 0.05
        Significance level for feedback decisions.

    Attributes
    ----------
    result_ : FeedbackTestResult or None
        Fitted structured result.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import FeedbackTest
    >>> rng = np.random.default_rng(42)
    >>> y = rng.normal(size=100)
    >>> exog = rng.normal(size=(100, 2))
    >>> result = FeedbackTest(y, exog, lags=2, exog_names=["price", "income"]).fit()
    >>> result.tests.shape[0]
    2
    """

    def __init__(
        self,
        y,
        exog,
        lags,
        *,
        exog_names=None,
        tested_inputs=None,
        trend="c",
        missing="raise",
        alpha=0.05,
    ):
        self.lags = _validate_lags(lags)
        self.alpha = _validate_alpha(alpha)
        if trend not in _TRENDS:
            raise ValueError("trend must be one of 'n', 'c', 't', or 'ct'")
        if missing not in {"raise", "drop"}:
            raise ValueError("missing must be 'raise' or 'drop'")
        self.trend = trend
        self.missing = missing

        y_series = _normalise_y(y)
        exog_frame = _normalise_exog(exog, exog_names)
        if len(y_series) != len(exog_frame):
            raise ValueError("y and exog must contain the same number of observations")
        if isinstance(y, pd.Series) and isinstance(exog, (pd.Series, pd.DataFrame)):
            if not y_series.index.equals(exog_frame.index):
                raise ValueError("y and exog pandas indexes must match exactly")
        elif isinstance(exog, (pd.Series, pd.DataFrame)):
            y_series.index = exog_frame.index
        elif isinstance(y, pd.Series):
            exog_frame.index = y_series.index

        if y_series.name in exog_frame.columns:
            raise ValueError("the y name must differ from every exog name")
        y_values = y_series.to_numpy()
        exog_values = exog_frame.to_numpy()
        if np.any(np.isinf(y_values)) or np.any(np.isinf(exog_values)):
            raise ValueError("y and exog must not contain infinite values")
        if missing == "raise" and (
            np.any(np.isnan(y_values)) or np.any(np.isnan(exog_values))
        ):
            raise ValueError("missing values require missing='drop'")

        self.y = y_series
        self.exog = exog_frame
        self.tested_inputs = _normalise_tested_inputs(
            tested_inputs, tuple(exog_frame.columns)
        )
        self.result_: FeedbackTestResult | None = None

    def _lagged_design(self):
        columns = {}
        for name in self.exog.columns:
            for lag in range(1, self.lags + 1):
                columns[f"{name}.L{lag}"] = self.exog[name].shift(lag)
        for lag in range(1, self.lags + 1):
            columns[f"{self.y.name}.L{lag}"] = self.y.shift(lag)
        design = pd.DataFrame(columns, index=self.exog.index)
        positions = np.arange(1, len(design) + 1, dtype=float)
        if "t" in self.trend:
            design.insert(0, "trend", positions)
        if "c" in self.trend:
            design.insert(0, "const", 1.0)
        return design

    def fit(self) -> FeedbackTestResult:
        """Fit every feedback equation and run the joint F restrictions.

        Returns
        -------
        FeedbackTestResult
            Full regressions and joint feedback tests.

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import FeedbackTest
        >>> rng = np.random.default_rng(42)
        >>> test = FeedbackTest(rng.normal(size=80), rng.normal(size=(80, 1)), lags=1, exog_names=["x"])
        >>> test.fit().nobs
        79
        """
        design = self._lagged_design()
        responses = self.exog.loc[:, list(self.tested_inputs)].add_prefix("response.")
        combined = pd.concat([responses, design], axis=1)
        combined = combined.iloc[self.lags :]
        if self.missing == "drop":
            combined = combined.dropna(axis=0, how="any")
        if combined.isna().any(axis=None):
            raise RuntimeError("feedback lag design unexpectedly contains missing values")

        x = combined.loc[:, design.columns]
        nobs, n_params = x.shape
        if nobs <= n_params:
            raise ValueError(
                "feedback regression has no positive residual degrees of freedom; "
                "reduce lags or provide more observations"
            )
        if np.linalg.matrix_rank(x.to_numpy()) < n_params:
            raise ValueError("feedback regression design matrix is rank deficient")

        y_lag_names = [
            f"{self.y.name}.L{lag}" for lag in range(1, self.lags + 1)
        ]
        restriction = np.zeros((self.lags, n_params))
        for row, name in enumerate(y_lag_names):
            restriction[row, x.columns.get_loc(name)] = 1.0

        equations = []
        residuals = []
        for input_name in self.tested_inputs:
            response = combined[f"response.{input_name}"]
            fitted = sm.OLS(response, x).fit()
            test = fitted.f_test(restriction)
            equation = FeedbackEquationResult(
                input_name=input_name,
                output_name=str(self.y.name),
                lags=self.lags,
                regression=fitted,
                f_statistic=float(np.asarray(test.fvalue).squeeze()),
                pvalue=float(np.asarray(test.pvalue).squeeze()),
                df_num=float(test.df_num),
                df_denom=float(test.df_denom),
                alpha=self.alpha,
                observation_index=combined.index.copy(),
            )
            equations.append(equation)
            residuals.append(np.asarray(fitted.resid, dtype=float))

        self.result_ = FeedbackTestResult(
            lags=self.lags,
            nobs=nobs,
            residuals=np.column_stack(residuals),
            equations=tuple(equations),
            alpha=self.alpha,
            output_name=str(self.y.name),
        )
        return self.result_
