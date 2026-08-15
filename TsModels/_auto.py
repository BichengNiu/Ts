"""Auto-estimation for SARIMAX and GARCH models.

Provides automatic optimal parameter selection via grid search over
user-specified parameter ranges and evaluation criteria.

Classes
-------
_BaseAutoModel
    Shared grid-search logic for AutoSARIMAX and AutoGARCH.
AutoModelResult
    Result container for auto-estimation (inherits BaseModelResult).
AutoSARIMAX
    Auto SARIMAX model selection via grid search.
AutoGARCH
    Auto GARCH model selection via grid search.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from Ts.TsUtils._validation import _resolve_missing_rows

from Ts.TsModels._base import (
    _GARCH_M_FORMS,
    _VOL_TYPES,
    BaseModel,
    BaseModelResult,
    _normalise_model_dates,
)
from Ts.TsModels._garch_base import _VolEvaluationMixin

_SUPPORTED_CRITERIA = frozenset({"aic", "bic", "hqic", "aicc"})


def _get_criterion_value(result: BaseModelResult, criterion: str) -> float:
    """Extract or compute the value of an information criterion.

    Reads ``aic`` / ``bic`` directly from *result*; computes ``hqic``
    and ``aicc`` from log-likelihood, parameter count, and sample size.

    Parameters
    ----------
    result : BaseModelResult
        Fitted model result.
    criterion : str
        One of ``"aic"``, ``"bic"``, ``"hqic"``, ``"aicc"``.

    Returns
    -------
    float
        Criterion value (lower is better).
    """
    if criterion == "aic":
        return float(result.aic)
    if criterion == "bic":
        return float(result.bic)

    k = len(result.params)
    n = result.nobs
    llf = result.log_likelihood

    if criterion == "hqic":
        import math

        return -2.0 * llf + 2.0 * k * math.log(max(1.0001, math.log(n)))

    if criterion == "aicc":
        correction = 2.0 * k * (k + 1.0) / max(1.0, n - k - 1.0)
        return float(result.aic) + correction

    raise ValueError(
        f"Unknown criterion: {criterion!r}. Supported: {sorted(_SUPPORTED_CRITERIA)}"
    )


@dataclass
class AutoModelResult(BaseModelResult):
    """Result container for automatic model selection.

    Inherits all estimation fields from :class:`BaseModelResult` (copied
    from the best model) and adds selection metadata.

    Use :meth:`from_search` to construct from a search result.

    Parameters
    ----------
    best_result : BaseModelResult
        The best model's full result object.
    best_order : tuple
        Optimal parameter combination (e.g. ``(1, 0, 0)`` for SARIMAX or
        ``(p, o, q)`` for GARCH / GJR-GARCH).
    candidate_results : list
        All successfully fitted :class:`BaseModelResult` objects.
    candidate_orders : list
        All parameter combinations that were attempted and succeeded.
    criterion_values : list
        Criterion value for each successfully fitted model.
    selection_criterion : str
        Criterion used for selection (``"aic"``, ``"bic"``, ``"hqic"``,
        ``"aicc"``).
    search_method : str
        Search strategy (``"grid"``).
    n_attempted : int, optional
        Number of order combinations attempted, including failed fits.
    best_seasonal_order : tuple, optional
        Seasonal order selected by :class:`AutoSARIMAX`, if seasonal search
        was enabled.
    candidate_seasonal_orders : list, optional
        Seasonal orders corresponding to successful AutoSARIMAX candidates.
    search_messages : list of str, optional
        Warnings and failed-fit messages collected during the search.
    model_type, params, std_errors, p_values : see BaseModelResult
    aic, bic, log_likelihood, residuals, fitted_values, nobs, data : see BaseModelResult
        Shared fields copied from the selected result.

    Examples
    --------
    >>> from Ts.TsModels import AutoModelResult, AutoSARIMAX
    >>> from Ts.TsSims import simulate_sarima
    >>> data = simulate_sarima(n=60, order=(1, 0, 0), seed=42).data
    >>> result = AutoSARIMAX(
    ...     data, p=(0, 1), d=(0, 0), q=(0, 0), P=(0, 0), D=(0, 0), Q=(0, 0)
    ... ).fit()
    >>> isinstance(result, AutoModelResult)
    True
    >>> result.best_order in {(0, 0, 0), (1, 0, 0)}
    True
    """

    best_result: BaseModelResult = field(default=None, repr=False)
    best_order: tuple = field(default=())
    candidate_results: list = field(default_factory=list, repr=False)
    candidate_orders: list = field(default_factory=list, repr=False)
    criterion_values: list = field(default_factory=list, repr=False)
    selection_criterion: str = ""
    search_method: str = ""
    n_attempted: int | None = None
    best_seasonal_order: tuple | None = None
    candidate_seasonal_orders: list = field(default_factory=list, repr=False)
    search_messages: list[str] = field(default_factory=list, repr=False)

    @classmethod
    def from_search(
        cls,
        best_result: BaseModelResult,
        best_order: tuple,
        candidate_results: list,
        candidate_orders: list,
        criterion_values: list,
        selection_criterion: str,
        search_method: str,
        n_attempted: int | None = None,
        best_seasonal_order: tuple | None = None,
        candidate_seasonal_orders: list | None = None,
        search_messages: list[str] | None = None,
    ) -> AutoModelResult:
        """Construct from a search loop outcome.

        Copies :class:`BaseModelResult` fields from *best_result* and
        stores selection metadata in the additional fields.

        Parameters
        ----------
        best_result : BaseModelResult
            Result of the best model found.
        best_order : tuple
            Parameter combination of the best model.
        candidate_results : list of BaseModelResult
            All successfully fitted results.
        candidate_orders : list of tuple
            Corresponding parameter combinations.
        criterion_values : list of float
            Criterion value for each candidate.
        selection_criterion : str
            Criterion name.
        search_method : str
            Search strategy name.
        n_attempted : int or None
            Total number of attempted candidates.
        best_seasonal_order : tuple or None
            Seasonal order of the selected model.
        candidate_seasonal_orders : list or None
            Seasonal orders corresponding to successful candidates.
        search_messages : list of str or None
            Diagnostics recorded for unsuccessful candidates.

        Returns
        -------
        AutoModelResult

        Examples
        --------
        >>> from Ts.TsModels import AutoModelResult, AutoSARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(n=60, seed=42).data
        >>> searched = AutoSARIMAX(data, p=(0, 0), d=(0, 0), q=(0, 0)).fit()
        >>> copied = AutoModelResult.from_search(
        ...     searched.best_result, searched.best_order,
        ...     searched.candidate_results, searched.candidate_orders,
        ...     searched.criterion_values, "aic", "grid",
        ... )
        >>> copied.best_order == searched.best_order
        True
        """
        return cls(
            model_type=best_result.model_type,
            params=best_result.params,
            std_errors=best_result.std_errors,
            p_values=best_result.p_values,
            aic=best_result.aic,
            bic=best_result.bic,
            log_likelihood=best_result.log_likelihood,
            residuals=best_result.residuals,
            fitted_values=best_result.fitted_values,
            nobs=best_result.nobs,
            data=best_result.data,
            best_result=best_result,
            best_order=best_order,
            candidate_results=candidate_results,
            candidate_orders=candidate_orders,
            criterion_values=criterion_values,
            selection_criterion=selection_criterion,
            search_method=search_method,
            n_attempted=n_attempted,
            best_seasonal_order=best_seasonal_order,
            candidate_seasonal_orders=(
                [] if candidate_seasonal_orders is None else candidate_seasonal_orders
            ),
            search_messages=[] if search_messages is None else search_messages,
        )

    def summary(self) -> str:
        """Return a formatted summary with selection header.

        Prepends auto-selection metadata (best order, criterion, success
        count) before the best model's full parameter table.

        Returns
        -------
        str

        Examples
        --------
        >>> from Ts.TsModels import AutoSARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(n=80, order=(1, 0, 0), ar=[0.5], seed=42).data
        >>> result = AutoSARIMAX(data, p=(0, 1), d=(0, 0), q=(0, 0)).fit()
        >>> isinstance(result.summary(), str)
        True
        """
        n_ok = len(self.candidate_results)
        n_total = n_ok if self.n_attempted is None else self.n_attempted

        lines = [
            f"Auto {self.model_type} Model Selection Result",
            "=" * 60,
            f"Search Method      : {self.search_method}",
            f"Selection Criterion: {self.selection_criterion.upper()}",
            f"Best Order         : {self.best_order}",
            f"Models Evaluated   : {n_ok}/{n_total} successful",
            f"Search Diagnostics : {len(self.search_messages)}",
            f"Best {self.selection_criterion.upper():>6s}        : {_get_criterion_value(self, self.selection_criterion):.4f}",
            "",
            "Best Model Details:",
            "-" * 60,
        ]
        if self.best_seasonal_order is not None:
            lines.insert(5, f"Best Seasonal Order: {self.best_seasonal_order}")
        if self.log:
            details_index = lines.index("Best Model Details:")
            lines.insert(
                details_index - 1,
                "Response Scale     : original (log fit; bias-adjusted mean)",
            )

        # Delegate to the base parameter table, which renders the shared
        # estimation fields identically for every model result.
        base_lines = super().summary().splitlines()

        return "\n".join(lines + base_lines)

    def predict(self, **kwargs):
        """Unified prediction: delegates to :meth:`best_result.predict`.

        Parameters
        ----------
        **kwargs
            Forwarded to the best model's ``predict()`` method.

        Returns
        -------
        PredictResult

        Examples
        --------
        >>> from Ts.TsModels import AutoSARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(n=60, seed=42).data
        >>> result = AutoSARIMAX(data, p=(0, 0), d=(0, 0), q=(0, 0)).fit()
        >>> result.predict(start=60, end=62).mean.shape
        (3,)
        """
        if self.best_result is None:
            raise RuntimeError("No best_result available")
        return self.best_result.predict(**kwargs)

    def parameter_correlation(self, parameters=None):
        """Return parameter correlations from the selected best model.

        Parameters
        ----------
        parameters : sequence of str, optional
            Ordered subset of parameter names. By default all estimated
            parameters from the selected model are returned.

        Returns
        -------
        pandas.DataFrame
            Correlation matrix delegated to ``best_result``.

        Examples
        --------
        >>> from Ts.TsModels import AutoSARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(n=60, seed=42).data
        >>> result = AutoSARIMAX(data, p=(0, 1), q=(0, 0)).fit()
        >>> result.parameter_correlation().shape[0] == len(result.params)
        True
        """
        if self.best_result is None:
            raise RuntimeError("No best_result available")
        return self.best_result.parameter_correlation(parameters=parameters)

    def plot_parameter_correlation(
        self,
        parameters=None,
        *,
        annotate=True,
        decimals=2,
        title=None,
        ax=None,
    ):
        """Plot parameter correlations from the selected best model.

        Parameters
        ----------
        parameters : sequence of str, optional
            Ordered subset of parameter names. By default all estimated
            parameters from the selected model are plotted.
        annotate : bool, default True
            Whether to write each correlation value inside its cell.
        decimals : int, default 2
            Number of decimal places used for cell annotations.
        title : str, optional
            Chart title. Uses the selected model's default when omitted.
        ax : matplotlib.axes.Axes, optional
            Existing axes. A new figure and axes are created when omitted.

        Returns
        -------
        tuple
            ``(figure, axes)`` containing the correlation heatmap.

        Examples
        --------
        >>> from Ts.TsModels import AutoGARCH
        >>> from Ts.TsSims import simulate_garch
        >>> data = simulate_garch(n=150, seed=42).data
        >>> result = AutoGARCH(data, p=(1, 1), q=(1, 1)).fit()
        >>> fig, ax = result.plot_parameter_correlation(annotate=False)
        """
        if self.best_result is None:
            raise RuntimeError("No best_result available")
        return self.best_result.plot_parameter_correlation(
            parameters=parameters,
            annotate=annotate,
            decimals=decimals,
            title=title,
            ax=ax,
        )

    @property
    def log(self):
        """Whether the selected model used a log-transformed response."""
        if self.best_result is None:
            return False
        return bool(getattr(self.best_result, "log", False))

    @property
    def level_intercept(self):
        """Delegate the fitted-response constant of the selected model."""
        if self.best_result is None:
            raise RuntimeError("No best_result available")
        return getattr(self.best_result, "level_intercept", None)

    @property
    def unconditional_log_variance(self):
        """Delegate the selected model's stationary log-response variance."""
        if self.best_result is None:
            raise RuntimeError("No best_result available")
        return getattr(self.best_result, "unconditional_log_variance", None)

    def level_intercept_inference(self, alpha=0.05):
        """Delegate fitted-response constant inference to the selected model.

        Parameters
        ----------
        alpha : float, default 0.05
            Two-sided significance level for the confidence interval.

        Returns
        -------
        dict or None
            Estimate and delta-method inference on the selected model's fitted
            response scale. With ``log=True``, that scale is the natural log.

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsModels import AutoSARIMAX
        >>> data = np.exp(np.linspace(1.0, 2.0, 40))
        >>> result = AutoSARIMAX(
        ...     data, p=(0, 0), d=(0, 0), q=(0, 0), log=True
        ... ).fit()
        >>> result.level_intercept_inference()["estimate"] == result.level_intercept
        True
        """
        if self.best_result is None:
            raise RuntimeError("No best_result available")
        method = getattr(self.best_result, "level_intercept_inference", None)
        if method is None:
            raise TypeError(
                "level_intercept_inference is only available for "
                "AutoSARIMAX results"
            )
        return method(alpha=alpha)

    @property
    def distributed_lags(self):
        """Delegate structured rational distributed-lag results."""
        if self.best_result is None:
            return {}
        return getattr(self.best_result, "distributed_lags", {})

    @property
    def distributed_lag_coefficients(self):
        """Delegate the complete transfer-polynomial coefficient table."""
        if self.best_result is None:
            return pd.DataFrame()
        return getattr(
            self.best_result,
            "distributed_lag_coefficients",
            pd.DataFrame(),
        )

    @property
    def steady_state_gains(self):
        """Delegate per-input steady-state gains."""
        if self.best_result is None:
            return pd.DataFrame()
        return getattr(self.best_result, "steady_state_gains", pd.DataFrame())

    def weights(self, steps):
        """Delegate rational distributed-lag impulse weights.

        Parameters
        ----------
        steps : int
            Strictly positive response horizon.

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        >>> from Ts.TsModels import AutoModelResult, RationalLagSpec, SARIMAX
        >>> from Ts.TsSims import RDLInputSpec, simulate_rdl
        >>> simulated = simulate_rdl(
        ...     n=80, distributed_lags={"x": RDLInputSpec({0: 1.0}, {1: .3})}, seed=42
        ... )
        >>> best = SARIMAX(
        ...     simulated.data, exog=simulated.exog,
        ...     distributed_lags={"x": RationalLagSpec(0, 1)},
        ... ).fit()
        >>> result = AutoModelResult.from_search(
        ...     best, (0, 0, 0), [best], [(0, 0, 0)], [best.aic], "aic", "grid"
        ... )
        >>> result.weights(3).shape
        (3, 1)
        """
        if self.best_result is None:
            raise RuntimeError("No best_result available")
        method = getattr(self.best_result, "weights", None)
        if method is None:
            raise TypeError("weights is only available for AutoSARIMAX RDL results")
        return method(steps)

    def feedback_test(self, lags, inputs=None, **kwargs):
        """Delegate conditional feedback testing to the selected model.

        Parameters
        ----------
        lags : int
            Positive common lag order.
        inputs : str or sequence of str, optional
            Original exogenous inputs to test.
        **kwargs
            Additional options forwarded to the selected model's
            :meth:`feedback_test` method.

        Returns
        -------
        FeedbackTestResult
            Full OLS regressions and joint feedback F tests.

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsModels import AutoSARIMAX
        >>> rng = np.random.default_rng(42)
        >>> result = AutoSARIMAX(rng.normal(size=60), exog=rng.normal(size=(60, 1)), exog_names=["x"], p=(0, 0), d=(0, 0), q=(0, 0)).fit()
        >>> result.feedback_test(1).input_names
        ('x',)
        """
        if self.best_result is None:
            raise RuntimeError("No best_result available")
        method = getattr(self.best_result, "feedback_test", None)
        if method is None:
            raise TypeError("selected model does not support feedback testing")
        return method(lags, inputs=inputs, **kwargs)

    def residual_ccf_test(
        self,
        input_models,
        lags=12,
        inputs=None,
        **kwargs,
    ):
        """Delegate RDL residual cross-correlation testing to the best model.

        Parameters
        ----------
        input_models : mapping
            Explicit fitted input prewhitening models keyed by RDL input.
        lags : int, default 12
            Positive maximum residual cross-correlation lag.
        inputs : str or sequence of str, optional
            RDL inputs to test.
        **kwargs
            Additional options forwarded to the selected model's
            :meth:`residual_ccf_test` method.

        Returns
        -------
        ResidualCCFTestResult
            Per-input residual CCFs and joint S* tests.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from Ts.TsModels import AutoSARIMAX, RationalLagSpec, SARIMAX
        >>> rng = np.random.default_rng(42)
        >>> x = pd.Series(rng.normal(size=60), name="x")
        >>> y = x.to_numpy() + rng.normal(scale=0.2, size=60)
        >>> result = AutoSARIMAX(y, exog=x, p=(0, 0), d=(0, 0), q=(0, 0), P=(0, 0), D=(0, 0), Q=(0, 0), trend="n", distributed_lags={"x": RationalLagSpec()}).fit()
        >>> input_model = SARIMAX(x, trend="n").fit()
        >>> result.residual_ccf_test({"x": input_model}, lags=3).input_names
        ('x',)
        """
        if self.best_result is None:
            raise RuntimeError("No best_result available")
        method = getattr(self.best_result, "residual_ccf_test", None)
        if method is None:
            raise TypeError(
                "residual_ccf_test is only available for AutoSARIMAX RDL results"
            )
        return method(
            input_models,
            lags=lags,
            inputs=inputs,
            **kwargs,
        )

    def plot_impulse_response(
        self,
        steps=None,
        inputs=None,
        sample_weights=None,
        **kwargs,
    ):
        """Delegate RDL impulse-response plotting to the selected model.

        Parameters
        ----------
        steps : int, optional
            Strictly positive response horizon. When omitted, the selected
            model infers it from ``sample_weights`` if supplied and otherwise
            defaults to 20.
        inputs : str or sequence of str, optional
            RDL inputs to plot.
        sample_weights : pandas.Series or pandas.DataFrame, optional
            Preliminary finite-lag estimates to plot as bars beneath the
            selected model's transfer-function weight lines.
        **kwargs
            Additional options forwarded to the selected model's
            :meth:`plot_impulse_response` method.

        Returns
        -------
        tuple
            Matplotlib figure and axis or axes.

        Examples
        --------
        >>> from Ts.TsModels import AutoModelResult, RationalLagSpec, SARIMAX
        >>> from Ts.TsSims import RDLInputSpec, simulate_rdl
        >>> simulated = simulate_rdl(n=60, distributed_lags={"x": RDLInputSpec({0: 1.0}, {})}, seed=42)
        >>> best = SARIMAX(simulated.data, exog=simulated.exog, distributed_lags={"x": RationalLagSpec()}).fit()
        >>> result = AutoModelResult.from_search(best, (0, 0, 0), [best], [(0, 0, 0)], [best.aic], "aic", "grid")
        >>> fig, ax = result.plot_impulse_response(3)
        """
        if self.best_result is None:
            raise RuntimeError("No best_result available")
        method = getattr(self.best_result, "plot_impulse_response", None)
        if method is None:
            raise TypeError("selected model does not support RDL impulse responses")
        return method(
            steps,
            inputs=inputs,
            sample_weights=sample_weights,
            **kwargs,
        )

    def long_run_equilibrium(self):
        """Return the long-run equilibrium of the best model.

        Delegates to :meth:`BaseModelResult.long_run_equilibrium` of the
        wrapped best result.

        Returns
        -------
        float or np.ndarray or None

        Examples
        --------
        >>> from Ts.TsModels import AutoSARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(n=100, order=(1, 0, 0), ar=[.5], seed=42).data
        >>> result = AutoSARIMAX(data, p=(1, 1), d=(0, 0), q=(0, 0)).fit()
        >>> result.long_run_equilibrium() is None
        False
        """
        if self.best_result is not None:
            return self.best_result.long_run_equilibrium()
        return None

    def cycle_period(self, *, seasonal=False):
        """Return the AR(2) cycle diagnostic of the selected SARIMAX model.

        Parameters
        ----------
        seasonal : bool, default False
            Inspect the seasonal rather than non-seasonal AR component.

        Returns
        -------
        ARCycleResult

        Examples
        --------
        >>> from Ts.TsModels import AutoSARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(n=150, order=(2, 0, 0), ar=[1.1, -.5], seed=42).data
        >>> result = AutoSARIMAX(data, p=(2, 2), d=(0, 0), q=(0, 0)).fit()
        >>> result.cycle_period().component
        'nonseasonal'
        """
        if self.best_result is None:
            raise RuntimeError("No best_result available")
        method = getattr(self.best_result, "cycle_period", None)
        if method is None:
            raise TypeError("cycle_period is only available for AutoSARIMAX results")
        return method(seasonal=seasonal)


_SUPPORTED_METHODS = frozenset({"grid"})


class _BaseAutoModel(BaseModel):
    """Shared grid-search logic for AutoSARIMAX and AutoGARCH."""

    def _validate_params(self, data, criterion, method):
        """Common validation shared by subclasses.

        Parameters
        ----------
        data : array-like
            Time series data.
        criterion : str
            Selection criterion.
        method : str
            Search strategy.

        Returns
        -------
        np.ndarray
            Cleaned 1-D data array.
        """
        y = np.asarray(data, dtype=float).ravel()
        if not np.all(np.isfinite(y)):
            raise ValueError("data must contain only finite values")
        if len(y) < 10:
            raise ValueError(f"Need at least 10 observations, got {len(y)}")
        if criterion not in _SUPPORTED_CRITERIA:
            raise ValueError(
                f"criterion must be one of {sorted(_SUPPORTED_CRITERIA)}, "
                f"got {criterion!r}"
            )
        if method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {sorted(_SUPPORTED_METHODS)}, got {method!r}"
            )
        return y

    def _run_grid_search(self, y, orders, build_model, criterion):
        """Common grid search loop.

        Parameters
        ----------
        y : np.ndarray
            1D array of data.
        orders : iterable of tuple
            Parameter combinations to try.
        build_model : callable(tuple) -> BaseModel
            Factory that creates a model from an order tuple.
        criterion : str
            Selection criterion.

        Returns
        -------
        tuple
            (best_result, best_order, candidate_results, candidate_orders,
            n_attempted, search_messages)
        """
        import warnings

        best_result = None
        best_order = None
        best_value = float("inf")
        candidate_results = []
        candidate_orders = []

        n_attempted = 0
        search_messages = []
        for order in orders:
            n_attempted += 1
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                try:
                    model = build_model(order)
                    result = model.fit()
                    value = _get_criterion_value(result, criterion)
                except Exception as error:
                    search_messages.append(f"{order}: {type(error).__name__}: {error}")
                    continue
            search_messages.extend(
                f"{order}: {warning.category.__name__}: {warning.message}"
                for warning in caught_warnings
            )
            candidate_results.append(result)
            candidate_orders.append(order)
            if value < best_value:
                best_value = value
                best_result = result
                best_order = order

        if best_result is None:
            raise RuntimeError("No model converged during grid search")

        return (
            best_result,
            best_order,
            candidate_results,
            candidate_orders,
            n_attempted,
            search_messages,
        )

    @staticmethod
    def _validate_range(rng, name, min_val=0):
        """Validate and convert a (min, max) range tuple."""
        if not isinstance(rng, (tuple, list)) or len(rng) != 2:
            raise ValueError(f"{name} must be a (min, max) tuple, got {rng}")
        lo, hi = int(rng[0]), int(rng[1])
        if lo < min_val:
            raise ValueError(f"{name} min must be >= {min_val}, got {lo}")
        if lo > hi:
            raise ValueError(f"{name}: min ({lo}) must be <= max ({hi})")
        return (lo, hi)


class AutoSARIMAX(_BaseAutoModel):
    """Auto SARIMAX model selection via grid search.

    Searches over a user-specified range of ``(p, d, q)`` and optional
    ``(P, D, Q, s)`` combinations, selecting the order that minimises
    the chosen information criterion.

    Parameters
    ----------
    data : array-like
        Time series data (1-D).
    p : tuple[int, int]
        AR order range ``(min, max)``. Default ``(0, 3)``.
    d : tuple[int, int]
        Differencing order range ``(min, max)``. Default ``(0, 1)``.
    q : tuple[int, int]
        MA order range ``(min, max)``. Default ``(0, 3)``.
    P : tuple[int, int]
        Seasonal AR range. Default ``(0, 1)``.
    D : tuple[int, int]
        Seasonal differencing range. Default ``(0, 1)``.
    Q : tuple[int, int]
        Seasonal MA range. Default ``(0, 1)``.
    s : int
        Seasonal period (0 = no seasonality). Default ``0``.
    trend : str
        Trend: ``"n"``, ``"c"``, ``"t"``, ``"ct"``. Default ``"c"``.
    criterion : str
        ``"aic"``, ``"bic"``, ``"hqic"``, or ``"aicc"``.
    method : str
        Search strategy. Currently only ``"grid"``.
    missing : {"raise", "drop"}
        Non-finite input policy. ``"drop"`` records removed zero-based rows
        in :attr:`dropped_positions`. Default ``"drop"``; use ``"raise"``
        to reject any sample change.
    log : bool
        Fit every candidate to the natural logarithm of the response. The
        input must be on its original positive scale. Fitted values,
        predictions, and intervals are returned on the original scale;
        prediction means use the horizon-specific lognormal bias correction.
        This is a fixed setting, not an automatically searched dimension.
        Default ``False``.
    dates : datetime-like sequence, optional
        Strict sample dates. A Series DatetimeIndex is inferred automatically.
        Array inputs may provide dates explicitly.
    exog : pandas.Series, pandas.DataFrame, or array-like, optional
        Ordinary exogenous variables shared by every candidate. A named
        Series or one-dimensional array represents one input. A dated Series
        or DataFrame may include rows after the estimation sample for default
        future forecasting.
    exog_names : sequence[str], optional
        Required column names for array exog and for an unnamed Series. Named
        Series and DataFrame labels are authoritative and must not be
        overridden.
    events : sequence[EventSpec], optional
        Event designs shared by every candidate.
    enforce_stationarity : bool
        Whether every candidate enforces AR stationarity. Default ``True``.
    enforce_invertibility : bool
        Whether every candidate enforces MA invertibility. Default ``True``.
    distributed_lags : mapping[str, RationalLagSpec], optional
        Fixed rational distributed-lag specifications shared by every
        candidate. Their orders are not included in the automatic search.
    enforce_distributed_lag_stability : bool
        Whether all candidates enforce and diagnose complete transfer
        denominator stability. Default ``True``.

    Examples
    --------
    Restrict ranges when the admissible model family is known.

    >>> from Ts.TsModels import AutoSARIMAX
    >>> from Ts.TsSims import simulate_sarima
    >>> data = simulate_sarima(n=70, order=(1, 0, 0), seed=42).data
    >>> result = AutoSARIMAX(
    ...     data, p=(0, 1), d=(0, 0), q=(0, 0), P=(0, 0), D=(0, 0), Q=(0, 0)
    ... ).fit()
    >>> len(result.candidate_orders)
    2
    """

    def __init__(
        self,
        data,
        p=(0, 3),
        d=(0, 1),
        q=(0, 3),
        P=(0, 1),
        D=(0, 1),
        Q=(0, 1),
        s=0,
        trend="c",
        criterion="aic",
        method="grid",
        *,
        dates=None,
        exog=None,
        exog_names=None,
        events=None,
        enforce_stationarity=True,
        enforce_invertibility=True,
        missing="drop",
        log=False,
        distributed_lags=None,
        enforce_distributed_lag_stability=True,
    ):
        from Ts.TsModels._sarimax import SARIMAX

        prototype = SARIMAX(
            data,
            order=(0, 0, 0),
            seasonal_order=(0, 0, 0, 0),
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            dates=dates,
            exog=exog,
            exog_names=exog_names,
            events=events,
            missing=missing,
            log=log,
            distributed_lags=distributed_lags,
            enforce_distributed_lag_stability=(enforce_distributed_lag_stability),
        )

        self.data = self._validate_params(prototype.data, criterion, method)
        self.missing = missing
        self.dropped_positions = prototype.dropped_positions
        self.log = prototype.log
        self.criterion = criterion
        self.method = method
        self.dates = None if prototype.dates is None else prototype.dates.copy()
        self.exog = None if prototype.exog is None else prototype.exog.copy()
        self.exog_names = prototype.exog_names
        self.future_exog = (
            None if prototype.future_exog is None else prototype.future_exog.copy()
        )
        self.events = prototype.events
        self.distributed_lags = prototype.distributed_lags
        self.enforce_distributed_lag_stability = (
            prototype.enforce_distributed_lag_stability
        )

        self.p_range = self._validate_range(p, "p")
        self.d_range = self._validate_range(d, "d")
        self.q_range = self._validate_range(q, "q")
        self.P_range = self._validate_range(P, "P")
        self.D_range = self._validate_range(D, "D")
        self.Q_range = self._validate_range(Q, "Q")
        self.s = s
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

    def _candidate_exog(self):
        """Return copied historical and default-future exog for one candidate."""
        if self.exog is None:
            return None
        if self.dates is None:
            return self.exog.copy()

        historical = pd.DataFrame(
            self.exog,
            index=self.dates.copy(),
            columns=self.exog_names,
        )
        if self.future_exog is None:
            return historical
        return pd.concat([historical, self.future_exog.copy()])

    def _clone_for_evaluation(self, data, exog=None, *, dates=None):
        """Rebuild automatic-selection state for one isolated training window."""
        return type(self)(
            data,
            p=self.p_range,
            d=self.d_range,
            q=self.q_range,
            P=self.P_range,
            D=self.D_range,
            Q=self.Q_range,
            s=self.s,
            trend=self.trend,
            criterion=self.criterion,
            method=self.method,
            dates=dates,
            exog=exog,
            exog_names=self.exog_names if exog is not None else None,
            events=self.events,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            missing="raise",
            log=self.log,
            distributed_lags=self.distributed_lags,
            enforce_distributed_lag_stability=(self.enforce_distributed_lag_stability),
        )

    def _evaluation_predict_kwargs(self, start, stop):
        """Return aligned future exogenous values and dates for evaluation."""
        from Ts.TsModels._sarimax import SARIMAX

        return SARIMAX._evaluation_predict_kwargs(self, start, stop)

    def fit(self):
        """Run grid search and return the best model.

        Returns
        -------
        AutoModelResult

        Examples
        --------
        >>> from Ts.TsModels import AutoModelResult, AutoSARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(n=60, seed=42).data
        >>> searched = AutoSARIMAX(data, p=(0, 0), d=(0, 0), q=(0, 0)).fit()
        >>> copied = AutoModelResult.from_search(
        ...     searched.best_result, searched.best_order,
        ...     searched.candidate_results, searched.candidate_orders,
        ...     searched.criterion_values, "aic", "grid",
        ... )
        >>> copied.best_order == searched.best_order
        True

        Examples
        --------
        >>> from Ts.TsModels import AutoSARIMAX
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(n=60, seed=42).data
        >>> result = AutoSARIMAX(
        ...     data, p=(0, 1), d=(0, 0), q=(0, 0),
        ...     P=(0, 0), D=(0, 0), Q=(0, 0),
        ... ).fit()
        >>> result.best_result is not None
        True
        """
        import itertools

        from Ts.TsModels._sarimax import SARIMAX

        p_lo, p_hi = self.p_range
        d_lo, d_hi = self.d_range
        q_lo, q_hi = self.q_range
        P_lo, P_hi = self.P_range
        D_lo, D_hi = self.D_range
        Q_lo, Q_hi = self.Q_range

        nonseasonal = itertools.product(
            range(p_lo, p_hi + 1),
            range(d_lo, d_hi + 1),
            range(q_lo, q_hi + 1),
        )

        if self.s > 0:
            seasonal = itertools.product(
                range(P_lo, P_hi + 1),
                range(D_lo, D_hi + 1),
                range(Q_lo, Q_hi + 1),
                [self.s],
            )
        else:
            seasonal = ((0, 0, 0, 0),)

        # Each order is a (nonseasonal, seasonal) pair
        orders = itertools.product(nonseasonal, seasonal)
        candidate_exog = self._candidate_exog()

        def build_model(order_tuple):
            ns_order, s_order = order_tuple
            return SARIMAX(
                self.data,
                order=ns_order,
                seasonal_order=s_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
                dates=self.dates,
                exog=candidate_exog,
                exog_names=(
                    self.exog_names
                    if candidate_exog is not None
                    and not isinstance(candidate_exog, pd.DataFrame)
                    else None
                ),
                events=self.events,
                missing="raise",
                log=self.log,
                distributed_lags=self.distributed_lags,
                enforce_distributed_lag_stability=(
                    self.enforce_distributed_lag_stability
                ),
            )

        (
            best_result,
            best_order_pair,
            candidate_results,
            candidate_orders,
            n_attempted,
            search_messages,
        ) = self._run_grid_search(self.data, orders, build_model, self.criterion)

        criterion_values = [
            _get_criterion_value(r, self.criterion) for r in candidate_results
        ]

        # Extract nonseasonal order for reporting
        best_order = best_order_pair[0]
        flat_orders = [o[0] for o in candidate_orders]
        seasonal_orders = [o[1] for o in candidate_orders]

        auto_result = AutoModelResult.from_search(
            best_result=best_result,
            best_order=best_order,
            candidate_results=candidate_results,
            candidate_orders=flat_orders,
            criterion_values=criterion_values,
            selection_criterion=self.criterion,
            search_method=self.method,
            n_attempted=n_attempted,
            best_seasonal_order=best_order_pair[1] if self.s > 0 else None,
            candidate_seasonal_orders=seasonal_orders if self.s > 0 else None,
            search_messages=search_messages,
        )

        self.result_ = auto_result
        return auto_result


class AutoGARCH(_VolEvaluationMixin, _BaseAutoModel):
    """Auto GARCH-family model selection via grid search.

    Searches over a user-specified range of ``(p, o, q)`` combinations,
    selecting the order that minimises the chosen information criterion.

    Supports all GARCH family variants:
    - Standard GARCH / ARCH / GJR-GARCH (``vol="GARCH"``, default)
    - EGARCH (``vol="EGARCH"``)
    - IGARCH (``igarch=True``)
    - GARCH-M / ARCH-in-mean (``garch_m=True``)

    Parameters
    ----------
    data : array-like
        Time series data (1-D).
    p : tuple[int, int]
        ARCH order range ``(min, max)``. Default ``(1, 4)``.
    q : tuple[int, int]
        GARCH order range ``(min, max)``. ``(0, 0)`` = pure ARCH only.
        Default ``(0, 4)``.
    o : tuple[int, int]
        Asymmetric (GJR) order range ``(min, max)``. Default ``(0, 0)``
        (standard symmetric GARCH only). Set to e.g. ``(0, 1)`` to
        compare symmetric vs asymmetric models.
    vol : str
        Volatility model type: ``"GARCH"`` (default, covers ARCH/GARCH/
        GJR-GARCH) or ``"EGARCH"``.
    mean : str
        Mean model: ``"Constant"``, ``"Zero"``, ``"AR"``, etc.
    dist : str
        Innovation distribution: ``"normal"``, ``"t"``, ``"skewt"``,
        ``"ged"``.
    criterion : str
        ``"aic"``, ``"bic"``, ``"hqic"``, or ``"aicc"``.
    method : str
        Search strategy. Currently only ``"grid"``.
    igarch : bool
        IGARCH constraint estimation (``sum(alpha)+sum(beta)=1``).
        Default ``False``.  Mutually exclusive with ``vol="EGARCH"``
        and ``garch_m=True``.
    garch_m : bool
        GARCH-in-Mean (ARCH-in-mean). Default ``False``.  Mutually
        exclusive with ``vol="EGARCH"`` and ``igarch=True``.
    garch_m_form : str
        Form for GARCH-M: ``"vol"``, ``"var"``, or ``"log"``.
        Default ``"vol"``.  Only used when ``garch_m=True``.
    ar_lags : int or list, optional
        AR lags for the mean equation (only used with ``garch_m=True``).
    exog : array-like, optional
        Exogenous regressors for the mean equation.
    dates : datetime-like sequence, optional
        Strict sample dates. A Series DatetimeIndex is inferred automatically.
        Array inputs may provide dates explicitly.
    missing : {"raise", "drop"}
        Joint non-finite policy for data and exog. ``"drop"`` records removed
        zero-based rows in :attr:`dropped_positions`. Default ``"drop"``;
        use ``"raise"`` to reject any sample change.

    Examples
    --------
    >>> from Ts.TsModels import AutoGARCH
    >>> from Ts.TsSims import simulate_garch
    >>> data = simulate_garch(n=150, p=1, q=1, seed=42).data
    >>> result = AutoGARCH(data, p=(1, 1), q=(0, 1), o=(0, 0)).fit()
    >>> len(result.candidate_orders)
    2
    """

    def __init__(
        self,
        data,
        p=(1, 4),
        q=(0, 4),
        o=(0, 0),
        vol="GARCH",
        mean="Constant",
        dist="normal",
        criterion="aic",
        method="grid",
        igarch=False,
        garch_m=False,
        garch_m_form="vol",
        ar_lags=None,
        exog=None,
        dates=None,
        missing="drop",
    ):
        raw_data = np.asarray(data, dtype=float).ravel()
        model_dates = _normalise_model_dates(data, dates, len(raw_data))
        if exog is not None:
            exog = np.asarray(exog, dtype=float)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            if exog.shape[0] != len(raw_data):
                raise ValueError(
                    f"exog must have {len(raw_data)} rows (same as data), "
                    f"got {exog.shape[0]}"
                )

        valid_rows = np.isfinite(raw_data)
        if exog is not None:
            valid_rows &= np.all(np.isfinite(exog), axis=1)
        dropped_positions = _resolve_missing_rows(
            valid_rows,
            missing,
            name="data or exog",
        )
        if missing == "drop":
            raw_data = raw_data[valid_rows]
            exog = None if exog is None else exog[valid_rows]
            if model_dates is not None:
                model_dates = model_dates[valid_rows].copy()
        self.missing = missing
        self.dropped_positions = dropped_positions
        self.dates = model_dates
        self.data = self._validate_params(raw_data, criterion, method)
        self.criterion = criterion
        self.method = method

        # --- GARCH-specific validation ---
        if vol not in _VOL_TYPES:
            raise ValueError(f"vol must be one of {sorted(_VOL_TYPES)}, got {vol!r}")
        if garch_m and garch_m_form not in _GARCH_M_FORMS:
            raise ValueError(
                f"garch_m_form must be one of {sorted(_GARCH_M_FORMS)}, "
                f"got {garch_m_form!r}"
            )
        if igarch and vol == "EGARCH":
            raise ValueError(
                "IGARCH is not supported for EGARCH models. "
                "Use vol='GARCH' for IGARCH estimation."
            )
        if igarch and garch_m:
            raise ValueError("IGARCH is not supported for GARCH-M (garch_m=True).")
        if garch_m and vol == "EGARCH":
            raise ValueError(
                "GARCH-M is not supported for EGARCH models. "
                "Use vol='GARCH' for ARCH-in-mean estimation."
            )

        self.p_range = self._validate_range(p, "p", min_val=1)
        self.q_range = self._validate_range(q, "q", min_val=0)
        self.o_range = self._validate_range(o, "o", min_val=0)
        self.vol = vol
        self.mean = mean
        self.dist = dist
        self.igarch = igarch
        self.garch_m = garch_m
        self.garch_m_form = garch_m_form
        self.ar_lags = ar_lags
        self.exog = exog

    def fit(self):
        """Run grid search and return the best model.

        Returns
        -------
        AutoModelResult

        Examples
        --------
        >>> from Ts.TsModels import AutoGARCH
        >>> from Ts.TsSims import simulate_garch
        >>> data = simulate_garch(n=120, seed=42).data
        >>> result = AutoGARCH(data, p=(1, 1), q=(0, 1), o=(0, 0)).fit()
        >>> result.best_result is not None
        True
        """
        import itertools

        from Ts.TsModels._garch import GARCH

        p_lo, p_hi = self.p_range
        q_lo, q_hi = self.q_range
        o_lo, o_hi = self.o_range

        orders = itertools.product(
            range(p_lo, p_hi + 1),
            range(o_lo, o_hi + 1),
            range(q_lo, q_hi + 1),
        )

        def build_model(order):
            return GARCH(
                self.data,
                p=order[0],
                o=order[1],
                q=order[2],
                vol=self.vol,
                mean=self.mean,
                dist=self.dist,
                igarch=self.igarch,
                garch_m=self.garch_m,
                garch_m_form=self.garch_m_form,
                ar_lags=self.ar_lags,
                exog=self.exog,
                dates=self.dates,
                compare_lags=False,
                missing="raise",
            )

        (
            best_result,
            best_order,
            candidate_results,
            candidate_orders,
            n_attempted,
            search_messages,
        ) = self._run_grid_search(self.data, orders, build_model, self.criterion)

        criterion_values = [
            _get_criterion_value(r, self.criterion) for r in candidate_results
        ]

        auto_result = AutoModelResult.from_search(
            best_result=best_result,
            best_order=best_order,
            candidate_results=candidate_results,
            candidate_orders=candidate_orders,
            criterion_values=criterion_values,
            selection_criterion=self.criterion,
            search_method=self.method,
            n_attempted=n_attempted,
            search_messages=search_messages,
        )

        self.result_ = auto_result
        return auto_result
