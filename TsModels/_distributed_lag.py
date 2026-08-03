"""Rational distributed-lag specifications and derived result quantities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import lfilter, lfilter_zi
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX as StatsmodelsSARIMAX
from statsmodels.tsa.statespace.tools import (
    constrain_stationary_univariate,
    unconstrain_stationary_univariate,
)


_RDL_INITIALIZATIONS = frozenset({"zero", "steady_state"})


def _normalise_nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _normalise_polynomial_lags(value, name, *, include_zero, allow_empty):
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        order = _normalise_nonnegative_integer(value, name)
        start = 0 if include_zero else 1
        return tuple(range(start, order + 1))
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an integer or an iterable of lags")
    try:
        lags = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer or an iterable of lags") from error
    if not lags:
        if allow_empty:
            return ()
        raise ValueError(f"{name} must contain at least one active lag")

    normalised = []
    for lag in lags:
        if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)):
            kind = "non-negative" if include_zero else "positive"
            raise TypeError(f"{name} lags must be {kind} integers")
        lag = int(lag)
        if (include_zero and lag < 0) or (not include_zero and lag <= 0):
            kind = "non-negative" if include_zero else "positive"
            raise ValueError(f"{name} lags must be {kind} integers")
        normalised.append(lag)
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"{name} lags must be unique")
    return tuple(sorted(normalised))


def _rdl_parameter_name(input_name, component, lag):
    token = "omega" if component == "numerator" else "delta"
    return f"rdl.{input_name}.{token}.L{lag}"


@dataclass(frozen=True)
class RationalLagSpec:
    """Immutable rational distributed-lag specification for one input.

    Integer numerator and denominator values denote contiguous polynomial
    orders. Iterables denote active sparse lags; omitted lags through the
    maximum active lag are fixed at zero.
    """

    numerator: object = 0
    denominator: object = 0
    delay: int = 0
    initialization: str = "zero"
    numerator_lags: tuple[int, ...] = field(init=False)
    denominator_lags: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        numerator_lags = _normalise_polynomial_lags(
            self.numerator,
            "numerator",
            include_zero=True,
            allow_empty=False,
        )
        denominator_lags = _normalise_polynomial_lags(
            self.denominator,
            "denominator",
            include_zero=False,
            allow_empty=True,
        )
        delay = _normalise_nonnegative_integer(self.delay, "delay")
        if self.initialization not in _RDL_INITIALIZATIONS:
            raise ValueError("initialization must be 'zero' or 'steady_state'")

        object.__setattr__(self, "numerator_lags", numerator_lags)
        object.__setattr__(self, "denominator_lags", denominator_lags)
        object.__setattr__(self, "delay", delay)

    @property
    def fixed_numerator_lags(self):
        maximum = max(self.numerator_lags)
        active = set(self.numerator_lags)
        return tuple(lag for lag in range(maximum + 1) if lag not in active)

    @property
    def fixed_denominator_lags(self):
        if not self.denominator_lags:
            return ()
        maximum = max(self.denominator_lags)
        active = set(self.denominator_lags)
        return tuple(lag for lag in range(1, maximum + 1) if lag not in active)

    def parameter_names(self, input_name):
        names = [
            _rdl_parameter_name(input_name, "numerator", lag)
            for lag in self.numerator_lags
        ]
        names.extend(
            _rdl_parameter_name(input_name, "denominator", lag)
            for lag in self.denominator_lags
        )
        return tuple(names)


def _filter_input(values, numerator, denominator, *, initialization):
    """Filter one input using explicit pre-sample initialization semantics."""
    values = np.asarray(values)
    numerator = np.asarray(numerator)
    denominator = np.asarray(denominator)
    if initialization == "zero":
        return lfilter(numerator, denominator, values)
    if initialization != "steady_state":
        raise ValueError("initialization must be 'zero' or 'steady_state'")

    initial_state = lfilter_zi(numerator, denominator) * values[0]
    if initial_state.size == 0:
        return lfilter(numerator, denominator, values)
    filtered, _ = lfilter(
        numerator,
        denominator,
        values,
        zi=initial_state,
    )
    return filtered


def _coefficient_arrays(spec, numerator, denominator):
    numerator_order = spec.delay + max(spec.numerator_lags)
    numerator_array = np.zeros(
        numerator_order + 1, dtype=np.result_type(*numerator.values())
    )
    for lag, value in numerator.items():
        numerator_array[spec.delay + lag] = value

    denominator_order = max(spec.denominator_lags, default=0)
    denominator_array = np.zeros(
        denominator_order + 1,
        dtype=np.result_type(1.0, *denominator.values()),
    )
    denominator_array[0] = 1.0
    for lag, value in denominator.items():
        denominator_array[lag] = -value
    return numerator_array, denominator_array


@dataclass
class RationalLagResult:
    """Estimated transfer-function quantities for one named input."""

    name: str
    spec: RationalLagSpec
    numerator: dict[int, float]
    denominator: dict[int, float]
    std_errors: dict[str, float] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)
    covariance: np.ndarray | None = None
    covariance_names: tuple[str, ...] = ()

    def __post_init__(self):
        self.numerator = dict(self.numerator)
        self.denominator = dict(self.denominator)
        self.std_errors = dict(self.std_errors)
        self.p_values = dict(self.p_values)
        self.covariance_names = tuple(self.covariance_names)
        if self.covariance is not None:
            covariance = np.asarray(self.covariance, dtype=float)
            expected = len(self.covariance_names)
            if covariance.shape != (expected, expected):
                raise ValueError("covariance shape must match covariance_names")
            self.covariance = covariance.copy()

    @property
    def fixed_params(self):
        fixed = {
            _rdl_parameter_name(self.name, "numerator", lag): 0.0
            for lag in self.spec.fixed_numerator_lags
        }
        fixed.update(
            {
                _rdl_parameter_name(self.name, "denominator", lag): 0.0
                for lag in self.spec.fixed_denominator_lags
            }
        )
        return fixed

    @property
    def coefficients(self):
        rows = []
        max_numerator = max(self.spec.numerator_lags)
        for lag in range(max_numerator + 1):
            parameter = _rdl_parameter_name(self.name, "numerator", lag)
            fixed = lag not in self.spec.numerator_lags
            rows.append(
                {
                    "input": self.name,
                    "component": "numerator",
                    "lag": lag,
                    "parameter": parameter,
                    "estimate": 0.0 if fixed else float(self.numerator[lag]),
                    "standard_error": (
                        np.nan if fixed else self.std_errors.get(parameter, np.nan)
                    ),
                    "p_value": np.nan
                    if fixed
                    else self.p_values.get(parameter, np.nan),
                    "fixed": fixed,
                }
            )
        max_denominator = max(self.spec.denominator_lags, default=0)
        for lag in range(1, max_denominator + 1):
            parameter = _rdl_parameter_name(self.name, "denominator", lag)
            fixed = lag not in self.spec.denominator_lags
            rows.append(
                {
                    "input": self.name,
                    "component": "denominator",
                    "lag": lag,
                    "parameter": parameter,
                    "estimate": 0.0 if fixed else float(self.denominator[lag]),
                    "standard_error": (
                        np.nan if fixed else self.std_errors.get(parameter, np.nan)
                    ),
                    "p_value": np.nan
                    if fixed
                    else self.p_values.get(parameter, np.nan),
                    "fixed": fixed,
                }
            )
        return pd.DataFrame(rows)

    @property
    def denominator_roots(self):
        _, denominator = _coefficient_arrays(
            self.spec,
            self.numerator,
            self.denominator,
        )
        if len(denominator) == 1:
            return np.array([], dtype=complex)
        return np.roots(denominator[::-1])

    @property
    def is_stable(self):
        roots = self.denominator_roots
        return bool(np.all(np.abs(roots) > 1.0))

    @property
    def steady_state_gain(self):
        if not self.is_stable:
            return np.nan
        denominator_at_one = 1.0 - float(sum(self.denominator.values()))
        if np.isclose(denominator_at_one, 0.0):
            return np.nan
        return float(sum(self.numerator.values()) / denominator_at_one)

    def weights(self, steps):
        steps = _normalise_nonnegative_integer(steps, "steps")
        if steps == 0:
            raise ValueError("steps must be strictly positive")
        numerator, denominator = _coefficient_arrays(
            self.spec,
            self.numerator,
            self.denominator,
        )
        impulse = np.zeros(steps)
        impulse[0] = 1.0
        values = lfilter(numerator, denominator, impulse)
        return pd.Series(values, index=pd.RangeIndex(steps, name="lag"), name=self.name)

    def filter(self, values):
        """Apply the fitted rational filter to one complete input path."""
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not len(values):
            raise ValueError("values must be a non-empty one-dimensional input path")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must contain only finite input values")
        numerator, denominator = _coefficient_arrays(
            self.spec,
            self.numerator,
            self.denominator,
        )
        return _filter_input(
            values,
            numerator,
            denominator,
            initialization=self.spec.initialization,
        )

    def gain(self, alpha=0.05):
        if (
            not isinstance(alpha, (int, float, np.integer, np.floating))
            or not 0 < alpha < 1
        ):
            raise ValueError("alpha must be between 0 and 1")
        estimate = self.steady_state_gain
        standard_error = np.nan
        if np.isfinite(estimate) and self.covariance is not None:
            denominator_at_one = 1.0 - float(sum(self.denominator.values()))
            numerator_at_one = float(sum(self.numerator.values()))
            gradient = np.zeros(len(self.covariance_names))
            positions = {name: i for i, name in enumerate(self.covariance_names)}
            for lag in self.spec.numerator_lags:
                name = _rdl_parameter_name(self.name, "numerator", lag)
                if name in positions:
                    gradient[positions[name]] = 1.0 / denominator_at_one
            for lag in self.spec.denominator_lags:
                name = _rdl_parameter_name(self.name, "denominator", lag)
                if name in positions:
                    gradient[positions[name]] = numerator_at_one / denominator_at_one**2
            variance = float(gradient @ self.covariance @ gradient)
            standard_error = float(np.sqrt(max(variance, 0.0)))

        critical = float(norm.ppf(1.0 - alpha / 2.0))
        lower = (
            np.nan
            if not np.isfinite(standard_error)
            else estimate - critical * standard_error
        )
        upper = (
            np.nan
            if not np.isfinite(standard_error)
            else estimate + critical * standard_error
        )
        return {
            "input": self.name,
            "estimate": estimate,
            "standard_error": standard_error,
            "lower": lower,
            "upper": upper,
            "stable": self.is_stable,
        }


def _lagged_values(values, lag, initialization):
    values = np.asarray(values)
    if lag == 0:
        return values.copy()
    fill = 0.0 if initialization == "zero" else values[0]
    shifted = np.full(len(values), fill, dtype=values.dtype)
    if lag < len(values):
        shifted[lag:] = values[:-lag]
    return shifted


class _RationalLagSARIMAX(StatsmodelsSARIMAX):
    """Statsmodels SARIMAX with jointly estimated rational input filters."""

    def __init__(
        self,
        endog,
        *,
        distributed_inputs,
        distributed_lags,
        enforce_distributed_lag_stability,
        **kwargs,
    ):
        input_names = tuple(distributed_lags)
        input_values = np.asarray(distributed_inputs, dtype=float)
        if input_values.ndim != 2 or input_values.shape[1] != len(input_names):
            raise ValueError("distributed_inputs must match distributed_lags")

        specs = tuple(distributed_lags[name] for name in input_names)
        self._rdl_input_names = ()
        self._rdl_specs = ()
        self._rdl_inputs = input_values
        self.enforce_distributed_lag_stability = bool(enforce_distributed_lag_stability)
        super().__init__(endog, **kwargs)

        self._base_k_params = self.k_params
        self._base_param_names = tuple(super().param_names)
        self._rdl_input_names = input_names
        self._rdl_specs = specs

        offset = self._base_k_params
        slices = []
        for spec in specs:
            numerator_slice = slice(offset, offset + len(spec.numerator_lags))
            offset = numerator_slice.stop
            denominator_slice = slice(offset, offset + len(spec.denominator_lags))
            offset = denominator_slice.stop
            slices.append((numerator_slice, denominator_slice))
        self._rdl_parameter_slices = tuple(slices)
        self.k_params = offset

    def clone(self, endog, exog=None, **kwargs):
        """Clone while preserving the private transfer-function specification."""
        kwargs.update(
            {
                "distributed_inputs": np.zeros(
                    (len(np.asarray(endog)), len(self._rdl_input_names))
                ),
                "distributed_lags": dict(
                    zip(self._rdl_input_names, self._rdl_specs, strict=True)
                ),
                "enforce_distributed_lag_stability": (
                    self.enforce_distributed_lag_stability
                ),
            }
        )
        return self._clone_from_init_kwds(endog, exog=exog, **kwargs)

    @property
    def param_names(self):
        if not self._rdl_specs:
            return list(super().param_names)
        names = list(self._base_param_names)
        for input_name, spec in zip(
            self._rdl_input_names,
            self._rdl_specs,
            strict=True,
        ):
            names.extend(spec.parameter_names(input_name))
        return names

    @property
    def start_params(self):
        base = np.asarray(super().start_params, dtype=float)
        if not self._rdl_specs:
            return base

        regressors = []
        if self._k_trend:
            regressors.append(np.asarray(self._trend_data, dtype=float))
        if self.exog is not None:
            regressors.append(np.asarray(self.exog, dtype=float))

        numerator_columns = []
        for values, spec in zip(self._rdl_inputs.T, self._rdl_specs, strict=True):
            numerator_columns.extend(
                [
                    _lagged_values(
                        values,
                        spec.delay + lag,
                        spec.initialization,
                    )
                    for lag in spec.numerator_lags
                ]
            )
        if numerator_columns:
            regressors.append(np.column_stack(numerator_columns))

        if regressors:
            design = np.column_stack(regressors)
            estimates = np.linalg.pinv(design).dot(np.asarray(self.endog).squeeze())
            numerator_start = estimates[-len(numerator_columns) :]
        else:
            numerator_start = np.zeros(len(numerator_columns))

        transfer_start = []
        numerator_position = 0
        for spec in self._rdl_specs:
            width = len(spec.numerator_lags)
            transfer_start.extend(
                numerator_start[numerator_position : numerator_position + width]
            )
            numerator_position += width
            transfer_start.extend(np.zeros(len(spec.denominator_lags)))
        return np.r_[base, transfer_start]

    def transform_params(self, unconstrained):
        unconstrained = np.asarray(unconstrained)
        if not self._rdl_specs:
            return super().transform_params(unconstrained)
        constrained = np.empty_like(unconstrained)
        constrained[: self._base_k_params] = super().transform_params(
            unconstrained[: self._base_k_params]
        )
        for _spec, (numerator_slice, denominator_slice) in zip(
            self._rdl_specs,
            self._rdl_parameter_slices,
            strict=True,
        ):
            constrained[numerator_slice] = unconstrained[numerator_slice]
            denominator = unconstrained[denominator_slice]
            if self.enforce_distributed_lag_stability and len(denominator):
                denominator = constrain_stationary_univariate(denominator)
            constrained[denominator_slice] = denominator
        return constrained

    def untransform_params(self, constrained):
        constrained = np.asarray(constrained)
        if not self._rdl_specs:
            return super().untransform_params(constrained)
        unconstrained = np.empty_like(constrained)
        unconstrained[: self._base_k_params] = super().untransform_params(
            constrained[: self._base_k_params]
        )
        for _spec, (numerator_slice, denominator_slice) in zip(
            self._rdl_specs,
            self._rdl_parameter_slices,
            strict=True,
        ):
            unconstrained[numerator_slice] = constrained[numerator_slice]
            denominator = constrained[denominator_slice]
            if self.enforce_distributed_lag_stability and len(denominator):
                denominator = unconstrain_stationary_univariate(denominator)
            unconstrained[denominator_slice] = denominator
        return unconstrained

    def _transfer_effect(self, params):
        effect = np.zeros(self.nobs, dtype=params.dtype)
        for values, spec, slices in zip(
            self._rdl_inputs.T,
            self._rdl_specs,
            self._rdl_parameter_slices,
            strict=True,
        ):
            numerator_slice, denominator_slice = slices
            numerator = dict(
                zip(
                    spec.numerator_lags,
                    params[numerator_slice],
                    strict=True,
                )
            )
            denominator = dict(
                zip(
                    spec.denominator_lags,
                    params[denominator_slice],
                    strict=True,
                )
            )
            numerator_array, denominator_array = _coefficient_arrays(
                spec,
                numerator,
                denominator,
            )
            effect += _filter_input(
                values,
                numerator_array,
                denominator_array,
                initialization=spec.initialization,
            )
        return effect

    def update(
        self,
        params,
        transformed=True,
        includes_fixed=False,
        complex_step=False,
    ):
        params = self.handle_params(
            params,
            transformed=transformed,
            includes_fixed=includes_fixed,
        )
        if not self._rdl_specs:
            return super().update(
                params,
                transformed=True,
                includes_fixed=includes_fixed,
                complex_step=complex_step,
            )

        self.ssm["obs_intercept"] = np.zeros((1, self.nobs), dtype=params.dtype)
        super().update(
            params[: self._base_k_params],
            transformed=True,
            includes_fixed=False,
            complex_step=complex_step,
        )
        base_intercept = np.asarray(self.ssm["obs_intercept"])
        if base_intercept.shape[-1] == 1:
            base_intercept = np.repeat(base_intercept, self.nobs, axis=1)
        self.ssm["obs_intercept"] = (
            base_intercept + self._transfer_effect(params)[None, :]
        )
        return params


def _make_rational_lag_results(fitted, distributed_lags):
    """Build structured per-input results from one joint fitted parameter set."""
    names = tuple(fitted.param_names)
    params = dict(zip(names, np.asarray(fitted.params), strict=True))
    std_errors = dict(zip(names, np.asarray(fitted.bse), strict=True))
    p_values = dict(zip(names, np.asarray(fitted.pvalues), strict=True))
    covariance = np.asarray(fitted.cov_params(), dtype=float)

    results = {}
    for input_name, spec in distributed_lags.items():
        numerator = {
            lag: float(params[_rdl_parameter_name(input_name, "numerator", lag)])
            for lag in spec.numerator_lags
        }
        numerator.update(dict.fromkeys(spec.fixed_numerator_lags, 0.0))
        denominator = {
            lag: float(params[_rdl_parameter_name(input_name, "denominator", lag)])
            for lag in spec.denominator_lags
        }
        denominator.update(dict.fromkeys(spec.fixed_denominator_lags, 0.0))
        relevant_names = spec.parameter_names(input_name)
        results[input_name] = RationalLagResult(
            name=input_name,
            spec=spec,
            numerator=dict(sorted(numerator.items())),
            denominator=dict(sorted(denominator.items())),
            std_errors={name: float(std_errors[name]) for name in relevant_names},
            p_values={name: float(p_values[name]) for name in relevant_names},
            covariance=covariance,
            covariance_names=names,
        )
    return results
