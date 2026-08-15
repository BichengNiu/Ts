"""Rational distributed-lag simulation using the TsModels parameterization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
import pandas as pd

from Ts.TsModels import RationalLagResult, RationalLagSpec

from ._base import BaseSimResult
from ._sarima import simulate_sarima
from ._validation import validate_sample


def _coefficient_mapping(values, name, *, positive_lags):
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a lag-to-coefficient mapping")
    if not values and not positive_lags:
        raise ValueError("numerator must contain at least one coefficient")

    result = {}
    for lag, coefficient in values.items():
        if isinstance(lag, bool) or not isinstance(lag, Integral):
            raise TypeError(f"{name} lags must be integers")
        lag = int(lag)
        if (positive_lags and lag <= 0) or (not positive_lags and lag < 0):
            kind = "positive" if positive_lags else "non-negative"
            raise ValueError(f"{name} lags must be {kind} integers")
        if isinstance(coefficient, bool) or not isinstance(coefficient, Real):
            raise TypeError(f"{name}[{lag}] must be a real number")
        coefficient = float(coefficient)
        if not np.isfinite(coefficient):
            raise ValueError(f"{name}[{lag}] must be finite")
        result[lag] = coefficient
    return dict(sorted(result.items()))


@dataclass
class RDLInputSpec:
    """True transfer-function coefficients for one simulated input.

    Mapping keys are active polynomial lags. Missing lags through the maximum
    key are fixed at zero, matching :class:`RationalLagSpec` in ``TsModels``.

    Parameters
    ----------
    numerator : mapping of int to float
        True numerator coefficients keyed by non-negative active lag.
    denominator : mapping of int to float, optional
        True denominator coefficients keyed by positive active lag.
    delay : int, default 0
        Pure input delay applied before the numerator polynomial.
    initialization : {"zero", "steady_state"}, default "zero"
        Transfer-filter state before the returned sample.

    Attributes
    ----------
    model_spec : RationalLagSpec
        Matching immutable estimator specification.

    Examples
    --------
    >>> from Ts.TsSims import RDLInputSpec
    >>> spec = RDLInputSpec(
    ...     numerator={0: 0.8, 1: 0.2}, denominator={1: 0.5}, delay=1
    ... )
    >>> spec.model_spec.delay
    1
    >>> spec.numerator
    {0: 0.8, 1: 0.2}
    """

    numerator: Mapping[int, float]
    denominator: Mapping[int, float] = field(default_factory=dict)
    delay: int = 0
    initialization: str = "zero"

    def __post_init__(self):
        self.numerator = _coefficient_mapping(
            self.numerator,
            "numerator",
            positive_lags=False,
        )
        self.denominator = _coefficient_mapping(
            self.denominator,
            "denominator",
            positive_lags=True,
        )
        # RationalLagSpec is the canonical validator for delays, sparse lags,
        # and initialization semantics shared with the estimator.
        model_spec = self.model_spec
        self.delay = model_spec.delay
        self.initialization = model_spec.initialization

    @property
    def model_spec(self):
        """Return the matching immutable estimator specification."""
        return RationalLagSpec(
            numerator=tuple(self.numerator),
            denominator=tuple(self.denominator),
            delay=self.delay,
            initialization=self.initialization,
        )

    def _result(self, name):
        return RationalLagResult(
            name=name,
            spec=self.model_spec,
            numerator=self.numerator,
            denominator=self.denominator,
        )


@dataclass
class SimRDLResult(BaseSimResult):
    """Container for a simulated RDL response, inputs, and components.

    Parameters
    ----------
    data, residuals : numpy.ndarray
        Simulated response and SARIMA innovations.
    params : dict
        Complete simulation settings.
    exog : pandas.DataFrame
        Input paths used by the transfer filters.
    components : pandas.DataFrame
        One transfer-effect column per input plus the disturbance column.
    input_specs : dict of str to RDLInputSpec
        True coefficient specifications keyed by input name.

    Attributes
    ----------
    distributed_lags : dict of str to RationalLagSpec
        Estimator-ready specifications matching the simulated filters.

    Examples
    --------
    >>> from Ts.TsSims import RDLInputSpec, simulate_rdl
    >>> specs = {"price": RDLInputSpec({0: 0.8}, {1: 0.4})}
    >>> result = simulate_rdl(n=30, distributed_lags=specs, seed=42)
    >>> result.get_exog().columns.tolist()
    ['price']
    >>> result.get_components().columns.tolist()
    ['price', 'noise']
    """

    exog: pd.DataFrame = field(default_factory=pd.DataFrame)
    components: pd.DataFrame = field(default_factory=pd.DataFrame)
    input_specs: dict[str, RDLInputSpec] = field(default_factory=dict)

    @property
    def distributed_lags(self):
        """Return specifications ready for ``SARIMAX(distributed_lags=...)``."""
        return {name: spec.model_spec for name, spec in self.input_specs.items()}

    def get_exog(self):
        """Return a copy of the simulated or supplied input paths.

        Returns
        -------
        pandas.DataFrame
            Input paths keyed by their public names.

        Examples
        --------
        >>> from Ts.TsSims import simulate_rdl
        >>> result = simulate_rdl(n=20, seed=42)
        >>> result.get_exog().shape
        (20, 1)
        """
        return self.exog.copy()

    def get_components(self):
        """Return a copy of each transfer effect and the SARIMA disturbance.

        Returns
        -------
        pandas.DataFrame
            Transfer effects followed by the ``"noise"`` component.

        Examples
        --------
        >>> from Ts.TsSims import simulate_rdl
        >>> result = simulate_rdl(n=20, seed=42)
        >>> result.get_components().columns.tolist()
        ['x', 'noise']
        """
        return self.components.copy()

    def summary(self):
        """Return the simulation settings and true transfer coefficients.

        Returns
        -------
        str
            Noise settings, input names, coefficients, gain, and seed.

        Examples
        --------
        >>> from Ts.TsSims import simulate_rdl
        >>> result = simulate_rdl(n=20, seed=42)
        >>> "Rational Distributed-Lag" in result.summary()
        True
        """
        lines = [
            "Rational Distributed-Lag Simulation Result",
            "=" * 42,
            f"Observations      : {len(self.data)}",
            f"Inputs            : {', '.join(self.input_specs)}",
            f"Noise order       : {self.params['noise']['order']}",
            f"Noise seasonal    : {self.params['noise']['seasonal_order']}",
            f"Noise sigma2      : {self.params['noise']['sigma2']:.4f}",
            f"Seed              : {self.params.get('seed', 'N/A')}",
        ]
        for name, spec in self.input_specs.items():
            result = spec._result(name)
            lines.extend(
                [
                    f"[{name}] numerator : {spec.numerator}",
                    f"[{name}] denominator: {spec.denominator}",
                    f"[{name}] delay/init : {spec.delay}/{spec.initialization}",
                    f"[{name}] gain       : {result.steady_state_gain:.6g}",
                ]
            )
        return "\n".join(lines)


def _normalise_input_specs(distributed_lags):
    if distributed_lags is None:
        distributed_lags = {"x": RDLInputSpec(numerator={0: 1.0}, denominator={1: 0.5})}
    if not isinstance(distributed_lags, Mapping) or not distributed_lags:
        raise ValueError("distributed_lags must be a non-empty mapping")

    result = {}
    for name, spec in distributed_lags.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("distributed-lag input names must be non-empty strings")
        if not isinstance(spec, RDLInputSpec):
            raise TypeError(f"distributed_lags[{name!r}] must be an RDLInputSpec")
        result[name] = spec
    return result


def _normalise_exog(exog, n, names, rng):
    if exog is None:
        return pd.DataFrame(
            rng.standard_normal((n, len(names))),
            columns=names,
        )
    if isinstance(exog, pd.DataFrame):
        if tuple(exog.columns) != names:
            raise ValueError(f"exog columns must be exactly {names}")
        frame = exog.copy()
    else:
        array = np.asarray(exog)
        if array.ndim != 2 or array.shape != (n, len(names)):
            raise ValueError(f"exog must have shape {(n, len(names))}")
        frame = pd.DataFrame(array, columns=names)
    if len(frame) != n:
        raise ValueError(f"exog must contain exactly {n} rows")
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("exog must contain only numeric values") from error
    if not np.all(np.isfinite(values)):
        raise ValueError("exog must contain only finite values")
    return pd.DataFrame(values, index=frame.index.copy(), columns=names)


def simulate_rdl(
    n: int = 200,
    distributed_lags: Mapping[str, RDLInputSpec] | None = None,
    *,
    exog=None,
    order: tuple[int, int, int] = (0, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    ar: list[float] | None = None,
    ma: list[float] | None = None,
    seasonal_ar: list[float] | None = None,
    seasonal_ma: list[float] | None = None,
    const: float = 0.0,
    sigma2: float = 1.0,
    seed: int | None = None,
    burn: int = 100,
    enforce_stability: bool = True,
) -> SimRDLResult:
    """Simulate one or more rational input effects plus a SARIMA disturbance.

    Supplied inputs are used as-is. If ``exog`` is omitted, independent
    standard-normal input paths are generated. Transfer filters begin at the
    returned sample boundary according to each specification's initialization;
    ``burn`` applies only to the SARIMA disturbance.

    Parameters
    ----------
    n : int, default 200
        Number of returned observations.
    distributed_lags : mapping of str to RDLInputSpec, optional
        True transfer functions. The default is one stable input named ``x``.
    exog : pandas.DataFrame or array-like, optional
        Complete input paths of shape ``(n, number_of_inputs)``. Independent
        standard-normal inputs are generated when omitted.
    order : tuple of int, default (0, 0, 0)
        Non-seasonal SARIMA disturbance order.
    seasonal_order : tuple of int, default (0, 0, 0, 0)
        Seasonal SARIMA disturbance order.
    ar, ma, seasonal_ar, seasonal_ma : sequence of float, optional
        Disturbance polynomial coefficients.
    const : float, default 0.0
        Disturbance constant or drift.
    sigma2 : float, default 1.0
        Disturbance innovation variance.
    seed : int, optional
        Seed used to derive independent input and disturbance streams.
    burn : int, default 100
        Burn-in applied only to the disturbance process.
    enforce_stability : bool, default True
        Reject unstable transfer denominators.

    Returns
    -------
    SimRDLResult
        Response, input paths, component contributions, and true parameters.

    Examples
    --------
    >>> from Ts.TsSims import RDLInputSpec, simulate_rdl
    >>> specs = {"policy": RDLInputSpec({0: 1.0}, {1: 0.5}, delay=1)}
    >>> result = simulate_rdl(n=40, distributed_lags=specs, seed=42)
    >>> result.data.shape
    (40,)
    >>> result.distributed_lags["policy"].delay
    1
    """
    n, burn = validate_sample(n, burn)
    specs = _normalise_input_specs(distributed_lags)
    if not isinstance(enforce_stability, (bool, np.bool_)):
        raise TypeError("enforce_stability must be a boolean")

    seed_sequence = np.random.SeedSequence(seed)
    input_seed, noise_seed = seed_sequence.spawn(2)
    inputs = _normalise_exog(
        exog,
        n,
        tuple(specs),
        np.random.default_rng(input_seed),
    )

    noise = simulate_sarima(
        n=n,
        order=order,
        seasonal_order=seasonal_order,
        ar=ar,
        ma=ma,
        seasonal_ar=seasonal_ar,
        seasonal_ma=seasonal_ma,
        const=const,
        sigma2=sigma2,
        seed=int(noise_seed.generate_state(1, dtype=np.uint32)[0]),
        burn=burn,
    )

    effects = {}
    for name, spec in specs.items():
        result = spec._result(name)
        if enforce_stability and not result.is_stable:
            raise ValueError(f"distributed-lag denominator for {name!r} is unstable")
        effects[name] = result.filter(inputs[name].to_numpy(dtype=float))

    components = pd.DataFrame(effects, index=inputs.index.copy())
    components["noise"] = noise.data
    data = components.sum(axis=1).to_numpy(dtype=float)
    params = {
        "n": n,
        "seed": seed,
        "burn": burn,
        "distributed_lags": {
            name: {
                "numerator": dict(spec.numerator),
                "denominator": dict(spec.denominator),
                "delay": spec.delay,
                "initialization": spec.initialization,
            }
            for name, spec in specs.items()
        },
        "noise": noise.get_params(),
        "enforce_stability": bool(enforce_stability),
    }
    return SimRDLResult(
        data=data,
        residuals=noise.residuals.copy(),
        params=params,
        exog=inputs,
        components=components,
        input_specs=dict(specs),
    )
