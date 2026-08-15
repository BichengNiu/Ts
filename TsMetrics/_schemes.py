"""Explicit time-ordered split schemes for forecast evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Ts.TsUtils._validation import validate_positive_int

from ._periods import _resolve_period, _validated_dates


def _validated_split_inputs(nobs, dates):
    """Return a positive sample size and an optional strict calendar."""
    nobs = validate_positive_int("nobs", nobs)
    if dates is None:
        return nobs, None
    return nobs, _validated_dates(dates, name="dates", expected_length=nobs)


def _nonnegative_int(name, value):
    """Return a non-negative integer after rejecting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _label(position, dates):
    """Return a calendar label or a positional integer."""
    if dates is None:
        return int(position)
    return dates[int(position)]


@dataclass(frozen=True)
class ForecastSplit:
    """One immutable training and forecast split resolved to positions."""

    split: int
    train_indices: np.ndarray
    target_indices: np.ndarray
    train_start: object
    train_end: object
    forecast_start: object
    forecast_end: object
    gap: int
    window: str

    def __post_init__(self):
        """Own read-only index arrays and validate generated metadata."""
        split = _nonnegative_int("split", self.split)
        gap = _nonnegative_int("gap", self.gap)
        train = np.array(self.train_indices, dtype=int, copy=True)
        target = np.array(self.target_indices, dtype=int, copy=True)
        if train.ndim != 1 or target.ndim != 1:
            raise ValueError("split indices must be one-dimensional")
        if train.size < 10:
            raise ValueError("training split must contain at least 10 observations")
        if target.size == 0:
            raise ValueError("target split must contain at least one observation")
        if np.any(np.diff(train) != 1) or np.any(np.diff(target) != 1):
            raise ValueError("split indices must be contiguous and increasing")
        if train[-1] >= target[0]:
            raise ValueError("target split must start strictly after training ends")
        if self.window not in {"holdout", "expanding", "rolling"}:
            raise ValueError("window must be holdout, expanding, or rolling")
        train.setflags(write=False)
        target.setflags(write=False)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "gap", gap)
        object.__setattr__(self, "train_indices", train)
        object.__setattr__(self, "target_indices", target)


@dataclass(frozen=True)
class Holdout:
    """One explicit closed training period followed by one test period.

    Parameters
    ----------
    train : tuple
        Inclusive positional or date bounds for model estimation.
    test : tuple
        Inclusive later bounds for forecast scoring.

    Examples
    --------
    >>> Holdout(train=(0, 19), test=(20, 24)).split(25)[0].gap
    0
    """

    train: tuple
    test: tuple

    def split(self, nobs, dates=None):
        """Resolve the public bounds into one immutable forecast split.

        Parameters
        ----------
        nobs : int
            Total number of observations.
        dates : sequence of datetime-like, optional
            Strict calendar aligned with the observations.

        Returns
        -------
        tuple of ForecastSplit
            The single resolved holdout split.

        Examples
        --------
        >>> Holdout((0, 9), (10, 11)).split(12)[0].target_indices.tolist()
        [10, 11]
        """
        nobs, dates = _validated_split_inputs(nobs, dates)
        data = np.empty(nobs)
        train_start, train_stop = _resolve_period("train", self.train, data, dates)
        test_start, test_stop = _resolve_period("test", self.test, data, dates)
        if train_stop - train_start < 10:
            raise ValueError("train must contain at least 10 observations")
        if test_start < train_stop:
            raise ValueError("test must start strictly later than train ends")
        return (
            ForecastSplit(
                split=0,
                train_indices=np.arange(train_start, train_stop, dtype=int),
                target_indices=np.arange(test_start, test_stop, dtype=int),
                train_start=_label(train_start, dates),
                train_end=_label(train_stop - 1, dates),
                forecast_start=_label(test_start, dates),
                forecast_end=_label(test_stop - 1, dates),
                gap=test_start - train_stop,
                window="holdout",
            ),
        )


@dataclass(frozen=True)
class RollingOrigin:
    """Regular expanding- or fixed-window historical forecast origins.

    Parameters
    ----------
    initial_window : int
        Observations available at the first training cutoff.
    horizon : int, default 1
        Forecast periods scored at each origin.
    step : int, default 1
        Distance between consecutive forecast origins.
    window : {"expanding", "rolling"}, default "expanding"
        Training-window update rule.
    window_size : int, optional
        Fixed training length when ``window="rolling"``.
    gap : int, default 0
        Unscored periods between the training cutoff and first target.

    Examples
    --------
    >>> scheme = RollingOrigin(initial_window=10, horizon=2, step=2)
    >>> len(scheme.split(16))
    3
    """

    initial_window: int
    horizon: int = 1
    step: int = 1
    window: str = "expanding"
    window_size: int | None = None
    gap: int = 0

    def __post_init__(self):
        """Normalise and validate the scheme independently of sample length."""
        initial_window = validate_positive_int(
            "initial_window",
            self.initial_window,
            minimum=10,
        )
        horizon = validate_positive_int("horizon", self.horizon)
        step = validate_positive_int("step", self.step)
        gap = _nonnegative_int("gap", self.gap)
        if self.window not in {"expanding", "rolling"}:
            raise ValueError("window must be either 'expanding' or 'rolling'")
        window_size = self.window_size
        if self.window == "expanding":
            if window_size is not None:
                raise ValueError("window_size is only valid when window='rolling'")
        elif window_size is None:
            window_size = initial_window
        else:
            window_size = validate_positive_int(
                "window_size",
                window_size,
                minimum=10,
            )
            if window_size > initial_window:
                raise ValueError("window_size must be <= initial_window")
        object.__setattr__(self, "initial_window", initial_window)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "gap", gap)
        object.__setattr__(self, "window_size", window_size)

    def split(self, nobs, dates=None):
        """Generate every complete historical forecast split.

        Parameters
        ----------
        nobs : int
            Total number of observations.
        dates : sequence of datetime-like, optional
            Strict calendar aligned with the observations.

        Returns
        -------
        tuple of ForecastSplit
            Complete splits in increasing forecast-origin order.

        Examples
        --------
        >>> splits = RollingOrigin(10, horizon=2).split(13)
        >>> splits[0].target_indices.tolist()
        [10, 11]
        """
        nobs, dates = _validated_split_inputs(nobs, dates)
        first_target = self.initial_window + self.gap
        origins = np.arange(
            first_target,
            nobs - self.horizon + 1,
            self.step,
            dtype=int,
        )
        if origins.size == 0:
            raise ValueError(
                "sample does not contain one complete forecast after the "
                "initial window and gap"
            )
        splits = []
        for split_number, origin in enumerate(origins):
            train_stop = int(origin) - self.gap
            train_start = (
                0
                if self.window == "expanding"
                else train_stop - int(self.window_size)
            )
            target_stop = int(origin) + self.horizon
            splits.append(
                ForecastSplit(
                    split=split_number,
                    train_indices=np.arange(train_start, train_stop, dtype=int),
                    target_indices=np.arange(origin, target_stop, dtype=int),
                    train_start=_label(train_start, dates),
                    train_end=_label(train_stop - 1, dates),
                    forecast_start=_label(origin, dates),
                    forecast_end=_label(target_stop - 1, dates),
                    gap=self.gap,
                    window=self.window,
                )
            )
        return tuple(splits)


__all__ = ["Holdout", "RollingOrigin"]
